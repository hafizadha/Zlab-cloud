"""Bound whole-MLP mean updates on sampled slow states; not a physics guarantee."""
import copy
import hashlib
import math
from pathlib import Path
import torch
from tensordict import TensorDict

class FunctionalBrakingGuard:
    def __init__(self, actor, calibration, anchor, budget, action_names, action_scale):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Functional braking budget must be finite and positive.')
        self.actor = actor
        self.device = next(actor.parameters()).device
        if self.device.type == 'cuda':torch.backends.cuda.matmul.allow_tf32 = False
        seed = torch.load(anchor, map_location=self.device, weights_only=True)['actor_state_dict']
        self.anchor = copy.deepcopy(actor)
        self.anchor.load_state_dict(seed)
        self.anchor.eval()
        for p in self.anchor.parameters():p.requires_grad_(False)
        self.parameters = {k:p for k,p in actor.named_parameters() if k.startswith('mlp.')}
        if not self.parameters:raise ValueError('Expected existing MLP mean parameters.')
        for k,p in actor.named_parameters():p.requires_grad_(k in self.parameters)
        actor.obs_normalizer.until = 0
        self.protected = {k:v.clone() for k,v in seed.items() if not k.startswith('mlp.')}
        self.seed = {k:seed[k].clone() for k in self.parameters}
        data = torch.load(calibration, map_location='cpu', weights_only=True)
        if list(data['action_names']) != list(action_names):raise ValueError('Functional guard action order mismatch.')
        self.scale = data['scale'].reshape(-1,len(action_names))[0].to(self.device)
        runtime = torch.as_tensor(action_scale,device=self.device).reshape(-1,len(action_names))
        if not torch.isfinite(self.scale).all() or (self.scale<0).any() or not (self.scale>0).any():raise ValueError('Invalid action scale.')
        if not torch.allclose(runtime,self.scale.expand_as(runtime),rtol=0,atol=1e-7):raise ValueError('Functional guard action scale mismatch.')
        raw = data['observations'].reshape(-1,data['observations'].shape[-1])
        self.fixed = raw[data['commands'].flatten()<1.5].to(self.device)
        if len(self.fixed)<128 or not torch.isfinite(self.fixed).all():raise ValueError('Need finite slow-state calibration.')
        self.recent = self.fixed[:0].clone()
        self.budget = float(budget)
        self.identity = dict(mode='functional',schema=1,budget=self.budget,action_names=list(action_names),calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),anchor_sha256=hashlib.sha256(Path(anchor).read_bytes()).hexdigest())

    @torch.no_grad()
    def observe(self, observations, mask):
        raw = observations['actor'][mask].detach()
        if len(raw):
            if not torch.isfinite(raw).all():raise ValueError('Nonfinite functional guard observations.')
            self.recent = torch.cat((self.recent,raw),0)[-4096:].clone()

    @torch.no_grad()
    def _bound(self, raw):
        maximum = 0.
        for batch in raw.split(4096):
            td = TensorDict({'actor':batch},batch_size=[len(batch)])
            difference = (self.actor(td,stochastic_output=False)-self.anchor(td,stochastic_output=False))*self.scale
            if not torch.isfinite(difference).all():return float('inf')
            maximum = max(maximum,float(difference.abs().max()))
        return maximum

    @torch.no_grad()
    def project(self):
        for k,v in self.protected.items():
            if not torch.equal(self.actor.state_dict()[k],v):raise ValueError('Functional guard protected state changed: '+k)
        proposal = {k:p.clone() for k,p in self.parameters.items()}
        if any(not torch.isfinite(v).all() for v in proposal.values()):
            for k,p in self.parameters.items():p.copy_(self.seed[k])
            raise ValueError('Nonfinite functional guard proposal; restored anchor mean.')
        raw = torch.cat((self.fixed,self.recent),0)
        retained = 1.
        for _ in range(17):
            for k,p in self.parameters.items():p.copy_(self.seed[k]+retained*(proposal[k]-self.seed[k]))
            maximum = self._bound(raw)
            if maximum<=self.budget*.999:break
            retained *= .5
        else:
            retained = 0.
            for k,p in self.parameters.items():p.copy_(self.seed[k])
            maximum = self._bound(raw)
        if maximum>self.budget:raise RuntimeError('Functional braking bound failed after rollback.')
        return dict(amp_braking_guard_max_rad=maximum,amp_braking_guard_retained_fraction=retained,amp_braking_guard_states=len(raw))

    def save(self):return dict(identity=self.identity,recent=self.recent.clone())

    def restore(self, state):
        if state['identity']!=self.identity:raise ValueError('Functional braking guard configuration changed.')
        raw=state['recent'].to(self.device)
        if raw.ndim!=2 or raw.shape[1]!=self.fixed.shape[1] or len(raw)>4096 or not torch.isfinite(raw).all():raise ValueError('Invalid functional braking replay observations.')
        self.recent=raw.clone()
