"""Bound restricted head drift on calibration and recent low-command states.

This is a sampled action constraint, not a guarantee on unseen states or physics.
"""
import hashlib
import math
from pathlib import Path

import torch
from tensordict import TensorDict
from scripts.restricted_actor import penultimate_features


class BrakingGuard:
    def __init__(self, actor, adapter, calibration, anchor, budget, action_scale=None):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Braking guard budget must be finite and positive.')
        self.actor, self.mask, self.head = actor, adapter.mask, actor.mlp[-1]
        self.budget = float(budget)
        self.identity = dict(calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                             anchor_sha256=hashlib.sha256(Path(anchor).read_bytes()).hexdigest(),
                             budget=self.budget, schema=1)
        seed = torch.load(anchor, map_location=self.head.weight.device, weights_only=True)['actor_state_dict']
        wk, bk = (f'mlp.{len(actor.mlp)-1}.weight', f'mlp.{len(actor.mlp)-1}.bias')
        for key, value in actor.state_dict().items():
            if key not in (wk, bk) and not torch.equal(value, seed[key]):
                raise ValueError('Braking guard requires the frozen anchor trunk and normalizer: '+key)
        self.anchor_weight, self.anchor_bias = seed[wk].clone(), seed[bk].clone()
        if not torch.equal(self.head.weight[~self.mask],self.anchor_weight[~self.mask]) or not torch.equal(self.head.bias[~self.mask],self.anchor_bias[~self.mask]):
            raise ValueError('Braking guard protected outputs differ from anchor.')
        data = torch.load(calibration, map_location='cpu', weights_only=True)
        if list(data['action_names']) != list(adapter.target_names):
            raise ValueError('Braking calibration action order mismatch.')
        self.scale = data['scale'].reshape(-1,len(self.mask))[0].to(self.head.weight.device)
        if not torch.isfinite(self.scale).all() or (self.scale[self.mask] <= 0).any():
            raise ValueError('Braking calibration requires finite positive selected action scales.')
        if action_scale is not None:
            runtime = torch.as_tensor(action_scale, device=self.scale.device).reshape(-1,len(self.mask))
            if not torch.allclose(runtime, self.scale.expand_as(runtime), rtol=0, atol=1e-7):
                raise ValueError('Braking calibration action scales differ from environment.')
        raw = data['observations'].reshape(-1,data['observations'].shape[-1])
        raw = raw[data['commands'].flatten()<1.5].to(self.head.weight.device)
        if len(raw)<128 or not torch.isfinite(raw).all():
            raise ValueError('Need at least 128 finite low-command calibration states.')
        obs=TensorDict({'actor':raw},batch_size=[len(raw)])
        with torch.no_grad():
            self.fixed = penultimate_features(actor,obs).detach()
        self.recent = self.fixed[:0].clone()
        self.check_obs = obs[:64].clone()
        self.check_features = self.fixed[:64]

    @torch.no_grad()
    def observe(self, obs, slow):
        if slow.any():
            features=penultimate_features(self.actor,obs[slow]).detach()
            self.recent=torch.cat((self.recent,features))[-4096:]

    @torch.no_grad()
    def project(self):
        features=torch.cat((self.fixed,self.recent))
        dw=self.head.weight-self.anchor_weight
        db=self.head.bias-self.anchor_bias
        drift=(features@dw.T+db)*self.scale
        maximum=float(drift[:,self.mask].abs().max())
        factor=min(1.,self.budget*.999/max(maximum,1e-12))
        self.head.weight[self.mask]=self.anchor_weight[self.mask]+dw[self.mask]*factor
        self.head.bias[self.mask]=self.anchor_bias[self.mask]+db[self.mask]*factor
        bound=((features@(self.head.weight-self.anchor_weight).T+self.head.bias-self.anchor_bias)*self.scale)[:,self.mask].abs().max()
        if not torch.isfinite(bound) or bound>self.budget+1e-5:
            raise RuntimeError('Braking guard failed its sampled target bound.')
        # Verify the actual actor's forward path, including repeated activations.
        expected=self.check_features@self.head.weight.T+self.head.bias
        torch.testing.assert_close(self.actor(self.check_obs,stochastic_output=False),expected,atol=2e-5,rtol=1e-5)
        return dict(amp_braking_guard_max_rad=float(bound),amp_braking_guard_retained_fraction=factor,
                    amp_braking_guard_states=len(features))

    def save(self):
        return dict(identity=self.identity,recent=self.recent.clone())

    def restore(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Braking guard configuration changed on resume.')
        recent=state['recent'].to(self.head.weight.device)
        if recent.ndim!=2 or recent.shape[1]!=self.fixed.shape[1] or len(recent)>4096 or not torch.isfinite(recent).all():
            raise ValueError('Invalid braking guard replay features.')
        self.recent=recent.clone()
