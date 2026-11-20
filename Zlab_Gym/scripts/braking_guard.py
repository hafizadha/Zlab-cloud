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
    def __init__(self, actor, adapter, calibration, anchor, budget, action_scale=None, projection_mode="uniform"):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Braking guard budget must be finite and positive.')
        # TF32's batch-shape-dependent rounding can invalidate cached features.
        # Guarded training uses full FP32 matmuls, including the live actor.
        if actor.mlp[-1].weight.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        self.actor, self.mask, self.head = actor, adapter.mask, actor.mlp[-1]
        if projection_mode not in ('uniform', 'phase'):
            raise ValueError('Unknown braking projection mode.')
        self.projection_mode = projection_mode
        self.budget = float(budget)
        self.identity = dict(calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                             anchor_sha256=hashlib.sha256(Path(anchor).read_bytes()).hexdigest(),
                             budget=self.budget, schema=1, projection_mode=projection_mode)
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
        if projection_mode == 'phase':
            run_raw=data['observations'].reshape(-1,data['observations'].shape[-1])
            run_raw=run_raw[data['commands'].flatten()>=2.].to(self.head.weight.device)
            if len(run_raw)<128 or not torch.isfinite(run_raw).all():
                raise ValueError('Phase projection needs 128 finite running calibration states.')
            with torch.no_grad():
                running=penultimate_features(actor,TensorDict({'actor':run_raw},batch_size=[len(run_raw)])).detach()
            # Solve the fixed calibration geometry accurately once on CPU.
            slow_cpu=self.fixed.cpu().double();running_cpu=running.cpu().double()
            mean=slow_cpu.mean(0);std=slow_cpu.std(0).clamp_min(.05)
            def augment(value):
                return torch.cat(((value-mean)/std,torch.ones(len(value),1,dtype=torch.float64)),1)
            zs,zr=augment(slow_cpu),augment(running_cpu)
            gs,gr=zs.T@zs/len(zs),zr.T@zr/len(zr)
            eye=torch.eye(len(gs),dtype=torch.float64)
            maps=[torch.linalg.solve(gr+weight*gs+.001*eye,gr) for weight in (16.,64.,256.)]
            if not all(torch.isfinite(value).all() for value in maps):
                raise ValueError('Nonfinite phase projection geometry.')
            device=self.head.weight.device
            self.phase_mean=mean.to(device=device,dtype=torch.float32)
            self.phase_std=std.to(device=device,dtype=torch.float32)
            self.phase_gram=gr.to(device=device,dtype=torch.float32)
            self.phase_maps=[value.to(device=device,dtype=torch.float32) for value in maps]

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
        chosen_weight,chosen_bias=dw[self.mask]*factor,db[self.mask]*factor
        selected=0
        chosen_factor=factor
        projection_error=0.
        if self.projection_mode == 'phase':
            proposal=torch.cat((self.phase_std[:,None]*dw[self.mask].T,
                                (self.phase_mean@dw[self.mask].T+db[self.mask])[None,:]),0)
            def error(coeff):
                delta=(coeff-proposal)*self.scale[self.mask]
                return float(((self.phase_gram@delta)*delta).sum().clamp_min(0)/len(self.scale[self.mask]))
            best_error=error(proposal*factor)
            if factor<1.:
                for index,mapping in enumerate(self.phase_maps,1):
                    coeff=mapping@proposal
                    weight=(coeff[:-1]/self.phase_std[:,None]).T
                    bias=coeff[-1]-self.phase_mean@(coeff[:-1]/self.phase_std[:,None])
                    trial_max=float(((features@weight.T+bias)*self.scale[self.mask]).abs().max())
                    trial_factor=min(1.,self.budget*.999/max(trial_max,1e-12))
                    trial_error=error(coeff*trial_factor)
                    if math.isfinite(trial_error) and trial_error<best_error:
                        best_error=trial_error
                        chosen_weight,chosen_bias=weight*trial_factor,bias*trial_factor
                        selected=index
                        chosen_factor=trial_factor
            projection_error=best_error**.5
        self.head.weight[self.mask]=self.anchor_weight[self.mask]+chosen_weight
        self.head.bias[self.mask]=self.anchor_bias[self.mask]+chosen_bias
        bound=((features@(self.head.weight-self.anchor_weight).T+self.head.bias-self.anchor_bias)*self.scale)[:,self.mask].abs().max()
        if not torch.isfinite(bound) or bound>self.budget+1e-5:
            raise RuntimeError('Braking guard failed its sampled target bound.')
        # Verify the actual actor's forward path, including repeated activations.
        expected=self.check_features@self.head.weight.T+self.head.bias
        torch.testing.assert_close(self.actor(self.check_obs,stochastic_output=False),expected,atol=2e-5,rtol=1e-5)
        metrics=dict(amp_braking_guard_max_rad=float(bound),
                     amp_braking_guard_retained_fraction=chosen_factor,
                     amp_braking_guard_states=len(features))
        if self.projection_mode == 'phase':
            metrics.update(amp_braking_guard_phase_selected=selected,
                           amp_braking_guard_running_projection_error_rad=projection_error)
        return metrics

    def save(self):
        return dict(identity=self.identity,recent=self.recent.clone())

    def restore(self, state):
        identity=dict(state['identity'])
        identity.setdefault('projection_mode','uniform')
        if identity != self.identity:
            raise ValueError('Braking guard configuration changed on resume.')
        recent=state['recent'].to(self.head.weight.device)
        if recent.ndim!=2 or recent.shape[1]!=self.fixed.shape[1] or len(recent)>4096 or not torch.isfinite(recent).all():
            raise ValueError('Invalid braking guard replay features.')
        self.recent=recent.clone()
