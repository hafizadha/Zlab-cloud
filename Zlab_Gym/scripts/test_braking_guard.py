"""Check full actor bounds with AdamW momentum and repeated activations."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from tensordict import TensorDict
from scripts.braking_guard import BrakingGuard
from scripts.restricted_actor import LegOutputAdapter

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        activation=nn.ELU()
        self.mlp=nn.Sequential(nn.Linear(7,16),activation,nn.Linear(16,16),activation,nn.Linear(16,29))
        self.obs_normalizer=SimpleNamespace(until=0)
    def get_latent(self,obs):return obs['actor']
    def forward(self,obs,stochastic_output=False):return self.mlp(self.get_latent(obs))

class BrakingGuardTest(unittest.TestCase):
    def test_actual_actor_bounds_and_restore(self):
        torch.manual_seed(341)
        names=[f'{side}_{joint}' for side in ('left','right') for joint in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]+[f'other_{i}' for i in range(17)]
        actor=Actor();teacher=copy.deepcopy(actor)
        optimizer=torch.optim.AdamW(actor.parameters(),lr=.05,weight_decay=.1)
        adapter=LegOutputAdapter(actor,optimizer,names)
        raw=torch.randn(32,16,7);commands=torch.zeros(32,16)
        batch=TensorDict({'actor':raw.flatten(0,1)},batch_size=[512])
        with tempfile.TemporaryDirectory() as temp:
            data,anchor=Path(temp)/'data.pt',Path(temp)/'anchor.pt'
            torch.save(dict(observations=raw,commands=commands,scale=torch.full((1,29),.25),action_names=names),data)
            torch.save(dict(actor_state_dict=teacher.state_dict()),anchor)
            guard=BrakingGuard(actor,adapter,data,anchor,.01,torch.full((1,29),.25))
            for _ in range(5):
                optimizer.zero_grad();(actor(batch)-2).square().mean().backward();optimizer.step()
                guard.observe(batch,torch.ones(512,dtype=torch.bool));guard.project()
                with torch.no_grad():delta=actor(batch)-teacher(batch)
                self.assertLessEqual(float((delta[:,adapter.mask]*.25).abs().max()),.01001)
                self.assertTrue(torch.equal(delta[:,~adapter.mask],torch.zeros_like(delta[:,~adapter.mask])))
            saved=guard.save()
            second=BrakingGuard(actor,adapter,data,anchor,.01);second.restore(saved)
            self.assertTrue(torch.equal(second.recent,guard.recent))
            with self.assertRaisesRegex(ValueError,'configuration'):BrakingGuard(actor,adapter,data,anchor,.02).restore(saved)
            with self.assertRaisesRegex(ValueError,'scales'):BrakingGuard(actor,adapter,data,anchor,.01,torch.ones(1,29))
            for budget in (0.,float('nan'),float('inf')):
                with self.assertRaises(ValueError):BrakingGuard(actor,adapter,data,anchor,budget)

if __name__=='__main__':unittest.main()
