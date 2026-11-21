import sys,tempfile,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from tensordict import TensorDict
from scripts.functional_braking_guard import FunctionalBrakingGuard
class Actor(nn.Module):
 def __init__(self):
  super().__init__();self.mlp=nn.Sequential(nn.Linear(4,8),nn.ELU(),nn.Linear(8,2));self.std_param=nn.Parameter(torch.zeros(2));self.obs_normalizer=nn.Module();self.obs_normalizer.register_buffer('mean',torch.zeros(4));self.obs_normalizer.until=100
 def forward(self,obs,stochastic_output=False):return self.mlp(obs['actor']-self.obs_normalizer.mean)
class TestGuard(unittest.TestCase):
 def test_full_forward_backtracking_replay_and_state_guards(self):
  torch.manual_seed(33);a=Actor();x=torch.randn(180,4);td=TensorDict({'actor':x},batch_size=[180])
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cp=root/'anchor.pt';cal=root/'cal.pt';torch.save({'actor_state_dict':copy.deepcopy(a.state_dict())},cp);torch.save({'observations':x.reshape(10,18,4),'commands':torch.zeros(180),'action_names':['a','b'],'scale':torch.tensor([.25,0.])},cal)
   guard=FunctionalBrakingGuard(a,cal,cp,.01,['a','b'],torch.tensor([.25,0.]));self.assertFalse(a.std_param.requires_grad);self.assertEqual(a.obs_normalizer.until,0)
   original={k:v.clone() for k,v in a.state_dict().items()};opt=torch.optim.AdamW(a.parameters(),lr=.1,weight_decay=.1)
   for _ in range(4):
    opt.zero_grad();a(td).square().mean().backward();opt.step();guard.observe(td,torch.ones(180,dtype=torch.bool));m=guard.project();self.assertLessEqual(m['amp_braking_guard_max_rad'],.01);self.assertLessEqual(guard._bound(x),.01)
   self.assertFalse(torch.equal(a.mlp[0].weight,original['mlp.0.weight']));self.assertTrue(torch.equal(a.std_param,original['std_param']))
   saved=guard.save();guard.restore(saved)
   bad=copy.deepcopy(saved);bad['identity']['budget']=.02
   with self.assertRaisesRegex(ValueError,'configuration'):guard.restore(bad)
   bad=copy.deepcopy(saved);bad['recent'][0,0]=float('nan')
   with self.assertRaisesRegex(ValueError,'replay'):guard.restore(bad)
   with torch.no_grad():a.std_param.add_(1.)
   with self.assertRaisesRegex(ValueError,'protected'):guard.project()
   a.load_state_dict(original)
   with torch.no_grad():a.mlp[0].weight[0,0]=float('nan')
   with self.assertRaisesRegex(ValueError,'restored'):guard.project()
   self.assertTrue(torch.equal(a.mlp[0].weight,original['mlp.0.weight']))
   with self.assertRaisesRegex(ValueError,'order'):FunctionalBrakingGuard(a,cal,cp,.01,['b','a'],torch.tensor([.25,0.]))
   with self.assertRaisesRegex(ValueError,'scale'):FunctionalBrakingGuard(a,cal,cp,.01,['a','b'],torch.tensor([.5,0.]))
if __name__=='__main__':unittest.main()
