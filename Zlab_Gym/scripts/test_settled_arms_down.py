import ast,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[1]
class SettledArmsTest(unittest.TestCase):
 def test_reward_targets_only_commanded_settled_upright_states(self):
  path=ROOT/'src/tasks/velocity/config/z01_std_add_head/ball_sprint.py'
  node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SettledArmsDown')
  scope={'torch':torch,'RelaxedStopArms':object};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
  obj=scope['SettledArmsDown']();q=torch.full((5,8),1.5,requires_grad=True);obj.pose_error=lambda:q
  velocity=torch.zeros(5,3);velocity[1,0]=1.;command=torch.zeros(5,3);command[2,0]=2.8;gravity=torch.tensor([[0.,0.,-1.]]*5);gravity[3,2]=-.5
  env=NS(scene={'robot':NS(data=NS(root_link_lin_vel_w=velocity,projected_gravity_b=gravity))},command_manager=NS(get_term=lambda _:NS(command=command,stop_steps=torch.zeros(5))))
  error=obj(env);self.assertGreater(float(error[0]),0);self.assertEqual(error[1:4].tolist(),[0.,0.,0.]);error.sum().backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertTrue((q.grad[0]>0).all())
  obj.pose_error=lambda:torch.zeros(5,8);self.assertEqual(obj(env).tolist(),[0.]*5)
if __name__=='__main__':unittest.main()
