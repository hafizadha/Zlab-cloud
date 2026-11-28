import ast,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
class AlignmentRecoveryTest(unittest.TestCase):
 def test_direction_and_phase_gates(self):
  p=Path(__file__).resolve().parents[1]/'src/tasks/velocity/config/z01_std_add_head/ball_sprint.py';nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='alignment_recovery_terms'];scope={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
  pos=torch.zeros(7,3);pos[:,1]=.5;pos[5,1]=-.5;vel=torch.zeros(7,3);vel[1,1]=-.2;vel[2,1]=.2;vel[5,1]=.2
  ball=torch.zeros(7,2);ball[:,0]=.8;ball[3,0]=3.;cmd=torch.zeros(7,3);cmd[4,0]=2.8;holds=torch.zeros(7);holds[6]=1
  robot=NS(root_link_pos_w=pos,root_link_lin_vel_w=vel,projected_gravity_b=torch.tensor([[0.,0.,-1.]]*7));term=NS(ball_spawn_xy=ball,command=cmd,stop_steps=holds);env=NS(scene={'robot':NS(data=robot)},command_manager=NS(get_term=lambda _:term));idle,progress=scope['alignment_recovery_terms'](env)
  self.assertEqual(float(idle[0]),1.);self.assertGreater(float(progress[1]),0);self.assertLess(float(progress[2]),0);self.assertGreater(float(progress[5]),0)
  self.assertEqual(idle[[3,4,6]].tolist(),[0.,0.,0.]);self.assertEqual(progress[[3,4,6]].tolist(),[0.,0.,0.])
if __name__=='__main__':unittest.main()
