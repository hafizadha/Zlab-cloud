"""Wall budget stops after a complete PPO update and saves its final model."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
class TestBudget(unittest.TestCase):
 def make(self):
  r=OnPolicyRunner.__new__(OnPolicyRunner);r.device='cpu';r.is_distributed=False;r.current_learning_iteration=7;r.cfg={'num_steps_per_env':1,'save_interval':100,'check_for_nan':False,'algorithm':{'rnd_cfg':None}}
  r.env=NS(device='cpu',get_observations=lambda:torch.zeros(1),step=lambda a:(torch.zeros(1),torch.zeros(1),torch.zeros(1),{}))
  r.alg=NS(train_mode=Mock(),act=lambda o:torch.zeros(1),process_env_step=Mock(),compute_returns=Mock(),update=Mock(return_value={}),learning_rate=5e-6,get_policy=lambda:NS(output_std=torch.ones(1)))
  r.logger=NS(writer=object(),log_dir='/tmp',init_logging_writer=Mock(),process_env_step=Mock(),log=Mock(),stop_logging_writer=Mock());r.save=Mock();return r
 def test_budget_and_final_checkpoint(self):
  r=self.make()
  with patch('rsl_rl.runners.on_policy_runner.time.monotonic',side_effect=[0.,1.,3.]):r.learn(100,max_wall_time_s=2.)
  self.assertEqual(r.alg.update.call_count,2);self.assertEqual(r.current_learning_iteration,8);self.assertTrue(r.save.call_args.args[0].endswith('model_8.pt'));r.logger.stop_logging_writer.assert_called_once()
 def test_default_iterations_and_invalid_budget(self):
  r=self.make();r.learn(3);self.assertEqual(r.alg.update.call_count,3)
  for bad in (0.,-1.,float('nan'),float('inf')):
   with self.assertRaises(ValueError):self.make().learn(100,max_wall_time_s=bad)
if __name__=='__main__':unittest.main()
