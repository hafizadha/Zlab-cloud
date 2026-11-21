"""Reference/policy agreement with deliberately reversed runtime joint order."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from scripts.amp_prior import MotionPriorPPO

class PPO:
    device='cpu'
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.zeros(1))
        self.optimizer=torch.optim.Adam([self.parameter],lr=2e-5)
    def save(self):return {'optimizer_state_dict':self.optimizer.state_dict()}
    def load(self,*args):return True

class FeatureModeTest(unittest.TestCase):
    def test_joint_mapping_boundaries_and_resume(self):
        names=json.loads((Path(__file__).resolve().parents[1]/'src/assets/robots/Z01_std_add_head/walking_motor_profile.json').read_text())['joint_names']
        q=np.arange(128*29,dtype=np.float32).reshape(128,29)*.001
        v=q*.1
        def find_joints(patterns,**kwargs):
            ids=[i for i in reversed(range(29)) if re.fullmatch(patterns[0],names[i])]
            return ids,[names[i] for i in ids]
        data=NS(joint_pos=torch.tensor(q[:2]),joint_vel=torch.tensor(v[:2]),
                root_link_quat_w=torch.tensor([[1.,0,0,0]]*2),root_link_pos_w=torch.zeros(2,3),
                root_link_lin_vel_b=torch.zeros(2,3),root_link_ang_vel_b=torch.zeros(2,3),
                projected_gravity_b=torch.zeros(2,3),site_pos_w=torch.zeros(2,2,3),site_lin_vel_w=torch.zeros(2,2,3))
        robot=NS(find_joints=find_joints,find_sites=lambda *a,**k:([0,1],None),data=data)
        env=NS(unwrapped=NS(scene={'robot':robot},step_dt=.02))
        with tempfile.TemporaryDirectory() as temp:
            clip=Path(temp)/'synthetic-schema-fixture.npz'
            np.savez(clip,joint_pos=q,joint_vel=v,fps=50.,clip_lengths=[64,64],body_features=np.zeros((128,21),np.float32),body_feature_schema='root-gravity-feet-v1')
            full=MotionPriorPPO(PPO(),env,clip)
            # This is the runner's real post-load LR override path.
            self.assertIs(full.optimizer,full.ppo.optimizer)
            for group in full.optimizer.param_groups:group['lr']=5e-6
            self.assertEqual(full.ppo.optimizer.param_groups[0]['lr'],5e-6)
            self.assertEqual(full.discriminator_optimizer.param_groups[0]['lr'],1e-4)
            rates=full.save()
            self.assertEqual(rates['optimizer_state_dict']['param_groups'][0]['lr'],5e-6)
            self.assertEqual(rates['motion_prior']['optimizer']['param_groups'][0]['lr'],1e-4)
            leg=MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion')
            joints=MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion_joints')
            self.assertFalse(joints.extended_features)
            self.assertEqual(joints.demo.shape,(126,48))
            self.assertEqual(leg.demo.shape,(126,90))
            self.assertEqual(full.demo.shape,(126,126))
            expected=torch.cat((torch.tensor(q[:2,leg.joint_ids]),torch.tensor(v[:2,leg.joint_ids]),torch.zeros(2,21)),1)
            torch.testing.assert_close(leg.features(),expected)
            torch.testing.assert_close(joints.features(),expected[:,:24])
            torch.testing.assert_close(joints.demo[0,:24],expected[0,:24])
            torch.testing.assert_close(leg.demo[0,:45],expected[0])
            # Segment boundary 63->64 must not appear in the reference transitions.
            torch.testing.assert_close(leg.demo[63,:12],torch.tensor(q[64,leg.joint_ids]))
            torch.testing.assert_close(leg.pose_similarity(expected),torch.ones(2))
            saved=leg.save();MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion').load(saved,None)
            with self.assertRaisesRegex(ValueError,'feature mode'):full.load(saved,None)
            with self.assertRaisesRegex(ValueError,'feature mode'):leg.load(joints.save(),None)
            MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion_joints').load(joints.save(),None)
            narrow=MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion',pose_guidance_weight=2.,pose_guidance_std=.35)
            shifted=expected[:1].repeat(2,1);shifted[:,:12]-=.2
            torch.testing.assert_close(narrow.pose_similarity(shifted),torch.full((2,),np.exp(-(.2/.35)**2)),atol=1e-6,rtol=1e-6)
            MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion',pose_guidance_weight=2.,pose_guidance_std=.35).load(narrow.save(),None)
            with self.assertRaisesRegex(ValueError,'pose guidance'):
                MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion',pose_guidance_weight=2.).load(narrow.save(),None)
            for invalid in (0.,-1.,float('nan'),float('inf')):
                with self.assertRaises(ValueError):MotionPriorPPO(PPO(),env,clip,pose_guidance_std=invalid)
            slow_judge=MotionPriorPPO(PPO(),env,clip,discriminator_learning_rate=1e-5)
            self.assertEqual(slow_judge.discriminator_optimizer.param_groups[0]['lr'],1e-5)
            slow_judge.optimizer.param_groups[0]['lr']=5e-6
            self.assertEqual(slow_judge.save()['motion_prior']['optimizer']['param_groups'][0]['lr'],1e-5)
            restored=MotionPriorPPO(PPO(),env,clip);restored.load(slow_judge.save(),None)
            self.assertEqual(restored.discriminator_optimizer.param_groups[0]['lr'],1e-5)
            for invalid in (0.,-1.,float('nan'),float('inf')):
                with self.assertRaises(ValueError):MotionPriorPPO(PPO(),env,clip,discriminator_learning_rate=invalid)
            bad=leg.save();bad['motion_prior']['reference_joint_names']=list(reversed(leg.reference_joint_names))
            with self.assertRaisesRegex(ValueError,'joint order'):leg.load(bad,None)
            legacy=full.save();legacy['motion_prior'].pop('feature_mode');legacy['motion_prior'].pop('reference_joint_names');full.load(legacy,None)

if __name__=='__main__':unittest.main()
