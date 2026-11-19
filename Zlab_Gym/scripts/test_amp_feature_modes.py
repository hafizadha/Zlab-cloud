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
    def save(self):return {}
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
            leg=MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion')
            self.assertEqual(leg.demo.shape,(126,90))
            self.assertEqual(full.demo.shape,(126,126))
            expected=torch.cat((torch.tensor(q[:2,leg.joint_ids]),torch.tensor(v[:2,leg.joint_ids]),torch.zeros(2,21)),1)
            torch.testing.assert_close(leg.features(),expected)
            torch.testing.assert_close(leg.demo[0,:45],expected[0])
            # Segment boundary 63->64 must not appear in the reference transitions.
            torch.testing.assert_close(leg.demo[63,:12],torch.tensor(q[64,leg.joint_ids]))
            torch.testing.assert_close(leg.pose_similarity(expected),torch.ones(2))
            saved=leg.save();MotionPriorPPO(PPO(),env,clip,feature_mode='locomotion').load(saved,None)
            with self.assertRaisesRegex(ValueError,'feature mode'):full.load(saved,None)
            bad=leg.save();bad['motion_prior']['reference_joint_names']=list(reversed(leg.reference_joint_names))
            with self.assertRaisesRegex(ValueError,'joint order'):leg.load(bad,None)
            legacy=full.save();legacy['motion_prior'].pop('feature_mode');legacy['motion_prior'].pop('reference_joint_names');full.load(legacy,None)

if __name__=='__main__':unittest.main()
