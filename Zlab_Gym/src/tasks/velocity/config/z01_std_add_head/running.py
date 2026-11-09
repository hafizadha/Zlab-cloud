"""Straight-line running practice for the pretrained Z01 walking actor."""

import math
from dataclasses import replace
from pathlib import Path
import re

import onnx
import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.velocity.mdp.rewards import feet_gait

from .env_cfgs import z01_std_add_head_flat_env_cfg

GAIT_PERIOD = 0.50
RUN_THRESHOLD = 1.10


def apply_walking_profile(cfg):
    """Match the original actor's joint offsets and motor gains."""
    policy_path = (
        Path(__file__).resolve().parents[4]
        / "assets/motions/z01_std_add_head/demo_act/z01_velocity_flat_base.onnx"
    )
    metadata = {item.key: item.value for item in onnx.load(policy_path).metadata_props}
    names = metadata["joint_names"].split(",")
    default_pose = dict(zip(names, map(float, metadata["default_joint_pos"].split(",")), strict=True))
    stiffness = dict(zip(names, map(float, metadata["joint_stiffness"].split(",")), strict=True))
    damping = dict(zip(names, map(float, metadata["joint_damping"].split(",")), strict=True))
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(robot.init_state, joint_pos=default_pose, pos=(0.0, 0.0, 0.85))
    actuators = []
    for actuator in robot.articulation.actuators:
        base = actuator.base_cfg
        for name in names:
            if any(re.fullmatch(pattern, name) for pattern in base.target_names_expr):
                actuators.append(replace(
                    actuator,
                    base_cfg=replace(base, target_names_expr=(name,),
                                     stiffness=stiffness[name], damping=damping[name]),
                ))
    robot.articulation = replace(robot.articulation, actuators=tuple(actuators))


class RunningOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the walking actor's input scale during PPO and checkpoint resume.
        self.alg.get_policy().obs_normalizer.until = 0


def walking_env_cfg(play: bool = False, max_speed: float = 0.8):
    cfg = z01_std_add_head_flat_env_cfg(play=play)
    apply_walking_profile(cfg)
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.2, max_speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.20
    command.init_velocity_prob = 0.0
    command.resampling_time_range = (3.0, 5.0)
    cfg.curriculum.pop("command_vel", None)
    cfg.events["reset_base"].params.update(pose_range={}, velocity_range={})
    cfg.events["reset_robot_joints"].params.update(
        position_range=(-0.03, 0.03), velocity_range=(-0.05, 0.05)
    )
    cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = False
    cfg.rewards.pop("foot_gait", None)
    return cfg


def gait_clock(env):
    phase = env.episode_length_buf * (2 * math.pi * env.step_dt / GAIT_PERIOD)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def walking_stall(env):
    command = env.command_manager.get_command("twist")[:, 0]
    forward = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    minimum_speed = (0.5 * command).clamp_min(0.1)
    return (command > 0.1) * (1.0 - forward / minimum_speed).clamp(0.0, 1.0)


def walking_landings(env, asset_cfg: SceneEntityCfg):
    """Measure how far a newly planted foot leads the supporting foot."""
    sensor = env.scene["feet_ground_contact"]
    robot = env.scene[asset_cfg.name].data
    feet = robot.site_pos_w[:, asset_cfg.site_ids]
    separation = quat_apply_inverse(robot.root_link_quat_w, feet[:, 0] - feet[:, 1])[:, 0]
    lead = torch.stack((separation, -separation), dim=1)
    supporting = sensor.data.current_contact_time.flip(dims=(1,)) > env.step_dt
    return sensor.compute_first_contact(env.step_dt) & supporting & (lead > 0), lead


def walking_stride(env, asset_cfg: SceneEntityCfg):
    command = env.command_manager.get_command("twist")[:, 0]
    landings, lead = walking_landings(env, asset_cfg)
    air_time = env.scene["feet_ground_contact"].data.last_air_time
    target_length = (0.33 * command).clamp(0.12, 0.53)
    length_score = torch.exp(-((lead - target_length[:, None]) / 0.13).square())
    swing_score = torch.exp(-((air_time - 0.22) / 0.09).square())
    return (command > 0.3) * (landings * length_score * swing_score).sum(dim=1)


def stride_env_cfg(play: bool = False):
    cfg = walking_env_cfg(play=play, max_speed=1.6)
    if play:
        # Keep the forward command constant in interactive playback.
        command = cfg.commands["twist"]
        command.ranges.lin_vel_x = (1.35, 1.35)
        command.rel_standing_envs = 0.0
    cfg.rewards["walking_stall"] = RewardTermCfg(func=walking_stall, weight=-2.0)
    cfg.rewards["walking_stride"] = RewardTermCfg(
        func=walking_stride,
        weight=8.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot"), preserve_order=True
        )},
    )
    return cfg


def counter_swing(env, asset_cfg: SceneEntityCfg):
    joints = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    left_arm, right_arm, left_hip, right_hip = joints.unbind(dim=-1)
    target = (0.35 * (right_hip - left_hip)).clamp(-0.5, 0.5)
    error = (left_arm - right_arm - target).square()
    running = env.command_manager.get_command("twist")[:, 0] > RUN_THRESHOLD
    return running * torch.exp(-error / 0.25**2)


def running_env_cfg(play: bool = False):
    cfg = z01_std_add_head_flat_env_cfg(play=play)
    apply_walking_profile(cfg)
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.4, 1.35)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.20
    command.init_velocity_prob = 0.0
    command.resampling_time_range = (3.0, 5.0)
    cfg.curriculum.pop("command_vel", None)
    cfg.events["reset_base"].params.update(pose_range={}, velocity_range={})
    cfg.events["reset_robot_joints"].params.update(
        position_range=(-0.03, 0.03), velocity_range=(-0.05, 0.05)
    )

    for group in cfg.observations.values():
        group.terms["gait_clock"] = ObservationTermCfg(func=gait_clock)
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=feet_gait,
        weight=1.5,
        params={
            "period": GAIT_PERIOD,
            "offset": [0.0, 0.5],
            "threshold": 0.43,
            "command_threshold": RUN_THRESHOLD,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards.pop("arm_symmetry", None)
    cfg.rewards["counter_swing"] = RewardTermCfg(
        func=counter_swing,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=("left_shoulder_pitch", "right_shoulder_pitch",
                             "left_hip_pitch", "right_hip_pitch"),
                preserve_order=True,
            ),
        },
    )
    return cfg
