"""Straight-line running practice for the pretrained Z01 walking actor."""

import math
from dataclasses import replace
from pathlib import Path
import re

import onnx
import torch
from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains import HfWaveTerrainCfg, TerrainGeneratorCfg
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


def walking_stride(env, asset_cfg: SceneEntityCfg, step_time: float = 0.33,
                   swing_time: float = 0.22, max_step_length: float = 0.53):
    command = env.command_manager.get_command("twist")[:, 0]
    landings, lead = walking_landings(env, asset_cfg)
    air_time = env.scene["feet_ground_contact"].data.last_air_time
    target_length = (step_time * command).clamp(0.12, max_step_length)
    length_score = torch.exp(-((lead - target_length[:, None]) / 0.13).square())
    swing_score = torch.exp(-((air_time - swing_time) / 0.09).square())
    return (command > 0.3) * (landings * length_score * swing_score).sum(dim=1)


def stride_env_cfg(play: bool = False, longer: bool = False):
    cfg = walking_env_cfg(play=play, max_speed=1.8 if longer else 1.6)
    if play:
        # Keep the forward command constant in interactive playback.
        command = cfg.commands["twist"]
        playback_speed = 1.55 if longer else 1.35
        command.ranges.lin_vel_x = (playback_speed, playback_speed)
        command.rel_standing_envs = 0.0
    cfg.rewards["walking_stall"] = RewardTermCfg(func=walking_stall, weight=-2.0)
    cfg.rewards["walking_stride"] = RewardTermCfg(
        func=walking_stride,
        weight=8.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot"), preserve_order=True
        )},
    )
    if longer:
        cfg.rewards["walking_stride"].params.update(
            step_time=0.36, swing_time=0.19, max_step_length=0.60
        )
    return cfg


def robust_stride_env_cfg(stage: int, play: bool = False):
    """Keep the stride task's actor layout while adding disturbances in stages."""
    cfg = stride_env_cfg(play=play, longer=True)
    cfg.events["foot_friction"].params["ranges"] = (
        (0.5, 0.9), (0.4, 1.0), (0.3, 1.1)
    )[stage - 1]
    cfg.events["joint_damping"].params["ranges"] = (
        (0.05, 0.15), (0.05, 0.22), (0.05, 0.30)
    )[stage - 1]
    if stage >= 2:
        push_speed = 0.12 if stage == 2 else 0.18
        cfg.events["push_robot"] = EventTermCfg(
            func=env_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 8.0),
            params={"velocity_range": {
                "x": (-push_speed, push_speed),
                "y": (-push_speed, push_speed),
            }},
        )
    if stage == 3:
        terrain = cfg.scene.terrain
        assert terrain is not None
        terrain.terrain_type = "generator"
        terrain.terrain_generator = TerrainGeneratorCfg(
            size=(80.0, 6.0),
            num_rows=2,
            num_cols=2,
            border_width=2.0,
            sub_terrains={
                "gentle_waves": HfWaveTerrainCfg(
                    amplitude_range=(0.008, 0.024),
                    num_waves=4,
                    vertical_scale=0.002,
                    border_width=0.2,
                ),
            },
        )
        cfg.events["randomize_terrain"] = EventTermCfg(func=env_mdp.randomize_terrain, mode="reset")
    return cfg


def backward_motion_cost(env):
    command = env.command_manager.get_command("twist")[:, 0]
    forward = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (command > 0.1) * (-forward - 0.15).clamp(0.0, 1.0)


def balance_recovery_env_cfg(play: bool = False):
    cfg = robust_stride_env_cfg(2, play=play)
    command = cfg.commands["twist"]
    if not play:
        command.rel_standing_envs = 0.35
    cfg.events["push_robot"].params["velocity_range"].update(
        roll=(-0.15, 0.15), pitch=(-0.15, 0.15)
    )
    cfg.rewards["upright"].weight = 2.5
    cfg.rewards["backward_motion"] = RewardTermCfg(func=backward_motion_cost, weight=-1.0)
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
