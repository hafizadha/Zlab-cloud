"""Stage 3 fast walking with uneven terrain and pushes."""

import json
import re
from dataclasses import replace
from pathlib import Path

import torch
from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains import HfWaveTerrainCfg, TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .env_cfgs import z01_std_add_head_flat_env_cfg


def apply_walking_profile(cfg):
    """Match the original actor's joint offsets and motor gains."""
    profile_path = (
        Path(__file__).resolve().parents[4]
        / "assets/robots/Z01_std_add_head/walking_motor_profile.json"
    )
    metadata = json.loads(profile_path.read_text())
    names = metadata["joint_names"]
    default_pose = dict(zip(names, map(float, metadata["default_joint_pos"]), strict=True))
    stiffness = dict(zip(names, map(float, metadata["joint_stiffness"]), strict=True))
    damping = dict(zip(names, map(float, metadata["joint_damping"]), strict=True))
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


class Stage3OnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the walking actor's input scale during PPO and checkpoint resume.
        self.alg.get_policy().obs_normalizer.until = 0


def _walking_env_cfg(play: bool):
    cfg = z01_std_add_head_flat_env_cfg(play=play)
    apply_walking_profile(cfg)
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.2, 1.8)
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


def walking_sideways_cost(env):
    """Penalize world-frame sideways velocity beyond small balance corrections."""
    forward_command = env.command_manager.get_command("twist")[:, 0]
    lateral_speed = env.scene["robot"].data.root_link_lin_vel_w[:, 1].abs()
    excess_speed = (lateral_speed - 0.04).clamp_min(0.0)
    return (forward_command > 0.1) * (excess_speed / 0.25).square().clamp_max(1.0)


def walking_heading_cost(env):
    """Keep the base facing world +X during commanded walking."""
    forward_command = env.command_manager.get_command("twist")[:, 0]
    heading = env.scene["robot"].data.heading_w
    return (forward_command > 0.1) * (1.0 - torch.exp(-(heading / 0.35).square()))


def stage3_env_cfg(play: bool = False):
    cfg = _walking_env_cfg(play)
    # Use the existing yaw-command channel to steer back toward world +X.
    command = cfg.commands["twist"]
    command.heading_command = True
    command.ranges.heading = (0.0, 0.0)
    command.rel_heading_envs = 1.0
    command.heading_control_stiffness = 1.0
    command.ranges.ang_vel_z = (-0.4, 0.4)
    cfg.rewards["walking_sideways"] = RewardTermCfg(func=walking_sideways_cost, weight=-1.0)
    cfg.rewards["walking_heading"] = RewardTermCfg(func=walking_heading_cost, weight=-1.5)
    if play:
        # Keep the forward command constant in interactive playback.
        command.ranges.lin_vel_x = (1.55, 1.55)
        command.rel_standing_envs = 0.0
    cfg.rewards["walking_stall"] = RewardTermCfg(func=walking_stall, weight=-2.0)
    cfg.rewards["walking_stride"] = RewardTermCfg(
        func=walking_stride,
        weight=8.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot"), preserve_order=True
        )},
    )
    cfg.rewards["walking_stride"].params.update(
        step_time=0.36, swing_time=0.19, max_step_length=0.60
    )
    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.1)
    cfg.events["joint_damping"].params["ranges"] = (0.05, 0.30)
    cfg.events["push_robot"] = EventTermCfg(
        func=env_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.18, 0.18), "y": (-0.18, 0.18)}},
    )

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
                # Coarser cells reduce the contacts generated per geom pair.
                horizontal_scale=0.2,
                vertical_scale=0.002,
                border_width=0.2,
            ),
        },
    )
    cfg.events["randomize_terrain"] = EventTermCfg(func=env_mdp.randomize_terrain, mode="reset")
    return cfg
