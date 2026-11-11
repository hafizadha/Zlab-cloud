"""Sprint to a randomly placed ball, then brake and settle in front of it."""

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import z01_std_add_head_flat_env_cfg


BALL_RADIUS = 0.11
STOP_DISTANCE = 0.65
MAX_SPRINT_SPEED = 3.0


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


class BallSprintOnPolicyRunner(VelocityOnPolicyRunner):
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


def _ball_spec() -> mujoco.MjSpec:
    # Match the mass and contact settings of the RoboCup soccer ball.
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="soccer_ball")
    body.add_freejoint(name="ball_free_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(BALL_RADIUS, 0.0, 0.0),
        mass=0.4,
        rgba=(0.2, 0.6, 0.9, 1.0),
        condim=6,
        friction=(0.8, 0.005, 0.004),
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        priority=1,
    )
    return spec


def ball_offset(env):
    ball = env.scene["ball"].data
    robot = env.scene["robot"].data
    return ball.root_link_pos_w[:, :2] - robot.root_link_pos_w[:, :2]


@dataclass(kw_only=True)
class BallApproachCommandCfg(CommandTermCfg):
    min_speed: float = 2.1
    max_speed: float = MAX_SPRINT_SPEED

    def build(self, env):
        return BallApproachCommand(self, env)


class BallApproachCommand(CommandTerm):
    """Convert simulated ball position into the existing forward and yaw commands."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self._command[env_ids] = 0
        self._speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.min_speed, self.cfg.max_speed
        )

    def _update_command(self):
        delta = ball_offset(self._env)
        distance = delta.norm(dim=-1)
        robot = self._env.scene["robot"].data
        error = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - robot.heading_w)

        # Turn first for targets at the side or behind. Slow down to stop at the ball.
        speed = (1.2 * (distance - STOP_DISTANCE)).clamp_min(0.0).minimum(self._speed)
        self._command[:, 0] = speed * error.cos().clamp_min(0.0).square()
        self._command[:, 2] = (1.5 * error).clamp(-0.85, 0.85) * (distance > STOP_DISTANCE)


def reset_sprint_ball(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ball = env.scene["ball"]
    state = ball.data.default_root_state[env_ids].clone()
    random = torch.rand((len(env_ids), 3), device=env.device)
    distance = 4.0 + 7.0 * random[:, 0]
    # Most targets are ahead. A third can appear anywhere, including behind.
    angle_range = torch.where(random[:, 1] < 2 / 3, math.pi / 3, math.pi)
    angle = (2.0 * random[:, 2] - 1.0) * angle_range
    state[:, :2] = torch.stack((distance * angle.cos(), distance * angle.sin()), dim=-1)
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 7:] = 0.0
    ball.write_root_state_to_sim(state, env_ids)


def ball_progress(env):
    delta = ball_offset(env)
    distance = delta.norm(dim=-1)
    velocity = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    closing = (velocity * delta).sum(dim=-1) / distance.clamp_min(1e-4)
    return closing.clamp(-1.0, MAX_SPRINT_SPEED) * (distance > STOP_DISTANCE)


def ball_reached(env):
    distance = ball_offset(env).norm(dim=-1)
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
    return (distance >= 0.45) & (distance <= 0.90) & (speed < 0.25)


def arrival_bonus(env):
    terminations = env.termination_manager
    if "ball_reached" not in terminations.active_terms:
        return ball_reached(env).float()
    return (terminations.get_term("ball_reached") & ~terminations.get_term("fell_over")).float()


def sprinting(env):
    command = env.command_manager.get_command("twist")[:, 0]
    forward = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (command > 1.8) & (forward > 1.6)


def sprint_flight(env):
    air_time = env.scene["feet_ground_contact"].data.current_air_time
    newest = air_time.min(dim=1).values
    staggered = (air_time[:, 0] - air_time[:, 1]).abs() > 0.05
    return sprinting(env) * (newest > 0.0) * (newest < 0.12) * staggered


def sprint_footfall(env, asset_cfg):
    sensor = env.scene["feet_ground_contact"]
    robot = env.scene["robot"].data
    feet = robot.site_pos_w[:, asset_cfg.site_ids]
    separation = quat_apply_inverse(robot.root_link_quat_w, feet[:, 0] - feet[:, 1])[:, 0]
    lead = torch.stack((separation, -separation), dim=1)
    alternating = sensor.data.current_air_time.flip(dims=(1,)) > 0.02
    landings = sensor.compute_first_contact(env.step_dt) & alternating
    reach = ((lead - 0.12) / 0.30).clamp(0.0, 1.0)
    return sprinting(env) * (landings * reach).sum(dim=1)


def ball_approach_env_cfg(play=False):
    cfg = _walking_env_cfg(play)
    cfg.rewards["walking_stall"] = RewardTermCfg(func=walking_stall, weight=-2.0)
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)
    cfg.events["joint_damping"].params["ranges"] = (0.05, 0.22)
    cfg.events["push_robot"] = EventTermCfg(
        func=env_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12)}},
    )
    cfg.episode_length_s = 16.0
    cfg.viewer.distance = 5.0
    cfg.commands["twist"] = BallApproachCommandCfg(
        resampling_time_range=(100.0, 100.0),
        min_speed=MAX_SPRINT_SPEED if play else 2.1,
    )
    cfg.scene.entities["ball"] = EntityCfg(
        spec_fn=_ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(5.0, 0.0, BALL_RADIUS)),
    )
    cfg.events["reset_ball"] = EventTermCfg(func=reset_sprint_ball, mode="reset")
    cfg.terminations["ball_reached"] = TerminationTermCfg(func=ball_reached)
    cfg.rewards["upright"].weight = 2.5
    cfg.rewards["body_ang_vel"].weight = -0.35
    cfg.rewards.update(
        ball_progress=RewardTermCfg(func=ball_progress, weight=1.6),
        ball_arrival=RewardTermCfg(func=arrival_bonus, weight=60.0),
        sprint_flight=RewardTermCfg(func=sprint_flight, weight=1.1),
        sprint_footfall=RewardTermCfg(
            func=sprint_footfall,
            weight=7.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot"), preserve_order=True,
            )},
        ),
    )
    return cfg
