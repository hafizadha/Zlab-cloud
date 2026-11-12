"""Sprint straight toward a ball, then brake and settle in front of it."""

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

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import _ROOT_BODY_NAME, z01_std_add_head_flat_env_cfg


BALL_RADIUS = 0.11
SPRINT_SPEED = 4.0
SPRINT_HOLD_SECONDS = 0.2
MAX_COMMAND_SPEED = 4.4


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


def _base_sprint_env_cfg(play: bool):
    cfg = z01_std_add_head_flat_env_cfg(play=play)
    apply_walking_profile(cfg)
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


@dataclass(kw_only=True)
class StraightBallSprintCommandCfg(CommandTermCfg):
    min_speed: float = 2.6
    max_speed: float = MAX_COMMAND_SPEED
    speed_ramp_steps: int = 24_000
    ball_distance_range: tuple[float, float] = (5.0, 11.0)

    def build(self, env):
        return StraightBallSprintCommand(self, env)


class StraightBallSprintCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        self.ball_spawn_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.fast_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stop_steps = torch.zeros_like(self.fast_steps)
        self.held_sprint = torch.zeros_like(self.fast_steps, dtype=torch.bool)
        self.sprint_steps_required = math.ceil(SPRINT_HOLD_SECONDS / env.step_dt)
        self.stop_steps_required = math.ceil(0.4 / env.step_dt)
        self.metrics["peak_forward_mps"] = torch.zeros_like(self._speed)
        self.metrics["target_speed_mps"] = torch.zeros_like(self._speed)
        self.metrics["sustained_4_mps"] = torch.zeros_like(self._speed)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        self._command[env_ids] = 0.0
        self.fast_steps[env_ids] = 0
        self.stop_steps[env_ids] = 0
        self.held_sprint[env_ids] = False

    def _update_command(self):
        env = self._env
        robot = env.scene["robot"].data
        ramp = min(env.common_step_counter / max(self.cfg.speed_ramp_steps, 1), 1.0)
        self._speed.fill_(self.cfg.min_speed + ramp * (self.cfg.max_speed - self.cfg.min_speed))
        forward_gap = self.ball_spawn_xy[:, 0] - robot.root_link_pos_w[:, 0]
        remaining = (forward_gap - 0.60).clamp_min(0.0)
        braking_speed = (7.0 * remaining).sqrt().minimum(1.8 * remaining)
        self._command[:, 0] = braking_speed.minimum(self._speed)

        lateral = robot.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
        lane_heading = (-0.20 * lateral).clamp(-0.30, 0.30)
        self._command[:, 2] = (1.5 * wrap_to_pi(lane_heading - robot.heading_w)).clamp(-0.6, 0.6)

    def _update_metrics(self):
        robot = self._env.scene["robot"].data
        forward = robot.root_link_lin_vel_w[:, 0]
        self.fast_steps.copy_(torch.where(forward >= SPRINT_SPEED, self.fast_steps + 1, 0))
        self.held_sprint.logical_or_(self.fast_steps >= self.sprint_steps_required)
        self.stop_steps.copy_(torch.where(self.in_stop_zone(), self.stop_steps + 1, 0))
        self.metrics["peak_forward_mps"].copy_(
            torch.maximum(self.metrics["peak_forward_mps"], forward)
        )
        self.metrics["target_speed_mps"].copy_(self._speed)
        self.metrics["sustained_4_mps"].copy_(self.held_sprint)

    def in_stop_zone(self):
        robot = self._env.scene["robot"].data
        ball_xy = self._env.scene["ball"].data.root_link_pos_w[:, :2]
        offset = ball_xy - robot.root_link_pos_w[:, :2]
        return (
            (offset[:, 0] >= 0.35)
            & (offset.norm(dim=1) <= 1.0)
            & (robot.root_link_lin_vel_w[:, :2].norm(dim=1) < 0.25)
            & (robot.projected_gravity_b[:, 2] < -0.9)
            & ~self.missed_ball()
        )

    def missed_ball(self):
        env = self._env
        ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
        robot_x = env.scene["robot"].data.root_link_pos_w[:, 0]
        return (
            (robot_x > self.ball_spawn_xy[:, 0] - 0.25)
            | ((ball_xy - self.ball_spawn_xy).norm(dim=1) > 0.15)
        )


def reset_straight_ball(env, env_ids):
    command = env.command_manager.get_term("twist")
    ball = env.scene["ball"]
    state = ball.data.default_root_state[env_ids].clone()
    distances = torch.empty(len(env_ids), device=env.device).uniform_(*command.cfg.ball_distance_range)
    state[:, 0] = distances
    state[:, 1] = 0.0
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 7:] = 0.0
    command.ball_spawn_xy[env_ids] = state[:, :2]
    ball.write_root_state_to_sim(state, env_ids)


def settled_at_ball(env):
    command = env.command_manager.get_term("twist")
    return (command.stop_steps >= command.stop_steps_required) & command.in_stop_zone()


def sprint_and_stop(env):
    command = env.command_manager.get_term("twist")
    return command.held_sprint & settled_at_ball(env)


def missed_ball(env):
    return env.command_manager.get_term("twist").missed_ball()


def sprint_success_reward(env):
    terminations = env.termination_manager
    return terminations.get_term("sprint_success") & ~terminations.get_term("fell_over")


def sprint_progress(env):
    robot = env.scene["robot"].data
    forward_command = env.command_manager.get_command("twist")[:, 0]
    return robot.root_link_lin_vel_w[:, 0].clamp_min(0.0).minimum(forward_command)


def settle_reward(env):
    command = env.command_manager.get_term("twist")
    robot = env.scene["robot"].data
    offset = command.ball_spawn_xy - robot.root_link_pos_w[:, :2]
    position_error = (offset[:, 0] - 0.65).square() + offset[:, 1].square()
    speed_error = robot.root_link_lin_vel_w[:, :2].square().sum(dim=1)
    return torch.exp(-position_error / 0.45**2 - speed_error / 0.5**2)


def arrival_bonus(env):
    terminations = env.termination_manager
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


class TorsoImpulse(env_mdp.apply_body_impulse):
    """Wait a random interval after each reset before the first push."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initial_delay_s = cfg.params["cooldown_s"]

    def reset(self, env_ids=None):
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        self._interval_time_left[indices] = torch.empty_like(
            self._interval_time_left[indices]
        ).uniform_(*self._initial_delay_s)


def ball_approach_env_cfg(play=False):
    cfg = _base_sprint_env_cfg(play)
    cfg.rewards["walking_stall"] = RewardTermCfg(func=walking_stall, weight=-2.0)
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["foot_friction"].params["ranges"] = (0.35, 1.15)
    cfg.events["joint_damping"].params["ranges"] = (0.04, 0.27)
    if not play:
        cfg.events["reset_base"].params.update(
            pose_range={"roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.06, 0.06)},
            velocity_range={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
        )
        cfg.events["torso_impulse"] = EventTermCfg(
            func=TorsoImpulse,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(_ROOT_BODY_NAME,)),
                "force_range": (-45.0, 45.0),
                "torque_range": (-3.0, 3.0),
                "duration_s": (0.08, 0.16),
                "cooldown_s": (0.45, 1.25),
                "body_point_offset": (0.0, 0.0, 0.10),
            },
        )
    cfg.episode_length_s = 20.0
    cfg.viewer.distance = 5.0
    cfg.commands["twist"] = StraightBallSprintCommandCfg(
        resampling_time_range=(100.0, 100.0),
        min_speed=MAX_COMMAND_SPEED if play else 2.6,
    )
    cfg.scene.entities["ball"] = EntityCfg(
        spec_fn=_ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(5.0, 0.0, BALL_RADIUS)),
    )
    cfg.events["reset_ball"] = EventTermCfg(func=reset_straight_ball, mode="reset")
    cfg.terminations.update(
        ball_reached=TerminationTermCfg(func=settled_at_ball),
        sprint_success=TerminationTermCfg(func=sprint_and_stop),
        missed_ball=TerminationTermCfg(func=missed_ball),
    )
    cfg.rewards["track_linear_velocity"].params["std"] = 0.65
    cfg.rewards["upright"].weight = 2.5
    cfg.rewards["body_ang_vel"].weight = -0.35
    cfg.rewards.update(
        ball_progress=RewardTermCfg(func=sprint_progress, weight=1.5),
        ball_arrival=RewardTermCfg(func=arrival_bonus, weight=300.0),
        settle_at_ball=RewardTermCfg(func=settle_reward, weight=0.8),
        sprint_success=RewardTermCfg(func=sprint_success_reward, weight=150.0),
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
