"""State observations, resets, rewards, and scoring for single-robot soccer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

BALL_RADIUS = 0.11
GOAL_X = 1.75
GOAL_HALF_WIDTH = 0.65
GOAL_HEIGHT = 0.90
POST_RADIUS = 0.025
FIELD_GOAL_X = 3.0
FIELD_GOAL_HALF_WIDTH = 1.30
FIELD_GOAL_HEIGHT = 1.20
FIELD_HALF_WIDTH = 2.75
SPRINT_FORWARD_SPEED = 1.8


def reset_ball(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    speed_range: tuple[float, float],
    spawn_clearance: float = 0.0,
) -> None:
    ball = env.scene["ball"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = ball.data.default_root_state[env_ids].clone()
    state[:, 0].uniform_(*x_range)
    state[:, 1].uniform_(*y_range)
    if spawn_clearance:
        # Keep the ball out of the robot's starting footprint. Draw replacements
        # from a safe edge of the spawn area without a GPU rejection loop.
        inside = state[:, :2].square().sum(-1) < spawn_clearance**2
        for axis, (low, high) in enumerate((x_range, y_range)):
            if low <= -spawn_clearance:
                safe_range = (low, min(high, -spawn_clearance))
                break
            if high >= spawn_clearance:
                safe_range = (max(low, spawn_clearance), high)
                break
        else:
            raise ValueError("Ball spawn ranges need an edge outside spawn_clearance.")
        positions = state[:, axis]
        positions.copy_(
            torch.where(
                inside, torch.empty_like(positions).uniform_(*safe_range), positions
            )
        )
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 7:9].uniform_(*speed_range)
    ball.write_root_state_to_sim(state, env_ids)


def _shot_direction(ball_xy: torch.Tensor, goal_x: float = GOAL_X) -> torch.Tensor:
    goal_xy = ball_xy.new_tensor((goal_x, 0.0))
    return torch.nn.functional.normalize(goal_xy - ball_xy, dim=-1)


def _approach_target(ball_xy: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    # Stand to the left of the ball so the right foot lines up with it.
    left = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    return ball_xy - 0.34 * direction + 0.10 * left


def _field_waypoint(
    robot_xy: torch.Tensor, ball_xy: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    setup = _approach_target(ball_xy, direction)
    left = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    relative = robot_xy - ball_xy
    along = (relative * direction).sum(-1)
    across = (relative * left).sum(-1)
    # Circle around the ball when the robot is between it and the goal.
    side = torch.where(across >= 0.0, 1.0, -1.0)
    detour = ball_xy - 0.28 * direction + (0.58 * side[:, None]) * left
    blocked = (along > -0.15) & (across.abs() < 0.43)
    return torch.where(blocked[:, None], detour, setup)


def approach_command(
    env: ManagerBasedRlEnv,
    goal_x: float = GOAL_X,
    field: bool = False,
    sprint: bool = False,
) -> torch.Tensor:
    """Body-relative twist command compatible with the flat walking actor."""
    robot = env.scene["robot"].data
    ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
    direction = _shot_direction(ball_xy, goal_x)
    robot_xy = robot.root_link_pos_w[:, :2]
    target = (
        _field_waypoint(robot_xy, ball_xy, direction)
        if field
        else _approach_target(ball_xy, direction)
    )
    error = target - robot_xy
    desired_heading = direction
    if field:
        toward_waypoint = torch.nn.functional.normalize(error, dim=-1)
        blend = ((error.norm(dim=-1, keepdim=True) - 0.30) / 0.35).clamp(0.0, 1.0)
        desired_heading = torch.lerp(direction, toward_waypoint, blend)
    heading = quat_apply(robot.root_link_quat_w, robot.forward_vec_b)[:, :2]
    heading = torch.nn.functional.normalize(heading, dim=-1)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    yaw_error = torch.atan2(
        (left * desired_heading).sum(-1), (heading * desired_heading).sum(-1)
    )
    forward = (heading * error).sum(-1)
    lateral = (left * error).sum(-1)
    forward_limit = 0.8
    if sprint:
        # Open up the speed range on straight approaches with room to brake.
        runway = ((forward - 0.65) / 0.95).clamp(0.0, 1.0)
        aligned = (1.0 - yaw_error.abs() / 0.65).clamp(0.0, 1.0)
        forward_limit = 0.8 + (SPRINT_FORWARD_SPEED - 0.8) * runway * aligned
    return torch.stack(
        (
            (1.5 * forward).clamp_min(-0.5).clamp_max(forward_limit),
            (1.5 * lateral).clamp(-0.5, 0.5),
            (2.0 * yaw_error).clamp(-0.8, 0.8),
        ),
        dim=-1,
    )


def sprint_velocity_tracking(env: ManagerBasedRlEnv, goal_x: float) -> torch.Tensor:
    """Track approach speed and braking until the robot reaches the kick setup."""
    robot = env.scene["robot"].data
    ball = env.scene["ball"].data
    ball_xy = ball.root_link_pos_w[:, :2]
    setup = _approach_target(ball_xy, _shot_direction(ball_xy, goal_x))
    setup_distance = (setup - robot.root_link_pos_w[:, :2]).norm(dim=-1)
    approaching = ((setup_distance - 0.15) / 0.25).clamp(0.0, 1.0)
    stationary = torch.exp(-ball.root_link_lin_vel_w[:, :2].square().sum(-1))
    target = approach_command(env, goal_x, field=True, sprint=True)[:, :2]
    velocity = quat_apply_inverse(robot.root_link_quat_w, robot.root_link_lin_vel_w)
    error = (target - velocity[:, :2]).square().sum(-1)
    return approaching * stationary * torch.exp(-error / 0.5**2)


def soccer_state(env: ManagerBasedRlEnv, goal_x: float = GOAL_X) -> torch.Tensor:
    robot = env.scene["robot"].data
    ball = env.scene["ball"].data
    offset = ball.root_link_pos_w - robot.root_link_pos_w
    shot = _shot_direction(ball.root_link_pos_w[:, :2], goal_x)
    shot_w = torch.cat((shot, torch.zeros_like(shot[:, :1])), dim=-1)
    return torch.cat(
        (
            quat_apply_inverse(robot.root_link_quat_w, offset),
            quat_apply_inverse(robot.root_link_quat_w, ball.root_link_lin_vel_w),
            quat_apply_inverse(robot.root_link_quat_w, shot_w)[:, :2],
        ),
        dim=-1,
    )


def approach_alignment(env: ManagerBasedRlEnv, goal_x: float = GOAL_X) -> torch.Tensor:
    robot = env.scene["robot"].data
    ball = env.scene["ball"].data
    ball_xy = ball.root_link_pos_w[:, :2]
    direction = _shot_direction(ball_xy, goal_x)
    error = _approach_target(ball_xy, direction) - robot.root_link_pos_w[:, :2]
    heading = quat_apply(robot.root_link_quat_w, robot.forward_vec_b)[:, :2]
    facing = (heading * direction).sum(-1).clamp(0.0, 1.0)
    # Release the approach incentive once the ball is rolling toward the goal.
    stationary = torch.exp(-ball.root_link_lin_vel_w[:, :2].square().sum(-1))
    return torch.exp(-4.0 * error.square().sum(-1)) * facing * stationary


def approach_eval_state(
    env: ManagerBasedRlEnv, goal_x: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return kick setup status and closing speed outside the braking zone."""
    robot = env.scene["robot"].data
    ball = env.scene["ball"].data
    ball_xy = ball.root_link_pos_w[:, :2]
    direction = _shot_direction(ball_xy, goal_x)
    setup = _approach_target(ball_xy, direction)
    offset = setup - robot.root_link_pos_w[:, :2]
    distance = offset.norm(dim=-1)
    velocity = robot.root_link_lin_vel_w[:, :2]
    heading = quat_apply(robot.root_link_quat_w, robot.forward_vec_b)[:, :2]
    facing = (torch.nn.functional.normalize(heading, dim=-1) * direction).sum(-1)
    stationary = ball.root_link_lin_vel_w[:, :2].norm(dim=-1) < 0.30
    slowed = velocity.norm(dim=-1) < 0.60
    at_setup = (distance < 0.25) & (facing > 0.85) & stationary & slowed
    closing_speed = (velocity * torch.nn.functional.normalize(offset, dim=-1)).sum(-1)
    closing_speed = torch.where((distance > 0.8) & stationary, closing_speed, 0.0)
    return at_setup, closing_speed


def approach_progress(env: ManagerBasedRlEnv, goal_x: float) -> torch.Tensor:
    robot = env.scene["robot"].data
    ball = env.scene["ball"].data
    ball_xy = ball.root_link_pos_w[:, :2]
    robot_xy = robot.root_link_pos_w[:, :2]
    error = (
        _field_waypoint(robot_xy, ball_xy, _shot_direction(ball_xy, goal_x)) - robot_xy
    )
    # Fade out close to the setup position so the foot swing can take over.
    direction = error / error.norm(dim=-1, keepdim=True).clamp_min(0.5)
    stationary = torch.exp(-ball.root_link_lin_vel_w[:, :2].square().sum(-1))
    return (robot.root_link_lin_vel_w[:, :2] * direction).sum(-1).clamp(
        -1.0, 1.0
    ) * stationary


def kick_swing(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, goal_x: float = GOAL_X
) -> torch.Tensor:
    """Reward a forward foot swing near a stationary ball."""
    robot = env.scene[asset_cfg.name].data
    ball = env.scene["ball"].data
    ball_xy = ball.root_link_pos_w[:, :2]
    direction = _shot_direction(ball_xy, goal_x)
    foot_pos = robot.site_pos_w[:, asset_cfg.site_ids[0], :2]
    foot_vel = robot.site_lin_vel_w[:, asset_cfg.site_ids[0], :2]
    contact_setup = ball_xy - 0.15 * direction
    distance_sq = (foot_pos - contact_setup).square().sum(-1)
    proximity = torch.exp(-distance_sq / 0.12**2)
    forward_speed = (foot_vel * direction).sum(-1).clamp(-1.5, 1.5)
    stationary = ball.root_link_lin_vel_w[:, :2].square().sum(-1) < 0.30**2
    return proximity * forward_speed * stationary


def ball_toward_goal(env: ManagerBasedRlEnv, goal_x: float = GOAL_X) -> torch.Tensor:
    ball = env.scene["ball"].data
    direction = _shot_direction(ball.root_link_pos_w[:, :2], goal_x)
    speed = (ball.root_link_lin_vel_w[:, :2] * direction).sum(-1)
    if "goal" not in env.termination_manager.active_terms:
        return torch.zeros_like(speed)
    goal_term = env.termination_manager.get_term_cfg("goal").func
    return goal_term.foot_touched * speed.clamp(-2.0, 4.0)


def ball_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    # History catches brief impacts between policy steps.
    force = env.scene[sensor_name].data.force_history
    return (force.square().sum(dim=-1) > 1.0).any(dim=(1, 2))


class GoalCrossing(ManagerTermBase):
    """Detect the ball clearing the plane inside the physical goal opening."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self.previous = torch.zeros((env.num_envs, 3), device=env.device)
        self.valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.foot_touched = torch.zeros_like(self.valid)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.valid[env_ids] = False
        self.foot_touched[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        goal_x: float = GOAL_X,
        half_width: float = GOAL_HALF_WIDTH,
        height: float = GOAL_HEIGHT,
    ) -> torch.Tensor:
        self.foot_touched |= ball_contact(env, "foot_ball_contact")
        pos = env.scene["ball"].data.root_link_pos_w
        plane = goal_x + BALL_RADIUS
        fraction = (
            (plane - self.previous[:, 0])
            / (pos[:, 0] - self.previous[:, 0]).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        crossing = self.previous.lerp(pos, fraction[:, None])
        scored = (
            self.valid
            & self.foot_touched
            & (self.previous[:, 0] < plane)
            & (pos[:, 0] >= plane)
            & (crossing[:, 1].abs() < half_width - POST_RADIUS - BALL_RADIUS)
            & (crossing[:, 2] < height - POST_RADIUS - BALL_RADIUS)
        )
        self.previous.copy_(pos)
        self.valid.fill_(True)
        return scored


def goal_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    if "goal" not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device)
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    return env.termination_manager.get_term("goal").float() / scale


def ball_out(
    env: ManagerBasedRlEnv,
    goal_x: float = GOAL_X,
    min_x: float = -1.0,
    half_width: float = 1.6,
) -> torch.Tensor:
    x, y, z = env.scene["ball"].data.root_link_pos_w.unbind(-1)
    return (x > goal_x + 0.6) | (x < min_x) | (y.abs() > half_width) | (z < 0.0)
