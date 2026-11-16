"""Play or evaluate a Z01 ball-sprint checkpoint."""
import json
import math
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project / "src"), str(project)]

from dataclasses import asdict, dataclass
from typing import Callable

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.debug_visualizer import DebugVisualizer

import src.tasks  # noqa: F401
from src.tasks.velocity.config.z01_std_add_head import TASK_ID
from src.tasks.velocity.config.z01_std_add_head.ball_sprint import (
    DEFAULT_FOOT_FRICTION, PITCH_HALF_LENGTH, PITCH_HALF_WIDTH, PITCH_START_X, STABILITY_SPEED,
    set_foot_friction, set_observation_noise, set_push_strength,
)


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: Path
    num_envs: int = 1
    device: str | None = None
    seed: int = 2026
    eval_episodes: int = 0
    """Run this many seeded ball-approach trials headlessly."""
    disturbances: bool | None = None
    """Randomize starting pose. Interactive playback defaults to the cloud pilot profile."""
    torso_push_scale: float | None = None
    """Set impulse strength when disturbances are enabled. The cloud pilot uses zero."""
    observation_noise_scale: float = 0.0
    foot_friction_min: float = DEFAULT_FOOT_FRICTION[0]
    foot_friction_max: float = DEFAULT_FOOT_FRICTION[1]
    metrics_file: Path | None = None
    lateral_correction: bool | None = None
    """Steer back toward the ball's lane. Enabled by default in interactive playback."""
    early_braking: bool = False
    """Start braking earlier and aim for a 0.75 m clearance."""
    recovery_starts: bool = False
    """Sample starts around the version 30 braking fall."""


_PITCH_CORNERS = np.array([
    (-PITCH_HALF_LENGTH, -PITCH_HALF_WIDTH, 0.005),
    (PITCH_HALF_LENGTH, -PITCH_HALF_WIDTH, 0.005),
    (PITCH_HALF_LENGTH, PITCH_HALF_WIDTH, 0.005),
    (-PITCH_HALF_LENGTH, PITCH_HALF_WIDTH, 0.005),
])
_PITCH_LINES = (
    *zip(_PITCH_CORNERS, np.roll(_PITCH_CORNERS, -1, axis=0)),
    ((_PITCH_CORNERS[0] + _PITCH_CORNERS[1]) / 2, (_PITCH_CORNERS[2] + _PITCH_CORNERS[3]) / 2),
)


class RoboCupPlaybackEnv(ManagerBasedRlEnv):
    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        for start, end in _PITCH_LINES:
            visualizer.add_cylinder(start, end, radius=0.035, color=(0.92, 0.95, 0.92, 1.0))


def run_play(cfg: PlayConfig) -> None:
    checkpoint = cfg.checkpoint_file.expanduser().resolve(strict=True)
    if cfg.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if cfg.eval_episodes < 0:
        raise ValueError("eval-episodes cannot be negative.")
    interactive = cfg.eval_episodes == 0
    disturbances = interactive if cfg.disturbances is None else cfg.disturbances
    push_scale = cfg.torso_push_scale
    if push_scale is None:
        push_scale = 0.0 if interactive and cfg.disturbances is None else 1.0
    if not 0.0 <= push_scale <= 1.0 or not 0.0 <= cfg.observation_noise_scale <= 1.0:
        raise ValueError("Torso push and observation noise scales must be between 0 and 1.")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK_ID, play=not disturbances)
    if disturbances:
        set_push_strength(env_cfg, push_scale)
    set_observation_noise(env_cfg, cfg.observation_noise_scale)
    set_foot_friction(env_cfg, (cfg.foot_friction_min, cfg.foot_friction_max))
    env_cfg.commands["twist"].lateral_correction = (
        interactive if cfg.lateral_correction is None else cfg.lateral_correction
    )
    env_cfg.commands["twist"].early_braking = cfg.early_braking
    env_cfg.commands["twist"].recovery_start_fraction = float(cfg.recovery_starts)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    rl_cfg = load_rl_cfg(TASK_ID)
    env_type = RoboCupPlaybackEnv if interactive else ManagerBasedRlEnv
    env = RslRlVecEnvWrapper(env_type(cfg=env_cfg, device=device), clip_actions=rl_cfg.clip_actions)
    try:
        if interactive:
            # Color only the viewer's floor. The physics model keeps the cloud terrain.
            ground = env.unwrapped.sim.mj_model.geom("terrain")
            ground.matid[:] = -1
            ground.rgba[:] = (0.13, 0.34, 0.19, 1.0)
        runner = load_runner_cls(TASK_ID)(env, asdict(rl_cfg), device=device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        runner.alg.get_policy().load_state_dict(state["actor_state_dict"], strict=True)
        policy = runner.get_inference_policy(device=device)
        if cfg.eval_episodes:
            metrics = evaluate_ball_sprint(env, policy, cfg.eval_episodes)
            if cfg.metrics_file is not None:
                cfg.metrics_file.write_text(json.dumps(metrics) + "\n")
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            NativeMujocoViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        env.close()


@torch.inference_mode()
def evaluate_ball_sprint(env: RslRlVecEnvWrapper, policy: Callable, episodes: int) -> dict[str, int | float | None]:
    simulation = env.unwrapped
    robot_entity = simulation.scene["robot"]
    robot = robot_entity.data
    foot_sites, _ = robot_entity.find_sites(("left_foot", "right_foot"), preserve_order=True)
    terminations = simulation.termination_manager
    command = simulation.command_manager.get_term("twist")
    arm_pose = simulation.reward_manager.get_term_cfg("relaxed_stop_arms").func
    relaxed_arm_steps_required = math.ceil(1.0 / simulation.step_dt)
    per_env, remainder = divmod(episodes, env.num_envs)
    remaining = torch.full((env.num_envs,), per_env, dtype=torch.long, device=env.device)
    remaining[:remainder] += 1
    peak_speed = torch.zeros(env.num_envs, device=env.device)
    lane_offset = torch.zeros_like(peak_speed)
    heading_error = torch.zeros_like(peak_speed)
    crossed_feet = torch.zeros_like(remaining, dtype=torch.bool)
    longest_hold = torch.zeros_like(remaining)
    relaxed_arm_steps = torch.zeros_like(remaining)
    final_arm_error = torch.zeros_like(peak_speed)
    backward_after_entry = torch.zeros_like(crossed_feet)
    results = []
    while remaining.any():
        distance = command.ball_spawn_xy[:, 0] - simulation.scene.env_origins[:, 0] - PITCH_START_X
        episode_time = (simulation.episode_length_buf + 1) * simulation.step_dt
        longest_hold = torch.maximum(longest_hold, command.stop_steps)
        arm_error = arm_pose.pose_error().abs()
        relaxed_arms = (arm_error <= 0.30).all(dim=1)
        relaxed_arm_steps = torch.where((command.stop_steps > 0) & relaxed_arms, relaxed_arm_steps + 1, 0)
        in_final_second = command.stop_steps >= command.stop_steps_required - relaxed_arm_steps_required
        final_arm_error = torch.where(in_final_second, final_arm_error + arm_error.mean(dim=1), 0.0)
        backward_after_entry |= (longest_hold > 0) & (robot.root_link_lin_vel_w[:, 0] < -0.25)
        peak_speed = torch.maximum(peak_speed, robot.root_link_lin_vel_w[:, 0])
        lateral_position = robot.root_link_pos_w[:, 1] - simulation.scene.env_origins[:, 1]
        lane_offset = torch.maximum(lane_offset, lateral_position.abs())
        heading_error = torch.maximum(heading_error, wrap_to_pi(robot.heading_w).abs())
        left, right = robot.site_pos_w[:, foot_sites, :2].unbind(dim=1)
        delta = left - right
        yaw = robot.heading_w
        crossed_feet |= delta[:, 1] * yaw.cos() - delta[:, 0] * yaw.sin() < 0.0
        _, _, done, _ = env.step(policy(env.get_observations()))
        finished = done.bool() & (remaining > 0)
        touched = terminations.get_term("ball_touched")
        fell = terminations.get_term("fell_over")
        stopped = terminations.get_term("ball_reached") & ~fell & ~touched
        missed = terminations.get_term("missed_ball") & ~fell & ~touched & ~stopped
        timed_out = ~(stopped | fell | touched | missed)
        results.append(torch.stack((
            distance, stopped, fell, touched, missed, timed_out, peak_speed,
            lane_offset, heading_error.rad2deg(), episode_time, crossed_feet,
            longest_hold * simulation.step_dt, backward_after_entry,
            relaxed_arm_steps >= relaxed_arm_steps_required,
            final_arm_error / relaxed_arm_steps_required,
            *(command.stop_errors & timed_out[:, None]).unbind(dim=1),
        ), dim=1)[finished])
        remaining -= finished.long()
        peak_speed.masked_fill_(finished, 0.0)
        lane_offset.masked_fill_(finished, 0.0)
        heading_error.masked_fill_(finished, 0.0)
        crossed_feet.masked_fill_(finished, False)
        longest_hold.masked_fill_(finished, 0)
        relaxed_arm_steps.masked_fill_(finished, 0)
        final_arm_error.masked_fill_(finished, 0.0)
        backward_after_entry.masked_fill_(finished, False)

    (distance, stops, falls, touches, misses, timeouts, peak, lateral, heading, duration, crossed,
     hold_seconds, reversed_after_entry, relaxed_arms_at_finish, arm_error_at_finish,
     timeout_gap, timeout_lateral, timeout_yaw, timeout_speed) = (
        torch.cat(results).T.cpu()
    )
    failed = ~stops.bool()
    entered = hold_seconds > 0
    metrics = {
        "episodes": episodes,
        "stops": int(stops.sum()),
        "stops_with_relaxed_arms": int((stops.bool() & relaxed_arms_at_finish.bool()).sum()),
        "mean_final_arm_error_rad": arm_error_at_finish[stops.bool()].mean().item() if stops.any() else None,
        "falls": int(falls.sum()),
        "touches": int(touches.sum()),
        "misses": int(misses.sum()),
        "timeouts": int(timeouts.sum()),
        "trials_with_crossed_feet": int(crossed.sum()),
        "failed_before_stop_entry": int((failed & ~entered).sum()),
        "failed_after_stop_entry": int((failed & entered).sum()),
        "falls_after_stop_entry": int((falls.bool() & entered).sum()),
        "failures_with_backward_drift_after_entry": int((failed & reversed_after_entry.bool()).sum()),
        "median_longest_hold_on_failed_entries_s": (
            hold_seconds[failed & entered].median().item() if (failed & entered).any() else None
        ),
        "timeout_outside_forward_gap": int(timeout_gap.sum()),
        "timeout_outside_lateral_band": int(timeout_lateral.sum()),
        "timeout_yaw_above_limit": int(timeout_yaw.sum()),
        "timeout_speed_above_limit": int(timeout_speed.sum()),
        "lane_offset_p95_m": lateral.quantile(0.95).item(),
        "heading_error_p95_deg": heading.quantile(0.95).item(),
        "median_stop_seconds": duration[stops.bool()].median().item() if stops.any() else None,
    }
    print(f"[BALL STABILITY] Command cap {STABILITY_SPEED:g} m/s")
    print(f"[BALL STABILITY] Clean stops {int(stops.sum())}/{episodes}, falls {int(falls.sum())}/{episodes}, "
          f"ball touches {int(touches.sum())}/{episodes}, other misses {int(misses.sum())}/{episodes}, "
          f"timeouts {int(timeouts.sum())}/{episodes}")
    print(f"[BALL STABILITY] Clean stops with arms down for the final second: "
          f"{metrics['stops_with_relaxed_arms']}/{metrics['stops']}")
    print(f"[BALL STABILITY] 95th percentile of per-trial maximum "
          f"lane offset {lateral.quantile(0.95):.3f} m, heading error {heading.quantile(0.95):.1f} deg")
    print(f"[BALL STABILITY] Mean peak forward speed {peak.mean():.2f} m/s")
    if stops.any():
        print(f"[BALL STABILITY] Median time to clean stop {duration[stops.bool()].median():.2f} s")
    for lower in (5, 7, 9):
        trials = (distance >= lower) & (distance < lower + 2)
        if trials.any():
            print(f"[BALL STABILITY] {lower}–{lower + 2} m: "
                  f"{int(stops[trials].sum())}/{int(trials.sum())} clean stops")

    return metrics


if __name__ == "__main__":
    run_play(tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS))
