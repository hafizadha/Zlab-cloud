"""Play or evaluate a Z01 ball-sprint checkpoint."""
import json
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project / "src"), str(project)]

from dataclasses import asdict, dataclass
from typing import Callable

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import src.tasks  # noqa: F401
from src.tasks.velocity.config.z01_std_add_head import TASK_ID
from src.tasks.velocity.config.z01_std_add_head.ball_sprint import (
    DEFAULT_FOOT_FRICTION, STABILITY_SPEED, set_foot_friction, set_observation_noise, set_push_strength,
)


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: Path
    num_envs: int = 1
    device: str | None = None
    seed: int = 2026
    eval_episodes: int = 0
    """Run this many seeded ball-approach trials headlessly."""
    disturbances: bool = False
    """Include the training torso impulses and randomized starting pose."""
    torso_push_scale: float = 1.0
    """Set impulse strength when disturbances are enabled."""
    observation_noise_scale: float = 0.0
    foot_friction_min: float = DEFAULT_FOOT_FRICTION[0]
    foot_friction_max: float = DEFAULT_FOOT_FRICTION[1]
    metrics_file: Path | None = None


def run_play(cfg: PlayConfig) -> None:
    checkpoint = cfg.checkpoint_file.expanduser().resolve(strict=True)
    if cfg.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if cfg.eval_episodes < 0:
        raise ValueError("eval-episodes cannot be negative.")
    if not 0.0 <= cfg.torso_push_scale <= 1.0 or not 0.0 <= cfg.observation_noise_scale <= 1.0:
        raise ValueError("Torso push and observation noise scales must be between 0 and 1.")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK_ID, play=not cfg.disturbances)
    if cfg.disturbances:
        set_push_strength(env_cfg, cfg.torso_push_scale)
    set_observation_noise(env_cfg, cfg.observation_noise_scale)
    set_foot_friction(env_cfg, (cfg.foot_friction_min, cfg.foot_friction_max))
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    rl_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
        clip_actions=rl_cfg.clip_actions,
    )
    try:
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
    robot = simulation.scene["robot"].data
    terminations = simulation.termination_manager
    command = simulation.command_manager.get_term("twist")
    per_env, remainder = divmod(episodes, env.num_envs)
    remaining = torch.full((env.num_envs,), per_env, dtype=torch.long, device=env.device)
    remaining[:remainder] += 1
    peak_speed = torch.zeros(env.num_envs, device=env.device)
    lane_offset = torch.zeros_like(peak_speed)
    heading_error = torch.zeros_like(peak_speed)
    results = []
    while remaining.any():
        distance = command.ball_spawn_xy[:, 0] - simulation.scene.env_origins[:, 0]
        episode_time = (simulation.episode_length_buf + 1) * simulation.step_dt
        peak_speed = torch.maximum(peak_speed, robot.root_link_lin_vel_w[:, 0])
        lateral_position = robot.root_link_pos_w[:, 1] - simulation.scene.env_origins[:, 1]
        lane_offset = torch.maximum(lane_offset, lateral_position.abs())
        heading_error = torch.maximum(heading_error, wrap_to_pi(robot.heading_w).abs())
        _, _, done, _ = env.step(policy(env.get_observations()))
        finished = done.bool() & (remaining > 0)
        touched = terminations.get_term("ball_touched")
        fell = terminations.get_term("fell_over")
        stopped = terminations.get_term("ball_reached") & ~fell & ~touched
        missed = terminations.get_term("missed_ball") & ~fell & ~touched & ~stopped
        timed_out = ~(stopped | fell | touched | missed)
        results.append(torch.stack((
            distance, stopped, fell, touched, missed, timed_out, peak_speed,
            lane_offset, heading_error.rad2deg(), episode_time,
        ), dim=1)[finished])
        remaining -= finished.long()
        peak_speed.masked_fill_(finished, 0.0)
        lane_offset.masked_fill_(finished, 0.0)
        heading_error.masked_fill_(finished, 0.0)

    distance, stops, falls, touches, misses, timeouts, peak, lateral, heading, duration = (
        torch.cat(results).T.cpu()
    )
    metrics = {
        "episodes": episodes,
        "stops": int(stops.sum()),
        "falls": int(falls.sum()),
        "touches": int(touches.sum()),
        "misses": int(misses.sum()),
        "timeouts": int(timeouts.sum()),
        "lane_offset_p95_m": lateral.quantile(0.95).item(),
        "heading_error_p95_deg": heading.quantile(0.95).item(),
        "median_stop_seconds": duration[stops.bool()].median().item() if stops.any() else None,
    }
    print(f"[BALL STABILITY] Command cap {STABILITY_SPEED:g} m/s")
    print(f"[BALL STABILITY] Clean stops {int(stops.sum())}/{episodes}, falls {int(falls.sum())}/{episodes}, "
          f"ball touches {int(touches.sum())}/{episodes}, other misses {int(misses.sum())}/{episodes}, "
          f"timeouts {int(timeouts.sum())}/{episodes}")
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
