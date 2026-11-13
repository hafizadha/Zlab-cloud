"""Play or evaluate a Z01 ball-sprint checkpoint."""
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
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import src.tasks  # noqa: F401
from src.tasks.velocity.config.z01_std_add_head import TASK_ID
from src.tasks.velocity.config.z01_std_add_head.ball_sprint import MAX_COMMAND_SPEED, set_push_strength


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
    command_speed: float = MAX_COMMAND_SPEED
    """Set a fixed speed cap for evaluation."""


def run_play(cfg: PlayConfig) -> None:
    checkpoint = cfg.checkpoint_file.expanduser().resolve(strict=True)
    if cfg.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if cfg.eval_episodes < 0:
        raise ValueError("eval-episodes cannot be negative.")
    if not 2.6 <= cfg.command_speed <= MAX_COMMAND_SPEED:
        raise ValueError("--command-speed must be between 2.6 and 4.4 m/s.")
    if not 0.0 <= cfg.torso_push_scale <= 1.0:
        raise ValueError("--torso-push-scale must be between 0 and 1.")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK_ID, play=not cfg.disturbances)
    if cfg.disturbances:
        set_push_strength(env_cfg, cfg.torso_push_scale)
    speed_command = env_cfg.commands["twist"]
    speed_command.speed_cap = cfg.command_speed
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
            evaluate_ball_sprint(env, policy, cfg.eval_episodes)
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            NativeMujocoViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        env.close()


@torch.inference_mode()
def evaluate_ball_sprint(env: RslRlVecEnvWrapper, policy: Callable, episodes: int) -> None:
    from src.tasks.velocity.config.z01_std_add_head.ball_sprint import (
        SPRINT_HOLD_SECONDS,
        SPRINT_SPEED,
    )

    simulation = env.unwrapped
    robot = simulation.scene["robot"].data
    terminations = simulation.termination_manager
    per_env, remainder = divmod(episodes, env.num_envs)
    remaining = torch.full((env.num_envs,), per_env, dtype=torch.long, device=env.device)
    remaining[:remainder] += 1
    peak_speed = torch.zeros(env.num_envs, device=env.device)
    # Keep counters on the simulation device until all trials finish.
    totals = torch.zeros(5, device=env.device)
    while remaining.any():
        peak_speed = torch.maximum(peak_speed, robot.root_link_lin_vel_w[:, 0])
        _, _, done, _ = env.step(policy(env.get_observations()))
        finished = done.bool() & (remaining > 0)
        fell = finished & terminations.get_term("fell_over")
        stopped = finished & terminations.get_term("ball_reached") & ~fell
        sprint_stop = stopped & terminations.get_term("sprint_success")
        missed = finished & terminations.get_term("missed_ball") & ~fell & ~stopped
        totals += torch.stack((
            stopped.sum(), sprint_stop.sum(), fell.sum(), missed.sum(),
            (peak_speed * finished).sum(),
        ))
        remaining -= finished.long()
        peak_speed.masked_fill_(finished, 0.0)
    stops, passes, falls, misses, peak_sum = totals.tolist()
    timeouts = episodes - int(stops + falls + misses)
    print(f"[BALL SPRINT] Held {SPRINT_SPEED:g} m/s for {SPRINT_HOLD_SECONDS:g} s "
          f"and stopped within 1 m: {int(passes)}/{episodes}")
    print(f"[BALL SPRINT] Clean stops {int(stops)}/{episodes}, falls {int(falls)}/{episodes}, "
          f"missed balls {int(misses)}/{episodes}, timeouts {timeouts}/{episodes}")
    print(f"[BALL SPRINT] Mean peak forward speed {peak_sum / episodes:.2f} m/s")


if __name__ == "__main__":
    run_play(tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS))
