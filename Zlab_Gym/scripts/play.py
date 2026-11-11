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

TASK = "Z01-StdAddHead-Soccer-BallSprint"


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: Path
    num_envs: int = 1
    device: str | None = None
    seed: int = 2026
    eval_episodes: int = 0
    """Run this many seeded ball-approach trials headlessly."""


def run_play(cfg: PlayConfig) -> None:
    checkpoint = cfg.checkpoint_file.expanduser().resolve(strict=True)
    if cfg.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if cfg.eval_episodes < 0:
        raise ValueError("eval-episodes cannot be negative.")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    rl_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
        clip_actions=rl_cfg.clip_actions,
    )
    try:
        runner = load_runner_cls(TASK)(env, asdict(rl_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
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
  from src.tasks.velocity.config.z01_std_add_head.ball_sprint import sprint_flight, sprinting

  simulation = env.unwrapped
  robot = simulation.scene["robot"].data
  terminations = simulation.termination_manager
  device = robot.root_link_lin_vel_b.device
  per_env, remainder = divmod(episodes, env.num_envs)
  remaining = torch.full((env.num_envs,), per_env, dtype=torch.int64, device=device)
  remaining[:remainder] += 1
  peak_speed = torch.zeros(env.num_envs, device=device)
  flight_steps = torch.zeros_like(remaining)
  successes = falls = sprint_successes = 0
  total_peak_speed = torso_rate = 0.0
  sprint_samples = 0
  while remaining.any():
    forward = robot.root_link_lin_vel_b[:, 0]
    peak_speed = torch.maximum(peak_speed, forward)
    flight_steps += sprint_flight(simulation)
    fast = sprinting(simulation) & (remaining > 0)
    sprint_samples += fast.sum().item()
    torso_rate += robot.root_link_ang_vel_b[fast, :2].norm(dim=-1).rad2deg().sum().item()
    _, _, done, _ = env.step(policy(env.get_observations()))
    finished = done.bool() & (remaining > 0)
    fell = finished & terminations.get_term("fell_over")
    arrived = finished & terminations.get_term("ball_reached") & ~fell
    successes += arrived.sum().item()
    falls += fell.sum().item()
    sprint_successes += (arrived & (flight_steps >= 3) & (peak_speed >= 2.0)).sum().item()
    total_peak_speed += peak_speed[finished].sum().item()
    remaining[finished] -= 1
    peak_speed[finished] = 0.0
    flight_steps[finished] = 0
  print(f"[BALL SPRINT] Reached and stopped {successes}/{episodes}, falls {falls}/{episodes}")
  print(f"[BALL SPRINT] Successful approaches with sprint flight {sprint_successes}/{episodes}, "
        f"mean peak forward speed {total_peak_speed / episodes:.2f} m/s")
  if sprint_samples:
    print(f"[BALL SPRINT] Mean torso roll/pitch rate during fast approaches {torso_rate / sprint_samples:.1f} deg/s")


if __name__ == "__main__":
    run_play(tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS))
