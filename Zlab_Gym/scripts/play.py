"""Play or evaluate a Z01 Stage 3 checkpoint."""
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

TASK = "Z01-StdAddHead-Velocity-RobustWalk3"


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: Path
    num_envs: int = 1
    device: str | None = None
    seed: int = 2026
    eval_run_speed: float | None = None
    """Run a 10-second speed ramp headlessly. Zero holds a stand command."""


def run_play(cfg: PlayConfig) -> None:
    checkpoint = cfg.checkpoint_file.expanduser().resolve(strict=True)
    if cfg.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if cfg.eval_run_speed is not None and not 0 <= cfg.eval_run_speed <= 1.8:
        raise ValueError("eval-run-speed must be between 0 and 1.8 m/s.")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    if cfg.eval_run_speed is not None:
        command = env_cfg.commands["twist"]
        command.resampling_time_range = (100.0, 100.0)
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.0, 0.0)
    rl_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
        clip_actions=rl_cfg.clip_actions,
    )
    try:
        runner = load_runner_cls(TASK)(env, asdict(rl_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
        policy = runner.get_inference_policy(device=device)
        if cfg.eval_run_speed is not None:
            evaluate_run(env, policy, cfg.eval_run_speed)
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            NativeMujocoViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        env.close()


@torch.inference_mode()
def evaluate_run(env: RslRlVecEnvWrapper, policy: Callable, speed: float) -> None:
  """Use one fixed command sequence and count only the first attempt per world."""
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.tasks.velocity.config.z01_std_add_head.stage3 import walking_landings

  scene = env.unwrapped.scene
  feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
  feet.resolve(scene)
  command = env.unwrapped.command_manager.get_command("twist")
  active = torch.ones(env.num_envs, dtype=torch.bool, device=command.device)
  robot = scene["robot"].data
  start_y = robot.root_link_pos_w[:, 1].clone()
  peak_drift = torch.zeros_like(start_y)
  world_lateral_speed = heading_error = 0.0
  contact = scene["feet_ground_contact"].data
  dt = env.unwrapped.step_dt
  cruise_samples = 0
  forward_speed = 0.0
  forward_error = sideways_speed = flight_samples = 0.0
  stalled_samples = backward_samples = touchdown_count = total_step_length = 0.0
  for step in range(round(10 / dt)):
    t = step * dt
    # Stand 1 s, ramp up 2 s, cruise 3 s, ramp down 2 s, stand 2 s.
    command[:, 0] = speed * min(1.0, max(0.0, (t - 1) / 2), max(0.0, (8 - t) / 2))
    if 3 <= t < 6:
      velocity = robot.root_link_lin_vel_b[active, :2]
      cruise_samples += velocity.shape[0]
      forward_speed += velocity[:, 0].sum().item()
      forward_error += (velocity[:, 0] - speed).abs().sum().item()
      sideways_speed += velocity[:, 1].abs().sum().item()
      world_lateral_speed += robot.root_link_lin_vel_w[active, 1].abs().sum().item()
      heading_error += robot.heading_w[active].abs().rad2deg().sum().item()
      flight_samples += (contact.current_contact_time[active] == 0).all(dim=1).sum().item()
      stalled_samples += (velocity[:, 0] < 0.5 * speed).sum().item()
      backward_samples += (velocity[:, 0] < -0.15).sum().item()
      landings, lead = walking_landings(env.unwrapped, feet)
      planted = landings & active[:, None]
      touchdown_count += planted.sum().item()
      total_step_length += lead[planted].sum().item()
    _, _, done, _ = env.step(policy(env.get_observations()))
    active &= ~done.bool()
    drift = (robot.root_link_pos_w[:, 1] - start_y).abs()
    peak_drift[active] = torch.maximum(peak_drift[active], drift[active])
    if not active.any():
      break
  stopped = robot.root_link_lin_vel_b[:, :2].norm(dim=1) < 0.15
  print(f"[RUN] {speed:.2f} m/s, completed {active.sum().item()}/{env.num_envs}, "
        f"completed and stopped { (active & stopped).sum().item()}/{env.num_envs}")
  if cruise_samples:
    print(f"[RUN] Surviving cruise samples: {cruise_samples}, "
          f"mean forward speed {forward_speed / cruise_samples:.3f} m/s, "
          f"forward MAE {forward_error / cruise_samples:.3f} m/s, "
          f"mean lateral speed {sideways_speed / cruise_samples:.3f} m/s, "
          f"both feet airborne {flight_samples / cruise_samples:.1%}")
    print(f"[RUN] World-frame mean lateral speed {world_lateral_speed / cruise_samples:.3f} m/s, "
          f"mean absolute heading error {heading_error / cruise_samples:.1f} deg")
  if active.any():
    print(f"[RUN] Completed trials: mean final sideways drift {drift[active].mean().item():.3f} m, "
          f"mean peak sideways drift {peak_drift[active].mean().item():.3f} m")
  if cruise_samples:
    if speed > 0:
      print(f"[RUN] Below half commanded speed {stalled_samples / cruise_samples:.1%}, "
            f"moving backward above 0.15 m/s {backward_samples / cruise_samples:.1%}, "
            f"supported forward steps {touchdown_count / (cruise_samples * dt):.2f}/s")
    if speed > 0 and touchdown_count:
      print(f"[RUN] Mean forward foot separation at landing {total_step_length / touchdown_count:.3f} m")


if __name__ == "__main__":
    run_play(tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS))
