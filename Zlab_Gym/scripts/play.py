
"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import tyro

# Prepend the repo's `src/` dir to sys.path BEFORE importing mjlab/rsl_rl so the
# local, patched `rsl_rl` (under src/rsl_rl) takes precedence over the copy
# installed in site-packages (rsl-rl-lib 5.0.1). The installed copy lacks the
# entropy-annealing / std-clamp fixes shipped in src/rsl_rl.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "walking", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  eval_episodes: int = 0
  """Run this many soccer episodes headlessly and report goals, falls, and setup time."""
  eval_run_speed: float | None = None
  """Evaluate stand, accelerate, cruise, and stop at this peak speed in m/s."""
  seed: int | None = None
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  no_motor_delay: bool = False
  """Bypass simulated motor command delays for interactive playback."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.no_motor_delay:
    from dataclasses import replace

    from mjlab.actuator import DelayedActuatorCfg

    robot = env_cfg.scene.entities["robot"]
    if robot.articulation is None:
      raise ValueError("--no-motor-delay requires a robot with actuators.")
    actuators = tuple(
      actuator.base_cfg if isinstance(actuator, DelayedActuatorCfg) else actuator
      for actuator in robot.articulation.actuators
    )
    robot.articulation = replace(robot.articulation, actuators=actuators)
    print("[INFO] Motor command delays bypassed for playback.")

  soccer_task = task_id.startswith("Z01-StdAddHead-Soccer-")
  running_task = task_id == "Z01-StdAddHead-Velocity-Run"
  walking_task = task_id in (
    "Z01-StdAddHead-Velocity-Walk",
    "Z01-StdAddHead-Velocity-FastWalk",
    "Z01-StdAddHead-Velocity-StrideWalk",
  )
  if cfg.eval_episodes < 0 or (
    cfg.eval_episodes and (not soccer_task or cfg.no_terminations)
  ):
    raise ValueError("--eval-episodes requires a soccer task with terminations enabled.")
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed
  if cfg.eval_run_speed is not None:
    if not (running_task or walking_task) or not 0 < cfg.eval_run_speed <= 3 or cfg.no_terminations:
      raise ValueError("--eval-run-speed requires a walking or running task, a speed in (0, 3], and terminations.")
    command = env_cfg.commands["twist"]
    command.resampling_time_range = (100.0, 100.0)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = cfg.agent == "trained"
  if cfg.agent == "walking" and not (soccer_task or running_task):
    raise ValueError("--agent walking requires a Z01 soccer or running task.")

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # A local motion file must be provided for tracking tasks
    # (works for both dummy and trained modes).
    if cfg.motion_file is None or not Path(cfg.motion_file).exists():
      raise FileNotFoundError(
        "Tracking tasks require `--motion-file /path/to/motion.npz`."
      )
    print(f"[INFO]: Using local motion file: {cfg.motion_file}")
    motion_cmd.motion_file = cfg.motion_file

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is None:
      raise ValueError(
        "`--checkpoint-file` is required when `--agent` is 'trained'. "
        "(WandB checkpoint download is not available in this base framework.)"
      )
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and not TRAINED_MODE:
    print(
      "[WARN] Video recording requires a checkpoint and its log directory."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    if cfg.agent == "walking":
      from src.tasks.soccer.walking import initialize_walking_actor

      initialize_walking_actor(runner.alg.get_policy(), extra_inputs=2 if running_task else 8)
    else:
      runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
      )
    policy = runner.get_inference_policy(device=device)

  if cfg.eval_episodes:
    from src.tasks.soccer.mdp import FIELD_GOAL_X, GOAL_X

    goal_x = GOAL_X if task_id.endswith("Soccer-Kick") else FIELD_GOAL_X
    evaluate_soccer(env, policy, cfg.eval_episodes, goal_x)
    env.close()
    return

  if cfg.eval_run_speed is not None:
    evaluate_run(env, policy, cfg.eval_run_speed)
    env.close()
    return

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


@torch.inference_mode()
def evaluate_run(env: RslRlVecEnvWrapper, policy: Callable, speed: float) -> None:
  """Use one fixed command sequence and count only the first attempt per world."""
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.tasks.velocity.config.z01_std_add_head.running import walking_landings

  scene = env.unwrapped.scene
  feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
  feet.resolve(scene)
  command = env.unwrapped.command_manager.get_command("twist")
  active = torch.ones(env.num_envs, dtype=torch.bool, device=command.device)
  robot = scene["robot"].data
  contact = scene["feet_ground_contact"].data
  dt = env.unwrapped.step_dt
  cruise_samples = 0
  forward_error = sideways_speed = flight_samples = 0.0
  stalled_samples = touchdown_count = total_step_length = 0.0
  for step in range(round(10 / dt)):
    t = step * dt
    # Stand 1 s, ramp up 2 s, cruise 3 s, ramp down 2 s, stand 2 s.
    command[:, 0] = speed * min(1.0, max(0.0, (t - 1) / 2), max(0.0, (8 - t) / 2))
    if 3 <= t < 6:
      velocity = robot.root_link_lin_vel_b[active, :2]
      cruise_samples += velocity.shape[0]
      forward_error += (velocity[:, 0] - speed).abs().sum().item()
      sideways_speed += velocity[:, 1].abs().sum().item()
      flight_samples += (contact.current_contact_time[active] == 0).all(dim=1).sum().item()
      stalled_samples += (velocity[:, 0] < 0.5 * speed).sum().item()
      landings, lead = walking_landings(env.unwrapped, feet)
      planted = landings & active[:, None]
      touchdown_count += planted.sum().item()
      total_step_length += lead[planted].sum().item()
    _, _, done, _ = env.step(policy(env.get_observations()))
    active &= ~done.bool()
    if not active.any():
      break
  stopped = robot.root_link_lin_vel_b[:, :2].norm(dim=1) < 0.15
  print(f"[RUN] {speed:.2f} m/s, completed {active.sum().item()}/{env.num_envs}, "
        f"completed and stopped { (active & stopped).sum().item()}/{env.num_envs}")
  if cruise_samples:
    print(f"[RUN] Surviving cruise samples: {cruise_samples}, "
          f"forward MAE {forward_error / cruise_samples:.3f} m/s, "
          f"mean lateral speed {sideways_speed / cruise_samples:.3f} m/s, "
          f"both feet airborne {flight_samples / cruise_samples:.1%}")
    print(f"[RUN] Below half commanded speed {stalled_samples / cruise_samples:.1%}, "
          f"supported forward steps {touchdown_count / (cruise_samples * dt):.2f}/s")
    if touchdown_count:
      print(f"[RUN] Mean forward foot separation at landing {total_step_length / touchdown_count:.3f} m")


@torch.inference_mode()
def evaluate_soccer(
  env: RslRlVecEnvWrapper, policy: Callable, episodes: int, goal_x: float
) -> None:
  from src.tasks.soccer.mdp import approach_eval_state

  observations = env.get_observations()
  terminations = env.unwrapped.termination_manager
  completed = goals = falls = 0
  reached = 0
  setup_seconds = peak_speed_sum = 0.0
  elapsed = torch.zeros(env.num_envs, device=env.unwrapped.device)
  setup_time = torch.full_like(elapsed, -1.0)
  peak_speed = torch.zeros_like(elapsed)
  per_world, remainder = divmod(episodes, env.num_envs)
  remaining = torch.full((env.num_envs,), per_world, device=elapsed.device, dtype=torch.long)
  remaining[:remainder] += 1
  while completed < episodes:
    at_setup, closing_speed = approach_eval_state(env.unwrapped, goal_x)
    peak_speed = torch.maximum(peak_speed, closing_speed)
    setup_time = torch.where(at_setup & (setup_time < 0), elapsed, setup_time)
    observations, _, dones, _ = env.step(policy(observations))
    done = dones.bool()
    # Give each world its assigned share so short episodes cannot dominate.
    finished = (done & (remaining > 0)).nonzero(as_tuple=True)[0]
    if finished.numel():
      remaining[finished] -= 1
      goals += terminations.get_term("goal")[finished].sum().item()
      falls += terminations.get_term("fell_over")[finished].sum().item()
      times = setup_time[finished]
      reached += (times >= 0).sum().item()
      setup_seconds += times.clamp_min(0).sum().item()
      peak_speed_sum += peak_speed[finished].sum().item()
      completed += finished.numel()
    elapsed = torch.where(done, 0.0, elapsed + env.unwrapped.step_dt)
    setup_time = torch.where(done, -1.0, setup_time)
    peak_speed = torch.where(done, 0.0, peak_speed)
  print(f"[SOCCER] {completed} episodes, goals {goals} ({goals / completed:.1%}), "
        f"falls {falls} ({falls / completed:.1%})")
  print(f"[SOCCER] Reached kick setup in {reached}/{completed} episodes"
        + (f", mean time {setup_seconds / reached:.2f} s" if reached else ""))
  print(f"[SOCCER] Mean peak closing speed {peak_speed_sum / completed:.2f} m/s")


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
