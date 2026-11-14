
"""Script to train RL agent with RSL-RL."""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load the bundled RSL-RL before importing mjlab. Its Gaussian distribution
# applies softplus to the seed checkpoint's exploration parameters.
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path[:0] = [_SRC_DIR, _PROJECT_ROOT]

from mjlab.utils.wrappers import VideoRecorder
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml
from mjlab.utils.gpu import select_gpus
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from scripts.checkpoint_upload import HuggingFaceCheckpoints
import mjlab
import torch
import tyro
from src.tasks.velocity.config.z01_std_add_head import TASK_ID
from src.tasks.velocity.config.z01_std_add_head.ball_sprint import (
    CHECKPOINT_TASK, DEFAULT_FOOT_FRICTION, STABILITY_SPEED, STABILITY_STAGE,
    set_foot_friction, set_observation_noise, set_push_strength,
)
from typing import Literal
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass, field
import logging
import math

os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")
@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    checkpoint: Path | None = None
    observation_noise_scale: float = 0.0
    foot_friction_min: float = DEFAULT_FOOT_FRICTION[0]
    foot_friction_max: float = DEFAULT_FOOT_FRICTION[1]
    torso_push_scale: float = 0.0
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    log_root: str | None = None
    log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(
        default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)



def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    if cfg.checkpoint is None:
        raise ValueError("Supply checkpoint 1400 or a saved stability checkpoint with --checkpoint.")
    checkpoint = cfg.checkpoint.expanduser().resolve(strict=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info = state.get("infos") or {}
    continuing = info.get("stage") == STABILITY_STAGE
    if info.get("task") != CHECKPOINT_TASK or not (continuing or state.get("iter") == 1400):
        raise ValueError("Expected the original straight BallSprint checkpoint 1400 or a stability checkpoint.")
    if not 0.0 <= cfg.torso_push_scale <= 1.0 or not 0.0 <= cfg.observation_noise_scale <= 1.0:
        raise ValueError("Torso push and observation noise scales must be between 0 and 1.")
    cfg.env.commands["twist"].speed_cap = STABILITY_SPEED
    set_push_strength(cfg.env, cfg.torso_push_scale)
    set_observation_noise(cfg.env, cfg.observation_noise_scale)
    set_foot_friction(cfg.env, (cfg.foot_friction_min, cfg.foot_friction_max))

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(
            f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")
        command = cfg.env.commands["twist"]
        print(
            f"[INFO] {task_id}: ball {command.ball_distance_range} m straight ahead, "
            f"command cap {command.speed_cap:g} m/s, "
            f"torso push scale {cfg.torso_push_scale:g}, "
            f"observation noise scale {cfg.observation_noise_scale:g}"
        )
    checkpoint_uploader = HuggingFaceCheckpoints.from_environment(log_dir.name) if rank == 0 else None

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner_kwargs = {}
    if checkpoint_uploader is not None:
        runner_kwargs["checkpoint_uploader"] = checkpoint_uploader
    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    runner.add_git_repo_to_log(__file__)
    # Warm-start the actor for the new reward/actuator setup. Refit the critic
    # from scratch so old sprint returns and optimizer moments do not carry over.
    runner.alg.load(state, {"actor": True, "critic": continuing}, strict=True)
    if continuing:
        runner.current_learning_iteration = state["iter"] + 1
    else:
        # Reduce stochastic exploration for the first stability block.
        with torch.no_grad():
            distribution = runner.alg.get_policy().distribution
            distribution.std_param.clamp_(max=math.log(math.expm1(0.12)))
    print(f"[INFO] {'Continued stability training' if continuing else 'Warm-started actor'} from {checkpoint}.")
    del state

    # Only write config files from rank 0 to avoid race conditions.
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    try:
        runner.learn(
            num_learning_iterations=cfg.agent.max_iterations
        )
    finally:
        if checkpoint_uploader is not None:
            checkpoint_uploader.close()
        env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

    # Create log directory once before launching workers.
    if args.log_dir == "":
        raise ValueError(
            "--log-dir was provided but empty. Check the environment variable "
            "used to populate it."
        )
    log_root_base = args.log_dir or args.log_root or os.environ.get("ZLAB_LOG_ROOT", "logs")
    log_root_path = Path(log_root_base).expanduser() / "rsl_rl" / args.agent.experiment_name
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

    if num_gpus <= 1:
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # Torchrunx starts a fresh Python interpreter. It can import RSL-RL
        # while deserializing the training config, before run_train executes.
        python_path = [_SRC_DIR, _PROJECT_ROOT]
        if os.environ.get("PYTHONPATH"):
            python_path.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(python_path)

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + (
                "PYTHONPATH", "MUJOCO*", "WARP*", "MJLAB_WARP_QUIET",
                "OMP_NUM_THREADS", "ZLAB_NOTEBOOK_PROGRESS",
            ),
        ).run(run_train, task_id, args, log_dir)


def main():
    args = tyro.cli(
        TrainConfig,
        default=TrainConfig.from_task(TASK_ID),
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id=TASK_ID, args=args)


if __name__ == "__main__":
    main()
