from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import z01_std_add_head_flat_tracking_env_cfg
from .rl_cfg import z01_std_add_head_tracking_ppo_runner_cfg

for residual in (False, True):
    rl_cfg = z01_std_add_head_tracking_ppo_runner_cfg()
    task_id = "Z01-StdAddHead-Tracking"
    if residual:
        task_id += "-Residual"
        rl_cfg.experiment_name += "_residual"
        rl_cfg.actor.distribution_cfg["init_std"] = 0.20
    register_mjlab_task(
        task_id=task_id,
        env_cfg=z01_std_add_head_flat_tracking_env_cfg(residual=residual),
        play_env_cfg=z01_std_add_head_flat_tracking_env_cfg(play=True, residual=residual),
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )
