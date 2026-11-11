"""Register Stage 3 robust walking."""
from mjlab.tasks.registry import register_mjlab_task

from .rl_cfg import z01_std_add_head_velocity_ppo_runner_cfg
from .stage3 import Stage3OnPolicyRunner, stage3_env_cfg

rl_cfg = z01_std_add_head_velocity_ppo_runner_cfg()
rl_cfg.experiment_name = "z01_robust_walking_3"
rl_cfg.actor.distribution_cfg["init_std"] = 0.15
rl_cfg.algorithm.learning_rate = 5.0e-5
rl_cfg.algorithm.schedule = "fixed"

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-RobustWalk3",
    env_cfg=stage3_env_cfg(),
    play_env_cfg=stage3_env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=Stage3OnPolicyRunner,
)
