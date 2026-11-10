from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    z01_std_add_head_flat_env_cfg,
    z01_std_add_head_rough_env_cfg,
)
from .rl_cfg import z01_std_add_head_velocity_ppo_runner_cfg
from .running import (
    RunningOnPolicyRunner, balance_recovery_env_cfg, robust_stride_env_cfg,
    running_env_cfg, stride_env_cfg, walking_env_cfg,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Rough",
    env_cfg=z01_std_add_head_rough_env_cfg(),
    play_env_cfg=z01_std_add_head_rough_env_cfg(play=True),
    rl_cfg=z01_std_add_head_velocity_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Flat",
    env_cfg=z01_std_add_head_flat_env_cfg(),
    play_env_cfg=z01_std_add_head_flat_env_cfg(play=True),
    rl_cfg=z01_std_add_head_velocity_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

running_rl_cfg = z01_std_add_head_velocity_ppo_runner_cfg()
running_rl_cfg.experiment_name = "z01_human_run_balanced"
running_rl_cfg.actor.distribution_cfg["init_std"] = 0.30
running_rl_cfg.algorithm.learning_rate = 1.0e-4
running_rl_cfg.algorithm.schedule = "fixed"
register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Run",
    env_cfg=running_env_cfg(),
    play_env_cfg=running_env_cfg(play=True),
    rl_cfg=running_rl_cfg,
    runner_cls=RunningOnPolicyRunner,
)

walking_rl_cfg = z01_std_add_head_velocity_ppo_runner_cfg()
walking_rl_cfg.experiment_name = "z01_walking"
walking_rl_cfg.actor.distribution_cfg["init_std"] = 0.15
walking_rl_cfg.algorithm.learning_rate = 5.0e-5
walking_rl_cfg.algorithm.schedule = "fixed"
register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Walk",
    env_cfg=walking_env_cfg(),
    play_env_cfg=walking_env_cfg(play=True),
    rl_cfg=walking_rl_cfg,
    runner_cls=RunningOnPolicyRunner,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-FastWalk",
    env_cfg=walking_env_cfg(max_speed=1.6),
    play_env_cfg=walking_env_cfg(play=True, max_speed=1.6),
    rl_cfg=replace(walking_rl_cfg, experiment_name="z01_fast_walking"),
    runner_cls=RunningOnPolicyRunner,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-StrideWalk",
    env_cfg=stride_env_cfg(),
    play_env_cfg=stride_env_cfg(play=True),
    rl_cfg=replace(walking_rl_cfg, experiment_name="z01_stride_walking"),
    runner_cls=RunningOnPolicyRunner,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-LongStrideWalk",
    env_cfg=stride_env_cfg(longer=True),
    play_env_cfg=stride_env_cfg(play=True, longer=True),
    rl_cfg=replace(walking_rl_cfg, experiment_name="z01_long_stride_walking"),
    runner_cls=RunningOnPolicyRunner,
)

for stage in (1, 2, 3):
    register_mjlab_task(
        task_id=f"Z01-StdAddHead-Velocity-RobustWalk{stage}",
        env_cfg=robust_stride_env_cfg(stage),
        play_env_cfg=robust_stride_env_cfg(stage, play=True),
        rl_cfg=replace(walking_rl_cfg, experiment_name=f"z01_robust_walking_{stage}"),
        runner_cls=RunningOnPolicyRunner,
    )

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-BalanceRecovery",
    env_cfg=balance_recovery_env_cfg(),
    play_env_cfg=balance_recovery_env_cfg(play=True),
    rl_cfg=replace(walking_rl_cfg, experiment_name="z01_balance_recovery"),
    runner_cls=RunningOnPolicyRunner,
)
