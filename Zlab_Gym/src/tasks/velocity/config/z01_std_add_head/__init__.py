"""Register the Z01 ball-sprint task."""
from mjlab.tasks.registry import register_mjlab_task

from .rl_cfg import z01_std_add_head_velocity_ppo_runner_cfg
from .ball_sprint import BallSprintOnPolicyRunner, ball_approach_env_cfg

TASK_ID = "Z01-StdAddHead-Soccer-StraightBallSprint"

rl_cfg = z01_std_add_head_velocity_ppo_runner_cfg()
rl_cfg.experiment_name = "z01_straight_ball_sprint"
rl_cfg.actor.distribution_cfg["init_std"] = 0.15
rl_cfg.algorithm.learning_rate = 5.0e-5
rl_cfg.algorithm.schedule = "fixed"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=ball_approach_env_cfg(),
    play_env_cfg=ball_approach_env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=BallSprintOnPolicyRunner,
)
