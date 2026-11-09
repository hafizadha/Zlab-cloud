"""Register the close-range kicking, approach, and sprint soccer tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.velocity.config.z01_std_add_head.rl_cfg import (
    z01_std_add_head_velocity_ppo_runner_cfg,
)

from .env_cfg import soccer_env_cfg


class SoccerOnPolicyRunner(VelocityOnPolicyRunner):
    """Keep the imported walking observation scale stable during PPO and resume."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.alg.get_policy().obs_normalizer.until = 0


for name in ("Soccer-Kick", "Soccer-Approach", "Soccer-Sprint"):
    approach = name != "Soccer-Kick"
    sprint = name == "Soccer-Sprint"
    rl_cfg = z01_std_add_head_velocity_ppo_runner_cfg()
    rl_cfg.experiment_name = f"z01_{name.lower().replace('-', '_')}"
    rl_cfg.actor.distribution_cfg["init_std"] = 0.45
    rl_cfg.algorithm.learning_rate = 1.0e-4
    rl_cfg.algorithm.schedule = "fixed"
    rl_cfg.algorithm.num_learning_epochs = 3
    register_mjlab_task(
        task_id=f"Z01-StdAddHead-{name}",
        env_cfg=soccer_env_cfg(approach=approach, sprint=sprint),
        play_env_cfg=soccer_env_cfg(play=True, approach=approach, sprint=sprint),
        rl_cfg=rl_cfg,
        runner_cls=SoccerOnPolicyRunner,
    )
