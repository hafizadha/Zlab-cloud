"""Motion-relative joint targets for dynamic tracking clips."""

from dataclasses import dataclass
from typing import cast

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.tracking.mdp import MotionCommand


@dataclass(kw_only=True)
class MotionResidualPositionActionCfg(JointPositionActionCfg):
    use_default_offset: bool = False

    def build(self, env) -> "MotionResidualPositionAction":
        return MotionResidualPositionAction(self, env)


class MotionResidualPositionAction(JointPositionAction):
    def apply_actions(self) -> None:
        motion = cast(MotionCommand, self._env.command_manager.get_term("motion"))
        reference = motion.joint_pos[:, self.target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self.target_ids]
        target = reference + self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self.target_ids)
