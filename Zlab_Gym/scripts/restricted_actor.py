"""Restrict an existing MLP actor to shoulder and elbow output adaptation."""

import re

import torch
from torch import nn


class ArmOutputAdapter:
    """Preserve non-arm action functions, including optimizer and normalizer state."""

    def __init__(self, actor, optimizer, target_names):
        head = [module for module in actor.mlp.modules()
                if isinstance(module, nn.Linear)][-1]
        if head.out_features != len(target_names):
            raise ValueError("Arm adaptation requires the existing joint-action layout.")
        mask = torch.tensor([
            bool(re.fullmatch(r"(left|right)_(shoulder_(pitch|roll|yaw)|elbow)",
                              name.split("/")[-1]))
            for name in target_names
        ], device=head.weight.device)
        if mask.sum().item() != 8:
            raise ValueError("Expected exactly eight shoulder/elbow action outputs.")
        if not hasattr(actor.obs_normalizer, "until"):
            raise ValueError("Arm adaptation requires a freezable observation normalizer.")
        self.mask = mask
        self.saved = {}
        self.handles = []
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        for parameter in (head.weight, head.bias):
            parameter.requires_grad_(True)
            self.saved[parameter] = parameter.detach().clone()
            shape = (-1,) + (1,) * (parameter.ndim - 1)
            self.handles.append(parameter.register_hook(
                lambda grad, shape=shape: grad * mask.reshape(shape)))
            # Loaded Adam momentum must not update protected rows.
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[~mask] = 0
        # Unlike eval(), this also survives a later actor.train() call.
        actor.obs_normalizer.until = 0
        self.handles.append(optimizer.register_step_post_hook(self.restore_rows))

    @torch.no_grad()
    def restore_rows(self, optimizer, args, kwargs):
        # Also protects against decoupled weight decay.
        for parameter, snapshot in self.saved.items():
            parameter[~self.mask] = snapshot[~self.mask]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
