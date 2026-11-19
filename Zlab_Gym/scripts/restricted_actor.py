"""Restrict selected action outputs without changing the existing MLP layout."""

import re

import torch
from torch import nn


def penultimate_features(actor, observations):
    """Match Sequential.forward, including repeated activation instances."""
    layers = list(actor.mlp)
    if not layers or not isinstance(layers[-1], nn.Linear):
        raise ValueError("Expected an MLP with a final linear action layer.")
    features = actor.get_latent(observations)
    # Module.children() deduplicates shared modules; Sequential iteration does not.
    for layer in layers[:-1]:
        features = layer(features)
    return features


class RestrictedOutputAdapter:
    """Preserve other action functions, optimizer rows and normalizer state."""

    def __init__(self, actor, optimizer, target_names, joint_pattern, expected_count):
        head = [module for module in actor.mlp.modules()
                if isinstance(module, nn.Linear)][-1]
        if head.out_features != len(target_names):
            raise ValueError("Restricted adaptation requires the existing joint-action layout.")
        mask = torch.tensor([
            bool(re.fullmatch(joint_pattern,
                              name.split("/")[-1]))
            for name in target_names
        ], device=head.weight.device)
        if mask.sum().item() != expected_count:
            raise ValueError(f"Expected exactly {expected_count} selected action outputs.")
        if not hasattr(actor.obs_normalizer, "until"):
            raise ValueError("Restricted adaptation requires a freezable observation normalizer.")
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


class ArmOutputAdapter(RestrictedOutputAdapter):
    def __init__(self, actor, optimizer, target_names):
        super().__init__(actor, optimizer, target_names,
                         r"(left|right)_(shoulder_(pitch|roll|yaw)|elbow)", 8)


class LegOutputAdapter(RestrictedOutputAdapter):
    def __init__(self, actor, optimizer, target_names):
        super().__init__(actor, optimizer, target_names,
                         r"(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))", 12)
