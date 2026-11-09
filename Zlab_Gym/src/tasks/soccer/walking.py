"""Seed the soccer actor with the bundled flat walking policy."""

from pathlib import Path

import onnx
import torch
from onnx import numpy_helper
from rsl_rl.models import MLPModel

WALKING_POLICY = (
    Path(__file__).resolve().parents[2]
    / "assets/motions/z01_std_add_head/demo_act/z01_velocity_flat_base.onnx"
)


@torch.no_grad()
def initialize_walking_actor(actor: MLPModel, extra_inputs: int = 8) -> None:
    """Preserve the walking output with zero initial weight on appended inputs."""
    graph = onnx.load(WALKING_POLICY).graph
    tensors = {
        tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy())
        for tensor in graph.initializer
    }
    divisor_name = next(node.input[1] for node in graph.node if node.op_type == "Div")
    mean = tensors["obs_normalizer._mean"]
    walking_dim = mean.shape[-1]
    if actor.obs_dim != walking_dim + extra_inputs:
        raise ValueError("Actor observation size does not match the walking input layout.")

    for name, param in actor.mlp.named_parameters():
        weights = tensors[f"mlp.{name}"]
        if name == "0.weight":
            param.zero_()
            param[:, :walking_dim].copy_(weights)
        else:
            param.copy_(weights)

    normalizer = actor.obs_normalizer
    std = (tensors[divisor_name] - normalizer.eps).clamp_min(0.0)
    normalizer._mean[:, :walking_dim].copy_(mean)
    normalizer._std[:, :walking_dim].copy_(std)
    normalizer._var[:, :walking_dim].copy_(std.square())
