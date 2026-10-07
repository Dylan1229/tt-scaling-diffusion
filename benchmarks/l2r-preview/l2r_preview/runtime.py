"""Reusable preview inference. Construct once and reuse throughout diffusion."""
from __future__ import annotations
import torch


def make_preview_decoder(model, compile_mode="none"):
    """Return native-latent -> clamped RGB [0,1] NCTHW callable.

    Call under torch.inference_mode(). Compilation specializes to shape/dtype;
    cold compilation is excluded from steady-state latency. Compiled BF16
    arithmetic has small rounding differences from eager execution.
    """
    if compile_mode not in ("none", "default", "max-autotune-no-cudagraphs"):
        raise ValueError("Unsupported compilation mode")
    model.eval().requires_grad_(False)
    def forward(latent):
        return model(latent).add(1).mul(0.5).clamp(0,1)
    if compile_mode == "none":
        return forward
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("Compiled preview path is validated only on CUDA")
    return torch.compile(forward, mode=compile_mode, fullgraph=True, dynamic=False)
