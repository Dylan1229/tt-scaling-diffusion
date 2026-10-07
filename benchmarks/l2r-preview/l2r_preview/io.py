from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .latents import WanLatentSpec
from .model import L2RDecoder
from .pyramid import PyramidPreviewDecoder


def load_latent(path: str | Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 5:
        raise ValueError("Expected a saved [B,C,T,H,W] tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("Latent must contain finite floating-point values")
    return value.float()


def save_checkpoint(path, model, spec, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": model.architecture,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "latent_spec": spec.to_dict(),
            "training": training,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    spec = WanLatentSpec.from_dict(data["latent_spec"])
    architecture = data["architecture"]
    if architecture.get("kind") == "pyramid_preview_v1":
        keys = ("latent_channels", "width", "blocks", "stage_widths")
        model = PyramidPreviewDecoder(**{k: architecture[k] for k in keys})
        for key in ("spatial_scale", "temporal_context", "spatial_halo"):
            if architecture[key] != model.architecture[key]:
                raise ValueError(f"Incorrect pyramid metadata: {key}")
    elif "kind" not in architecture:
        model = L2RDecoder(**architecture)
    else:
        raise ValueError("Unsupported decoder architecture")
    if model.latent_channels != spec.latent_channels or model.spatial_scale != spec.spatial_scale:
        raise ValueError("Checkpoint architecture and VAE latent spec disagree")
    model.load_state_dict(data["state_dict"], strict=True)
    return model.to(device).eval(), spec, data["training"]


def rgb_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    if rgb.ndim != 5 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
        raise ValueError("Export expects RGB [1,3,T,H,W]")
    return (
        ((rgb.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 3, 0)
        .numpy()
    )


def save_video(path, rgb, fps):
    if fps <= 0:
        raise ValueError("fps must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        rgb_to_uint8(rgb),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
