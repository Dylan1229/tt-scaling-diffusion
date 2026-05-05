"""Adapter around the diffusers Wan 2.2 TI2V-5B pipeline.

Used in T2V mode (no input image). Exposes a callback hook so that runners
and search policies can capture or modify intermediate latents at chosen
denoising steps.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch

DEFAULT_MODEL_PATH = (
    "/data/datasets/fanjiang/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
)


def _resolved_model_path() -> str:
    return os.environ.get("WAN22_MODEL_PATH", DEFAULT_MODEL_PATH)


@dataclass
class GenerationOutput:
    frames: torch.Tensor  # (T, H, W, 3) uint8 or float in [0, 1]
    latents_by_step: dict[int, torch.Tensor] = field(default_factory=dict)


class Wan22Adapter:
    """Wraps `diffusers.WanPipeline` (or the unified TI2V pipeline) for T2V mode."""

    def __init__(
        self,
        model_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.model_path = model_path or _resolved_model_path()
        self.dtype = dtype
        self.device = device
        self._pipe = None  # lazily loaded

    def _load(self):
        if self._pipe is not None:
            return
        # Lazy import so the package can be installed without diffusers present.
        from diffusers import DiffusionPipeline

        # Resolve snapshot directory inside the HF cache dir.
        path = self.model_path
        snap_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snap_dir):
            snaps = sorted(os.listdir(snap_dir))
            if snaps:
                path = os.path.join(snap_dir, snaps[-1])

        self._pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=self.dtype)
        self._pipe.to(self.device)

    def generate(
        self,
        prompt: str,
        seed: int,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        snapshot_steps: Iterable[int] = (),
        on_step_end: Callable | None = None,
    ) -> GenerationOutput:
        """Run one T2V generation. `snapshot_steps` are step indices whose latents
        we keep in CPU memory; `on_step_end` (if provided) is called after every
        step with (pipeline, step_idx, timestep, callback_kwargs) — letting
        search policies replace/branch the latent in-place."""
        self._load()
        snapshot_set = set(snapshot_steps)
        captured: dict[int, torch.Tensor] = {}

        def _cb(pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if latents is not None and step_idx in snapshot_set:
                captured[step_idx] = latents.detach().to("cpu", dtype=torch.float16).clone()
            if on_step_end is not None:
                callback_kwargs = on_step_end(pipe, step_idx, timestep, callback_kwargs)
            return callback_kwargs

        gen = torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # `result.frames` is typically (B, T, H, W, 3) for video pipelines.
        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        return GenerationOutput(frames=frames, latents_by_step=captured)
