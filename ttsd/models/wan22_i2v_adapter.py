"""Small Wan 2.2 image-to-video helper used by long-video continuation.

This stays separate from `Wan22Adapter` so the existing T2V path remains
unchanged. For the long-video runner it can wrap the already-loaded T2V pipeline
components, avoiding a second copy of the 5B model on the same GPU.
"""

from __future__ import annotations

import os

import torch

from ttsd.models.wan22_adapter import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEDULER_KIND,
    SCHEDULER_KINDS,
    GenerationOutput,
    _build_alt_scheduler,
)


def _snapshot_model_path(model_path: str) -> str:
    snap_dir = os.path.join(model_path, "snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            return os.path.join(snap_dir, snaps[-1])
    return model_path


def _config_value(config, key: str, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class Wan22I2VAdapter:
    """Minimal wrapper around `diffusers.WanImageToVideoPipeline`."""

    def __init__(
        self,
        model_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        scheduler_kind: str = DEFAULT_SCHEDULER_KIND,
    ):
        if scheduler_kind not in SCHEDULER_KINDS:
            raise ValueError(
                f"scheduler_kind={scheduler_kind!r} not in {SCHEDULER_KINDS}"
            )
        self.model_path = model_path or os.environ.get("WAN22_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.dtype = dtype
        self.device = device
        self.scheduler_kind = scheduler_kind
        self._pipe = None

    @classmethod
    def from_t2v_adapter(cls, t2v_adapter) -> "Wan22I2VAdapter":
        """Build an I2V pipeline view from the already-loaded T2V components."""
        from diffusers import WanImageToVideoPipeline

        t2v_adapter._load()
        t2v_pipe = t2v_adapter._pipe
        obj = cls(
            model_path=t2v_adapter.model_path,
            dtype=t2v_adapter.dtype,
            device=t2v_adapter.device,
            scheduler_kind=t2v_adapter.scheduler_kind,
        )
        obj._pipe = WanImageToVideoPipeline(
            tokenizer=t2v_pipe.tokenizer,
            text_encoder=t2v_pipe.text_encoder,
            vae=t2v_pipe.vae,
            scheduler=t2v_pipe.scheduler,
            transformer=getattr(t2v_pipe, "transformer", None),
            transformer_2=getattr(t2v_pipe, "transformer_2", None),
            boundary_ratio=_config_value(t2v_pipe.config, "boundary_ratio", None),
            expand_timesteps=bool(_config_value(t2v_pipe.config, "expand_timesteps", False)),
        )
        obj._pipe.to(obj.device)
        return obj

    def _load(self):
        if self._pipe is not None:
            return
        from diffusers import WanImageToVideoPipeline

        path = _snapshot_model_path(self.model_path)
        self._pipe = WanImageToVideoPipeline.from_pretrained(path, torch_dtype=self.dtype)
        alt = _build_alt_scheduler(self._pipe.scheduler.config, self.scheduler_kind)
        if alt is not None:
            self._pipe.scheduler = alt
        self._pipe.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image,
        seed: int,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
    ) -> GenerationOutput:
        self._load()
        gen = torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipe(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        return GenerationOutput(frames=frames)
