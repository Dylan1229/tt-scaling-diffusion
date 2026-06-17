"""ModelAdapter Protocol + the Wan 2.2 adapter that conforms to it.

This is the *one* surface between the pipeline orchestrator and any
diffusion backbone. To add a new backbone (CogVideoX, HunyuanVideo, ...):
write a class that satisfies the Protocol and register it with
`@register_model("name")`.

The wrapper around `Wan22Adapter` is thin — it just translates between the
pipeline's `(GenerationRequest, on_step)` interface and Wan22Adapter's
`(prompt, seed, ..., on_step_end)` keyword interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from ttsd.pipeline.core import AbortTrajectory, StepDirective, StepState
from ttsd.pipeline.registry import register_model


@dataclass
class GenerationRequest:
    """Pipeline-facing generation parameters. Backend-agnostic."""
    prompt: str
    seed: int
    num_frames: int = 81
    height: int = 480
    width: int = 832
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    initial_latent: torch.Tensor | None = None    # for Trial 1 anchor injection (P3)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """Pipeline-facing generation output. Backend-agnostic."""
    frames: torch.Tensor                          # (T, H, W, 3) uint8 or float
    final_latent: torch.Tensor | None = None
    aborted: bool = False                         # set if on_step requested abort
    abort_at_step: int | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    """Any class with these methods is a valid backbone."""
    name: str

    def generate(
        self,
        request: GenerationRequest,
        on_step: Callable[[StepState], StepDirective | None] | None = None,
    ) -> GenerationOutput: ...


@register_model("wan22_ti2v_5b")
class WanModelAdapter:
    """Wraps Wan22Adapter so the pipeline can drive it."""

    name = "wan22_ti2v_5b"

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "bf16",
        device: str = "cuda",
        scheduler: str = "unipc",
    ):
        from ttsd.models.wan22_adapter import Wan22Adapter

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._impl = Wan22Adapter(
            model_path=model_path,
            dtype=dtype_map[dtype],
            device=device,
            scheduler_kind=scheduler,
        )

    def generate(
        self,
        request: GenerationRequest,
        on_step: Callable[[StepState], StepDirective | None] | None = None,
    ) -> GenerationOutput:
        aborted_at: list[int] = []

        def _cb(pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if on_step is None:
                return callback_kwargs
            state = StepState(
                step=step_idx,
                total_steps=request.num_inference_steps,
                timestep=timestep,
                latent=latents,
            )
            directive = on_step(state)
            if directive is None:
                return callback_kwargs
            if directive.replace_latent is not None:
                callback_kwargs["latents"] = directive.replace_latent
            if directive.abort:
                aborted_at.append(step_idx)
                raise AbortTrajectory(f"aborted at step {step_idx}")
            return callback_kwargs

        try:
            result = self._impl.generate(
                prompt=request.prompt,
                seed=request.seed,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                on_step_end=_cb,
            )
        except AbortTrajectory:
            # The pipeline raised inside the diffusers callback. Return a
            # partial output so the caller knows the abort happened cleanly.
            return GenerationOutput(
                frames=torch.empty(0),
                aborted=True,
                abort_at_step=aborted_at[-1] if aborted_at else None,
            )

        return GenerationOutput(frames=result.frames)
