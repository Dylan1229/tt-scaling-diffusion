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
        capture: set[str] | None = None,
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
        capture: set[str] | None = None,
    ) -> GenerationOutput:
        """If `capture` includes 'posterior_mean', we wrap scheduler.step to
        compute x0_hat each step and stuff the freshest one into the next
        StepState handed to `on_step`. Memory is O(1) — only the most recent
        posterior_mean is held, then replaced on the next step."""
        capture = capture or set()
        need_pm = "posterior_mean" in capture
        aborted_at: list[int] = []
        last_pm: list[torch.Tensor | None] = [None]
        step_counter: list[int] = [0]
        original_scheduler_step = None

        pipe = self._impl._pipe
        if need_pm:
            # Lazy-load so the scheduler reference is valid.
            self._impl._load()
            pipe = self._impl._pipe
            original_scheduler_step = pipe.scheduler.step

            def _wrapped_step(model_output, timestep, sample, **kwargs):
                # Compute x0_hat at this step (cheap algebra; no extra forward).
                # step_idx tracking matters only for Euler schedulers where the
                # fallback formula reads scheduler.sigmas[step_idx]. For UniPC
                # (our default) it's ignored because convert_model_output is
                # self-sufficient.
                from ttsd.models.wan22_adapter import _posterior_mean_from_step

                try:
                    posterior = _posterior_mean_from_step(
                        pipe.scheduler, model_output, sample, step_idx=step_counter[0]
                    )
                    last_pm[0] = posterior.detach()
                except Exception:    # pragma: no cover — never fail generation on capture issues
                    last_pm[0] = None
                step_counter[0] += 1
                return original_scheduler_step(model_output, timestep, sample, **kwargs)

            pipe.scheduler.step = _wrapped_step

        def _cb(pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if on_step is None:
                return callback_kwargs
            state = StepState(
                step=step_idx,
                total_steps=request.num_inference_steps,
                timestep=timestep,
                latent=latents,
                posterior_mean=last_pm[0] if need_pm else None,
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
                initial_latent=request.initial_latent,
            )
        except AbortTrajectory:
            # The pipeline raised inside the diffusers callback. Return a
            # partial output so the caller knows the abort happened cleanly.
            return GenerationOutput(
                frames=torch.empty(0),
                aborted=True,
                abort_at_step=aborted_at[-1] if aborted_at else None,
            )
        finally:
            if need_pm and original_scheduler_step is not None:
                self._impl._pipe.scheduler.step = original_scheduler_step

        return GenerationOutput(frames=result.frames)
