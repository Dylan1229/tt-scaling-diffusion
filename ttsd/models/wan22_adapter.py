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

# Supported sampler families:
#   "unipc"      — UniPCMultistepScheduler, deterministic 2nd-order ODE.
#                  This is Wan 2.2's bundled default.
#   "euler"      — FlowMatchEulerDiscreteScheduler, deterministic 1st-order ODE.
#                  Useful as a same-family ODE control when ablating against SDE.
#   "euler_sde"  — FlowMatchEulerDiscreteScheduler with stochastic_sampling=True.
#                  True SDE: Euler step + ancestral Gaussian noise injection at
#                  every step. Reproducible via the seeded torch.Generator.
SCHEDULER_KINDS = ("unipc", "euler", "euler_sde")
DEFAULT_SCHEDULER_KIND = "unipc"


def _resolved_model_path() -> str:
    return os.environ.get("WAN22_MODEL_PATH", DEFAULT_MODEL_PATH)


def _build_alt_scheduler(orig_config, kind: str):
    """Build a replacement scheduler that preserves the model's trained-time
    schedule (num_train_timesteps + flow shift) but switches solver family."""
    if kind == "unipc":
        return None  # keep the model's bundled scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler

    if kind in ("euler", "euler_sde"):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=getattr(orig_config, "num_train_timesteps", 1000),
            shift=getattr(orig_config, "flow_shift", 1.0),
            stochastic_sampling=(kind == "euler_sde"),
        )
    raise ValueError(f"unknown scheduler kind: {kind!r}; expected one of {SCHEDULER_KINDS}")


def _posterior_mean_from_step(scheduler, model_output, sample, step_idx: int) -> torch.Tensor:
    """Recover x0_hat = E[x_0 | x_t, c] from scheduler state + model output.

    - UniPC: use the scheduler's own `convert_model_output` (handles
      flow_prediction parametrization internally).
    - FlowMatchEuler (no convert_model_output): use the closed-form flow-matching
      identity `x_0 = x_t - sigma * v_theta`. `sigma` is the current sigma read
      from `scheduler.sigmas[step_idx]`.
    """
    if hasattr(scheduler, "convert_model_output"):
        return scheduler.convert_model_output(model_output, sample=sample)
    sigma = scheduler.sigmas[step_idx].to(device=sample.device, dtype=sample.dtype)
    return sample - sigma * model_output


@dataclass
class GenerationOutput:
    frames: torch.Tensor  # (T, H, W, 3) uint8 or float in [0, 1]
    latents_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    posterior_means_by_step: dict[int, torch.Tensor] = field(default_factory=dict)


class Wan22Adapter:
    """Wraps `diffusers.WanPipeline` (or the unified TI2V pipeline) for T2V mode.

    `scheduler_kind` selects the sampler family (see SCHEDULER_KINDS). The model's
    trained-time schedule (flow shift, num_train_timesteps) is preserved when
    swapping; only the solver discretization changes.
    """

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
        self.model_path = model_path or _resolved_model_path()
        self.dtype = dtype
        self.device = device
        self.scheduler_kind = scheduler_kind
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
        # Optionally swap the bundled scheduler — done BEFORE .to(device) so
        # any scheduler buffers also move to the GPU.
        alt = _build_alt_scheduler(self._pipe.scheduler.config, self.scheduler_kind)
        if alt is not None:
            self._pipe.scheduler = alt
        self._pipe.to(self.device)

    @staticmethod
    def _capture_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu", dtype=torch.float16).clone()

    @torch.no_grad()
    def generate_with_posterior_means(
        self,
        prompt: str,
        seed: int,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        snapshot_steps: Iterable[int] = (),
        posterior_mean_steps: Iterable[int] = (),
    ) -> GenerationOutput:
        """Run one T2V generation and additionally capture UniPC x0 predictions.

        This wraps the scheduler used by the original pipeline call instead of
        reimplementing the denoising loop, so the final sample stays on the same
        execution path as the baseline run.
        """
        self._load()
        pipe = self._pipe
        snapshot_set = set(snapshot_steps)
        posterior_set = set(posterior_mean_steps)
        captured: dict[int, torch.Tensor] = {}
        posterior_means: dict[int, torch.Tensor] = {}
        original_step = pipe.scheduler.step
        step_idx = 0

        def wrapped_step(*args, **kwargs):
            """Generic wrapper — forwards every arg/kwarg the pipeline passes to
            the underlying scheduler.step (signatures differ across UniPC and
            FlowMatchEuler; we just splat). We extract `model_output` (first
            positional) and `sample` (third positional or `sample=` kwarg) for
            the posterior-mean computation."""
            nonlocal step_idx
            model_output = args[0] if args else kwargs["model_output"]
            if len(args) >= 3:
                sample = args[2]
            elif "sample" in kwargs:
                sample = kwargs["sample"]
            else:
                raise RuntimeError("scheduler.step called without a recognizable `sample` arg")

            if step_idx in posterior_set:
                posterior = _posterior_mean_from_step(pipe.scheduler, model_output, sample, step_idx)
                posterior_means[step_idx] = self._capture_tensor(posterior)

            result = original_step(*args, **kwargs)
            return_dict = kwargs.get("return_dict", True)
            prev_sample = result.prev_sample if return_dict else result[0]

            if step_idx in snapshot_set:
                captured[step_idx] = self._capture_tensor(prev_sample)

            step_idx += 1
            return result

        pipe.scheduler.step = wrapped_step
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        finally:
            pipe.scheduler.step = original_step

        frames = result.frames[0] if hasattr(result, "frames") else result[0]

        return GenerationOutput(
            frames=frames,
            latents_by_step=captured,
            posterior_means_by_step=posterior_means,
        )

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
