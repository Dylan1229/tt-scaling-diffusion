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

from ttsd.search.late_branching import (
    BranchSpec,
    LateBranchConfig,
    denoising_step_equivalents,
    fork_latents,
    renoise_latents,
    sigma_after_step,
)

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


def _scheduler_meta(scheduler) -> dict:
    """Snapshot the coarse UniPC schedule + config needed to rebuild teacher sub-grids."""
    cfg = scheduler.config

    def _to_list(x):
        return [float(v) for v in x.tolist()] if x is not None else None

    return {
        "sigmas": _to_list(getattr(scheduler, "sigmas", None)),
        "timesteps": _to_list(getattr(scheduler, "timesteps", None)),
        "flow_shift": float(getattr(cfg, "flow_shift", 0.0)),
        "num_train_timesteps": int(cfg.num_train_timesteps),
        "solver_order": int(cfg.solver_order),
        "solver_type": cfg.solver_type,
        "prediction_type": cfg.prediction_type,
        "predict_x0": bool(cfg.predict_x0),
        "final_sigmas_type": cfg.final_sigmas_type,
        "use_flow_sigmas": bool(getattr(cfg, "use_flow_sigmas", False)),
    }


@dataclass
class GenerationOutput:
    frames: torch.Tensor  # (T, H, W, 3) uint8 or float in [0, 1]
    latents_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    posterior_means_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_latents_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    model_outputs_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    scheduler_meta: dict = field(default_factory=dict)


@dataclass
class LateBranchGenerationOutput:
    frames_by_branch: list
    branch_specs: list[BranchSpec]
    branch_step: int
    branch_sigma: float
    denoising_step_equivalents: int
    posterior_means_by_step: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class RenoiseGenerationOutput:
    frames_by_amplitude: list
    amplitudes: tuple[float, ...]
    branch_step: int
    branch_sigma: float
    noise_seed: int


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

    def _decode_latents_in_chunks(
        self, latents: torch.Tensor, batch_size: int
    ) -> list:
        if batch_size < 1:
            raise ValueError("decode_batch_size must be positive")
        pipe = self._pipe
        if not hasattr(pipe, "vae"):
            return [latents[i] for i in range(len(latents))]

        torch.cuda.empty_cache()
        frames: list = []
        for start in range(0, len(latents), batch_size):
            chunk = latents[start : start + batch_size].to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(chunk.device, chunk.dtype)
            )
            latents_std = (
                1.0
                / torch.tensor(pipe.vae.config.latents_std)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(chunk.device, chunk.dtype)
            )
            decoded = pipe.vae.decode(
                chunk / latents_std + latents_mean, return_dict=False
            )[0]
            videos = pipe.video_processor.postprocess_video(decoded, output_type="np")
            frames.extend(videos[i] for i in range(len(videos)))
            del chunk, decoded, videos
            torch.cuda.empty_cache()
        return frames

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
        raw_latent_steps: Iterable[int] = (),
        model_output_steps: Iterable[int] = (),
    ) -> GenerationOutput:
        """Run one T2V generation and additionally capture UniPC x0 predictions.

        This wraps the scheduler used by the original pipeline call instead of
        reimplementing the denoising loop, so the final sample stays on the same
        execution path as the baseline run.

        `snapshot_steps` capture the post-step latent (z_{i+1}); `raw_latent_steps`
        capture the pre-step latent (z_i, the interval anchor); `model_output_steps`
        capture the raw guided model output (o_i) fed into `scheduler.step`;
        `posterior_mean_steps` capture the scheduler's x0_hat. Indices are 0-based
        over the `num_inference_steps` schedule.
        """
        self._load()
        pipe = self._pipe
        snapshot_set = set(snapshot_steps)
        posterior_set = set(posterior_mean_steps)
        raw_latent_set = set(raw_latent_steps)
        model_output_set = set(model_output_steps)
        captured: dict[int, torch.Tensor] = {}
        posterior_means: dict[int, torch.Tensor] = {}
        raw_latents: dict[int, torch.Tensor] = {}
        model_outputs: dict[int, torch.Tensor] = {}
        scheduler_meta: dict = {}
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
            timestep = args[1] if len(args) >= 2 else kwargs.get("timestep")
            if len(args) >= 3:
                sample = args[2]
            elif "sample" in kwargs:
                sample = kwargs["sample"]
            else:
                raise RuntimeError("scheduler.step called without a recognizable `sample` arg")

            sched = pipe.scheduler
            if step_idx == 0 and not scheduler_meta and self.scheduler_kind == "unipc":
                scheduler_meta.update(_scheduler_meta(sched))
            if step_idx in raw_latent_set:
                raw_latents[step_idx] = self._capture_tensor(sample)
            if step_idx in model_output_set:
                model_outputs[step_idx] = self._capture_tensor(model_output)
            if step_idx in posterior_set:
                # convert_model_output reads scheduler.step_index; ensure it is
                # initialized (the pipeline's first step() would otherwise set it).
                if sched.step_index is None:
                    sched._init_step_index(timestep)
                posterior = _posterior_mean_from_step(sched, model_output, sample, step_idx)
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
            raw_latents_by_step=raw_latents,
            model_outputs_by_step=model_outputs,
            scheduler_meta=scheduler_meta,
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

    @torch.no_grad()
    def generate_with_renoise_branches(
        self,
        prompt: str,
        seed: int,
        amplitudes: Iterable[float],
        branch_step: int = 2,
        noise_seed: int = 10_000_000,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        decode_batch_size: int = 8,
    ) -> RenoiseGenerationOutput:
        """Fork one trajectory by rotating its implied noise after a step."""

        if not 1 <= branch_step < num_inference_steps:
            raise ValueError("branch_step must be in [1, num_inference_steps - 1]")
        amplitudes = tuple(float(amplitude) for amplitude in amplitudes)
        self._load()
        branch_sigma: float | None = None
        branch_posterior: torch.Tensor | None = None
        original_step = self._pipe.scheduler.step
        scheduler_step_idx = 0

        def wrapped_step(*args, **kwargs):
            nonlocal branch_posterior, scheduler_step_idx
            model_output = args[0] if args else kwargs["model_output"]
            timestep = args[1] if len(args) >= 2 else kwargs.get("timestep")
            if len(args) >= 3:
                sample = args[2]
            elif "sample" in kwargs:
                sample = kwargs["sample"]
            else:
                raise RuntimeError(
                    "scheduler.step called without a recognizable `sample` arg"
                )

            sched = self._pipe.scheduler
            if scheduler_step_idx == branch_step - 1:
                if getattr(sched, "step_index", None) is None and hasattr(
                    sched, "_init_step_index"
                ):
                    sched._init_step_index(timestep)
                branch_posterior = _posterior_mean_from_step(
                    sched, model_output, sample, scheduler_step_idx
                )

            result = original_step(*args, **kwargs)
            scheduler_step_idx += 1
            return result

        def _cb(pipe, step_idx, timestep, callback_kwargs):
            del timestep
            nonlocal branch_sigma
            if step_idx != branch_step - 1:
                return callback_kwargs
            if branch_posterior is None:
                raise RuntimeError("Step posterior was not captured before RENOISE")

            branch_sigma = sigma_after_step(pipe.scheduler, step_idx)
            callback_kwargs["latents"] = renoise_latents(
                callback_kwargs["latents"],
                posterior=branch_posterior,
                sigma=branch_sigma,
                amplitudes=amplitudes,
                noise_seed=noise_seed,
            )
            for key in ("prompt_embeds", "negative_prompt_embeds"):
                embeds = callback_kwargs.get(key)
                if embeds is not None:
                    callback_kwargs[key] = embeds.repeat_interleave(
                        len(amplitudes), dim=0
                    )
            return callback_kwargs

        self._pipe.scheduler.step = wrapped_step
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self._pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=[
                    "latents",
                    "prompt_embeds",
                    "negative_prompt_embeds",
                ],
                output_type="latent",
            )
        finally:
            self._pipe.scheduler.step = original_step

        if branch_sigma is None:
            raise RuntimeError("RENOISE callback was never reached")
        final_latents = result.frames if hasattr(result, "frames") else result[0]
        if len(final_latents) != len(amplitudes):
            raise RuntimeError(
                "pipeline returned an unexpected number of final latents: "
                f"expected {len(amplitudes)}, got {len(final_latents)}"
            )
        return RenoiseGenerationOutput(
            frames_by_amplitude=self._decode_latents_in_chunks(
                final_latents, decode_batch_size
            ),
            amplitudes=amplitudes,
            branch_step=branch_step,
            branch_sigma=branch_sigma,
            noise_seed=noise_seed,
        )

    @torch.no_grad()
    def generate_with_late_branches(
        self,
        prompt: str,
        seed: int,
        branch_config: LateBranchConfig,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        posterior_mean_offsets: Iterable[int] = (),
        decode_batch_size: int = 8,
    ) -> LateBranchGenerationOutput:
        """Share one prefix, fork the latent once, then batch all suffixes.

        The unperturbed branch is part of the same expanded batch as the noisy
        candidates. It diagnoses batch-dependent numerical drift, but is not a
        substitute for a separately generated batch-one baseline. The callback
        repeats both conditional and unconditional prompt embeddings.
        """

        branch_config.validate(num_inference_steps)
        self._load()
        branch_specs: list[BranchSpec] = []
        branch_sigma: float | None = None
        posterior_offsets = set(int(offset) for offset in posterior_mean_offsets)
        if any(offset < 1 for offset in posterior_offsets):
            raise ValueError("posterior_mean_offsets must contain positive integers")
        posterior_steps = {
            branch_config.branch_step - 1 + offset for offset in posterior_offsets
        }
        if any(step >= num_inference_steps for step in posterior_steps):
            raise ValueError(
                "posterior_mean_offsets extend beyond num_inference_steps"
            )
        posterior_means: dict[int, torch.Tensor] = {}
        original_step = self._pipe.scheduler.step
        scheduler_step_idx = 0

        def wrapped_step(*args, **kwargs):
            nonlocal scheduler_step_idx
            model_output = args[0] if args else kwargs["model_output"]
            timestep = args[1] if len(args) >= 2 else kwargs.get("timestep")
            if len(args) >= 3:
                sample = args[2]
            elif "sample" in kwargs:
                sample = kwargs["sample"]
            else:
                raise RuntimeError(
                    "scheduler.step called without a recognizable `sample` arg"
                )

            sched = self._pipe.scheduler
            if scheduler_step_idx in posterior_steps:
                if getattr(sched, "step_index", None) is None and hasattr(
                    sched, "_init_step_index"
                ):
                    sched._init_step_index(timestep)
                posterior = _posterior_mean_from_step(
                    sched, model_output, sample, scheduler_step_idx
                )
                posterior_means[scheduler_step_idx] = self._capture_tensor(posterior)

            result = original_step(*args, **kwargs)
            scheduler_step_idx += 1
            return result

        def _cb(pipe, step_idx, timestep, callback_kwargs):
            del timestep
            nonlocal branch_specs, branch_sigma
            if step_idx != branch_config.branch_step - 1:
                return callback_kwargs

            branch_sigma = sigma_after_step(pipe.scheduler, step_idx)
            branched_latents, branch_specs = fork_latents(
                callback_kwargs["latents"],
                root_seed=seed,
                sigma=branch_sigma,
                config=branch_config,
            )
            callback_kwargs["latents"] = branched_latents
            for key in ("prompt_embeds", "negative_prompt_embeds"):
                embeds = callback_kwargs.get(key)
                if embeds is not None:
                    callback_kwargs[key] = embeds.repeat_interleave(
                        branch_config.total_branches, dim=0
                    )
            return callback_kwargs

        self._pipe.scheduler.step = wrapped_step
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self._pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=[
                    "latents",
                    "prompt_embeds",
                    "negative_prompt_embeds",
                ],
                output_type="latent",
            )
        finally:
            self._pipe.scheduler.step = original_step

        if branch_sigma is None or not branch_specs:
            raise RuntimeError("late-branch callback was never reached")
        final_latents = result.frames if hasattr(result, "frames") else result[0]
        if len(final_latents) != branch_config.total_branches:
            raise RuntimeError(
                "pipeline returned an unexpected number of final latents: "
                f"expected {branch_config.total_branches}, got {len(final_latents)}"
            )
        frames = self._decode_latents_in_chunks(final_latents, decode_batch_size)

        return LateBranchGenerationOutput(
            frames_by_branch=frames,
            branch_specs=branch_specs,
            branch_step=branch_config.branch_step,
            branch_sigma=branch_sigma,
            denoising_step_equivalents=denoising_step_equivalents(
                num_inference_steps,
                branch_config.branch_step,
                branch_config.total_branches,
            ),
            posterior_means_by_step=posterior_means,
        )
