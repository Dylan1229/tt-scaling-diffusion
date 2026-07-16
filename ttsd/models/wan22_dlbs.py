from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch


class VideoRewardFn(Protocol):
    def __call__(self, video_tensor: torch.Tensor, prompt: str) -> tuple[float, dict[str, float]]: ...


@dataclass(frozen=True)
class WanDLBSConfig:
    num_beams: int = 4
    num_candidates: int = 2
    num_lookahead_steps: int = 6
    branch_noise_scale: float = 1.0
    branch_noise_std: float | None = None
    include_deterministic_candidate: bool = False
    reward_frame_stride: int = 2
    reward_max_frames: int | None = 16
    max_sequence_length: int = 512
    score_last_step: bool = True
    higher_is_better: bool = True
    output_type: str = "np"
    trace_path: Path | None = None

    def __post_init__(self) -> None:
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.num_lookahead_steps <= 0:
            raise ValueError("num_lookahead_steps must be positive")
        if self.branch_noise_scale < 0:
            raise ValueError("branch_noise_scale must be non-negative")
        if self.branch_noise_std is not None and self.branch_noise_std < 0:
            raise ValueError("branch_noise_std must be non-negative")
        if self.reward_frame_stride <= 0:
            raise ValueError("reward_frame_stride must be positive")


@dataclass
class _Beam:
    latent: torch.Tensor
    scheduler: Any
    score: float = 0.0
    path: list[int] = field(default_factory=list)


@dataclass
class WanDLBSOutput:
    frames: Any
    latents_by_step: dict[int, torch.Tensor]
    trace: list[dict[str, Any]]


def _as_float(value: int | float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").item())
    return float(value)


def _as_timestep_tensor(value: int | float | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, device=device)


def _zero_like_timestep(timestep: int | float | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        return torch.zeros((), device=device, dtype=timestep.dtype)
    return torch.tensor(0, device=device)


@contextmanager
def _cache_context(model: Any, name: str):
    if hasattr(model, "cache_context"):
        with model.cache_context(name):
            yield
    else:
        yield


def _adjust_video_shape(pipe: Any, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
    if num_frames % pipe.vae_scale_factor_temporal != 1:
        num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    patch_size = transformer.config.patch_size
    h_multiple_of = pipe.vae_scale_factor_spatial * patch_size[1]
    w_multiple_of = pipe.vae_scale_factor_spatial * patch_size[2]
    height = height // h_multiple_of * h_multiple_of
    width = width // w_multiple_of * w_multiple_of
    return height, width, num_frames


def _choose_model_and_guidance(
    pipe: Any,
    timestep: torch.Tensor,
    guidance_scale: float,
    guidance_scale_2: float | None,
) -> tuple[Any, float]:
    boundary_ratio = getattr(pipe.config, "boundary_ratio", None)
    if boundary_ratio is None:
        return pipe.transformer, guidance_scale

    boundary_timestep = boundary_ratio * pipe.scheduler.config.num_train_timesteps
    if _as_float(timestep) >= boundary_timestep:
        return pipe.transformer, guidance_scale
    return pipe.transformer_2, guidance_scale if guidance_scale_2 is None else guidance_scale_2


def _expanded_timestep(pipe: Any, timestep: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if getattr(pipe.config, "expand_timesteps", False):
        temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
        return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
    return timestep.expand(latents.shape[0])


def _predict_noise(
    pipe: Any,
    model: Any,
    latents: torch.Tensor,
    timestep: int | float | torch.Tensor,
    *,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
    attention_kwargs: dict[str, Any] | None,
    transformer_dtype: torch.dtype,
) -> torch.Tensor:
    timestep_tensor = _as_timestep_tensor(timestep, device=latents.device)
    timestep_input = _expanded_timestep(pipe, timestep_tensor, latents, mask)
    latent_model_input = latents.to(transformer_dtype)

    with _cache_context(model, "cond"):
        noise_pred = model(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    if negative_prompt_embeds is not None and guidance_scale > 1.0:
        with _cache_context(model, "uncond"):
            noise_uncond = model(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

    return noise_pred


def _scheduler_step(
    scheduler: Any,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    try:
        result = scheduler.step(noise_pred, timestep, latents, return_dict=False, generator=generator)
    except TypeError:
        result = scheduler.step(noise_pred, timestep, latents, return_dict=False)
    return result[0] if isinstance(result, tuple) else result.prev_sample


def _sigma_from_scheduler(scheduler: Any, step_index: int, timestep: torch.Tensor) -> float:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is not None and len(sigmas) > 0:
        idx = max(0, min(step_index, len(sigmas) - 1))
        return _as_float(sigmas[idx])

    num_train_timesteps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    return max(_as_float(timestep), 0.0) / num_train_timesteps


def _branch_noise_std(
    scheduler: Any,
    step_index: int,
    timestep: torch.Tensor,
    config: WanDLBSConfig,
) -> float:
    if config.branch_noise_std is not None:
        return config.branch_noise_std * config.branch_noise_scale
    if config.branch_noise_scale == 0:
        return 0.0

    current_sigma = _sigma_from_scheduler(scheduler, step_index, timestep)
    next_sigma = _sigma_from_scheduler(scheduler, step_index + 1, timestep)
    variance = max((current_sigma * current_sigma) - (next_sigma * next_sigma), 0.0)
    return math.sqrt(variance) * config.branch_noise_scale


def _add_branch_noise(
    latents: torch.Tensor,
    *,
    std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if std == 0:
        return latents
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return latents + std * noise


def _flow_step_between(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
) -> torch.Tensor:
    num_train_timesteps = float(getattr(scheduler.config, "num_train_timesteps", 1000))
    sigma_t = _as_float(timestep) / num_train_timesteps
    sigma_next = _as_float(next_timestep) / num_train_timesteps
    return sample + (sigma_next - sigma_t) * model_output


def _coarse_preview_x0(
    pipe: Any,
    scheduler: Any,
    latents: torch.Tensor,
    next_timestep: torch.Tensor,
    *,
    mask: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
    guidance_scale_2: float | None,
    attention_kwargs: dict[str, Any] | None,
    transformer_dtype: torch.dtype,
    num_lookahead_steps: int,
) -> torch.Tensor:
    preview = latents
    base_t = _as_float(next_timestep)
    for lookahead_idx in range(num_lookahead_steps, 0, -1):
        t_value = base_t * lookahead_idx / num_lookahead_steps
        next_t_value = base_t * (lookahead_idx - 1) / num_lookahead_steps
        t_tensor = torch.tensor(t_value, device=latents.device, dtype=next_timestep.dtype)
        next_t_tensor = torch.tensor(next_t_value, device=latents.device, dtype=next_timestep.dtype)
        model, current_guidance = _choose_model_and_guidance(
            pipe,
            t_tensor,
            guidance_scale,
            guidance_scale_2,
        )
        noise_pred = _predict_noise(
            pipe,
            model,
            preview,
            t_tensor,
            mask=mask,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=current_guidance,
            attention_kwargs=attention_kwargs,
            transformer_dtype=transformer_dtype,
        )
        preview = _flow_step_between(preview, noise_pred, scheduler, t_tensor, next_t_tensor)
    return preview


def _unscale_wan_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    if not hasattr(pipe.vae.config, "latents_mean") or not hasattr(pipe.vae.config, "latents_std"):
        return latents
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return latents / latents_std + latents_mean


def _decode_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    vae_latents = _unscale_wan_latents(pipe, latents.to(pipe.vae.dtype))
    return pipe.vae.decode(vae_latents, return_dict=False)[0]


def _reward_video_tensor(decoded_video: torch.Tensor, config: WanDLBSConfig) -> torch.Tensor:
    video = decoded_video[0].permute(1, 0, 2, 3)
    if config.reward_frame_stride > 1:
        video = video[:: config.reward_frame_stride]
    if config.reward_max_frames is not None:
        video = video[: config.reward_max_frames]
    return (((video + 1.0) / 2.0) * 255.0).clamp(0, 255).to(dtype=torch.uint8)


def _score_preview(
    pipe: Any,
    preview_latents: torch.Tensor,
    prompt: str,
    reward_model: VideoRewardFn,
    config: WanDLBSConfig,
) -> tuple[float, dict[str, float]]:
    decoded = _decode_latents(pipe, preview_latents)
    reward_video = _reward_video_tensor(decoded, config)
    return reward_model(reward_video, prompt)


def _write_trace_line(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


@torch.no_grad()
def generate_wan22_dlbs(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    seed: int,
    reward_model: VideoRewardFn,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: float | None = None,
    generator: torch.Generator | None = None,
    latents: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    snapshot_steps: set[int] | None = None,
    config: WanDLBSConfig | None = None,
) -> WanDLBSOutput:
    cfg = config or WanDLBSConfig()
    snapshot_steps = snapshot_steps or set()
    device = pipe._execution_device
    generator = generator or torch.Generator(device=device).manual_seed(seed)

    if cfg.trace_path is not None and cfg.trace_path.exists():
        cfg.trace_path.unlink()

    height, width, num_frames = _adjust_video_shape(pipe, height, width, num_frames)
    if pipe.config.boundary_ratio is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    pipe._guidance_scale = guidance_scale
    pipe._guidance_scale_2 = guidance_scale_2
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    batch_size = 1
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=guidance_scale > 1.0,
        num_videos_per_prompt=1,
        max_sequence_length=cfg.max_sequence_length,
        device=device,
    )
    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    transformer_dtype = transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)
    timesteps = pipe.scheduler.timesteps
    pipe._num_timesteps = len(timesteps)

    num_channels_latents = transformer.config.in_channels
    init_latents = pipe.prepare_latents(
        batch_size * cfg.num_beams,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )
    if init_latents.shape[0] == 1 and cfg.num_beams > 1:
        init_latents = init_latents.repeat(cfg.num_beams, 1, 1, 1, 1)
    if init_latents.shape[0] != cfg.num_beams:
        raise ValueError(
            f"DLBS expected {cfg.num_beams} initial beam latents, got batch {init_latents.shape[0]}"
        )
    mask = torch.ones(init_latents[:1].shape, dtype=torch.float32, device=device)
    beams = [
        _Beam(
            latent=init_latents[index : index + 1].clone(),
            scheduler=copy.deepcopy(pipe.scheduler),
            score=0.0,
            path=[],
        )
        for index in range(cfg.num_beams)
    ]

    captured: dict[int, torch.Tensor] = {}
    trace: list[dict[str, Any]] = []
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for step_index, timestep in enumerate(timesteps):
            pipe._current_timestep = timestep
            next_beams: list[_Beam] = []
            candidate_logs: list[dict[str, Any]] = []
            next_timestep = (
                timesteps[step_index + 1]
                if step_index + 1 < len(timesteps)
                else _zero_like_timestep(timestep, device=device)
            )

            for beam_index, beam in enumerate(beams):
                model, current_guidance = _choose_model_and_guidance(
                    pipe,
                    timestep,
                    guidance_scale,
                    guidance_scale_2,
                )
                noise_pred = _predict_noise(
                    pipe,
                    model,
                    beam.latent,
                    timestep,
                    mask=mask,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=current_guidance,
                    attention_kwargs=attention_kwargs,
                    transformer_dtype=transformer_dtype,
                )
                branch_std = _branch_noise_std(beam.scheduler, step_index, timestep, cfg)

                for candidate_index in range(cfg.num_candidates):
                    candidate_scheduler = copy.deepcopy(beam.scheduler)
                    candidate_latent = _scheduler_step(
                        candidate_scheduler,
                        noise_pred,
                        timestep,
                        beam.latent,
                        generator=generator,
                    )
                    should_keep_deterministic = (
                        cfg.include_deterministic_candidate and candidate_index == 0
                    )
                    if not should_keep_deterministic:
                        candidate_latent = _add_branch_noise(
                            candidate_latent,
                            std=branch_std,
                            generator=generator,
                        )

                    if step_index == len(timesteps) - 1 and not cfg.score_last_step:
                        score = beam.score
                        details: dict[str, float] = {"reward": score}
                    else:
                        preview_latent = _coarse_preview_x0(
                            pipe,
                            candidate_scheduler,
                            candidate_latent,
                            next_timestep,
                            mask=mask,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            guidance_scale=guidance_scale,
                            guidance_scale_2=guidance_scale_2,
                            attention_kwargs=attention_kwargs,
                            transformer_dtype=transformer_dtype,
                            num_lookahead_steps=cfg.num_lookahead_steps,
                        )
                        score, details = _score_preview(
                            pipe,
                            preview_latent,
                            prompt,
                            reward_model,
                            cfg,
                        )

                    path = beam.path + [candidate_index]
                    next_beams.append(
                        _Beam(
                            latent=candidate_latent,
                            scheduler=candidate_scheduler,
                            score=float(score),
                            path=path,
                        )
                    )
                    candidate_logs.append(
                        {
                            "beam_index": beam_index,
                            "candidate_index": candidate_index,
                            "score": float(score),
                            "score_details": {key: float(value) for key, value in details.items()},
                            "path": path,
                        }
                    )

            ranked = sorted(next_beams, key=lambda beam: beam.score, reverse=cfg.higher_is_better)
            beams = ranked[: cfg.num_beams]
            selected_paths = [beam.path for beam in beams]
            step_log = {
                "step": step_index,
                "timestep": _as_float(timestep),
                "branch_noise_std": float(branch_std),
                "candidates": candidate_logs,
                "selected_paths": selected_paths,
                "selected_scores": [float(beam.score) for beam in beams],
            }
            trace.append(step_log)
            _write_trace_line(cfg.trace_path, step_log)

            if step_index in snapshot_steps:
                captured[step_index] = beams[0].latent.detach().to("cpu", dtype=torch.float16).clone()

            if step_index == len(timesteps) - 1 or (
                (step_index + 1) > num_warmup_steps and (step_index + 1) % pipe.scheduler.order == 0
            ):
                progress_bar.update()

    pipe._current_timestep = None
    best_latents = beams[0].latent
    if cfg.output_type != "latent":
        decoded = _decode_latents(pipe, best_latents)
        frames = pipe.video_processor.postprocess_video(decoded, output_type=cfg.output_type)
    else:
        frames = best_latents

    pipe.maybe_free_model_hooks()
    return WanDLBSOutput(frames=frames, latents_by_step=captured, trace=trace)
