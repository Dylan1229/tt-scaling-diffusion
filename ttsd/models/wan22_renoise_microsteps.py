from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ttsd.models.wan22_dlbs import (
    _adjust_video_shape,
    _as_float,
    _as_timestep_tensor,
    _choose_model_and_guidance,
    _decode_latents,
    _flow_step_between,
    _predict_noise,
    _scheduler_step,
    _sigma_from_scheduler,
    _zero_like_timestep,
)
from ttsd.search.diffusion_resampling import renoise_with_sigma_gap
from ttsd.search.renoise_microsteps import (
    RenoiseMicrostepWindow,
    build_renoise_replay_segment,
)


@dataclass(frozen=True)
class WanRenoiseMicrostepsConfig:
    trigger_step: int = 20
    rollback_to_step: int = 18
    extra_microsteps: int = 5
    noise_scale: float = 1.0
    index_base: int = 1
    max_sequence_length: int = 512
    output_type: str = "np"
    trace_path: Path | None = None

    def window(self) -> RenoiseMicrostepWindow:
        return RenoiseMicrostepWindow(
            trigger_step=self.trigger_step,
            rollback_to_step=self.rollback_to_step,
            extra_microsteps=self.extra_microsteps,
            noise_scale=self.noise_scale,
            index_base=self.index_base,
        )


@dataclass
class WanRenoiseMicrostepsOutput:
    frames: Any
    latents_by_step: dict[int, torch.Tensor]
    trace: list[dict[str, Any]]


def _write_trace_line(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


@torch.no_grad()
def generate_wan22_renoise_microsteps(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    seed: int,
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
    config: WanRenoiseMicrostepsConfig | None = None,
) -> WanRenoiseMicrostepsOutput:
    """Generate with a local re-noise restart followed by replay microsteps.

    This is intentionally implemented on the first-order Wan loop rather than
    through ``set_timesteps(sigmas=...)``. The intervention must run the trigger
    step normally, then reset the latent to an earlier noise level and replay the
    rollback window. A static non-monotone scheduler would alter the trigger
    step's own ``sigma_next`` and would not represent that algorithm.
    """

    cfg = config or WanRenoiseMicrostepsConfig()
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
    if getattr(pipe.scheduler, "order", 1) != 1:
        raise ValueError(
            "Renoise+microsteps currently requires a first-order scheduler "
            "(use model.scheduler=euler or euler_sde)."
        )
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(0)
    timesteps = pipe.scheduler.timesteps
    pipe._num_timesteps = len(timesteps)
    segment = build_renoise_replay_segment(timesteps, cfg.window())

    num_channels_latents = transformer.config.in_channels
    current_latents = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )
    mask = torch.ones(current_latents.shape, dtype=torch.float32, device=device)

    captured: dict[int, torch.Tensor] = {}
    trace: list[dict[str, Any]] = []
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    progress_total = num_inference_steps + segment.extra_nfe

    with pipe.progress_bar(total=progress_total) as progress_bar:
        for step_index, timestep in enumerate(timesteps):
            pipe._current_timestep = timestep
            model, current_guidance = _choose_model_and_guidance(
                pipe,
                timestep,
                guidance_scale,
                guidance_scale_2,
            )
            noise_pred = _predict_noise(
                pipe,
                model,
                current_latents,
                timestep,
                mask=mask,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=current_guidance,
                attention_kwargs=attention_kwargs,
                transformer_dtype=transformer_dtype,
            )
            current_latents = _scheduler_step(
                pipe.scheduler,
                noise_pred,
                timestep,
                current_latents,
                generator=generator,
            )

            step_log: dict[str, Any] = {
                "step": step_index,
                "timestep": _as_float(timestep),
                "action": "base_step",
            }

            if step_index == segment.trigger_index:
                next_timestep = (
                    timesteps[step_index + 1]
                    if step_index + 1 < len(timesteps)
                    else _zero_like_timestep(timestep, device=device)
                )
                current_sigma = _sigma_from_scheduler(
                    pipe.scheduler,
                    segment.resume_index,
                    _as_timestep_tensor(next_timestep, device=device),
                )
                target_sigma = _sigma_from_scheduler(
                    pipe.scheduler,
                    segment.rollback_index,
                    _as_timestep_tensor(segment.rollback_timestep, device=device),
                )
                current_latents = renoise_with_sigma_gap(
                    current_latents,
                    current_sigma=current_sigma,
                    target_sigma=target_sigma,
                    generator=generator,
                    noise_scale=cfg.noise_scale,
                )

                replay_logs: list[dict[str, float | int]] = []
                replay_values = list(segment.replay_timesteps)
                for replay_index, replay_timestep_value in enumerate(replay_values):
                    replay_timestep = _as_timestep_tensor(replay_timestep_value, device=device)
                    next_value = (
                        replay_values[replay_index + 1]
                        if replay_index + 1 < len(replay_values)
                        else segment.resume_timestep
                    )
                    next_replay_timestep = _as_timestep_tensor(next_value, device=device)
                    model, replay_guidance = _choose_model_and_guidance(
                        pipe,
                        replay_timestep,
                        guidance_scale,
                        guidance_scale_2,
                    )
                    replay_noise_pred = _predict_noise(
                        pipe,
                        model,
                        current_latents,
                        replay_timestep,
                        mask=mask,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        guidance_scale=replay_guidance,
                        attention_kwargs=attention_kwargs,
                        transformer_dtype=transformer_dtype,
                    )
                    current_latents = _flow_step_between(
                        current_latents,
                        replay_noise_pred,
                        pipe.scheduler,
                        replay_timestep,
                        next_replay_timestep,
                    )
                    replay_logs.append(
                        {
                            "replay_index": replay_index,
                            "timestep": float(replay_timestep_value),
                            "next_timestep": float(next_value),
                        }
                    )
                    progress_bar.update()

                step_log.update(
                    {
                        "action": "renoise_replay",
                        "rollback_step": cfg.rollback_to_step,
                        "trigger_step": cfg.trigger_step,
                        "current_sigma": float(current_sigma),
                        "target_sigma": float(target_sigma),
                        "noise_scale": float(cfg.noise_scale),
                        "extra_microsteps": int(cfg.extra_microsteps),
                        "extra_nfe": int(segment.extra_nfe),
                        "replay": replay_logs,
                    }
                )

            trace.append(step_log)
            _write_trace_line(cfg.trace_path, step_log)

            if step_index in snapshot_steps:
                captured[step_index] = current_latents.detach().to("cpu", dtype=torch.float16).clone()

            if step_index == len(timesteps) - 1 or (
                (step_index + 1) > num_warmup_steps and (step_index + 1) % pipe.scheduler.order == 0
            ):
                progress_bar.update()

    pipe._current_timestep = None
    if cfg.output_type != "latent":
        decoded = _decode_latents(pipe, current_latents)
        frames = pipe.video_processor.postprocess_video(decoded, output_type=cfg.output_type)
    else:
        frames = current_latents

    pipe.maybe_free_model_hooks()
    cfg_payload = asdict(cfg)
    if cfg_payload.get("trace_path") is not None:
        cfg_payload["trace_path"] = str(cfg_payload["trace_path"])
    trace_meta = {
        "config": cfg_payload,
        "segment": {
            "rollback_index": segment.rollback_index,
            "trigger_index": segment.trigger_index,
            "resume_index": segment.resume_index,
            "rollback_timestep": segment.rollback_timestep,
            "resume_timestep": segment.resume_timestep,
            "replay_timesteps": segment.replay_timesteps,
        },
    }
    trace.insert(0, {"action": "config", **trace_meta})
    return WanRenoiseMicrostepsOutput(frames=frames, latents_by_step=captured, trace=trace)
