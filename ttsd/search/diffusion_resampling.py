from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ttsd.search.common import InterventionAction, InterventionContext, InterventionResult


def _as_float(value: float | torch.Tensor, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


def renoise_with_sigma_gap(
    latent: torch.Tensor,
    *,
    current_sigma: float | torch.Tensor,
    target_sigma: float | torch.Tensor,
    generator: torch.Generator | None = None,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Add restart noise matching sqrt(target_sigma^2 - current_sigma^2)."""

    current = _as_float(current_sigma, name="current_sigma")
    target = _as_float(target_sigma, name="target_sigma")
    if current < 0 or target < 0:
        raise ValueError("sigmas must be non-negative")
    if target <= current:
        raise ValueError(
            f"target_sigma must be greater than current_sigma, got {target} <= {current}"
        )
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")

    std = math.sqrt((target * target) - (current * current)) * noise_scale
    noise = torch.randn(
        latent.shape,
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    return latent + std * noise


@dataclass(frozen=True)
class ResamplingResult:
    latents: torch.Tensor
    indices: torch.Tensor
    normalized_weights: torch.Tensor


def multinomial_resample(
    latents: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    num_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> ResamplingResult:
    """Particle-filter style resampling from unnormalized log weights."""

    if latents.shape[0] != log_weights.numel():
        raise ValueError(
            f"latents first dim ({latents.shape[0]}) must match log_weights ({log_weights.numel()})"
        )
    count = int(num_samples if num_samples is not None else latents.shape[0])
    if count <= 0:
        raise ValueError("num_samples must be positive")

    weights = torch.softmax(log_weights.reshape(-1).to(dtype=torch.float32), dim=0)
    indices = torch.multinomial(
        weights,
        num_samples=count,
        replacement=True,
        generator=generator,
    )
    return ResamplingResult(
        latents=latents[indices],
        indices=indices,
        normalized_weights=weights,
    )


@dataclass(frozen=True)
class DiffusionResamplingConfig:
    target_sigma: float | None = None
    sigma_multiplier: float | None = 1.25
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.target_sigma is None and self.sigma_multiplier is None:
            raise ValueError("target_sigma or sigma_multiplier must be set")
        if self.sigma_multiplier is not None and self.sigma_multiplier <= 1.0:
            raise ValueError("sigma_multiplier must be > 1.0")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")


class DiffusionResamplingIntervention:
    """Restart-style intervention that re-noises a latent to a larger sigma."""

    def __init__(self, config: DiffusionResamplingConfig | None = None) -> None:
        self.config = config or DiffusionResamplingConfig()

    def intervene(
        self,
        ctx: InterventionContext,
        *,
        generator: torch.Generator | None = None,
    ) -> InterventionResult:
        if ctx.sigma is None:
            raise ValueError("InterventionContext.sigma is required for re-noise")

        current_sigma = _as_float(ctx.sigma, name="ctx.sigma")
        target_sigma = (
            self.config.target_sigma
            if self.config.target_sigma is not None
            else current_sigma * float(self.config.sigma_multiplier)
        )
        latent = renoise_with_sigma_gap(
            ctx.latent,
            current_sigma=current_sigma,
            target_sigma=target_sigma,
            generator=generator,
            noise_scale=self.config.noise_scale,
        )
        return InterventionResult(
            latent=latent,
            action=InterventionAction.RENOISE,
            extra_nfe=0,
            metadata={
                "current_sigma": current_sigma,
                "target_sigma": target_sigma,
                "noise_scale": self.config.noise_scale,
            },
        )
