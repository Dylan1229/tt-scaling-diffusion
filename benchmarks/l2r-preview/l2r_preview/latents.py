"""Explicit bridge from TTS/Wan diffusion-space latents to VAE-space latents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class WanLatentSpec:
    latent_channels: int
    spatial_scale: int
    temporal_scale: int
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self):
        if min(self.latent_channels, self.spatial_scale, self.temporal_scale) < 1:
            raise ValueError("Channel count and compression scales must be positive")
        if len(self.mean) != self.latent_channels or len(self.std) != self.latent_channels:
            raise ValueError("Mean/std lengths must match the latent channel count")
        values = torch.tensor([*self.mean, *self.std], dtype=torch.float64)
        if not torch.isfinite(values).all() or any(v <= 0 for v in self.std):
            raise ValueError("Normalization values must be finite and std must be positive")

    @classmethod
    def from_vae_config(cls, config: dict | str | Path) -> WanLatentSpec:
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text())
        return cls(
            latent_channels=int(config["z_dim"]),
            spatial_scale=int(config["scale_factor_spatial"]),
            temporal_scale=int(config["scale_factor_temporal"]),
            mean=tuple(config["latents_mean"]),
            std=tuple(config["latents_std"]),
        )

    @classmethod
    def from_dict(cls, data: dict) -> WanLatentSpec:
        return cls(**{**data, "mean": tuple(data["mean"]), "std": tuple(data["std"])})

    def to_dict(self) -> dict:
        return asdict(self)

    def _stats(self, z: torch.Tensor):
        if z.ndim != 5 or z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected [B,{self.latent_channels},T,H,W]")
        mean = z.new_tensor(self.mean).view(1, -1, 1, 1, 1)
        std = z.new_tensor(self.std).view(1, -1, 1, 1, 1)
        return mean, std

    def denormalize(self, diffusion_latent: torch.Tensor) -> torch.Tensor:
        """Matches the TTS native Wan decoder: z_vae = z_diffusion * std + mean."""
        mean, std = self._stats(diffusion_latent)
        return diffusion_latent * std + mean

    def normalize(self, vae_latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(vae_latent)
        return (vae_latent - mean) / std

    def rgb_frame_indices(self, latent_frames: int) -> torch.Tensor:
        """Causal VAE endpoint convention: 0, stride, 2*stride, ..."""
        return torch.arange(latent_frames) * self.temporal_scale


def flow_prediction_to_clean(
    sample: torch.Tensor, velocity: torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    """For z_sigma=(1-sigma)*z0+sigma*noise and v=noise-z0, return z0_hat.

    This matches TTS's FlowMatchEuler branch. Sample, velocity and sigma must
    belong to the SAME pre-step state; post-step callback latents cannot be
    combined with a pre-step model output. For UniPC, capture its own converted
    x0 prediction as TTS already does.
    """
    if sample.shape != velocity.shape or sample.ndim != 5:
        raise ValueError("Sample and velocity must have the same [B,C,T,H,W] shape")
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    if not torch.isfinite(sigma).all() or (sigma < 0).any() or (sigma > 1).any():
        raise ValueError("Flow sigma must be finite and in [0,1]")
    if sigma.ndim == 1:
        if sigma.numel() != sample.shape[0]:
            raise ValueError("A sigma vector must have one value per batch element")
        sigma = sigma.view(-1, 1, 1, 1, 1)
    elif sigma.ndim != 0:
        raise ValueError("Use a scalar sigma or a batch-length vector")
    return sample - sigma * velocity
