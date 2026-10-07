"""Independent implementation of Fig. S1 in arXiv:2603.14320.

The paper does not specify subpixel layout, channel sharing, padding, or the
projection activation. Our explicit choices are documented in README.md.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv3d(nn.Module):
    """A 2x3x3 convolution; replicate padding on space and the past only."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (2, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (1, 1, 1, 1, 1, 0), mode="replicate"))


class CausalResidualBlock(nn.Module):
    """Fig. S1(b): three convolutions on the main path, one on the skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            CausalConv3d(channels, channels),
            nn.ReLU(),
            CausalConv3d(channels, channels),
            nn.ReLU(),
            CausalConv3d(channels, channels),
        )
        self.skip = CausalConv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + self.skip(x))


class LearnedSpatialUpsample(nn.Module):
    """Learn a past/current 3x3 stencil for each spatial subpixel.

    Filters are shared across latent channels: a small Conv3d(1, scale**2)
    applied to each channel separately, followed by per-frame pixel shuffle.
    This is our compact realization of Fig. S1(c), not released author code.
    """

    def __init__(self, scale: int):
        super().__init__()
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
            raise ValueError("scale must be a positive integer")
        self.scale = scale
        self.stencil = CausalConv3d(1, scale * scale)
        # Start from nearest-neighbor spatial enlargement, then learn the stencil.
        nn.init.zeros_(self.stencil.conv.weight)
        nn.init.zeros_(self.stencil.conv.bias)
        with torch.no_grad():
            self.stencil.conv.weight[:, 0, 1, 1, 1] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        phases = self.stencil(x.reshape(b * c, 1, t, h, w))
        phases = phases.permute(0, 2, 1, 3, 4).reshape(b * c * t, self.scale**2, h, w)
        enlarged = F.pixel_shuffle(phases, self.scale)
        return enlarged.reshape(b, c, t, h * self.scale, w * self.scale)


class L2RDecoder(nn.Module):
    """Native VAE latents [B,C,T,H,W] -> RGB [B,3,T,sH,sW].

    No temporal upsampling. Output is unconstrained during training; clamp to
    [-1,1] only when exporting. Width stays at C until the final RGB projection.
    """

    temporal_context = 7  # 3 (pre block) + 1 (upsampler) + 3 (post block)

    def __init__(self, latent_channels: int = 16, spatial_scale: int = 8):
        super().__init__()
        if not isinstance(latent_channels, int) or isinstance(latent_channels, bool) or latent_channels < 1:
            raise ValueError("latent_channels must be a positive integer")
        self.latent_channels = latent_channels
        self.spatial_scale = spatial_scale
        self.pre = CausalResidualBlock(latent_channels)
        self.upsample = LearnedSpatialUpsample(spatial_scale)
        self.post = CausalResidualBlock(latent_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(latent_channels, 3, 3, padding=1),
        )

    @property
    def architecture(self) -> dict:
        return {"latent_channels": self.latent_channels, "spatial_scale": self.spatial_scale}

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(f"Expected [B,{self.latent_channels},T,H,W], got {tuple(latent.shape)}")
        if min(latent.shape) < 1 or not latent.is_floating_point():
            raise ValueError("Latent must be a nonempty floating-point tensor")
        x = self.post(self.upsample(self.pre(latent)))
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        rgb = self.projection(frames)
        return rgb.reshape(b, t, 3, h, w).permute(0, 2, 1, 3, 4)

    def iter_chunks(self, latent: torch.Tensor, chunk_frames: int = 4):
        """Yield (start, RGB) with the causal context needed for exact chunking.

        The caller can immediately move each output to CPU to limit device RAM.
        This recomputes overlap instead of keeping architecture-specific caches.
        """
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        for start in range(0, latent.shape[2], chunk_frames):
            context_start = max(0, start - self.temporal_context)
            stop = min(start + chunk_frames, latent.shape[2])
            decoded = self(latent[:, :, context_start:stop])
            yield start, decoded[:, :, start - context_start :]
