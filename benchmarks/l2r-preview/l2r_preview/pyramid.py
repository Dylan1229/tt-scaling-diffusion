"""Efficient experimental preview architecture; quality depends on trained weights.

All temporal mixing happens at latent resolution. Spatial reconstruction uses
progressive x2 subpixel stages with shrinking width, ending directly in RGB.
This is a new experimental design, not the paper's L2R architecture or weights.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LowResolutionCausalBlock(nn.Module):
    """Spatial residual processing plus channel-wise causal temporal mixing."""
    def __init__(self, width: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), groups=width, bias=False)
        # Initially act on the current frame; past-frame taps remain trainable.
        nn.init.zeros_(self.temporal.weight)
        with torch.no_grad():
            self.temporal.weight[:, 0, -1, 0, 0] = 1

    def forward(self, frames, batch, time):
        spatial = self.spatial(frames)
        _, channels, height, width = spatial.shape
        history = spatial.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
        history = self.temporal(F.pad(history, (0, 0, 0, 0, 2, 0), mode="replicate"))
        mixed = history.permute(0, 2, 1, 3, 4).reshape_as(frames)
        return F.relu(frames + 0.1 * mixed)


class SubpixelUp(nn.Module):
    """Learned x2 enlargement; equal initial phases avoid random checkerboard starts."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 4*out_channels, 3, padding=1)
        base = torch.empty(out_channels, in_channels, 3, 3)
        nn.init.kaiming_normal_(base, nonlinearity="relu")
        with torch.no_grad():
            self.projection.weight.copy_(base.repeat_interleave(4, dim=0))
            self.projection.bias.zero_()

    def forward(self, x):
        return F.pixel_shuffle(self.projection(x), 2)


class SpatialRefinement(nn.Module):
    """Cheap per-frame detail refinement after an upsampling stage."""
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x):
        return F.relu(x + 0.1*self.layers(x))


class PyramidPreviewDecoder(nn.Module):
    """Native Wan latents [B,48,T,H,W] -> RGB [B,3,T,16H,16W].

    Produces the same 21-frame preview contract as the original L2R adaptation.
    No spatial convolution operates at final RGB resolution. RGB output is
    unconstrained; clamp to [-1,1] only for export.
    """
    def __init__(self, latent_channels=48, width=96, blocks=4, stage_widths=(64, 32, 16)):
        super().__init__()
        if min(latent_channels, width, blocks, *stage_widths) < 1 or len(stage_widths) != 3:
            raise ValueError("Positive widths and three spatial feature stages are required")
        self.latent_channels = latent_channels
        self.width, self.blocks = width, blocks
        self.stage_widths = tuple(stage_widths)
        self.spatial_scale = 16
        self.temporal_context = 2 * blocks
        # Conservative latent-space halo for exact interior crop supervision.
        self.spatial_halo = 2 * blocks + 4
        self.stem = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.trunk = nn.ModuleList([LowResolutionCausalBlock(width) for _ in range(blocks)])
        stages = []
        previous = width
        for current in stage_widths:
            stages.append(nn.Sequential(SubpixelUp(previous, current), nn.ReLU(inplace=True),
                                        SpatialRefinement(current)))
            previous = current
        self.stages = nn.Sequential(*stages)
        self.rgb = SubpixelUp(previous, 3)

    @classmethod
    def from_variant(cls, name):
        configs = {
            "compact": {"width": 64, "blocks": 3, "stage_widths": (32, 16, 8)},
            "balanced": {"width": 96, "blocks": 4, "stage_widths": (64, 32, 16)},
            "quality": {"width": 128, "blocks": 4, "stage_widths": (96, 48, 24)},
            "wide": {"width": 192, "blocks": 6, "stage_widths": (128, 64, 32)},
        }
        if name not in configs:
            raise ValueError(f"Unknown variant: {name}")
        return cls(**configs[name])

    @property
    def architecture(self):
        return {"kind": "pyramid_preview_v1", "latent_channels": self.latent_channels,
                "width": self.width, "blocks": self.blocks, "stage_widths": self.stage_widths,
                "spatial_scale": 16, "temporal_context": self.temporal_context,
                "spatial_halo": self.spatial_halo}

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, latent, output_start=0, output_frames=None):
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(f"Expected [B,{self.latent_channels},T,H,W]")
        if min(latent.shape) < 1 or not latent.is_floating_point():
            raise ValueError("Expected a nonempty floating-point latent")
        batch, channels, time, height, width = latent.shape
        frames = latent.permute(0, 2, 1, 3, 4).reshape(batch*time, channels, height, width)
        frames = F.relu(self.stem(frames))
        for block in self.trunk:
            frames = block(frames, batch, time)
        count = time-output_start if output_frames is None else output_frames
        if output_start < 0 or count < 1 or output_start+count > time:
            raise ValueError("Invalid requested output frame range")
        # All causal mixing is complete: discarded history needs no RGB synthesis.
        frames = frames.reshape(batch, time, self.width, height, width)[:, output_start:output_start+count]
        frames = frames.reshape(batch*count, self.width, height, width)
        rgb = self.rgb(self.stages(frames))
        return rgb.reshape(batch, count, 3, height*16, width*16).permute(0, 2, 1, 3, 4)

    def iter_chunks(self, latent, chunk_frames=4):
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        for start in range(0, latent.shape[2], chunk_frames):
            context_start = max(0, start-self.temporal_context)
            stop = min(latent.shape[2], start+chunk_frames)
            rgb = self(latent[:, :, context_start:stop])
            yield start, rgb[:, :, start-context_start:]
