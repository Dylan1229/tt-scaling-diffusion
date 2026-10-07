"""Full-video fidelity metrics for aligned RGB previews in [0, 1]."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def rgb_ssim(reference, candidate):
    """Wang SSIM: 11x11 Gaussian, sigma=1.5, population variance, valid interior."""
    coordinate = torch.arange(11, device=reference.device, dtype=torch.float32)-5
    kernel = torch.exp(-coordinate.square()/(2*1.5**2))
    kernel = kernel/kernel.sum()
    weight = (kernel[:, None]*kernel[None, :]).expand(3, 1, 11, 11).contiguous()
    a, b = reference.float(), candidate.float()
    def filt(x):
        return F.conv2d(x, weight, groups=3)
    ma, mb = filt(a), filt(b)
    va, vb = filt(a*a)-ma.square(), filt(b*b)-mb.square()
    cov = filt(a*b)-ma*mb
    score = ((2*ma*mb+0.01**2)*(2*cov+0.03**2))/(
        (ma.square()+mb.square()+0.01**2)*(va+vb+0.03**2))
    return float(score.mean())


def full_video_metrics(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 5:
        raise ValueError("Expected matching NCTHW RGB")
    reference, candidate = reference.float(), candidate.float()
    mse = float((reference-candidate).square().mean())
    indices = sorted({0, reference.shape[2]//2, reference.shape[2]-1})
    a = reference[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, 3, *reference.shape[-2:])
    b = candidate[:, :, indices].permute(0, 2, 1, 3, 4).reshape_as(a)
    return {"full_mse_01": mse, "full_psnr_db": -10*math.log10(max(mse, 1e-12)),
            "ssim": rgb_ssim(a, b), "ssim_frames": indices,
            "temporal_delta_mae": float((torch.diff(reference, dim=2)-torch.diff(candidate, dim=2)).abs().mean())}
