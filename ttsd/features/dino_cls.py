"""Reusable DINOv2 CLS extraction and reduction helpers.

The runners in :mod:`ttsd.runners.features` handle CLI/file IO.  This module
keeps the reusable feature math small enough for verifiers and tests to use
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_DINO_MODEL_NAME = "facebook/dinov2-base"


def l2_normalize(values: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Normalize the last axis of ``values``."""
    arr = np.asarray(values, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + eps)


def cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity along the last axis."""
    aa = l2_normalize(np.asarray(a), eps=eps)
    bb = l2_normalize(np.asarray(b), eps=eps)
    return np.sum(aa * bb, axis=-1)


def diagonal_similarity(cls_grid: np.ndarray) -> np.ndarray:
    """cos(F[step, frame], F[step + 1, frame + 1]).

    ``cls_grid`` must have shape ``(steps, frames, dim)``.
    """
    grid = _as_step_frame_grid(cls_grid)
    return cosine(grid[:-1, :-1, :], grid[1:, 1:, :])


def frame_neighbor_similarity(cls_grid: np.ndarray) -> np.ndarray:
    """cos(F[step, frame], F[step, frame + 1]).

    ``cls_grid`` may have shape ``(steps, frames, dim)`` or
    ``(frames, dim)``.  The output has one fewer frame entry.
    """
    arr = _as_frame_array(cls_grid)
    return cosine(arr[..., :-1, :], arr[..., 1:, :])


def posterior_neighbor_similarity(cls_grid: np.ndarray) -> np.ndarray:
    """cos(F[step, frame], F[step + 1, frame]).

    ``cls_grid`` must have shape ``(steps, frames, dim)``.
    """
    grid = _as_step_frame_grid(cls_grid)
    return cosine(grid[:-1, :, :], grid[1:, :, :])


def gap_cosine_profile(cls_step: np.ndarray, gap: int = 1) -> np.ndarray:
    """Frame-pair cosine profile for one or more CLS trajectories."""
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    arr = _as_frame_array(cls_step)
    if arr.shape[-2] <= gap:
        raise ValueError(f"Need more than {gap} frames, got {arr.shape[-2]}")
    return cosine(arr[..., :-gap, :], arr[..., gap:, :])


def frame_cosine_mean(cls_step: np.ndarray, gap: int = 1) -> float:
    """Mean adjacent-frame cosine similarity."""
    return float(gap_cosine_profile(cls_step, gap=gap).mean())


def frame_cosine_tail_low(cls_step: np.ndarray, tail_fraction: float = 0.20, gap: int = 1) -> float:
    """Mean of the lowest ``tail_fraction`` adjacent-frame cosine values."""
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    flat = np.sort(gap_cosine_profile(cls_step, gap=gap).reshape(-1))
    tail_count = max(1, int(np.ceil(flat.size * tail_fraction)))
    return float(flat[:tail_count].mean())


def frame_cosine_min(cls_step: np.ndarray, gap: int = 1) -> float:
    """Minimum adjacent-frame cosine similarity."""
    return float(gap_cosine_profile(cls_step, gap=gap).min())


def frame_cosine_std(cls_step: np.ndarray, gap: int = 1) -> float:
    """Standard deviation of adjacent-frame cosine similarity."""
    return float(gap_cosine_profile(cls_step, gap=gap).std())


def temporal_quantile_features(
    cls_step: np.ndarray,
    quantiles: Iterable[float] = (0.05, 0.50, 0.95),
) -> np.ndarray:
    """Concatenate per-dimension temporal quantiles over the frame axis."""
    arr = _as_frame_array(cls_step).astype(np.float32, copy=False)
    pieces = [np.quantile(arr, q, axis=-2).astype(np.float32) for q in quantiles]
    return np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)


def select_step(cls_by_step: np.ndarray, step_index: int | None) -> np.ndarray:
    """Select one step from ``(steps, frames, dim)`` or pass through ``(frames, dim)``."""
    arr = np.asarray(cls_by_step)
    if arr.ndim == 2:
        if step_index is not None:
            raise ValueError("step_index was provided for an already single-step CLS array")
        return arr
    if arr.ndim < 3:
        raise ValueError(f"Expected CLS array with at least 2 dims, got shape {arr.shape}")
    if step_index is None:
        raise ValueError("step_index is required for a multi-step CLS array")
    return np.take(arr, int(step_index), axis=-3)


def upto_gap_cosine_profiles(
    cls_by_step: np.ndarray,
    step_indices: Iterable[int],
    gap: int = 1,
) -> np.ndarray:
    """Concatenate gap-cosine profiles from selected posterior steps.

    ``cls_by_step`` is interpreted as ``(..., steps, frames, dim)``.  The
    returned feature vector has shape ``(..., len(step_indices) * (frames-gap))``.
    """
    indices = [int(i) for i in step_indices]
    if not indices:
        raise ValueError("step_indices must not be empty")
    arr = np.asarray(cls_by_step)
    if arr.ndim < 3:
        raise ValueError(f"Expected CLS grid with shape (..., steps, frames, dim), got {arr.shape}")
    selected = np.take(arr, indices, axis=-3)
    profiles = gap_cosine_profile(selected, gap=gap).astype(np.float32, copy=False)
    leading_shape = profiles.shape[:-2]
    return profiles.reshape(*leading_shape, -1).astype(np.float32, copy=False)


def candidate_devices(device_arg: str, gpu_indices: str | None) -> list[str]:
    """Resolve a CLI device argument to concrete devices."""
    if device_arg != "cuda":
        return [device_arg]

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false")

    if gpu_indices:
        indices = [part.strip() for part in gpu_indices.split(",") if part.strip()]
    else:
        indices = [str(i) for i in range(torch.cuda.device_count())]

    if not indices:
        raise ValueError("No CUDA devices were selected")
    return [f"cuda:{idx}" for idx in indices]


@dataclass
class DinoClsExtractor:
    """DINOv2 CLS extractor for in-memory video frames."""

    model_name: str = DEFAULT_DINO_MODEL_NAME
    batch_size: int = 32
    candidate_devices: tuple[str, ...] = ("cuda",)

    def extract_grid(self, frame_grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Extract normalized CLS features from ``(steps, frames, H, W, C)`` frames."""
        grid = np.asarray(frame_grid)
        if grid.ndim != 5:
            raise ValueError(f"Expected frame grid with shape (steps, frames, H, W, C), got {grid.shape}")
        features, device = self.extract_frames(grid.reshape(-1, *grid.shape[2:]))
        return features.reshape(grid.shape[0], grid.shape[1], -1), device

    def extract_frames(self, frames: np.ndarray) -> tuple[np.ndarray, str]:
        """Extract normalized CLS features from ``(frames, H, W, C)`` frames."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel

        frame_array = _as_uint8_frames(np.asarray(frames))
        if frame_array.ndim != 4:
            raise ValueError(f"Expected frames with shape (frames, H, W, C), got {frame_array.shape}")

        images = [Image.fromarray(frame) for frame in frame_array]
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        last_error: Exception | None = None

        for device in self.candidate_devices:
            model = None
            try:
                model = AutoModel.from_pretrained(self.model_name).to(device)
                model.eval()
                features = []
                with torch.no_grad():
                    for start in range(0, len(images), self.batch_size):
                        batch = images[start : start + self.batch_size]
                        inputs = processor(images=batch, return_tensors="pt")
                        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                        outputs = model(**inputs)
                        features.append(outputs.last_hidden_state[:, 0].cpu())
                feature_tensor = F.normalize(torch.cat(features, dim=0), dim=-1)
                return feature_tensor.numpy(), device
            except torch.OutOfMemoryError as exc:
                last_error = exc
                print(f"[DinoClsExtractor] OOM on {device}; trying next device")
            finally:
                if model is not None:
                    del model
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise RuntimeError(f"DINO CLS extraction failed on all candidate devices: {self.candidate_devices}") from last_error


def _as_frame_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least (frames, dim), got shape {arr.shape}")
    return arr


def _as_step_frame_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"Expected shape (steps, frames, dim), got {arr.shape}")
    return arr


def _as_uint8_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)
