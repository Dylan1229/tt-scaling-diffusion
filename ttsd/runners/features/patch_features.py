"""Extract DINOv2 *patch* tokens for one seed across the full posterior trajectory.

For each posterior step i (11 total) and each *subsampled* frame k we run
DINOv2 and keep `last_hidden_state[:, 1:]` (the patch tokens; the CLS token is
omitted).  Output is saved as float16 with shape (T, N_sub, P, D), where for
DINOv2-base at 224x224 input P = 256 and D = 768.

Frames are subsampled by `--frame-stride` (default 4 -> 21 of 81 frames) to
keep storage manageable; subsampled frame indices are stored alongside the
tensor in the metadata JSON.

Usage:
    python -m ttsd.runners.features.patch_features \
        --dino-input-frames-dir /path/to/dino_input_frames/seed_dir \
        --output-dir /path/to/seed_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
PATCH_FEATURES_FILE = "posterior_mean_patch_features.npy"
PATCH_META_FILE = "posterior_mean_patch_features_metadata.json"


def _sorted_video_files(video_dir: Path) -> list[Path]:
    files = sorted(video_dir.glob("step_*.mp4"))
    if not files:
        raise FileNotFoundError(f"No step_*.mp4 files found under {video_dir}")
    return files


def _candidate_devices(device_arg: str, gpu_indices: str | None) -> list[str]:
    if device_arg != "cuda":
        return [device_arg]
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false")
    if gpu_indices:
        indices = [p.strip() for p in gpu_indices.split(",") if p.strip()]
    else:
        indices = [str(i) for i in range(torch.cuda.device_count())]
    return [f"cuda:{idx}" for idx in indices]


def _load_subsampled_frame_grid(video_files: list[Path], frame_stride: int):
    grid = []
    step_labels = []
    frame_indices: list[int] | None = None
    for path in video_files:
        frames = np.stack(list(iio.imiter(path)))
        idxs = list(range(0, frames.shape[0], frame_stride))
        if frame_indices is None:
            frame_indices = idxs
        elif idxs != frame_indices:
            raise ValueError(f"Frame-count mismatch in {path}")
        grid.append(frames[idxs])
        step_labels.append(path.stem.replace("step_", ""))
    return np.stack(grid), step_labels, frame_indices


def _extract_patch_features(
    frame_grid: np.ndarray,
    model_name: str,
    batch_size: int,
    candidate_devices: list[str],
) -> tuple[np.ndarray, str]:
    T, N = frame_grid.shape[0], frame_grid.shape[1]
    images = [Image.fromarray(frame) for frame in frame_grid.reshape(-1, *frame_grid.shape[2:])]
    processor = AutoImageProcessor.from_pretrained(model_name)
    last_error: Exception | None = None
    for device in candidate_devices:
        model = None
        try:
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(images), batch_size):
                    batch = images[start : start + batch_size]
                    inputs = processor(images=batch, return_tensors="pt")
                    inputs = {n: t.to(device) for n, t in inputs.items()}
                    out = model(**inputs)
                    # drop CLS token
                    patch = out.last_hidden_state[:, 1:]
                    patch = F.normalize(patch, dim=-1)
                    chunks.append(patch.to(torch.float16).cpu())
            tensor = torch.cat(chunks, dim=0)  # (T*N, P, D)
            P, D = tensor.shape[1], tensor.shape[2]
            return tensor.numpy().reshape(T, N, P, D), device
        except torch.OutOfMemoryError as exc:
            last_error = exc
            print(f"[patch_features] OOM on {device}; trying next")
        finally:
            if model is not None:
                del model
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"Patch extraction failed on all devices: {candidate_devices}") from last_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-input-frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-indices", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args(argv)

    video_files = _sorted_video_files(args.dino_input_frames_dir)
    grid, step_labels, frame_indices = _load_subsampled_frame_grid(video_files, args.frame_stride)
    devices = _candidate_devices(args.device, args.gpu_indices)
    feats, used_device = _extract_patch_features(grid, args.model_name, args.batch_size, devices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / PATCH_FEATURES_FILE
    np.save(out_path, feats)

    meta = {
        "model_name": args.model_name,
        "device_used": used_device,
        "step_labels": step_labels,
        "frame_indices": frame_indices,
        "frame_stride": args.frame_stride,
        "shape": list(feats.shape),
        "dtype": str(feats.dtype),
        "patch_features_file": PATCH_FEATURES_FILE,
    }
    (args.output_dir / PATCH_META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"[patch_features] wrote {out_path}  shape={feats.shape}  dtype={feats.dtype}")


if __name__ == "__main__":
    main()
