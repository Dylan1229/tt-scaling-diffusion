"""Build a posterior-mean DINOv2 similarity heatmap for one seed.

Usage:
    python -m ttsd.runners.posterior_mean_heatmap \
        --posterior-mean-video-dir /path/to/decoded/posterior_means \
        --output-dir /path/to/seed_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"


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
        indices = [part.strip() for part in gpu_indices.split(",") if part.strip()]
    else:
        indices = [str(i) for i in range(torch.cuda.device_count())]

    if not indices:
        raise ValueError("No CUDA devices were selected")

    return [f"cuda:{idx}" for idx in indices]


def _load_frame_grid(video_files: list[Path]) -> tuple[np.ndarray, list[str]]:
    frame_grid = []
    step_labels = []
    expected_frame_count = None

    for video_path in video_files:
        frames = np.stack(list(iio.imiter(video_path)))
        if expected_frame_count is None:
            expected_frame_count = frames.shape[0]
        elif frames.shape[0] != expected_frame_count:
            raise ValueError(
                f"Frame-count mismatch for {video_path}: expected {expected_frame_count}, got {frames.shape[0]}"
            )
        frame_grid.append(frames)
        step_labels.append(video_path.stem.replace("step_", ""))

    return np.stack(frame_grid), step_labels


def _extract_features(
    frame_grid: np.ndarray,
    model_name: str,
    batch_size: int,
    candidate_devices: list[str],
) -> tuple[np.ndarray, str]:
    images = [Image.fromarray(frame) for frame in frame_grid.reshape(-1, *frame_grid.shape[2:])]
    processor = AutoImageProcessor.from_pretrained(model_name)
    last_error: Exception | None = None

    for device in candidate_devices:
        model = None
        try:
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            features = []
            with torch.no_grad():
                for start in range(0, len(images), batch_size):
                    batch = images[start : start + batch_size]
                    inputs = processor(images=batch, return_tensors="pt")
                    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                    outputs = model(**inputs)
                    batch_features = outputs.last_hidden_state[:, 0]
                    features.append(batch_features.cpu())
            feature_tensor = torch.cat(features, dim=0)
            feature_tensor = F.normalize(feature_tensor, dim=-1)
            return feature_tensor.numpy().reshape(frame_grid.shape[0], frame_grid.shape[1], -1), device
        except torch.OutOfMemoryError as exc:
            last_error = exc
            print(f"[posterior_mean_heatmap] OOM on {device}; trying next device")
        finally:
            if model is not None:
                del model
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"Feature extraction failed on all candidate devices: {candidate_devices}") from last_error


def _compute_diagonal_similarity(feature_grid: np.ndarray) -> np.ndarray:
    current = feature_grid[:-1, :-1, :]
    next_diag = feature_grid[1:, 1:, :]
    return np.sum(current * next_diag, axis=-1)


def _save_heatmap(similarity: np.ndarray, step_labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(similarity, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Posterior-mean step transition")
    ax.set_title("DINOv2 diagonal neighbor cosine similarity")
    ax.set_xticks(np.arange(similarity.shape[1]))
    ax.set_yticks(np.arange(similarity.shape[0]))
    ax.set_yticklabels([f"{step_labels[i]}->{step_labels[i + 1]}" for i in range(len(step_labels) - 1)])
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-mean-video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-indices", type=str, default="4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    video_files = _sorted_video_files(args.posterior_mean_video_dir)
    frame_grid, step_labels = _load_frame_grid(video_files)
    candidate_devices = _candidate_devices(args.device, args.gpu_indices)
    feature_grid, used_device = _extract_features(
        frame_grid=frame_grid,
        model_name=args.model_name,
        batch_size=args.batch_size,
        candidate_devices=candidate_devices,
    )
    similarity = _compute_diagonal_similarity(feature_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "posterior_mean_diagonal_similarity.npy"
    heatmap_path = args.output_dir / "posterior_mean_diagonal_similarity_heatmap.png"
    metadata_path = args.output_dir / "posterior_mean_diagonal_similarity_metadata.json"

    np.save(matrix_path, similarity)
    _save_heatmap(similarity, step_labels, heatmap_path)

    metadata = {
        "posterior_mean_video_dir": str(args.posterior_mean_video_dir.resolve()),
        "model_name": args.model_name,
        "used_device": used_device,
        "grid_shape": {
            "k": int(feature_grid.shape[0]),
            "n": int(feature_grid.shape[1]),
            "d": int(feature_grid.shape[2]),
        },
        "similarity_shape": list(similarity.shape),
        "step_labels": step_labels,
        "definition": "similarity[i, j] = cosine(feature[i, j], feature[i + 1, j + 1])",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps({
        "matrix_path": str(matrix_path),
        "heatmap_path": str(heatmap_path),
        "metadata_path": str(metadata_path),
        "used_device": used_device,
        "similarity_shape": list(similarity.shape),
    }, indent=2))


if __name__ == "__main__":
    main()