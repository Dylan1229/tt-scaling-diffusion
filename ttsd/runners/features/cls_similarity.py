"""Build posterior-mean DINOv2 similarity heatmaps for one seed.

Usage:
    python -m ttsd.runners.features.cls_similarity \
        --dino-input-frames-dir /path/to/dino_input_frames/seed_dir \
        --output-dir /path/to/seed_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from ttsd.features.dino_cls import (
    DinoClsExtractor,
    candidate_devices as _resolve_candidate_devices,
    diagonal_similarity,
    frame_neighbor_similarity,
    posterior_neighbor_similarity,
)

MODEL_NAME = "facebook/dinov2-base"


def _sorted_video_files(video_dir: Path) -> list[Path]:
    files = sorted(video_dir.glob("step_*.mp4"))
    if not files:
        raise FileNotFoundError(f"No step_*.mp4 files found under {video_dir}")
    return files


def _candidate_devices(device_arg: str, gpu_indices: str | None) -> list[str]:
    return _resolve_candidate_devices(device_arg, gpu_indices)


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
    extractor = DinoClsExtractor(
        model_name=model_name,
        batch_size=batch_size,
        candidate_devices=tuple(candidate_devices),
    )
    return extractor.extract_grid(frame_grid)


def _compute_diagonal_similarity(feature_grid: np.ndarray) -> np.ndarray:
    return diagonal_similarity(feature_grid)


def _compute_frame_neighbor_similarity(feature_grid: np.ndarray) -> np.ndarray:
    return frame_neighbor_similarity(feature_grid)


def _compute_posterior_neighbor_similarity(feature_grid: np.ndarray) -> np.ndarray:
    return posterior_neighbor_similarity(feature_grid)


def _save_heatmap(
    similarity: np.ndarray,
    step_labels: list[str],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ytick_mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(similarity, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(similarity.shape[1]))
    ax.set_yticks(np.arange(similarity.shape[0]))
    if ytick_mode == "transitions":
        ax.set_yticklabels([f"{step_labels[i]}->{step_labels[i + 1]}" for i in range(len(step_labels) - 1)])
    elif ytick_mode == "steps":
        ax.set_yticklabels(step_labels)
    else:
        raise ValueError(f"Unsupported ytick_mode: {ytick_mode}")
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-input-frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-indices", type=str, default="4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    video_files = _sorted_video_files(args.dino_input_frames_dir)
    frame_grid, step_labels = _load_frame_grid(video_files)
    candidate_devices = _candidate_devices(args.device, args.gpu_indices)
    feature_grid, used_device = _extract_features(
        frame_grid=frame_grid,
        model_name=args.model_name,
        batch_size=args.batch_size,
        candidate_devices=candidate_devices,
    )
    diagonal_similarity = _compute_diagonal_similarity(feature_grid)
    frame_neighbor_similarity = _compute_frame_neighbor_similarity(feature_grid)
    posterior_neighbor_similarity = _compute_posterior_neighbor_similarity(feature_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagonal_matrix_path = args.output_dir / "posterior_mean_diagonal_similarity.npy"
    diagonal_heatmap_path = args.output_dir / "posterior_mean_diagonal_similarity_heatmap.png"
    frame_matrix_path = args.output_dir / "posterior_mean_frame_neighbor_similarity.npy"
    frame_heatmap_path = args.output_dir / "posterior_mean_frame_neighbor_similarity_heatmap.png"
    posterior_matrix_path = args.output_dir / "posterior_mean_posterior_neighbor_similarity.npy"
    posterior_heatmap_path = args.output_dir / "posterior_mean_posterior_neighbor_similarity_heatmap.png"
    features_path = args.output_dir / "posterior_mean_features.npy"
    metadata_path = args.output_dir / "posterior_mean_similarity_metadata.json"

    np.save(diagonal_matrix_path, diagonal_similarity)
    np.save(frame_matrix_path, frame_neighbor_similarity)
    np.save(posterior_matrix_path, posterior_neighbor_similarity)
    # store as float16 to keep disk usage modest (~1.4 MB per seed)
    np.save(features_path, feature_grid.astype(np.float16))
    _save_heatmap(
        diagonal_similarity,
        step_labels,
        diagonal_heatmap_path,
        title="DINOv2 diagonal neighbor cosine similarity",
        xlabel="Frame index",
        ylabel="Posterior-mean step transition",
        ytick_mode="transitions",
    )
    _save_heatmap(
        frame_neighbor_similarity,
        step_labels,
        frame_heatmap_path,
        title="DINOv2 frame-neighbor cosine similarity",
        xlabel="Frame transition",
        ylabel="Posterior-mean step",
        ytick_mode="steps",
    )
    _save_heatmap(
        posterior_neighbor_similarity,
        step_labels,
        posterior_heatmap_path,
        title="DINOv2 posterior-neighbor cosine similarity",
        xlabel="Frame index",
        ylabel="Posterior-mean step transition",
        ytick_mode="transitions",
    )

    metadata = {
        "dino_input_frames_dir": str(args.dino_input_frames_dir.resolve()),
        "model_name": args.model_name,
        "used_device": used_device,
        "grid_shape": {
            "k": int(feature_grid.shape[0]),
            "n": int(feature_grid.shape[1]),
            "d": int(feature_grid.shape[2]),
        },
        "step_labels": step_labels,
        "features": {
            "path": str(features_path.resolve()),
            "dtype": "float16",
            "shape": list(feature_grid.shape),
            "definition": "DINOv2 CLS feature[posterior_step, frame_index, :], L2-normalized",
        },
        "matrices": {
            "diagonal": {
                "shape": list(diagonal_similarity.shape),
                "definition": "diagonal[i, j] = cosine(feature[i, j], feature[i + 1, j + 1])",
                "matrix_path": str(diagonal_matrix_path.resolve()),
                "heatmap_path": str(diagonal_heatmap_path.resolve()),
            },
            "frame_neighbor": {
                "shape": list(frame_neighbor_similarity.shape),
                "definition": "frame_neighbor[i, j] = cosine(feature[i, j], feature[i, j + 1])",
                "matrix_path": str(frame_matrix_path.resolve()),
                "heatmap_path": str(frame_heatmap_path.resolve()),
            },
            "posterior_neighbor": {
                "shape": list(posterior_neighbor_similarity.shape),
                "definition": "posterior_neighbor[i, j] = cosine(feature[i, j], feature[i + 1, j])",
                "matrix_path": str(posterior_matrix_path.resolve()),
                "heatmap_path": str(posterior_heatmap_path.resolve()),
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps({
        "diagonal_matrix_path": str(diagonal_matrix_path),
        "frame_matrix_path": str(frame_matrix_path),
        "posterior_matrix_path": str(posterior_matrix_path),
        "metadata_path": str(metadata_path),
        "used_device": used_device,
        "diagonal_similarity_shape": list(diagonal_similarity.shape),
        "frame_neighbor_shape": list(frame_neighbor_similarity.shape),
        "posterior_neighbor_shape": list(posterior_neighbor_similarity.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
