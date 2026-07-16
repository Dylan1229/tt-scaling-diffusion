"""Extract causal step-10 verifier features from an Euler baseline trajectory.

The extractor decodes saved posterior means at denoising steps 5 and 10,
computes DINOv2 CLS trajectories plus lightweight pixel/motion statistics, and
writes one compressed feature file per video. It never reads the final VBench
score or the Renoise output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ttsd.features.dino_cls import gap_cosine_profile, l2_normalize
from ttsd.runners.generate.decode_latents import (
    DEFAULT_MODEL_PATH,
    _decode_latents,
    _load_decoder,
)


def _iter_samples(run_root: Path) -> list[Path]:
    return sorted(
        seed_dir
        for seed_dir in run_root.glob("p*/seed*")
        if (seed_dir / "meta.json").exists()
    )


def _as_uint8(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


class _PersistentDino:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def extract(self, frames: np.ndarray) -> np.ndarray:
        images = [Image.fromarray(frame) for frame in _as_uint8(frames)]
        chunks: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            chunks.append(outputs.last_hidden_state[:, 0].float().cpu())
        return l2_normalize(torch.cat(chunks, dim=0).numpy())


def _summary_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_max": float(arr.max()),
    }


def _dino_scalars(prefix: str, cls: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for gap in (1, 2, 4):
        profile = gap_cosine_profile(cls, gap=gap)
        out.update(_summary_stats(f"{prefix}_dino_gap{gap}_cos", profile))
        tail_count = max(1, int(np.ceil(profile.size * 0.20)))
        out[f"{prefix}_dino_gap{gap}_tail20_low"] = float(
            np.sort(profile.reshape(-1))[:tail_count].mean()
        )
    velocity = np.linalg.norm(np.diff(cls, axis=0), axis=1)
    out.update(_summary_stats(f"{prefix}_dino_velocity", velocity))
    return out


def _pixel_scalars(prefix: str, frames: np.ndarray) -> dict[str, float]:
    sampled = _as_uint8(frames)[::4]
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in sampled])
    hsv = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) for frame in sampled])

    lap_var = np.asarray([cv2.Laplacian(frame, cv2.CV_64F).var() for frame in gray])
    edge_density = np.asarray(
        [(cv2.Canny(frame, 80, 160) > 0).mean() for frame in gray],
        dtype=np.float64,
    )
    temporal_absdiff = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))

    flow_values: list[float] = []
    for first, second in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flow_values.append(float(np.linalg.norm(flow, axis=2).mean()))

    out: dict[str, float] = {}
    out.update(_summary_stats(f"{prefix}_gray_mean", gray.mean(axis=(1, 2)) / 255.0))
    out.update(_summary_stats(f"{prefix}_gray_std", gray.std(axis=(1, 2)) / 255.0))
    out.update(_summary_stats(f"{prefix}_saturation", hsv[..., 1].mean(axis=(1, 2)) / 255.0))
    out.update(_summary_stats(f"{prefix}_lap_var", lap_var))
    out.update(_summary_stats(f"{prefix}_edge_density", edge_density))
    out.update(_summary_stats(f"{prefix}_temporal_absdiff", temporal_absdiff / 255.0))
    out.update(_summary_stats(f"{prefix}_flow", np.asarray(flow_values)))
    return out


def _cross_step_scalars(early: np.ndarray, current: np.ndarray) -> dict[str, float]:
    same_frame_cos = np.sum(early * current, axis=1)
    same_frame_delta = np.linalg.norm(current - early, axis=1)
    out = _summary_stats("trajectory_step5_to_step10_cos", same_frame_cos)
    out.update(_summary_stats("trajectory_step5_to_step10_delta", same_frame_delta))
    return out


def _decode_posterior(
    seed_dir: Path,
    step_index: int,
    *,
    vae,
    video_processor,
    device: str,
) -> np.ndarray:
    path = seed_dir / "posterior_means" / f"step_{step_index:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    latent = torch.load(path, map_location="cpu")
    decoded = _decode_latents(vae, video_processor, latent, device, output_type="np")
    return _as_uint8(np.asarray(decoded[0]))


def _process_sample(
    seed_dir: Path,
    output_path: Path,
    *,
    step_indices: list[int],
    vae,
    video_processor,
    dino: _PersistentDino,
    device: str,
) -> None:
    meta = json.loads((seed_dir / "meta.json").read_text())
    cls_by_step: list[np.ndarray] = []
    scalar_features: dict[str, float] = {}

    for step_index in step_indices:
        frames = _decode_posterior(
            seed_dir,
            step_index,
            vae=vae,
            video_processor=video_processor,
            device=device,
        )
        cls = dino.extract(frames)
        cls_by_step.append(cls)
        user_step = step_index + 1
        prefix = f"step{user_step:02d}"
        scalar_features.update(_dino_scalars(prefix, cls))
        scalar_features.update(_pixel_scalars(prefix, frames))

    if len(cls_by_step) >= 2:
        scalar_features.update(_cross_step_scalars(cls_by_step[-2], cls_by_step[-1]))
        early_prefix = f"step{step_indices[-2] + 1:02d}"
        current_prefix = f"step{step_indices[-1] + 1:02d}"
        shared_suffixes = sorted(
            {
                name.removeprefix(f"{current_prefix}_")
                for name in scalar_features
                if name.startswith(f"{current_prefix}_")
                and f"{early_prefix}_{name.removeprefix(f'{current_prefix}_')}"
                in scalar_features
            }
        )
        for suffix in shared_suffixes:
            scalar_features[f"trajectory_delta_{suffix}"] = (
                scalar_features[f"{current_prefix}_{suffix}"]
                - scalar_features[f"{early_prefix}_{suffix}"]
            )

    names = sorted(scalar_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cls=np.stack(cls_by_step).astype(np.float16),
        step_indices=np.asarray(step_indices, dtype=np.int16),
        scalar_names=np.asarray(names),
        scalar_values=np.asarray([scalar_features[name] for name in names], dtype=np.float32),
        prompt_id=np.asarray(meta["prompt_id"]),
        prompt_text=np.asarray(meta["prompt_text"]),
        axis=np.asarray(meta.get("axis", "")),
        seed_idx=np.asarray(int(meta["seed"]), dtype=np.int16),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", default="4,9", help="Zero-based posterior step indices.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("shard-index must be in [0, num-shards)")
    step_indices = [int(part.strip()) for part in args.steps.split(",") if part.strip()]
    if len(step_indices) < 1:
        raise SystemExit("--steps must contain at least one index")

    samples = _iter_samples(args.baseline_run)
    samples = [
        seed_dir
        for index, seed_dir in enumerate(samples)
        if index % args.num_shards == args.shard_index
    ]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    vae, video_processor = _load_decoder(
        args.model_path,
        dtype,
        args.device,
        128,
        128,
        96,
        96,
    )
    dino = _PersistentDino(args.dino_model, args.device, args.batch_size)

    print(
        f"[renoise_online_features] shard={args.shard_index}/{args.num_shards} "
        f"samples={len(samples)} steps={step_indices} device={args.device}"
    )
    for seed_dir in samples:
        rel = seed_dir.relative_to(args.baseline_run)
        output_path = args.output_dir / rel / "online_features.npz"
        if args.skip_existing and output_path.exists():
            print(f"[renoise_online_features] SKIP {rel}")
            continue
        print(f"[renoise_online_features] {rel}")
        _process_sample(
            seed_dir,
            output_path,
            step_indices=step_indices,
            vae=vae,
            video_processor=video_processor,
            dino=dino,
            device=args.device,
        )


if __name__ == "__main__":
    main()
