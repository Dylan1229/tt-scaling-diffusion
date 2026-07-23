"""Extract cheap causal features from late-branch posterior-mean latents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def _sample_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
        raise TypeError(f"expected a 5D tensor in {path}")
    # Preserve every latent frame but subsample channels and space. This keeps
    # extraction cheap while retaining temporal and spatial structure.
    return tensor[:, ::4, :, ::2, ::2].float()


def _safe_std(tensor: torch.Tensor) -> float:
    return float(tensor.std(unbiased=False))


def _snapshot_features(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.flatten()
    temporal = tensor[:, :, 1:] - tensor[:, :, :-1]
    spatial_h = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    spatial_w = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
    frame_rms = tensor.square().mean(dim=(0, 1, 3, 4)).sqrt()
    quantiles = torch.quantile(flat[::17], torch.tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(tensor.mean()),
        "std": _safe_std(tensor),
        "rms": float(tensor.square().mean().sqrt()),
        "abs_mean": float(tensor.abs().mean()),
        "temporal_rms": float(temporal.square().mean().sqrt()),
        "spatial_h_rms": float(spatial_h.square().mean().sqrt()),
        "spatial_w_rms": float(spatial_w.square().mean().sqrt()),
        "first_last_rms": float(
            (tensor[:, :, -1] - tensor[:, :, 0]).square().mean().sqrt()
        ),
        "frame_rms_std": _safe_std(frame_rms),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
    }


def _trajectory_features(
    earlier: torch.Tensor, later: torch.Tensor
) -> dict[str, float]:
    delta = later - earlier
    earlier_flat = earlier.flatten()
    later_flat = later.flatten()
    denominator = float(earlier.square().mean().sqrt()) + 1e-8
    cosine = torch.nn.functional.cosine_similarity(
        earlier_flat, later_flat, dim=0
    )
    return {
        "relative_l2": float(delta.square().mean().sqrt()) / denominator,
        "cosine": float(cosine),
        "delta_mean": float(delta.mean()),
        "delta_std": _safe_std(delta),
    }


def _extract_candidate(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    posterior_dir = meta_path.parent / "posterior_means"
    paths = sorted(posterior_dir.glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no posterior means under {posterior_dir}")

    row = {
        "prompt_id": meta["prompt_id"],
        "prompt_text": meta["prompt_text"],
        "root_seed": int(meta["root_seed"]),
        "candidate_index": int(meta["candidate_index"]),
        "candidate_seed": int(meta["seed"]),
        "branch_kind": meta["branch_kind"],
    }
    sampled: list[tuple[int, torch.Tensor]] = []
    for path in paths:
        step = int(path.stem.split("_")[-1])
        tensor = _sample_tensor(path)
        sampled.append((step, tensor))
        for name, value in _snapshot_features(tensor).items():
            row[f"s{step:03d}_{name}"] = value

    for (early_step, early), (late_step, late) in zip(
        sampled, sampled[1:], strict=False
    ):
        for name, value in _trajectory_features(early, late).items():
            row[f"s{early_step:03d}_to_s{late_step:03d}_{name}"] = value
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    meta_paths = sorted(args.run.glob("*/seed*/meta.json"))
    if not meta_paths:
        raise SystemExit(f"no candidate metadata found under {args.run}")

    rows = []
    for index, meta_path in enumerate(meta_paths, start=1):
        rows.append(_extract_candidate(meta_path))
        if index % 100 == 0 or index == len(meta_paths):
            print(f"[online-features] {index}/{len(meta_paths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[online-features] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
