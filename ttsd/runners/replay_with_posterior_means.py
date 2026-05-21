"""Backfill posterior means for a single existing baseline (prompt, seed).

This reproduces one generation from a prior run using the exact same sampling
settings and seed, then additionally saves the UniPC scheduler's x0-like
intermediate predictions (the "posterior mean" / Tweedie estimate ẑ_{0|t}).

DEPRECATED FOR NEW RUNS. For any run started after 2026-05-05, prefer setting
`snapshots.posterior_means: true` in the baseline config (or pass
`--capture-posterior-means` to `ttsd.runners.baseline`) — that captures the
posterior means inline at essentially zero extra cost. This replay script
exists only to backfill the original 2026-05-04 Phase 0 run, which was
generated before that flag was wired in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from ttsd.models.wan22_adapter import Wan22Adapter
from ttsd.runners.baseline import _save_video


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-seed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    source_seed_dir = args.source_seed_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latents").mkdir(exist_ok=True)
    (output_dir / "posterior_means").mkdir(exist_ok=True)

    meta = json.loads((source_seed_dir / "meta.json").read_text())
    run_root = source_seed_dir.parents[1]
    cfg = yaml.safe_load((run_root / "config.snapshot.yaml").read_text())

    model_cfg = cfg["model"]
    dtype_name = model_cfg["dtype"]
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=_dtype_from_name(dtype_name),
        device=model_cfg["device"],
    )

    result = adapter.generate_with_posterior_means(
        prompt=meta["prompt_text"],
        seed=meta["seed"],
        num_frames=meta["num_frames"],
        height=meta["height"],
        width=meta["width"],
        num_inference_steps=meta["num_inference_steps"],
        guidance_scale=meta["guidance_scale"],
        snapshot_steps=meta["snapshot_steps"],
        posterior_mean_steps=meta["snapshot_steps"],
    )

    video_path = output_dir / "video.mp4"
    _save_video(result.frames, video_path)

    for step_idx, latent in result.latents_by_step.items():
        torch.save(latent, output_dir / "latents" / f"step_{step_idx:03d}.pt")

    for step_idx, posterior in result.posterior_means_by_step.items():
        torch.save(posterior, output_dir / "posterior_means" / f"step_{step_idx:03d}.pt")

    source_video = source_seed_dir / "video.mp4"
    summary = {
        "source_seed_dir": str(source_seed_dir),
        "replayed_video": str(video_path),
        "source_video_sha256": _sha256(source_video) if source_video.exists() else None,
        "replayed_video_sha256": _sha256(video_path),
        "snapshot_steps": meta["snapshot_steps"],
    }
    (output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()