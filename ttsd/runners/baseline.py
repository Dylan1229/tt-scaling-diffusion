"""Phase 0 baseline runner.

Generates a sweep of (prompt × seed) clips on Wan 2.2 5B at 480p, persisting
intermediate latents at every-N-steps for the Phase 1 verifier.

Headline artifact: per-prompt VBench-score variance across seeds → the
TTS-necessity argument for the rest of the project.

Usage:
    python -m ttsd.runners.baseline --config configs/baseline_wan22_480p.yaml
    python -m ttsd.runners.baseline --config configs/baseline_wan22_480p.yaml --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class RunMeta:
    prompt_id: str
    prompt_text: str
    axis: str
    seed: int
    model: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    snapshot_steps: list[int]
    timestamp: str


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _save_video(frames, path: Path, fps: int = 16) -> None:
    import imageio.v3 as iio
    import numpy as np

    # Normalize to uint8 H×W×3.
    arr = frames if hasattr(frames, "__array__") else frames.cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255).astype("uint8")
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    iio.imwrite(path, arr, fps=fps, codec="libx264")


def _snapshot_step_indices(num_steps: int, every_n: int, also_keep: list[int]) -> list[int]:
    s = set(range(every_n - 1, num_steps, every_n))
    s.update(i for i in also_keep if 0 <= i < num_steps)
    return sorted(s)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--smoke", action="store_true", help="One prompt, one seed — sanity check.")
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--limit-seeds", type=int, default=None)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    # Lazy import so --help works even if torch/diffusers aren't installed.
    from ttsd.models.wan22_adapter import Wan22Adapter

    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    snap_cfg = cfg["snapshots"]
    out_cfg = cfg["output"]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    adapter = Wan22Adapter(model_path=model_cfg["path"], dtype=dtype, device=model_cfg["device"])

    prompts = _load_prompts(cfg["prompts"]["source"])
    seeds = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]

    if args.smoke:
        prompts = prompts[:1]
        seeds = seeds[:1]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]

    snapshot_steps = _snapshot_step_indices(
        gen_cfg["num_inference_steps"],
        snap_cfg["every_n_steps"],
        snap_cfg.get("also_keep", []),
    )

    run_id = out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg))

    print(f"[baseline] run_root={run_root}")
    print(f"[baseline] {len(prompts)} prompts × {len(seeds)} seeds = {len(prompts) * len(seeds)} clips")
    print(f"[baseline] snapshotting steps: {snapshot_steps}")

    height, width = gen_cfg["resolution"]

    for prompt in prompts:
        for seed in seeds:
            out_dir = run_root / prompt["id"] / f"seed{seed:04d}"
            if (out_dir / "DONE").exists():
                print(f"[baseline] SKIP {prompt['id']} seed={seed} (already done)")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "latents").mkdir(exist_ok=True)

            print(f"[baseline] ▶ {prompt['id']} seed={seed} :: {prompt['text'][:60]}…")
            result = adapter.generate(
                prompt=prompt["text"],
                seed=seed,
                num_frames=gen_cfg["num_frames"],
                height=height,
                width=width,
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                snapshot_steps=snapshot_steps,
            )

            if out_cfg.get("save_video", True):
                _save_video(result.frames, out_dir / "video.mp4")
            if out_cfg.get("save_latents", True):
                for step_idx, latent in result.latents_by_step.items():
                    torch.save(latent, out_dir / "latents" / f"step_{step_idx:03d}.pt")

            meta = RunMeta(
                prompt_id=prompt["id"],
                prompt_text=prompt["text"],
                axis=prompt.get("axis", ""),
                seed=seed,
                model=model_cfg["name"],
                height=height,
                width=width,
                num_frames=gen_cfg["num_frames"],
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                snapshot_steps=snapshot_steps,
                timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            )
            (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
            (out_dir / "DONE").touch()

    print(f"[baseline] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
