"""Run VBench evaluation over a Phase-0 baseline run.

Phase 0 produces `runs/baseline/<run_id>/<prompt_id>/seed<NNNN>/video.mp4`.
This module:

  1. Stages videos into a VBench-friendly directory (one flat dir per
     dimension, with files named `<prompt_text>-<seed_idx>.mp4` — VBench's
     custom_input mode keys on the filename stem).
  2. Calls VBench dimension-by-dimension (each dimension owns its own
     submodel: ViCLIP, RAFT, DINO, etc.).
  3. Aggregates per-(prompt, seed, dim) scores into a long-format CSV plus a
     wide summary CSV with per-prompt mean/std across seeds — the headline
     "TTS-necessity" artifact.

VBench downloads ~10–20 GB of pretrained checkpoints on first use. Set
`HF_HOME=/data/datasets/fanjiang/.cache/huggingface` to keep them off /home.

Usage:
    python -m ttsd.eval.vbench --run runs/baseline/<run_id>
    python -m ttsd.eval.vbench --run runs/baseline/<run_id> --dimensions subject_consistency,human_action
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

# VBench dimensions we map our dev_set axes onto. Order matters only for
# logging — each dim is independent.
DEFAULT_DIMENSIONS: list[str] = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "overall_consistency",
]


def _slug(prompt: str) -> str:
    """VBench keys videos on the prompt-as-filename — keep the filename
    reversible to the original prompt text. We use the prompt verbatim,
    only stripping characters that break filesystems."""
    s = prompt.strip()
    s = re.sub(r"[/\\\0]", " ", s)
    return s


def _iter_clips(run_dir: Path) -> Iterable[tuple[dict, Path]]:
    for prompt_dir in sorted(run_dir.iterdir()):
        if not prompt_dir.is_dir() or prompt_dir.name.startswith("_"):
            continue
        for seed_dir in sorted(prompt_dir.iterdir()):
            video = seed_dir / "video.mp4"
            meta_p = seed_dir / "meta.json"
            if not (video.exists() and meta_p.exists()):
                continue
            yield json.loads(meta_p.read_text()), video


def stage_videos_by_dimension(
    run_dir: Path, staging_root: Path, dimensions: list[str]
) -> dict[str, Path]:
    """Symlink each (prompt, seed) clip into one staging dir per dimension
    that prompt belongs to. Returns {dimension: staging_dir}."""
    # VBench's custom_input mode reads filename stem as the prompt. Multiple
    # videos for one prompt are disambiguated by the suffix `-<idx>`.
    by_dim: dict[str, list[tuple[str, int, Path]]] = defaultdict(list)
    for meta, video in _iter_clips(run_dir):
        axis = meta["axis"]
        if axis in dimensions:
            by_dim[axis].append((meta["prompt_text"], meta["seed"], video))

    out: dict[str, Path] = {}
    for dim, clips in by_dim.items():
        d = staging_root / dim
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        # Group by prompt to assign sequential seed indices for the filename suffix.
        per_prompt: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for prompt, seed, video in clips:
            per_prompt[prompt].append((seed, video))
        for prompt, items in per_prompt.items():
            for idx, (_seed, video) in enumerate(sorted(items)):
                fname = f"{_slug(prompt)}-{idx}.mp4"
                (d / fname).symlink_to(video.resolve())
        out[dim] = d
    return out


def run_vbench_for_dimension(
    dimension: str,
    staging_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """Invoke VBench on one dimension's staging dir. Returns the parsed
    eval-results dict keyed by video filename → score."""
    # Imported lazily — VBench import is heavy and pulls torch/CLIP.
    from vbench import VBench

    output_dir.mkdir(parents=True, exist_ok=True)
    import vbench as _vb  # type: ignore

    pkg_dir = Path(_vb.__file__).parent
    full_info_path = pkg_dir / "VBench_full_info.json"

    name = f"{dimension}_{staging_dir.name}"
    vb = VBench(device, str(full_info_path), str(output_dir))
    vb.evaluate(
        videos_path=str(staging_dir),
        name=name,
        dimension_list=[dimension],
        mode="custom_input",
    )

    # VBench writes <name>_eval_results.json under output_dir.
    candidates = list(output_dir.glob(f"{name}_eval_results.json"))
    if not candidates:
        candidates = list(output_dir.glob(f"{name}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"VBench did not produce a results JSON in {output_dir} for {name}"
        )
    return json.loads(candidates[0].read_text())


def aggregate_to_csv(
    run_dir: Path, results_by_dim: dict[str, dict], output_dir: Path
) -> tuple[Path, Path]:
    """Write two CSVs:
      - long-format: one row per (prompt_id, prompt_text, seed, dim, score)
      - summary: one row per (prompt_id, prompt_text, dim) with mean/std/min/max.
    """
    import statistics

    long_rows: list[dict] = []
    summary: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    # Build a (prompt_text → prompt_id, axis) lookup from the run.
    text_to_meta: dict[str, dict] = {}
    for meta, _ in _iter_clips(run_dir):
        text_to_meta.setdefault(meta["prompt_text"], meta)

    for dim, results in results_by_dim.items():
        # VBench JSON shape varies a bit across versions. Common shapes:
        #   {dim: [score, [{"video_path":..., "score":...}, ...]]}
        #   {dim: {"avg_score": x, "score_per_video": {...}}}
        per_video: list[tuple[str, float]] = []
        block = results.get(dim, results)
        if isinstance(block, list) and len(block) >= 2 and isinstance(block[1], list):
            for entry in block[1]:
                vp = entry.get("video_path") or entry.get("video") or ""
                sc = entry.get("video_results") if "video_results" in entry else entry.get("score")
                if vp and sc is not None:
                    per_video.append((vp, float(sc)))
        elif isinstance(block, dict) and "score_per_video" in block:
            for vp, sc in block["score_per_video"].items():
                per_video.append((vp, float(sc)))
        else:
            print(f"[vbench-agg] WARN: unrecognized result shape for {dim}: {type(block)}")
            continue

        for vp, score in per_video:
            stem = Path(vp).stem  # "<prompt_text>-<idx>"
            m = re.match(r"^(.*)-(\d+)$", stem)
            prompt_text = m.group(1) if m else stem
            seed_idx = int(m.group(2)) if m else -1
            meta = text_to_meta.get(prompt_text, {})
            row = {
                "prompt_id": meta.get("prompt_id", "?"),
                "prompt_text": prompt_text,
                "axis": meta.get("axis", ""),
                "seed_idx": seed_idx,
                "dimension": dim,
                "score": score,
            }
            long_rows.append(row)
            summary[(row["prompt_id"], prompt_text, dim)].append(score)

    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "vbench_scores_long.csv"
    summ_csv = output_dir / "vbench_scores_summary.csv"

    with long_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "prompt_text", "axis", "seed_idx", "dimension", "score"])
        w.writeheader()
        w.writerows(sorted(long_rows, key=lambda r: (r["prompt_id"], r["dimension"], r["seed_idx"])))

    with summ_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt_text", "dimension", "n_seeds", "mean", "std", "min", "max"])
        for (pid, ptext, dim), scores in sorted(summary.items()):
            mean = statistics.fmean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            w.writerow([pid, ptext, dim, len(scores), f"{mean:.4f}", f"{std:.4f}", f"{min(scores):.4f}", f"{max(scores):.4f}"])

    return long_csv, summ_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path, help="Path to runs/baseline/<run_id>")
    p.add_argument("--dimensions", default=None,
                   help="Comma-separated VBench dimensions to score. Default: dimensions matching the prompt axes used in this run.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, type=Path,
                   help="Where to write VBench raw results + aggregated CSVs. Default: <run>/vbench/")
    p.add_argument("--skip-staged", action="store_true",
                   help="Reuse existing staging dirs instead of re-symlinking.")
    args = p.parse_args(argv)

    run_dir: Path = args.run
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    # Pick dimensions — by default, those matching the axes used in this run.
    if args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    else:
        axes_used = sorted({m["axis"] for m, _ in _iter_clips(run_dir) if m.get("axis")})
        dimensions = [d for d in DEFAULT_DIMENSIONS if d in axes_used]
        # Always also score model-quality dimensions on every video.
        for d in ("subject_consistency", "background_consistency", "motion_smoothness",
                  "dynamic_degree", "aesthetic_quality", "imaging_quality"):
            if d not in dimensions:
                dimensions.append(d)

    out_root = args.output or (run_dir / "vbench")
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root = out_root / "_staging"
    raw_root = out_root / "raw"

    print(f"[vbench] run={run_dir}")
    print(f"[vbench] dimensions={dimensions}")
    print(f"[vbench] output={out_root}")

    # 1) Stage videos. (For dims that aren't tied to a specific axis — e.g.
    #     subject_consistency, motion_smoothness — we restage ALL clips into
    #     a `_all/` dir so the dim sees every video.)
    if not args.skip_staged:
        per_axis_staging = stage_videos_by_dimension(run_dir, staging_root, dimensions)
        # Restage everything for axis-agnostic dimensions.
        all_dir = staging_root / "_all"
        if all_dir.exists():
            shutil.rmtree(all_dir)
        all_dir.mkdir(parents=True)
        per_prompt_seq: dict[str, int] = defaultdict(int)
        for meta, video in _iter_clips(run_dir):
            i = per_prompt_seq[meta["prompt_text"]]
            per_prompt_seq[meta["prompt_text"]] += 1
            (all_dir / f"{_slug(meta['prompt_text'])}-{i}.mp4").symlink_to(video.resolve())
    else:
        per_axis_staging = {d: staging_root / d for d in dimensions if (staging_root / d).exists()}

    # 2) Score each dimension. Axis-bound dims use per-axis staging; the
    #    "model quality" dims use the _all dir.
    axis_bound = {"object_class", "multiple_objects", "human_action", "color",
                  "spatial_relationship", "scene", "appearance_style", "overall_consistency"}

    results_by_dim: dict[str, dict] = {}
    for dim in dimensions:
        if dim in axis_bound:
            sdir = per_axis_staging.get(dim)
            if sdir is None:
                print(f"[vbench] SKIP {dim} (no prompts of this axis in run)")
                continue
        else:
            sdir = staging_root / "_all"
        print(f"[vbench] ▶ scoring dim={dim} dir={sdir}")
        try:
            results_by_dim[dim] = run_vbench_for_dimension(dim, sdir, raw_root, device=args.device)
        except Exception as e:  # noqa: BLE001
            print(f"[vbench] ERROR on {dim}: {type(e).__name__}: {e}")

    # 3) Aggregate.
    long_csv, summ_csv = aggregate_to_csv(run_dir, results_by_dim, out_root)
    print(f"[vbench] long CSV: {long_csv}")
    print(f"[vbench] summary CSV: {summ_csv}")


if __name__ == "__main__":
    # Default HF cache to the project's data dir so downloads don't fill /home.
    os.environ.setdefault("HF_HOME", "/data/datasets/fanjiang/.cache/huggingface")
    main()
