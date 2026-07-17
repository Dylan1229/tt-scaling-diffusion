"""Run VBench evaluation over a Phase-0 baseline run.

Phase 0 produces `runs/baseline/<run_id>/<prompt_id>/seed<NNNN>/video.mp4`.
This module:

  1. Stages videos into a VBench-friendly directory (one flat dir per
     dimension, with files named `<prompt_text>-<seed_idx>.mp4`).
  2. Calls VBench dimension-by-dimension (each dimension owns its own
     submodel: ViCLIP, RAFT, DINO, etc.), passing an explicit prompt map so
     VBench never has to recover the prompt from the filename.
  3. Aggregates per-(prompt, seed, dim) scores into a long-format CSV, a wide
     summary CSV with per-prompt mean/std across seeds, and `vbench_targets.csv`
     — one row per clip carrying the three per-video targets (`vbench_quality`,
     `dynamic_degree`, `overall_consistency`) alongside every input to them.

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

# Dimensions scored on EVERY clip (prompt-agnostic): the model-quality metrics plus
# overall video<->text consistency. To score a new metric on all videos, add it here
# (it must NOT also be in `axis_bound` below).
ALL_VIDEO_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)

# VBench's own (Min, Max) normalization bounds for the QUALITY_LIST dimensions we
# score, copied from external/VBench/scripts/constant.py (NORMALIZE_DIC). That
# directory has no __init__.py, so it is not importable. Every one of these five
# carries DIM_WEIGHT 1.0, so `vbench_quality` is a plain mean of the terms.
#
# VBench's QUALITY_LIST has two more members. `temporal_flickering` is only
# meaningful on still-frame prompts. `dynamic_degree` is boolean per video, so
# folding it in (weight 0.5) would bimodalize a per-video score; it is reported
# on its own instead.
QUALITY_NORMALIZE: dict[str, tuple[float, float]] = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
}

# The six subscores the retired `avg_vbench_z` averaged. Kept only to reproduce it
# under --legacy-avg-vbench-z.
LEGACY_Z_SUBSCORES: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)


def normalized_quality_terms(clip: dict) -> dict[str, float]:
    """(x - Min) / (Max - Min) for each quality dimension of one video.

    `imaging_quality` is a 0-100 MUSIQ score per video — VBench divides by 100 only
    when aggregating (vbench/imaging_quality.py:55) — so it is rescaled here, the one
    place quality is computed. Terms are deliberately left unclamped: a few clips
    exceed `motion_smoothness`'s Max, and clamping would break the identity below.
    """
    terms: dict[str, float] = {}
    for dim, (lo, hi) in QUALITY_NORMALIZE.items():
        x = float(clip[dim])
        if dim == "imaging_quality":
            x /= 100.0
        terms[dim] = (x - lo) / (hi - lo)
    return terms


def vbench_quality(clip: dict) -> float:
    """VBench's Quality score for a single video, over 5 of its 7 quality dimensions.

    The normalization is affine and the weights are all 1.0, so averaging this over any
    set of videos reproduces exactly what external/VBench/scripts/cal_final_score.py
    computes for Quality from the dimension averages. It is therefore VBench's Quality
    decomposed into per-video contributions, not an approximation of it.

    It is never VBench's Total, which additionally needs the Semantic factor — and 8 of
    those 9 dimensions require prompt-specific `auxiliary_info`.
    """
    terms = normalized_quality_terms(clip)
    return sum(terms.values()) / len(terms)


def _slug(prompt: str) -> str:
    """VBench keys videos on the prompt-as-filename — keep the filename
    reversible to the original prompt text. We use the prompt verbatim,
    only stripping characters that break filesystems."""
    s = prompt.strip()
    s = re.sub(r"[/\\\0]", " ", s)
    return s


def parse_staged_video_stem(stem: str) -> tuple[str, int]:
    """Recover prompt and raw seed from current and legacy staging names."""

    current = re.match(r"^(.*)-seed(\d+)$", stem)
    if current:
        return current.group(1), int(current.group(2))
    legacy = re.match(r"^(.*)-(\d+)$", stem)
    if legacy:
        return legacy.group(1), int(legacy.group(2))
    return stem, -1


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
        # Bake the raw seed into the filename so `aggregate_to_csv` can recover
        # it. Earlier versions used a sequential per-prompt index, which silently
        # mislabels the CSV's `seed_idx` whenever seeds aren't 0..N-1.
        per_prompt: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for prompt, seed, video in clips:
            per_prompt[prompt].append((seed, video))
        for prompt, items in per_prompt.items():
            for seed, video in sorted(items):
                fname = f"{_slug(prompt)}-seed{seed:04d}.mp4"
                (d / fname).symlink_to(video.resolve())
        out[dim] = d
    return out


def run_vbench_for_dimension(
    dimension: str,
    staging_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    prompt_list: dict[str, str] | None = None,
) -> dict:
    """Invoke VBench on one dimension's staging dir. Returns the parsed
    eval-results dict keyed by video filename → score.

    `prompt_list` maps a staged filename to its clean prompt text. Without it VBench
    falls back to `get_prompt_from_filename`, which strips only a trailing `-<digits>`
    and so leaves our `-seedNNNN` suffix in the prompt fed to ViCLIP. Only the
    text-conditioned dimensions read it; for the rest it is inert.
    """
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
        prompt_list=prompt_list or {},
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


def _add_legacy_avg_vbench_z(clips: list[dict]) -> None:
    """DEPRECATED. Per-prompt z-normalized mean of the six subscores.

    Retired as a target because it is identically zero for every prompt by
    construction — so it cannot compare sampling strategies — and because three of
    its six terms (subject_consistency, background_consistency, motion_smoothness)
    all measure frame-to-frame similarity, which it therefore triple-counts.

    Emitted only under --legacy-avg-vbench-z so the existing run can be compared
    against `vbench_quality`. No analysis or report runner computes it any more.
    """
    import statistics

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        by_prompt[clip["prompt_id"]].append(clip)

    for group in by_prompt.values():
        zs: dict[str, list[float]] = {}
        for metric in LEGACY_Z_SUBSCORES:
            vals = [float(c[metric]) for c in group]
            mu = statistics.fmean(vals)
            sd = statistics.pstdev(vals)
            zs[metric] = [0.0] * len(vals) if sd == 0 else [(v - mu) / sd for v in vals]
        for i, clip in enumerate(group):
            clip["avg_vbench_z"] = statistics.fmean([zs[m][i] for m in LEGACY_Z_SUBSCORES])


def pivot_to_clips(long_rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Collapse the long rows into one dict per clip, keyed by (prompt_id, seed_idx)."""
    by_clip: dict[tuple[str, int], dict] = {}
    for row in long_rows:
        clip = by_clip.setdefault(
            (row["prompt_id"], row["seed_idx"]),
            {k: row[k] for k in ("prompt_id", "prompt_text", "seed_idx")},
        )
        clip[row["dimension"]] = row["score"]
    return by_clip


def add_quality_to_clips(by_clip: dict[tuple[str, int], dict]) -> bool:
    """Annotate each clip with its norm_* terms and vbench_quality, in place.

    Returns False (annotating nothing) when a quality dimension was not scored, which
    is the normal case for a run with a partial `--dimensions` list.
    """
    if not by_clip:
        return False
    missing = [d for d in QUALITY_NORMALIZE if any(d not in c for c in by_clip.values())]
    if missing:
        print(f"[vbench-agg] no vbench_quality: {', '.join(missing)} not scored")
        return False

    for clip in by_clip.values():
        clip.update({f"norm_{d}": v for d, v in normalized_quality_terms(clip).items()})
        clip["vbench_quality"] = vbench_quality(clip)
    return True


def write_targets_csv(
    by_clip: dict[tuple[str, int], dict], output_dir: Path, legacy_avg_vbench_z: bool = False
) -> Path:
    """Write one row per clip: the three per-video targets and every input to them."""
    cols = ["prompt_id", "prompt_text", "seed_idx"]
    cols += [*QUALITY_NORMALIZE, "overall_consistency", "dynamic_degree"]
    cols += [f"norm_{d}" for d in QUALITY_NORMALIZE]
    cols += ["vbench_quality"]

    if legacy_avg_vbench_z:
        absent = [m for m in LEGACY_Z_SUBSCORES if any(m not in c for c in by_clip.values())]
        if absent:
            print(f"[vbench-agg] skipping avg_vbench_z: no {', '.join(absent)} scores")
        else:
            _add_legacy_avg_vbench_z(list(by_clip.values()))
            cols += ["avg_vbench_z"]

    targets_csv = output_dir / "vbench_targets.csv"
    with targets_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(clip for _, clip in sorted(by_clip.items()))

    return targets_csv


def aggregate_to_csv(
    run_dir: Path,
    results_by_dim: dict[str, dict],
    output_dir: Path,
    legacy_avg_vbench_z: bool = False,
) -> tuple[Path, Path, Path | None]:
    """Write three CSVs:
      - long: one row per (clip, VBench dimension). Raw VBench outputs only.
      - summary: one row per (prompt, dimension) with mean/std/min/max across seeds.
        `dimension` also carries `vbench_quality`, whose across-seed spread is the
        quantity a test-time-scaling search is trying to exploit.
      - targets: one row per clip — the three per-video targets and their inputs.
    """
    import statistics

    long_rows: list[dict] = []
    summary: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    # Build a prompt_text → prompt_id lookup from the run.
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
            stem = Path(vp).stem  # "<prompt_text>-seed<NNNN>"
            prompt_text, seed_idx = parse_staged_video_stem(stem)
            meta = text_to_meta.get(prompt_text, {})
            row = {
                "prompt_id": meta.get("prompt_id", "?"),
                "prompt_text": prompt_text,
                "seed_idx": seed_idx,
                "dimension": dim,
                "score": score,
            }
            long_rows.append(row)
            summary[(row["prompt_id"], prompt_text, dim)].append(score)

    by_clip = pivot_to_clips(long_rows)
    has_quality = add_quality_to_clips(by_clip)
    if has_quality:
        for clip in by_clip.values():
            summary[(clip["prompt_id"], clip["prompt_text"], "vbench_quality")].append(
                clip["vbench_quality"]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "vbench_scores_long.csv"
    summ_csv = output_dir / "vbench_scores_summary.csv"

    with long_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "prompt_text", "seed_idx", "dimension", "score"])
        w.writeheader()
        w.writerows(sorted(long_rows, key=lambda r: (r["prompt_id"], r["dimension"], r["seed_idx"])))

    with summ_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt_text", "dimension", "n_seeds", "mean", "std", "min", "max"])
        for (pid, ptext, dim), scores in sorted(summary.items()):
            mean = statistics.fmean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            w.writerow([pid, ptext, dim, len(scores), f"{mean:.4f}", f"{std:.4f}", f"{min(scores):.4f}", f"{max(scores):.4f}"])

    targets_csv = write_targets_csv(by_clip, output_dir, legacy_avg_vbench_z) if has_quality else None

    return long_csv, summ_csv, targets_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path, help="Path to runs/baseline/<run_id>")
    p.add_argument("--dimensions", default=None,
                   help="Comma-separated VBench dimensions to score. Default: dimensions matching the prompt axes used in this run.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, type=Path,
                   help="Where to write VBench raw results + aggregated CSVs. Default: runs/vbench/<run_id>/")
    p.add_argument("--skip-staged", action="store_true",
                   help="Reuse existing staging dirs instead of re-symlinking.")
    p.add_argument("--legacy-avg-vbench-z", action="store_true",
                   help="DEPRECATED. Emit the retired per-prompt z-mean of six subscores as an "
                        "extra vbench_targets.csv column, to compare against vbench_quality on "
                        "an existing run. Do not use for new sweeps.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip scoring; rebuild the CSVs from the existing raw/*.json. No GPU.")
    args = p.parse_args(argv)

    run_dir: Path = args.run
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    # Pick dimensions — by default, those matching the axes used in this run.
    if args.aggregate_only:
        dimensions = []
    elif args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    else:
        axes_used = sorted({m["axis"] for m, _ in _iter_clips(run_dir) if m.get("axis")})
        dimensions = [d for d in DEFAULT_DIMENSIONS if d in axes_used]
        # Always also score the all-video dimensions on every clip.
        for d in ALL_VIDEO_DIMENSIONS:
            if d not in dimensions:
                dimensions.append(d)

    # Default output is a top-level sibling of baseline/, keyed by run_id:
    #   runs/baseline/<run_id>  ->  runs/vbench/<run_id>
    out_root = args.output or (run_dir.parent.parent / "vbench" / run_dir.name)
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root = out_root / "_staging"
    raw_root = out_root / "raw"

    print(f"[vbench] run={run_dir}")
    print(f"[vbench] dimensions={dimensions}")
    print(f"[vbench] output={out_root}")

    # 1) Stage videos. (For dims that aren't tied to a specific axis — e.g.
    #     subject_consistency, motion_smoothness — we restage ALL clips into
    #     a `_all/` dir so the dim sees every video.)
    if args.aggregate_only:
        per_axis_staging = {}
    elif not args.skip_staged:
        per_axis_staging = stage_videos_by_dimension(run_dir, staging_root, dimensions)
        # Restage everything for axis-agnostic dimensions.
        all_dir = staging_root / "_all"
        if all_dir.exists():
            shutil.rmtree(all_dir)
        all_dir.mkdir(parents=True)
        for meta, video in _iter_clips(run_dir):
            fname = f"{_slug(meta['prompt_text'])}-seed{int(meta['seed']):04d}.mp4"
            (all_dir / fname).symlink_to(video.resolve())
    else:
        per_axis_staging = {d: staging_root / d for d in dimensions if (staging_root / d).exists()}

    # 2) Score each dimension. Axis-bound dims use per-axis staging (only their axis's
    #    clips); ALL_VIDEO_DIMENSIONS use the _all dir (every clip).
    axis_bound = {"object_class", "multiple_objects", "human_action", "color",
                  "spatial_relationship", "scene", "appearance_style"}

    # Hand VBench the prompt for each staged file rather than letting it parse the
    # filename: `get_prompt_from_filename` strips only a trailing `-<digits>`, so our
    # `-seedNNNN` suffix would end up inside the prompt text.
    prompt_list = {
        f"{_slug(m['prompt_text'])}-seed{int(m['seed']):04d}.mp4": m["prompt_text"]
        for m, _ in _iter_clips(run_dir)
    }

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
            results_by_dim[dim] = run_vbench_for_dimension(
                dim, sdir, raw_root, device=args.device, prompt_list=prompt_list
            )
        except Exception as e:  # noqa: BLE001
            print(f"[vbench] ERROR on {dim}: {type(e).__name__}: {e}")

    # 3) Pull in dimensions we did not re-score this run, so a partial `--dimensions`
    #    run still emits every column. Keyed on `dimensions` rather than
    #    `results_by_dim` so a dimension that errored above stays absent instead of
    #    silently resurrecting its previous result.
    for jf in sorted(raw_root.glob("*__all_eval_results.json")):
        dim = jf.name[: -len("__all_eval_results.json")]
        if dim in dimensions:
            continue
        results_by_dim.setdefault(dim, json.loads(jf.read_text()))
        print(f"[vbench] reusing {dim} from {jf.name}")

    if args.aggregate_only and not results_by_dim:
        raise SystemExit(f"--aggregate-only: no *__all_eval_results.json under {raw_root}")

    # 4) Aggregate.
    long_csv, summ_csv, targets_csv = aggregate_to_csv(
        run_dir, results_by_dim, out_root, legacy_avg_vbench_z=args.legacy_avg_vbench_z
    )
    print(f"[vbench] long CSV: {long_csv}")
    print(f"[vbench] summary CSV: {summ_csv}")
    if targets_csv is not None:
        print(f"[vbench] targets CSV: {targets_csv}")


if __name__ == "__main__":
    # Default HF cache to the project's data dir so downloads don't fill /home.
    os.environ.setdefault("HF_HOME", "/data/datasets/fanjiang/.cache/huggingface")
    main()
