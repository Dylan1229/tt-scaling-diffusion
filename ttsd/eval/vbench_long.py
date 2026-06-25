"""Run VBench-Long evaluation over a long-video baseline run.

Expected input layout:
    runs/baseline_long/<run_id>/<prompt_id>/seed<NNNN>/video.mp4

This mirrors `ttsd.eval.vbench`, but uses VBench-Long in `long_custom_input`
mode and aggregates scores back to the original prompt/seed metadata.

Usage:
    python -m ttsd.eval.vbench_long --run runs/baseline_long/<run_id>
    python -m ttsd.eval.vbench_long --run runs/baseline_long/<run_id> --dimensions subject_consistency
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def _slug(prompt: str) -> str:
    s = prompt.strip()
    return re.sub(r"[/\\\0]", " ", s)


def _parse_csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {item.strip() for item in value.split(",") if item.strip()}
    return out or None


def _iter_clips(run_dir: Path, prompt_ids: set[str] | None = None):
    for prompt_dir in sorted(run_dir.iterdir()):
        if not prompt_dir.is_dir() or prompt_dir.name.startswith("_"):
            continue
        for seed_dir in sorted(prompt_dir.iterdir()):
            video = seed_dir / "video.mp4"
            meta_path = seed_dir / "meta.json"
            if not (video.exists() and meta_path.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            if prompt_ids is not None and str(meta.get("prompt_id", "")) not in prompt_ids:
                continue
            yield meta, video


def _seed_values(run_dir: Path, prompt_ids: set[str] | None = None) -> list[int]:
    return sorted({int(meta["seed"]) for meta, _ in _iter_clips(run_dir, prompt_ids)})


def _select_seeds(available: list[int], selector: str | None) -> list[int]:
    if not selector:
        return available

    selected: list[int] = []
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending seed range: {part}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))

    available_set = set(available)
    missing = sorted(set(selected) - available_set)
    if missing:
        raise ValueError(f"Requested seeds not found in run: {missing}; available={available}")

    out: list[int] = []
    seen: set[int] = set()
    for seed in selected:
        if seed not in seen:
            out.append(seed)
            seen.add(seed)
    return out


def _stage_seed_videos(
    run_dir: Path,
    staging_dir: Path,
    seed: int,
    skip_staged: bool,
    prompt_ids: set[str] | None = None,
) -> dict[str, dict]:
    if prompt_ids is not None:
        skip_staged = False
    if staging_dir.exists() and not skip_staged:
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    stage_to_meta: dict[str, dict] = {}
    if skip_staged and any(staging_dir.glob("*.mp4")):
        for meta, _ in _iter_clips(run_dir, prompt_ids):
            if int(meta["seed"]) != seed:
                continue
            slug = _slug(meta["prompt_text"])
            stage_to_meta[slug] = meta
            stage_to_meta[f"{slug}-0"] = meta
        return stage_to_meta

    for meta, video in _iter_clips(run_dir, prompt_ids):
        if int(meta["seed"]) != seed:
            continue
        slug = _slug(meta["prompt_text"])
        # The -0 suffix lets VBench-Long recognize the staged file as one
        # sample and reuse split clips across dimensions.
        stage_name = f"{slug}-0"
        staged = staging_dir / f"{stage_name}.mp4"
        staged.symlink_to(video.resolve())
        stage_to_meta[slug] = meta
        stage_to_meta[stage_name] = meta

    if not stage_to_meta:
        raise ValueError(f"No videos found for seed {seed} under {run_dir}")
    return stage_to_meta


def _vbench_long_kwargs(args: argparse.Namespace) -> dict:
    import vbench2_beta_long as _vbl  # type: ignore

    cur_dir = Path(_vbl.__file__).parent
    return {
        "sb_clip2clip_feat_extractor": args.sb_clip2clip_feat_extractor,
        "bg_clip2clip_feat_extractor": args.bg_clip2clip_feat_extractor,
        "imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode,
        "clip_length_config": args.clip_length_config,
        "w_inclip": args.w_inclip,
        "w_clip2clip": args.w_clip2clip,
        "use_semantic_splitting": args.use_semantic_splitting,
        "slow_fast_eval_config": str(cur_dir / "configs" / "slow_fast_params.yaml"),
        "dev_flag": args.dev_flag,
        "sb_mapping_file_path": str(cur_dir / "configs" / "subject_mapping_table.yaml"),
        "bg_mapping_file_path": str(cur_dir / "configs" / "background_mapping_table.yaml"),
        "num_of_samples_per_prompt": 1,
        "static_filter_flag": False,
    }


def _result_path(dimension: str, staging_dir: Path, output_dir: Path) -> Path:
    return output_dir / f"{dimension}_{staging_dir.name}_eval_results.json"


def run_vbench_long_for_dimension(
    dimension: str,
    staging_dir: Path,
    output_dir: Path,
    device: str,
    kwargs: dict,
    skip_existing: bool,
) -> dict:
    import torch
    import vbench2_beta_long as _vbl  # type: ignore
    from vbench2_beta_long import VBenchLong  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(dimension, staging_dir, output_dir)
    if skip_existing and result_path.exists():
        print(f"[vbench-long] reuse raw result: {result_path}")
        return json.loads(result_path.read_text())

    full_info_path = Path(_vbl.__file__).parent / "VBench_full_info.json"
    name = f"{dimension}_{staging_dir.name}"
    evaluator = VBenchLong(torch.device(device), str(full_info_path), str(output_dir))
    evaluator.evaluate(
        videos_path=str(staging_dir),
        name=name,
        dimension_list=[dimension],
        mode="long_custom_input",
        **kwargs,
    )

    result_path = _result_path(dimension, staging_dir, output_dir)
    if not result_path.exists():
        candidates = list(output_dir.glob(f"{name}*_eval_results.json"))
        if not candidates:
            raise FileNotFoundError(f"VBench-Long did not produce results for {name}")
        result_path = candidates[0]
    return json.loads(result_path.read_text())


def _score_entries(block) -> list[dict]:
    if isinstance(block, list):
        if len(block) >= 3 and isinstance(block[2], list):
            return block[2]
        if len(block) >= 2 and isinstance(block[1], list):
            return block[1]
    if isinstance(block, dict) and isinstance(block.get("score_per_video"), dict):
        return [
            {"video_path": path, "video_results": score}
            for path, score in block["score_per_video"].items()
        ]
    return []


def _dimension_score(dimension: str, raw_result: dict) -> float | None:
    block = raw_result.get(dimension, raw_result)
    if isinstance(block, list) and block and isinstance(block[0], (int, float, bool)):
        return float(block[0])
    if isinstance(block, dict) and isinstance(block.get("score"), (int, float, bool)):
        return float(block["score"])
    return None


def _candidate_stage_keys(video_path: str) -> list[str]:
    stem = Path(video_path).stem
    out = [stem]
    out.append(re.sub(r"_\d{3}$", "", out[-1]))
    out.append(re.sub(r"_full$", "", out[-1]))
    out.append(re.sub(r"-\d+$", "", out[-1]))
    deduped = []
    for item in out:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _rows_from_result(
    dimension: str,
    raw_result: dict,
    seed: int,
    stage_to_meta: dict[str, dict],
) -> list[dict]:
    block = raw_result.get(dimension, raw_result)
    grouped: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for entry in _score_entries(block):
        video_path = entry.get("video_path") or entry.get("video") or ""
        score = entry.get("video_results") if "video_results" in entry else entry.get("score")
        if not video_path or score is None:
            continue

        meta = None
        for key in _candidate_stage_keys(video_path):
            meta = stage_to_meta.get(key)
            if meta is not None:
                break
        if meta is None:
            print(f"[vbench-long-agg] WARN: could not map result path to prompt: {video_path}")
            continue

        score_value = float(score)
        if dimension == "imaging_quality":
            score_value /= 100.0

        grouped[
            (
                str(meta.get("prompt_id", "?")),
                str(meta.get("prompt_text", "")),
                str(meta.get("axis", "")),
                seed,
                dimension,
            )
        ].append(score_value)

    rows = []
    for (prompt_id, prompt_text, axis, seed_idx, dim), scores in grouped.items():
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "axis": axis,
                "seed_idx": seed_idx,
                "dimension": dim,
                "score": statistics.fmean(scores),
                "n_clips": len(scores),
            }
        )
    return rows


def _write_csvs(rows: list[dict], seed_scores: list[dict], output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "vbench_long_scores_long.csv"
    summary_csv = output_dir / "vbench_long_scores_summary.csv"
    by_seed_csv = output_dir / "vbench_long_scores_by_seed.csv"
    overall_csv = output_dir / "vbench_long_scores_overall.csv"

    with long_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "prompt_text", "axis", "seed_idx", "dimension", "score", "n_clips"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["prompt_id"], r["dimension"], r["seed_idx"])))

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_id"], row["prompt_text"], row["dimension"])].append(float(row["score"]))

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt_text", "dimension", "n_seeds", "mean", "std", "min", "max"])
        for (prompt_id, prompt_text, dimension), scores in sorted(grouped.items()):
            mean = statistics.fmean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            writer.writerow(
                [
                    prompt_id,
                    prompt_text,
                    dimension,
                    len(scores),
                    f"{mean:.4f}",
                    f"{std:.4f}",
                    f"{min(scores):.4f}",
                    f"{max(scores):.4f}",
                ]
            )

    with by_seed_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_idx", "dimension", "score"])
        writer.writeheader()
        writer.writerows(sorted(seed_scores, key=lambda r: (r["dimension"], r["seed_idx"])))

    grouped_seed_scores: dict[str, list[float]] = defaultdict(list)
    for row in seed_scores:
        grouped_seed_scores[row["dimension"]].append(float(row["score"]))

    with overall_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "n_seeds", "mean", "std", "min", "max"])
        dimension_means = []
        for dimension, scores in sorted(grouped_seed_scores.items()):
            mean = statistics.fmean(scores)
            dimension_means.append(mean)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            writer.writerow(
                [
                    dimension,
                    len(scores),
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    f"{min(scores):.6f}",
                    f"{max(scores):.6f}",
                ]
            )
        if dimension_means:
            writer.writerow(["overall_mean", len(dimension_means), f"{statistics.fmean(dimension_means):.6f}", "", "", ""])

    return long_csv, summary_csv, by_seed_csv, overall_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Path to runs/baseline_long/<run_id>")
    parser.add_argument(
        "--dimensions",
        default=",".join(DEFAULT_DIMENSIONS),
        help="Comma-separated VBench-Long dimensions. Defaults to custom-input supported dimensions.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--skip-staged", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing raw *_eval_results.json files.")
    parser.add_argument("--merge-only", action="store_true", help="Only aggregate existing raw results into CSVs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list/ranges, e.g. 0,3-5.")
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated prompt ids to score, e.g. p01,p03.")
    parser.add_argument("--use-semantic-splitting", action="store_true")
    parser.add_argument("--clip-length-config", default="clip_length_mix.yaml")
    parser.add_argument("--imaging-quality-preprocessing-mode", default="longer")
    parser.add_argument("--sb-clip2clip-feat-extractor", default="dinov2", choices=["dino", "dinov2", "dreamsim"])
    parser.add_argument("--bg-clip2clip-feat-extractor", default="dreamsim", choices=["clip", "dreamsim"])
    parser.add_argument("--w-inclip", type=float, default=1.0)
    parser.add_argument("--w-clip2clip", type=float, default=0.0)
    parser.add_argument("--no-dev-flag", dest="dev_flag", action="store_false", default=True)
    args = parser.parse_args(argv)

    run_dir = args.run
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    prompt_ids = _parse_csv_set(args.prompt_ids)
    available_prompt_ids = {str(meta.get("prompt_id", "")) for meta, _ in _iter_clips(run_dir)}
    if prompt_ids is not None:
        missing_prompts = sorted(prompt_ids - available_prompt_ids)
        if missing_prompts:
            raise SystemExit(
                f"Requested prompt ids not found: {missing_prompts}; "
                f"available={sorted(available_prompt_ids)}"
            )

    dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    unsupported = sorted(set(dimensions) - set(DEFAULT_DIMENSIONS))
    if unsupported:
        raise SystemExit(
            "This custom-input evaluator supports only "
            f"{', '.join(DEFAULT_DIMENSIONS)}; got unsupported: {', '.join(unsupported)}"
        )

    out_root = args.output or (run_dir.parent.parent / "vbench_long" / run_dir.name)
    staging_root = out_root / "_staging"
    raw_root = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    seeds = _select_seeds(_seed_values(run_dir, prompt_ids), args.seeds)
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]

    print(f"[vbench-long] run={run_dir}")
    print(f"[vbench-long] dimensions={dimensions}")
    print(f"[vbench-long] prompt_ids={sorted(prompt_ids) if prompt_ids else 'all'}")
    print(f"[vbench-long] seeds={seeds}")
    print(f"[vbench-long] output={out_root}")

    kwargs = _vbench_long_kwargs(args)
    rows: list[dict] = []
    seed_scores: list[dict] = []
    prompt_tag = ""
    if prompt_ids:
        prompt_tag = "_prompts-" + "-".join(sorted(prompt_ids))
    for seed in seeds:
        staging_dir = staging_root / f"seed{seed:04d}{prompt_tag}"
        stage_to_meta = _stage_seed_videos(run_dir, staging_dir, seed, args.skip_staged, prompt_ids)
        for dimension in dimensions:
            if args.merge_only:
                result_path = _result_path(dimension, staging_dir, raw_root)
                if not result_path.exists():
                    print(f"[vbench-long] WARN missing raw result: {result_path}")
                    continue
                raw_result = json.loads(result_path.read_text())
                dimension_score = _dimension_score(dimension, raw_result)
                if dimension_score is not None:
                    seed_scores.append({"seed_idx": seed, "dimension": dimension, "score": dimension_score})
                rows.extend(_rows_from_result(dimension, raw_result, seed, stage_to_meta))
                continue

            print(f"[vbench-long] > seed={seed} dim={dimension}")
            try:
                raw_result = run_vbench_long_for_dimension(
                    dimension=dimension,
                    staging_dir=staging_dir,
                    output_dir=raw_root,
                    device=args.device,
                    kwargs=kwargs,
                    skip_existing=args.skip_existing,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[vbench-long] ERROR seed={seed} dim={dimension}: {type(exc).__name__}: {exc}")
                continue
            dimension_score = _dimension_score(dimension, raw_result)
            if dimension_score is not None:
                seed_scores.append({"seed_idx": seed, "dimension": dimension, "score": dimension_score})
            rows.extend(_rows_from_result(dimension, raw_result, seed, stage_to_meta))

    long_csv, summary_csv, by_seed_csv, overall_csv = _write_csvs(rows, seed_scores, out_root)
    print(f"[vbench-long] long CSV: {long_csv}")
    print(f"[vbench-long] summary CSV: {summary_csv}")
    print(f"[vbench-long] by-seed CSV: {by_seed_csv}")
    print(f"[vbench-long] overall CSV: {overall_csv}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/data/datasets/fanjiang/.cache/huggingface")
    main()
