"""Rank seeds per prompt using a clean posterior-mean heatmap metric.

The default metric is LateTailMean_q: restrict the heatmap to the last few
posterior-mean transition rows, flatten them, take the lowest q fraction of
values, and average them. Higher is better.

Usage:
    python -m ttsd.runners.analysis.similarity_tail_ranking \
        --heatmap-run-root /path/to/cls_features/<run_id> \
        --vbench-long-csv /path/to/vbench_scores_long.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttsd.runners.utilities.run_layout import resolve_run_id, stage_output_dir
from ttsd.runners.utilities.seed_vbench_loaders import _iter_seed_dirs, _load_vbench_rows, _seed_idx_from_name


def _tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    flattened = np.sort(values.reshape(-1))
    tail_count = max(1, math.ceil(flattened.size * tail_fraction))
    return float(flattened[:tail_count].mean())


def _metric_output_stem(score_type: str, tail_fraction: float, late_rows: int) -> str:
    tail_pct = int(round(tail_fraction * 100))
    if score_type == "tail_mean_q":
        return f"tail_mean{tail_pct}"
    if score_type == "late_tail_mean_q":
        return f"late_tail{tail_pct}_last{late_rows}"
    if score_type == "late_mean":
        return f"late_mean_last{late_rows}"
    if score_type == "global_mean":
        return "global_mean"
    raise ValueError(f"Unsupported score type: {score_type}")


def _score_value(heatmap: np.ndarray, score_type: str, tail_fraction: float, late_rows: int) -> float:
    if score_type == "tail_mean_q":
        return _tail_mean(heatmap, tail_fraction)
    if score_type == "late_tail_mean_q":
        return _tail_mean(heatmap[-late_rows:, :], tail_fraction)
    if score_type == "late_mean":
        return float(heatmap[-late_rows:, :].mean())
    if score_type == "global_mean":
        return float(heatmap.mean())
    raise ValueError(f"Unsupported score type: {score_type}")


def _metric_row_for_seed(
    seed_dir: Path,
    score_type: str,
    tail_fraction: float,
    late_rows: int,
    vbench_rows: dict[tuple[str, int], dict[str, object]],
) -> dict[str, object] | None:
    prompt_id = seed_dir.parent.name
    seed_idx = _seed_idx_from_name(seed_dir)
    heatmap_path = seed_dir / "posterior_mean_diagonal_similarity.npy"

    if not heatmap_path.exists():
        return None

    heatmap = np.load(heatmap_path)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap at {heatmap_path}, got shape {heatmap.shape}")

    row = dict(vbench_rows.get((prompt_id, seed_idx), {}))
    row.setdefault("prompt_id", prompt_id)
    row.setdefault("prompt_text", "")
    row.setdefault("axis", "")
    row["seed_idx"] = seed_idx
    row["heatmap_path"] = str(heatmap_path)
    row["score_name"] = _metric_output_stem(score_type, tail_fraction, late_rows)
    row["score_value"] = _score_value(heatmap, score_type, tail_fraction, late_rows)
    row["tail_fraction"] = tail_fraction
    row["tail_mean_q"] = _tail_mean(heatmap, tail_fraction)
    row["late_rows"] = late_rows
    row["late_mean"] = float(heatmap[-late_rows:, :].mean())
    row["late_tail_mean_q"] = _tail_mean(heatmap[-late_rows:, :], tail_fraction)
    row["heatmap_mean"] = float(heatmap.mean())
    row["min_similarity"] = float(heatmap.min())
    row["max_similarity"] = float(heatmap.max())
    row["heatmap_rows"] = int(heatmap.shape[0])
    row["heatmap_cols"] = int(heatmap.shape[1])
    return row


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(row)

    ranked_rows: list[dict[str, object]] = []
    for prompt_id, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                -float(row["score_value"]),
                -float(row["late_tail_mean_q"]),
                -float(row["heatmap_mean"]),
                -float(row["min_similarity"]),
                int(row["seed_idx"]),
            ),
        )
        prompt_size = len(ordered)
        for rank, row in enumerate(ordered, start=1):
            row["tail_mean_rank"] = rank
            row["prompt_size"] = prompt_size
            ranked_rows.append(row)
    return ranked_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-run-root", type=Path, default=None)
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--score-type",
        choices=["tail_mean_q", "late_tail_mean_q", "late_mean", "global_mean"],
        default="late_tail_mean_q",
    )
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--late-rows", type=int, default=2)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args(argv)
    resolve_run_id(args, parser, needs=["heatmap_run_root", "vbench_long_csv"])

    if not (0.0 < args.tail_fraction <= 1.0):
        raise ValueError("--tail-fraction must be in (0, 1]")
    if args.late_rows <= 0:
        raise ValueError("--late-rows must be positive")

    heatmap_run_root = args.heatmap_run_root.resolve()
    output_stem = _metric_output_stem(args.score_type, args.tail_fraction, args.late_rows)
    output_dir = (args.output_dir or stage_output_dir(heatmap_run_root, "analysis", __file__)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vbench_rows = _load_vbench_rows(args.vbench_long_csv.resolve())
    seed_dirs = _iter_seed_dirs(heatmap_run_root)

    rows: list[dict[str, object]] = []
    missing_heatmaps: list[str] = []
    for seed_dir in seed_dirs:
        row = _metric_row_for_seed(
            seed_dir,
            args.score_type,
            args.tail_fraction,
            args.late_rows,
            vbench_rows,
        )
        if row is None:
            missing_heatmaps.append(str(seed_dir))
            continue
        rows.append(row)

    ranked_rows = _rank_rows(rows)
    winners: list[dict[str, object]] = [row for row in ranked_rows if int(row["tail_mean_rank"]) == 1]

    fieldnames = [
        "prompt_id",
        "prompt_text",
        "axis",
        "seed_idx",
        "prompt_size",
        "tail_mean_rank",
        "score_name",
        "score_value",
        "tail_fraction",
        "late_rows",
        "tail_mean_q",
        "late_mean",
        "late_tail_mean_q",
        "heatmap_mean",
        "min_similarity",
        "max_similarity",
        "heatmap_rows",
        "heatmap_cols",
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "aesthetic_quality",
        "imaging_quality",
        "heatmap_path",
    ]

    all_rows_csv = output_dir / f"{output_stem}_seed_rankings.csv"
    winners_csv = output_dir / f"{output_stem}_prompt_winners.csv"
    metadata_json = output_dir / f"{output_stem}_metadata.json"

    _write_csv(all_rows_csv, ranked_rows, fieldnames)
    _write_csv(winners_csv, winners, fieldnames)

    metadata = {
        "heatmap_run_root": str(heatmap_run_root),
        "vbench_long_csv": str(args.vbench_long_csv.resolve()),
        "score_type": args.score_type,
        "tail_fraction": args.tail_fraction,
        "late_rows": args.late_rows,
        "output_stem": output_stem,
        "available_heatmaps": len(rows),
        "missing_heatmaps": len(missing_heatmaps),
        "missing_heatmap_seed_dirs": missing_heatmaps,
        "output_files": {
            "seed_rankings_csv": str(all_rows_csv),
            "prompt_winners_csv": str(winners_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()