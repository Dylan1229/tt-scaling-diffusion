"""Grid-search window size and tail_k for posterior-mean similarity matrices.

Each candidate score is:
    1. take the last n rows or columns of the heatmap,
    2. flatten,
    3. sort,
    4. take the lowest k fraction,
    5. average them.

The script measures how each (n, k) aligns with the existing VBench seed
scores and writes one wide CSV row per candidate.
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
from ttsd.runners.utilities.seed_vbench_loaders import (
    SUBSCORES,
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
)

VALID_DIRECTIONS = {"rows", "cols"}
VALID_MATRIX_TYPES = {"diagonal", "frame_neighbor", "posterior_neighbor"}

MATRIX_FILE_BY_TYPE = {
    "diagonal": "posterior_mean_diagonal_similarity.npy",
    "frame_neighbor": "posterior_mean_frame_neighbor_similarity.npy",
    "posterior_neighbor": "posterior_mean_posterior_neighbor_similarity.npy",
}

SCORE_PREFIX_BY_MATRIX_TYPE = {
    "diagonal": {"rows": "late", "cols": "frame"},
    "frame_neighbor": {"rows": "posterior", "cols": "frame"},
    "posterior_neighbor": {"rows": "posterior", "cols": "frame"},
}


def _tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    flattened = np.sort(values.reshape(-1))
    tail_count = max(1, math.ceil(flattened.size * tail_fraction))
    return float(flattened[:tail_count].mean())


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _parse_tail_fractions(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid tail fraction: {value}")
        values.append(value)
    if not values:
        raise ValueError("No tail fractions provided")
    return values


def _default_tail_fractions() -> list[float]:
    return [round(step / 100.0, 2) for step in range(5, 101, 5)]


def _load_seed_records(
    heatmap_run_root: Path,
    vbench_rows: dict[tuple[str, int], dict[str, object]],
    matrix_type: str,
) -> tuple[list[dict[str, object]], int, int]:
    seed_records = []
    max_rows = 0
    max_cols = 0
    matrix_filename = MATRIX_FILE_BY_TYPE[matrix_type]
    for seed_dir in _iter_seed_dirs(heatmap_run_root):
        heatmap_path = seed_dir / matrix_filename
        if not heatmap_path.exists():
            continue
        prompt_id = seed_dir.parent.name
        seed_idx = _seed_idx_from_name(seed_dir)
        heatmap = np.load(heatmap_path)
        if heatmap.ndim != 2:
            raise ValueError(f"Expected 2D heatmap at {heatmap_path}, got {heatmap.shape}")
        max_rows = max(max_rows, int(heatmap.shape[0]))
        max_cols = max(max_cols, int(heatmap.shape[1]))
        record = dict(vbench_rows[(prompt_id, seed_idx)])
        record["prompt_id"] = prompt_id
        record["seed_idx"] = seed_idx
        record["heatmap"] = heatmap
        seed_records.append(record)

    grouped = defaultdict(list)
    for record in seed_records:
        grouped[str(record["prompt_id"])].append(record)

    for _, group in grouped.items():
        for metric in SUBSCORES:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            mean = values.mean()
            std = values.std(ddof=0)
            z_values = np.zeros_like(values) if std == 0 else (values - mean) / std
            for row, z_value in zip(group, z_values):
                row[f"z_{metric}"] = float(z_value)
        for row in group:
            row["avg_vbench_z"] = float(np.mean([row[f"z_{metric}"] for metric in SUBSCORES]))
            row["avg_vbench_raw"] = float(np.mean([float(row[metric]) for metric in SUBSCORES]))

    return seed_records, max_rows, max_cols


def _candidate_score(heatmap: np.ndarray, last_n: int, tail_fraction: float, direction: str) -> float:
    if direction == "rows":
        values = heatmap[-last_n:, :]
    elif direction == "cols":
        values = heatmap[:, -last_n:]
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return _tail_mean(values, tail_fraction)


def _alignment_stats(by_prompt_subset: dict[str, list[dict[str, object]]],
                     metric: str) -> dict[str, object]:
    prompt_spearmans: list[float] = []
    winner_match = 0
    top3_hit = 0
    score_top5_contains_target_best = 0
    score_top5_overlap_total = 0.0
    for _, group in sorted(by_prompt_subset.items()):
        score_rank_values = np.array([-row["score_value"] for row in group], dtype=float)
        metric_rank_values = np.array([-row[metric] for row in group], dtype=float)
        prompt_spearmans.append(
            float(np.corrcoef(_rankdata(score_rank_values), _rankdata(metric_rank_values))[0, 1])
        )
        score_winner = max(group, key=lambda row: (row["score_value"], -int(row["seed_idx"])))
        score_top5 = sorted(group, key=lambda row: (-row["score_value"], int(row["seed_idx"])))[:5]
        metric_sorted = sorted(group, key=lambda row: (-row[metric], int(row["seed_idx"])))
        metric_best = metric_sorted[0]
        metric_top5 = metric_sorted[:5]
        metric_rank = next(
            index + 1
            for index, row in enumerate(metric_sorted)
            if int(row["seed_idx"]) == int(score_winner["seed_idx"])
        )
        if metric_rank == 1:
            winner_match += 1
        if metric_rank <= 3:
            top3_hit += 1
        if any(int(row["seed_idx"]) == int(metric_best["seed_idx"]) for row in score_top5):
            score_top5_contains_target_best += 1
        score_top5_seed_ids = {int(row["seed_idx"]) for row in score_top5}
        metric_top5_seed_ids = {int(row["seed_idx"]) for row in metric_top5}
        score_top5_overlap_total += len(score_top5_seed_ids & metric_top5_seed_ids) / 5.0
    n = len(by_prompt_subset)
    return {
        "prompt_mean_spearman": float(np.mean(prompt_spearmans)) if prompt_spearmans else float("nan"),
        "prompt_median_spearman": float(np.median(prompt_spearmans)) if prompt_spearmans else float("nan"),
        "winner_match": winner_match,
        "winner_match_rate": winner_match / n if n else float("nan"),
        "top3_hit": top3_hit,
        "top3_hit_rate": top3_hit / n if n else float("nan"),
        "score_top5_contains_target_best": score_top5_contains_target_best,
        "score_top5_contains_target_best_rate": score_top5_contains_target_best / n if n else float("nan"),
        "score_top5_overlap_rate": score_top5_overlap_total / n if n else float("nan"),
        "n_prompts": n,
    }


def _compute_alignment_rows(
    seed_records: list[dict[str, object]],
    last_n_values: list[int],
    tail_fractions: list[float],
    direction: str,
    matrix_type: str,
) -> list[dict[str, object]]:
    by_prompt = defaultdict(list)
    for row in seed_records:
        by_prompt[str(row["prompt_id"])].append(row)

    target_metrics = SUBSCORES + ["avg_vbench_z", "avg_vbench_raw"]
    summary_rows: list[dict[str, object]] = []

    for last_n in last_n_values:
        for tail_fraction in tail_fractions:
            candidate_rows = []
            for row in seed_records:
                score_value = _candidate_score(row["heatmap"], last_n, tail_fraction, direction)
                candidate = {
                    "prompt_id": row["prompt_id"],
                    "seed_idx": row["seed_idx"],
                    "score_value": score_value,
                }
                for metric in target_metrics:
                    candidate[metric] = float(row[metric])
                candidate_rows.append(candidate)

            score_values = np.array([row["score_value"] for row in candidate_rows], dtype=float)
            prefix_name = SCORE_PREFIX_BY_MATRIX_TYPE[matrix_type][direction]
            summary = {
                "matrix_type": matrix_type,
                "direction": direction,
                "last_n": last_n,
                "tail_k": tail_fraction,
                "score_name": f"{prefix_name}_tail{int(round(tail_fraction * 100))}_last{last_n}",
            }

            for metric in target_metrics:
                metric_values = np.array([row[metric] for row in candidate_rows], dtype=float)
                pearson = float(np.corrcoef(score_values, metric_values)[0, 1])
                spearman = float(np.corrcoef(_rankdata(score_values), _rankdata(metric_values))[0, 1])

                prompt_spearmans = []
                winner_match = 0
                top3_hit = 0
                score_top5_contains_target_best = 0
                score_top5_overlap_total = 0.0
                grouped_candidates = defaultdict(list)
                for row in candidate_rows:
                    grouped_candidates[str(row["prompt_id"])].append(row)

                for prompt_id, group in sorted(grouped_candidates.items()):
                    score_rank_values = np.array([-row["score_value"] for row in group], dtype=float)
                    metric_rank_values = np.array([-row[metric] for row in group], dtype=float)
                    prompt_spearmans.append(
                        float(np.corrcoef(_rankdata(score_rank_values), _rankdata(metric_rank_values))[0, 1])
                    )

                    score_winner = max(group, key=lambda row: (row["score_value"], -int(row["seed_idx"])))
                    score_top5 = sorted(
                        group,
                        key=lambda row: (-row["score_value"], int(row["seed_idx"])),
                    )[:5]
                    metric_sorted = sorted(group, key=lambda row: (-row[metric], int(row["seed_idx"])))
                    metric_best = metric_sorted[0]
                    metric_top5 = metric_sorted[:5]
                    metric_rank = next(
                        index + 1
                        for index, row in enumerate(metric_sorted)
                        if int(row["seed_idx"]) == int(score_winner["seed_idx"])
                    )
                    if metric_rank == 1:
                        winner_match += 1
                    if metric_rank <= 3:
                        top3_hit += 1
                    if any(int(row["seed_idx"]) == int(metric_best["seed_idx"]) for row in score_top5):
                        score_top5_contains_target_best += 1
                    score_top5_seed_ids = {int(row["seed_idx"]) for row in score_top5}
                    metric_top5_seed_ids = {int(row["seed_idx"]) for row in metric_top5}
                    score_top5_overlap_total += len(score_top5_seed_ids & metric_top5_seed_ids) / 5.0

                prefix = metric
                summary["n_prompts"] = len(by_prompt)
                summary[f"{prefix}_pearson"] = pearson
                summary[f"{prefix}_spearman"] = spearman
                summary[f"{prefix}_prompt_mean_spearman"] = float(np.mean(prompt_spearmans))
                summary[f"{prefix}_prompt_median_spearman"] = float(np.median(prompt_spearmans))
                summary[f"{prefix}_winner_match"] = winner_match
                summary[f"{prefix}_winner_match_rate"] = winner_match / len(by_prompt)
                summary[f"{prefix}_top3_hit"] = top3_hit
                summary[f"{prefix}_top3_hit_rate"] = top3_hit / len(by_prompt)
                summary[f"{prefix}_score_top5_contains_target_best"] = score_top5_contains_target_best
                summary[f"{prefix}_score_top5_contains_target_best_rate"] = (
                    score_top5_contains_target_best / len(by_prompt)
                )
                summary[f"{prefix}_score_top5_overlap_rate"] = score_top5_overlap_total / len(by_prompt)

                sizes = sorted({len(g) for g in grouped_candidates.values()})
                for sz in sizes:
                    sub = {pid: g for pid, g in grouped_candidates.items() if len(g) == sz}
                    b = _alignment_stats(sub, metric)
                    summary[f"n_prompts_n{sz}"] = b["n_prompts"]
                    summary[f"{prefix}_prompt_mean_spearman_n{sz}"] = b["prompt_mean_spearman"]
                    summary[f"{prefix}_prompt_median_spearman_n{sz}"] = b["prompt_median_spearman"]
                    summary[f"{prefix}_winner_match_n{sz}"] = b["winner_match"]
                    summary[f"{prefix}_winner_match_rate_n{sz}"] = b["winner_match_rate"]
                    summary[f"{prefix}_top3_hit_n{sz}"] = b["top3_hit"]
                    summary[f"{prefix}_top3_hit_rate_n{sz}"] = b["top3_hit_rate"]
                    summary[f"{prefix}_score_top5_contains_target_best_n{sz}"] = b["score_top5_contains_target_best"]
                    summary[f"{prefix}_score_top5_contains_target_best_rate_n{sz}"] = b["score_top5_contains_target_best_rate"]
                    summary[f"{prefix}_score_top5_overlap_rate_n{sz}"] = b["score_top5_overlap_rate"]

            summary_rows.append(summary)

    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    matrix_type = rows[0]["matrix_type"]
    direction = rows[0]["direction"]
    axis_label = "rows" if direction == "rows" else "columns"
    n_prompts = rows[0].get("n_prompts", "?") if rows else "?"
    best_avg = max(rows, key=lambda row: row["avg_vbench_z_prompt_mean_spearman"])
    best_subj = max(rows, key=lambda row: row["subject_consistency_prompt_mean_spearman"])
    best_avg_winner = max(rows, key=lambda row: row["avg_vbench_z_winner_match"])
    best_subj_winner = max(rows, key=lambda row: row["subject_consistency_winner_match"])
    best_avg_top5 = max(rows, key=lambda row: row["avg_vbench_z_score_top5_contains_target_best"])
    best_subj_top5 = max(rows, key=lambda row: row["subject_consistency_score_top5_contains_target_best"])
    best_avg_top5_overlap = max(rows, key=lambda row: row["avg_vbench_z_score_top5_overlap_rate"])
    best_subj_top5_overlap = max(rows, key=lambda row: row["subject_consistency_score_top5_overlap_rate"])

    lines = [
        "# Last-N / Tail-K Grid Search",
        "",
        "This table searches the metric:",
        "",
        f"1. take the last `n` heatmap {axis_label},",
        "2. flatten,",
        "3. sort,",
        "4. take the lowest `k` fraction,",
        "5. average them.",
        "",
        f"Matrix type: `{matrix_type}`",
        f"Direction: `{direction}`",
        f"Number of prompts: `{n_prompts}`",
        "",
        "## Best by prompt-local average VBench z Spearman",
        "",
        f"- score: `{best_avg['score_name']}`",
        f"- `avg_vbench_z_prompt_mean_spearman = {best_avg['avg_vbench_z_prompt_mean_spearman']:.4f}`",
        f"- `avg_vbench_z_winner_match = {best_avg['avg_vbench_z_winner_match']}/{n_prompts}`",
        "",
        "## Best by subject consistency Spearman",
        "",
        f"- score: `{best_subj['score_name']}`",
        f"- `subject_consistency_prompt_mean_spearman = {best_subj['subject_consistency_prompt_mean_spearman']:.4f}`",
        f"- `subject_consistency_winner_match = {best_subj['subject_consistency_winner_match']}/{n_prompts}`",
        "",
        "## Best by prompt-local average VBench winner match",
        "",
        f"- score: `{best_avg_winner['score_name']}`",
        f"- `avg_vbench_z_winner_match = {best_avg_winner['avg_vbench_z_winner_match']}/{n_prompts}`",
        f"- `avg_vbench_z_prompt_mean_spearman = {best_avg_winner['avg_vbench_z_prompt_mean_spearman']:.4f}`",
        "",
        "## Best by subject consistency winner match",
        "",
        f"- score: `{best_subj_winner['score_name']}`",
        f"- `subject_consistency_winner_match = {best_subj_winner['subject_consistency_winner_match']}/{n_prompts}`",
        f"- `subject_consistency_prompt_mean_spearman = {best_subj_winner['subject_consistency_prompt_mean_spearman']:.4f}`",
        "",
        "## Best by prompt-local average VBench top-5 containment",
        "",
        f"- score: `{best_avg_top5['score_name']}`",
        f"- `avg_vbench_z_score_top5_contains_target_best = {best_avg_top5['avg_vbench_z_score_top5_contains_target_best']}/{n_prompts}`",
        f"- `avg_vbench_z_prompt_mean_spearman = {best_avg_top5['avg_vbench_z_prompt_mean_spearman']:.4f}`",
        "",
        "## Best by subject consistency top-5 containment",
        "",
        f"- score: `{best_subj_top5['score_name']}`",
        f"- `subject_consistency_score_top5_contains_target_best = {best_subj_top5['subject_consistency_score_top5_contains_target_best']}/{n_prompts}`",
        f"- `subject_consistency_prompt_mean_spearman = {best_subj_top5['subject_consistency_prompt_mean_spearman']:.4f}`",
        "",
        "## Best by prompt-local average VBench top-5 overlap",
        "",
        f"- score: `{best_avg_top5_overlap['score_name']}`",
        f"- `avg_vbench_z_score_top5_overlap_rate = {best_avg_top5_overlap['avg_vbench_z_score_top5_overlap_rate']:.4f}`",
        "",
        "## Best by subject consistency top-5 overlap",
        "",
        f"- score: `{best_subj_top5_overlap['score_name']}`",
        f"- `subject_consistency_score_top5_overlap_rate = {best_subj_top5_overlap['subject_consistency_score_top5_overlap_rate']:.4f}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-run-root", type=Path, default=None)
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-last-n", type=int, default=None)
    parser.add_argument("--tail-fractions", type=str, default=None)
    parser.add_argument("--direction", type=str, choices=sorted(VALID_DIRECTIONS), default="rows")
    parser.add_argument("--matrix-type", type=str, choices=sorted(VALID_MATRIX_TYPES), default="diagonal")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args(argv)
    resolve_run_id(args, parser, needs=["heatmap_run_root", "vbench_long_csv"])

    heatmap_run_root = args.heatmap_run_root.resolve()
    variant = f"{args.matrix_type}_{args.direction}"
    output_dir = (args.output_dir or stage_output_dir(heatmap_run_root, "analysis", __file__) / variant).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vbench_rows = _load_vbench_rows(args.vbench_long_csv.resolve())
    seed_records, max_rows, max_cols = _load_seed_records(heatmap_run_root, vbench_rows, args.matrix_type)

    max_window = max_rows if args.direction == "rows" else max_cols
    last_n_values = list(range(1, (args.max_last_n or max_window) + 1))
    tail_fractions = _parse_tail_fractions(args.tail_fractions) if args.tail_fractions else _default_tail_fractions()

    rows = _compute_alignment_rows(seed_records, last_n_values, tail_fractions, args.direction, args.matrix_type)
    rows = sorted(
        rows,
        key=lambda row: (
            -row["avg_vbench_z_prompt_mean_spearman"],
            -row["avg_vbench_z_winner_match"],
            row["last_n"],
            row["tail_k"],
        ),
    )

    csv_path = output_dir / "lastn_tailk_alignment_table.csv"
    metadata_path = output_dir / "lastn_tailk_alignment_metadata.json"
    markdown_path = output_dir / "lastn_tailk_alignment_summary.md"

    _write_csv(csv_path, rows)
    _write_markdown_summary(markdown_path, rows)

    metadata = {
        "heatmap_run_root": str(heatmap_run_root),
        "vbench_long_csv": str(args.vbench_long_csv.resolve()),
        "available_heatmaps": len(seed_records),
        "max_heatmap_rows": max_rows,
        "max_heatmap_cols": max_cols,
        "matrix_type": args.matrix_type,
        "direction": args.direction,
        "searched_last_n": last_n_values,
        "searched_tail_k": tail_fractions,
        "output_files": {
            "alignment_table_csv": str(csv_path),
            "alignment_summary_md": str(markdown_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()