"""Offline oracle analysis for a late-branching feasibility run.

This module deliberately uses final VBench targets. It estimates whether useful
branches exist; it is not an online verifier and must not be reported as one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TARGET_COLUMNS = ("vbench_quality", "dynamic_degree", "overall_consistency")


def _read_candidate_meta(run_dir: Path) -> dict[tuple[str, int], dict]:
    candidates: dict[tuple[str, int], dict] = {}
    for path in sorted(run_dir.glob("*/seed*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("experiment") != "late_branching_best_of_m":
            continue
        key = (meta["prompt_id"], int(meta["seed"]))
        if key in candidates:
            raise ValueError(f"duplicate candidate identity {key}")
        meta["_meta_path"] = str(path)
        candidates[key] = meta
    if not candidates:
        raise ValueError(f"no late-branch candidate metadata found under {run_dir}")
    return candidates


def _read_targets(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    targets: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["prompt_id"], int(row["seed_idx"]))
            if key in targets:
                raise ValueError(f"duplicate target identity {key} in {path}")
            try:
                targets[key] = {column: float(row[column]) for column in TARGET_COLUMNS}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid target row for {key}: {exc}") from exc
    return targets


def _delta(candidate: dict, control: dict, metric: str) -> float:
    return candidate[metric] - control[metric]


def analyze_groups(
    candidates: dict[tuple[str, int], dict],
    targets: dict[tuple[str, int], dict[str, float]],
    baseline_targets: dict[tuple[str, int], dict[str, float]],
    *,
    dynamic_tolerance: float = 0.0,
    overall_tolerance: float = 0.0,
    win_epsilon: float = 1e-12,
) -> list[dict]:
    """Return one oracle-analysis row per (prompt, root seed) group."""

    missing_candidates = sorted(set(candidates) - set(targets))
    if missing_candidates:
        preview = ", ".join(
            f"{pid}/seed{seed}" for pid, seed in missing_candidates[:5]
        )
        raise ValueError(
            "candidate VBench targets are missing "
            f"{len(missing_candidates)} generated videos: {preview}"
        )

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for key, meta in candidates.items():
        record = {**meta, **targets[key]}
        grouped[(meta["prompt_id"], int(meta["root_seed"]))].append(record)

    missing_baselines = sorted(set(grouped) - set(baseline_targets))
    if missing_baselines:
        preview = ", ".join(
            f"{pid}/seed{seed}" for pid, seed in missing_baselines[:5]
        )
        raise ValueError(
            f"baseline VBench targets are missing {len(missing_baselines)} roots: {preview}"
        )

    rows: list[dict] = []
    for (prompt_id, root_seed), records in sorted(grouped.items()):
        batched_controls = [
            record for record in records if record["branch_kind"] == "batched_control"
        ]
        noises = [record for record in records if record["branch_kind"] == "noise"]
        if len(batched_controls) != 1 or not noises:
            raise ValueError(
                f"{prompt_id}/root_seed{root_seed} needs one batched control and >=1 "
                f"noise branch; found {len(batched_controls)} and {len(noises)}"
            )
        batched_control = batched_controls[0]
        baseline = {
            "seed": root_seed,
            "branch_kind": "true_baseline",
            **baseline_targets[(prompt_id, root_seed)],
        }
        options = [baseline, *noises]

        noise_quality_best = max(noises, key=lambda record: record["vbench_quality"])
        quality_best = max(options, key=lambda record: record["vbench_quality"])
        same_dynamic = [
            record
            for record in options
            if abs(_delta(record, baseline, "dynamic_degree")) <= dynamic_tolerance
        ]
        same_dynamic_best = max(
            same_dynamic, key=lambda record: record["vbench_quality"]
        )
        safe_options = [
            record
            for record in options
            if _delta(record, baseline, "dynamic_degree") >= -dynamic_tolerance
            and _delta(record, baseline, "overall_consistency") >= -overall_tolerance
        ]
        safe_best = max(safe_options, key=lambda record: record["vbench_quality"])

        pareto_candidates = [
            record
            for record in noises
            if all(
                _delta(record, baseline, metric) >= -tolerance
                for metric, tolerance in (
                    ("vbench_quality", 0.0),
                    ("dynamic_degree", dynamic_tolerance),
                    ("overall_consistency", overall_tolerance),
                )
            )
            and any(
                _delta(record, baseline, metric) > win_epsilon
                for metric in TARGET_COLUMNS
            )
        ]

        row = {
            "prompt_id": prompt_id,
            "prompt_text": batched_control["prompt_text"],
            "axis": batched_control.get("axis", ""),
            "root_seed": root_seed,
            "branch_step": int(batched_control["branch_step"]),
            "num_noise_branches": len(noises),
            "baseline_seed": root_seed,
            "baseline_vbench_quality": baseline["vbench_quality"],
            "baseline_dynamic_degree": baseline["dynamic_degree"],
            "baseline_overall_consistency": baseline["overall_consistency"],
            "batched_control_candidate_seed": int(batched_control["seed"]),
            "batched_control_quality_delta": _delta(
                batched_control, baseline, "vbench_quality"
            ),
            "batched_control_dynamic_delta": _delta(
                batched_control, baseline, "dynamic_degree"
            ),
            "batched_control_overall_delta": _delta(
                batched_control, baseline, "overall_consistency"
            ),
            "best_noise_candidate_seed": int(noise_quality_best["seed"]),
            "best_noise_vs_batched_control_quality_delta": _delta(
                noise_quality_best, batched_control, "vbench_quality"
            ),
            "best_noise_vs_batched_control_dynamic_delta": _delta(
                noise_quality_best, batched_control, "dynamic_degree"
            ),
            "best_noise_vs_batched_control_overall_delta": _delta(
                noise_quality_best, batched_control, "overall_consistency"
            ),
            "quality_best_candidate_seed": int(quality_best["seed"]),
            "quality_best_is_baseline": quality_best is baseline,
            "quality_oracle_delta": _delta(quality_best, baseline, "vbench_quality"),
            "quality_oracle_dynamic_delta": _delta(
                quality_best, baseline, "dynamic_degree"
            ),
            "quality_oracle_overall_delta": _delta(
                quality_best, baseline, "overall_consistency"
            ),
            "same_dynamic_best_candidate_seed": int(same_dynamic_best["seed"]),
            "same_dynamic_best_is_baseline": same_dynamic_best is baseline,
            "same_dynamic_quality_delta": _delta(
                same_dynamic_best, baseline, "vbench_quality"
            ),
            "safe_best_candidate_seed": int(safe_best["seed"]),
            "safe_best_is_baseline": safe_best is baseline,
            "safe_quality_delta": _delta(safe_best, baseline, "vbench_quality"),
            "safe_dynamic_delta": _delta(safe_best, baseline, "dynamic_degree"),
            "safe_overall_delta": _delta(
                safe_best, baseline, "overall_consistency"
            ),
            "has_pareto_improvement": bool(pareto_candidates),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict], *, win_epsilon: float = 1e-12) -> dict:
    if not rows:
        return {"n_groups": 0}

    def win_rate(column: str) -> float:
        return sum(row[column] > win_epsilon for row in rows) / len(rows)

    return {
        "n_groups": len(rows),
        "branch_steps": sorted({int(row["branch_step"]) for row in rows}),
        "quality_oracle_win_rate": win_rate("quality_oracle_delta"),
        "quality_oracle_mean_delta": statistics.fmean(
            row["quality_oracle_delta"] for row in rows
        ),
        "quality_selected_mean_dynamic_delta": statistics.fmean(
            row["quality_oracle_dynamic_delta"] for row in rows
        ),
        "quality_selected_mean_overall_delta": statistics.fmean(
            row["quality_oracle_overall_delta"] for row in rows
        ),
        "batched_control_mean_quality_delta": statistics.fmean(
            row["batched_control_quality_delta"] for row in rows
        ),
        "batched_control_mean_dynamic_delta": statistics.fmean(
            row["batched_control_dynamic_delta"] for row in rows
        ),
        "batched_control_mean_overall_delta": statistics.fmean(
            row["batched_control_overall_delta"] for row in rows
        ),
        "best_noise_over_batched_control_win_rate": win_rate(
            "best_noise_vs_batched_control_quality_delta"
        ),
        "best_noise_over_batched_control_mean_quality_delta": statistics.fmean(
            row["best_noise_vs_batched_control_quality_delta"] for row in rows
        ),
        "same_dynamic_quality_win_rate": win_rate("same_dynamic_quality_delta"),
        "same_dynamic_quality_mean_delta": statistics.fmean(
            row["same_dynamic_quality_delta"] for row in rows
        ),
        "safe_quality_win_rate": win_rate("safe_quality_delta"),
        "safe_quality_mean_delta": statistics.fmean(
            row["safe_quality_delta"] for row in rows
        ),
        "pareto_improvement_rate": statistics.fmean(
            float(row["has_pareto_improvement"]) for row in rows
        ),
    }


def build_summary(
    rows: list[dict],
    *,
    bottom_n: int = 15,
    bottom_fraction: float = 0.25,
    win_epsilon: float = 1e-12,
) -> dict:
    ordered = sorted(rows, key=lambda row: row["baseline_vbench_quality"])
    bottom_n_rows = ordered[: min(bottom_n, len(ordered))]
    quartile_count = max(1, math.ceil(len(ordered) * bottom_fraction)) if ordered else 0
    bottom_fraction_rows = ordered[:quartile_count]
    return {
        "interpretation": (
            "Offline final-metric oracle upper bound only. It tests whether a useful "
            "late branch exists relative to a separate batch-one baseline; it is not "
            "a causal or online verifier."
        ),
        "scope": (
            "Results apply only to the branch_step values listed in each stratum. "
            "They do not establish behavior at other intervention steps."
        ),
        "strata": {
            "all": summarize_rows(rows, win_epsilon=win_epsilon),
            f"bottom_{bottom_n}_by_baseline_quality": summarize_rows(
                bottom_n_rows, win_epsilon=win_epsilon
            ),
            f"bottom_{bottom_fraction:.0%}_by_baseline_quality": summarize_rows(
                bottom_fraction_rows, win_epsilon=win_epsilon
            ),
        },
    }


def write_analysis(rows: list[dict], summary: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "late_branch_oracle_per_group.csv"
    json_path = output_dir / "late_branch_oracle_summary.json"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    json_path.write_text(json.dumps(summary, indent=2))
    return csv_path, json_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--targets",
        required=True,
        type=Path,
        help="vbench_targets.csv produced by the updated dev-branch evaluator.",
    )
    parser.add_argument(
        "--baseline-targets",
        required=True,
        type=Path,
        help="vbench_targets.csv for the original batch-one baseline run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bottom-n", type=int, default=15)
    parser.add_argument("--bottom-fraction", type=float, default=0.25)
    parser.add_argument("--dynamic-tolerance", type=float, default=0.0)
    parser.add_argument("--overall-tolerance", type=float, default=0.0)
    parser.add_argument("--win-epsilon", type=float, default=1e-12)
    args = parser.parse_args(argv)

    if args.bottom_n < 1:
        raise SystemExit("--bottom-n must be positive")
    if not 0 < args.bottom_fraction <= 1:
        raise SystemExit("--bottom-fraction must be in (0, 1]")
    if args.dynamic_tolerance < 0 or args.overall_tolerance < 0:
        raise SystemExit("metric tolerances must be non-negative")

    candidates = _read_candidate_meta(args.run)
    targets = _read_targets(args.targets)
    baseline_targets = _read_targets(args.baseline_targets)
    rows = analyze_groups(
        candidates,
        targets,
        baseline_targets,
        dynamic_tolerance=args.dynamic_tolerance,
        overall_tolerance=args.overall_tolerance,
        win_epsilon=args.win_epsilon,
    )
    summary = build_summary(
        rows,
        bottom_n=args.bottom_n,
        bottom_fraction=args.bottom_fraction,
        win_epsilon=args.win_epsilon,
    )
    output_dir = args.output or (args.run / "_analysis")
    csv_path, json_path = write_analysis(rows, summary, output_dir)
    print(f"[late_branch_oracle] groups={len(rows)}")
    print(f"[late_branch_oracle] per-group CSV: {csv_path}")
    print(f"[late_branch_oracle] summary JSON: {json_path}")


if __name__ == "__main__":
    main()
