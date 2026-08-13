"""Summarize baseline vs Step35 branching wall-clock benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NUM_STEPS = 50
BRANCH_STEP = 35


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("no timing values")
    arr = np.asarray(values, dtype=float)
    return {
        "n_roots": int(len(values)),
        "mean_seconds": float(arr.mean()),
        "median_seconds": float(np.median(arr)),
        "p90_seconds": _percentile(values, 90),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "total_seconds": float(arr.sum()),
    }


def _baseline_rows(run: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(run.glob("*/seed*/meta.json")):
        meta = _load_json(meta_path)
        elapsed = meta.get("elapsed_seconds")
        if elapsed is None:
            continue
        rows.append(
            {
                "method": "baseline",
                "prompt_id": meta["prompt_id"],
                "root_seed": int(meta["seed"]),
                "elapsed_seconds": float(elapsed),
                "denoising_step_equivalents": NUM_STEPS,
                "theoretical_ratio_with_control": 1.0,
                "theoretical_ratio_without_control": 1.0,
                "total_candidates_written": 1,
            }
        )
    return rows


def _branch_rows(run: Path, label: str) -> list[dict]:
    grouped: dict[tuple[str, int], dict] = {}
    for meta_path in sorted(run.glob("*/seed*/meta.json")):
        meta = _load_json(meta_path)
        key = (str(meta["prompt_id"]), int(meta["root_seed"]))
        if key in grouped:
            continue
        total_candidates = int(meta["total_candidates"])
        num_noise_branches = total_candidates - 1
        without_control = (
            BRANCH_STEP + (NUM_STEPS - BRANCH_STEP) * num_noise_branches
        ) / NUM_STEPS
        grouped[key] = {
            "method": label,
            "prompt_id": key[0],
            "root_seed": key[1],
            "elapsed_seconds": float(meta["elapsed_seconds_for_group"]),
            "denoising_step_equivalents": int(meta["denoising_step_equivalents"]),
            "theoretical_ratio_with_control": float(meta["compute_ratio_vs_baseline"]),
            "theoretical_ratio_without_control": float(without_control),
            "total_candidates_written": total_candidates,
        }
    return list(grouped.values())


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_seconds(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{seconds:.1f}s ({minutes:.2f}m)"


def _plot(summary_rows: list[dict], output: Path) -> None:
    labels = [row["method"] for row in summary_rows]
    baseline_median = summary_rows[0]["median_seconds"]
    measured_ratios = [row["median_seconds"] / baseline_median for row in summary_rows]
    theory = [row["theoretical_ratio_with_control"] for row in summary_rows]
    theory_no_control = [
        row["theoretical_ratio_without_control"] for row in summary_rows
    ]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - width, measured_ratios, width, label="measured median wall-clock")
    ax.bar(x, theory, width, label="step-equivalent with diagnostic control")
    ax.bar(x + width, theory_no_control, width, label="step-equivalent without control")
    ax.set_ylabel("relative cost vs baseline")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(max(measured_ratios), max(theory)) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Wan2.2 5B 480p: Step35 Branching Compute Cost")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--m2-run", type=Path, required=True)
    parser.add_argument("--m4-run", type=Path, required=True)
    parser.add_argument("--m8-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_root = []
    per_root.extend(_baseline_rows(args.baseline_run))
    per_root.extend(_branch_rows(args.m2_run, "step35_m2_plus_control"))
    per_root.extend(_branch_rows(args.m4_run, "step35_m4_plus_control"))
    per_root.extend(_branch_rows(args.m8_run, "step35_m8_plus_control"))

    order = [
        "baseline",
        "step35_m2_plus_control",
        "step35_m4_plus_control",
        "step35_m8_plus_control",
    ]
    summary_rows = []
    for method in order:
        rows = [row for row in per_root if row["method"] == method]
        stats = _summarize([float(row["elapsed_seconds"]) for row in rows])
        first = rows[0]
        summary_rows.append(
            {
                "method": method,
                **stats,
                "theoretical_ratio_with_control": first[
                    "theoretical_ratio_with_control"
                ],
                "theoretical_ratio_without_control": first[
                    "theoretical_ratio_without_control"
                ],
                "total_candidates_written": first["total_candidates_written"],
            }
        )

    baseline_median = summary_rows[0]["median_seconds"]
    for row in summary_rows:
        row["measured_median_ratio_vs_baseline"] = (
            row["median_seconds"] / baseline_median
            if baseline_median and not math.isnan(baseline_median)
            else float("nan")
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "compute_cost_per_root.csv", per_root)
    _write_csv(args.output / "compute_cost_summary.csv", summary_rows)
    _plot(summary_rows, args.output / "compute_cost_tradeoff.png")

    markdown = [
        "# Compute Cost Benchmark",
        "",
        "Wan2.2 5B, 480p, 81 frames, 50 UniPC steps. Step35 branching runs",
        "one batched unperturbed diagnostic control plus M noisy suffix branches.",
        "",
        "| method | n | median | p90 | measured ratio | theory ratio + control | theory ratio no control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            "| {method} | {n_roots} | {median} | {p90} | {measured:.2f}x | "
            "{theory:.2f}x | {theory_no_control:.2f}x |".format(
                method=row["method"],
                n_roots=row["n_roots"],
                median=_format_seconds(row["median_seconds"]),
                p90=_format_seconds(row["p90_seconds"]),
                measured=row["measured_median_ratio_vs_baseline"],
                theory=row["theoretical_ratio_with_control"],
                theory_no_control=row["theoretical_ratio_without_control"],
            )
        )
    (args.output / "compute_cost_benchmark.md").write_text("\n".join(markdown) + "\n")

    print(f"[compute_cost] per-root: {args.output / 'compute_cost_per_root.csv'}")
    print(f"[compute_cost] summary: {args.output / 'compute_cost_summary.csv'}")
    print(f"[compute_cost] chart: {args.output / 'compute_cost_tradeoff.png'}")
    print(f"[compute_cost] markdown: {args.output / 'compute_cost_benchmark.md'}")


if __name__ == "__main__":
    main()
