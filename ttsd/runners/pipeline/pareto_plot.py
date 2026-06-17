"""Pareto plot generator for pipeline runs.

Reads `result.json` + `config.snapshot.json` from one or more
`runs/pipeline/<run_id>/` directories, groups by strategy, and plots the
(cost, score) tradeoff. The headline artifact for comparing EFD&I-tiered
vs. naive BoN (and any future strategies).

Two views:
  - Run-level (default): one point per run. (cost = total wall-clock,
    score = final_score recorded in result.json).
  - Candidate-level (`--per-candidate`): for ParallelCandidateSearch runs,
    one point per candidate from events.jsonl (cost = per-candidate
    wall_s, score = per-candidate verifier score). Useful for visualizing
    "compute budget vs. best-so-far" within a single BoN run.

Usage:
    python -m ttsd.runners.pipeline.pareto_plot \\
        --runs runs/pipeline/efdi_dino_smoke runs/pipeline/bon_dino_smoke \\
        --output-dir runs/pipeline/_analysis/pareto

    python -m ttsd.runners.pipeline.pareto_plot \\
        --runs runs/pipeline/bon_dino_smoke --per-candidate

Outputs into --output-dir:
    pareto.png       — scatter plot with per-strategy Pareto frontier
    pareto_data.csv  — long-format underlying data (one row per point)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Point:
    """One scatter-plot point."""
    strategy: str
    run_id: str
    label: str        # human-readable point label (run_id or candidate idx)
    cost_s: float
    score: float
    success: bool


def _read_run_level(run_dir: Path) -> list[Point]:
    """One Point per run, taken from result.json + config.snapshot.json."""
    res_path = run_dir / "result.json"
    cfg_path = run_dir / "config.snapshot.json"
    if not (res_path.exists() and cfg_path.exists()):
        return []
    res = json.loads(res_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    strategy = cfg.get("strategy", {}).get("kind", "unknown")
    cost = float(res.get("budget", {}).get("spent", {}).get("wall_clock_s", 0.0))
    score = res.get("final_score")
    success = bool(res.get("success", False))
    if score is None or not success:
        # Mark unsuccessful runs with NaN score so they show on the plot as
        # "ran for this much time but produced no usable output". This makes
        # the cost-of-failure visible.
        score = float("nan")
    return [Point(
        strategy=strategy,
        run_id=run_dir.name,
        label=run_dir.name,
        cost_s=cost,
        score=float(score),
        success=success,
    )]


def _read_per_candidate(run_dir: Path) -> list[Point]:
    """One Point per BoN candidate. Reads events.jsonl for per-candidate
    wall_s + score. Only meaningful for ParallelCandidateSearch runs."""
    events_path = run_dir / "events.jsonl"
    cfg_path = run_dir / "config.snapshot.json"
    if not (events_path.exists() and cfg_path.exists()):
        return []
    cfg = json.loads(cfg_path.read_text())
    strategy = cfg.get("strategy", {}).get("kind", "unknown")
    if strategy != "parallel_candidates":
        # Per-candidate view only makes sense for BoN-like strategies.
        return []
    points: list[Point] = []
    with events_path.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "candidate_end":
                continue
            score = ev.get("score")
            if score is None or score != score:    # NaN-skip
                continue
            points.append(Point(
                strategy=strategy,
                run_id=run_dir.name,
                label=f"{run_dir.name}/cand{int(ev['candidate']):02d}",
                cost_s=float(ev.get("wall_s", 0.0)),
                score=float(score),
                success=not bool(ev.get("aborted", False)),
            ))
    return points


def _pareto_frontier(points: list[Point]) -> list[Point]:
    """Upper-left envelope: for each cost, keep only the best score that
    is not dominated by any cheaper point. Returns points sorted by cost."""
    if not points:
        return []
    sorted_pts = sorted(
        [p for p in points if p.score == p.score],    # drop NaN
        key=lambda p: p.cost_s,
    )
    frontier: list[Point] = []
    best_score_so_far = float("-inf")
    for p in sorted_pts:
        if p.score > best_score_so_far:
            frontier.append(p)
            best_score_so_far = p.score
    return frontier


def _write_csv(points: list[Point], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "run_id", "label", "cost_s", "score", "success"])
        for p in sorted(points, key=lambda x: (x.strategy, x.cost_s)):
            w.writerow([p.strategy, p.run_id, p.label, f"{p.cost_s:.3f}",
                        f"{p.score:.6f}" if p.score == p.score else "",
                        int(p.success)])


def _plot(points: list[Point], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    by_strategy: dict[str, list[Point]] = defaultdict(list)
    for p in points:
        by_strategy[p.strategy].append(p)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
    cmap = plt.get_cmap("tab10")
    for i, (strategy, pts) in enumerate(sorted(by_strategy.items())):
        color = cmap(i % 10)
        valid = [p for p in pts if p.score == p.score]
        failed = [p for p in pts if p.score != p.score]
        # Scatter all points.
        if valid:
            ax.scatter(
                [p.cost_s for p in valid],
                [p.score for p in valid],
                color=color, s=70, edgecolor="white", linewidth=1.0,
                label=f"{strategy} (n={len(pts)})", zorder=3,
            )
        if failed:
            ax.scatter(
                [p.cost_s for p in failed],
                [0] * len(failed),    # plot failures at y=0 (visible at the floor)
                color=color, marker="x", s=80, label=f"{strategy} failed", zorder=3,
            )
        # Frontier line.
        frontier = _pareto_frontier(valid)
        if len(frontier) >= 2:
            ax.plot(
                [p.cost_s for p in frontier],
                [p.score for p in frontier],
                color=color, linewidth=1.5, linestyle="--", alpha=0.6, zorder=2,
            )

    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("verifier final score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, nargs="+", type=Path,
                   help="One or more runs/pipeline/<run_id>/ directories to plot.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write pareto.png + pareto_data.csv. "
                        "Default: runs/pipeline/_analysis/pareto_<timestamp>/")
    p.add_argument("--per-candidate", action="store_true",
                   help="Plot one point per BoN candidate (from events.jsonl) "
                        "instead of one point per run.")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title override. Default is auto-generated.")
    args = p.parse_args(argv)

    reader = _read_per_candidate if args.per_candidate else _read_run_level

    points: list[Point] = []
    for run in args.runs:
        if not run.is_dir():
            print(f"[pareto] WARN: not a directory: {run}")
            continue
        run_points = reader(run)
        if not run_points:
            print(f"[pareto] WARN: no usable data in {run} "
                  f"(strategy mismatch or missing result.json?)")
        points.extend(run_points)

    if not points:
        raise SystemExit("[pareto] no data points to plot")

    out_dir = args.output_dir or (
        Path("runs/pipeline/_analysis")
        / f"pareto_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or (
        f"Pareto: cost vs. score "
        f"({'per-candidate' if args.per_candidate else 'per-run'}, "
        f"{len(set(p.strategy for p in points))} strategies, {len(points)} points)"
    )
    _plot(points, out_dir / "pareto.png", title)
    _write_csv(points, out_dir / "pareto_data.csv")
    print(f"[pareto] wrote {out_dir / 'pareto.png'}")
    print(f"[pareto] wrote {out_dir / 'pareto_data.csv'}")
    print(f"[pareto] {len(points)} points across {len(set(p.strategy for p in points))} strategies")


if __name__ == "__main__":
    main()
