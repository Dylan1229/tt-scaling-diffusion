"""Sweep richer reductions over the per-seed similarity heatmaps.

Each reduction produces ONE scalar score per seed from one or both clean
matrices (`frame_neighbor`, `posterior_neighbor`). We then align each
reduction's per-seed scores with VBench, identical to
``posterior_mean_metric_grid_search``.

Reductions sweepable here:
- col_min_tail:   per-column min over last_n posterior rows, tail-mean over cols
- col_quantile_tail: per-column q-quantile over last_n rows, tail-mean over cols
- worst_m_cols:   per-column min over last_n rows, mean of lowest M cols
- harmonic_tail:  harmonic mean of the lowest `tail_k` fraction (last_n rows)
- softmin:        -T * log(mean(exp(-x/T))) over last_n rows
- late_minus_early: mean(last L rows) - mean(first E rows)
- combined_tail:  rank-mean of frame_neighbor tail + posterior_neighbor tail

Writes a wide CSV plus a markdown summary highlighting all-time bests
on (a) avg_vbench_z prompt-mean Spearman and (b) avg_vbench_z winner_match.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from ttsd.runners.posterior_mean_tail_rank import _iter_seed_dirs, _load_vbench_rows, _seed_idx_from_name

SUBSCORES = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]

FRAME_FILE = "posterior_mean_frame_neighbor_similarity.npy"
POST_FILE = "posterior_mean_posterior_neighbor_similarity.npy"


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


def _tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    flat = np.sort(values.reshape(-1))
    n = max(1, math.ceil(flat.size * tail_fraction))
    return float(flat[:n].mean())


def _harmonic_tail_mean(values: np.ndarray, tail_fraction: float, eps: float = 1e-6) -> float:
    flat = np.sort(values.reshape(-1))
    n = max(1, math.ceil(flat.size * tail_fraction))
    tail = np.clip(flat[:n], eps, None)
    return float(n / (1.0 / tail).sum())


def _softmin(values: np.ndarray, temperature: float) -> float:
    x = values.reshape(-1).astype(np.float64)
    shifted = -x / max(temperature, 1e-6)
    m = shifted.max()
    return float(-temperature * (math.log(np.exp(shifted - m).mean()) + m))


# --- Reductions -----------------------------------------------------------

def _r_col_min_tail(frame: np.ndarray, post: np.ndarray, *, last_n: int, tail_k: float) -> float:
    sub = frame[-last_n:, :]
    col_min = sub.min(axis=0)
    return _tail_mean(col_min, tail_k)


def _r_col_quantile_tail(frame: np.ndarray, post: np.ndarray, *, last_n: int, q: float, tail_k: float) -> float:
    sub = frame[-last_n:, :]
    col_q = np.quantile(sub, q, axis=0)
    return _tail_mean(col_q, tail_k)


def _r_worst_m_cols(frame: np.ndarray, post: np.ndarray, *, last_n: int, m: int) -> float:
    sub = frame[-last_n:, :]
    col_min = np.sort(sub.min(axis=0))
    return float(col_min[:m].mean())


def _r_harmonic_tail(frame: np.ndarray, post: np.ndarray, *, last_n: int, tail_k: float) -> float:
    sub = frame[-last_n:, :]
    return _harmonic_tail_mean(sub, tail_k)


def _r_softmin(frame: np.ndarray, post: np.ndarray, *, last_n: int, temperature: float) -> float:
    sub = frame[-last_n:, :]
    return _softmin(sub, temperature)


def _r_late_minus_early(frame: np.ndarray, post: np.ndarray, *, late: int, early: int) -> float:
    late_mean = float(frame[-late:, :].mean())
    early_mean = float(frame[:early, :].mean())
    return late_mean - early_mean


def _r_late_plus_col_min(frame: np.ndarray, post: np.ndarray, *, last_n: int, tail_k: float, alpha: float) -> float:
    """Sum of original frame tail mean and per-col-min tail mean (rescaled)."""
    sub = frame[-last_n:, :]
    flat_tail = _tail_mean(sub, tail_k)
    col_tail = _tail_mean(sub.min(axis=0), tail_k)
    return (1 - alpha) * flat_tail + alpha * col_tail


def _r_frame_plus_post_tail(frame: np.ndarray, post: np.ndarray, *, last_n: int, tail_k: float, alpha: float) -> float:
    f_tail = _tail_mean(frame[-last_n:, :], tail_k)
    # posterior_neighbor: row index advances posterior step, cols are frames; take same row window
    p_window = min(last_n, post.shape[0])
    p_tail = _tail_mean(post[-p_window:, :], tail_k)
    return (1 - alpha) * f_tail + alpha * p_tail


# --- Sweep definitions ----------------------------------------------------

def _build_sweep() -> list[tuple[str, Callable, dict]]:
    out: list[tuple[str, Callable, dict]] = []

    for last_n in (2, 3, 5, 8, 11):
        for tail_k in (0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
            out.append((f"col_min_tail_n{last_n}_k{int(tail_k*100)}",
                        _r_col_min_tail, dict(last_n=last_n, tail_k=tail_k)))

    for last_n in (2, 5, 8, 11):
        for q in (0.0, 0.1, 0.25, 0.5):
            for tail_k in (0.2, 0.5, 1.0):
                out.append((f"col_q{int(q*100)}_tail_n{last_n}_k{int(tail_k*100)}",
                            _r_col_quantile_tail, dict(last_n=last_n, q=q, tail_k=tail_k)))

    for last_n in (2, 5, 8, 11):
        for m in (1, 2, 4, 8, 16):
            out.append((f"worst_m{m}_cols_n{last_n}",
                        _r_worst_m_cols, dict(last_n=last_n, m=m)))

    for last_n in (2, 5, 11):
        for tail_k in (0.2, 0.5, 0.8, 1.0):
            out.append((f"harmonic_tail_n{last_n}_k{int(tail_k*100)}",
                        _r_harmonic_tail, dict(last_n=last_n, tail_k=tail_k)))

    for last_n in (2, 5, 11):
        for temperature in (0.005, 0.01, 0.02, 0.05, 0.1):
            t_tag = str(temperature).replace(".", "p")
            out.append((f"softmin_n{last_n}_T{t_tag}",
                        _r_softmin, dict(last_n=last_n, temperature=temperature)))

    for late in (2, 3, 5):
        for early in (1, 2, 3):
            out.append((f"late{late}_minus_early{early}",
                        _r_late_minus_early, dict(late=late, early=early)))

    for last_n in (5, 11):
        for tail_k in (0.3, 0.5, 0.8):
            for alpha in (0.25, 0.5, 0.75):
                out.append((f"latecol_n{last_n}_k{int(tail_k*100)}_a{int(alpha*100)}",
                            _r_late_plus_col_min,
                            dict(last_n=last_n, tail_k=tail_k, alpha=alpha)))

    for last_n in (5, 11):
        for tail_k in (0.3, 0.5, 0.8):
            for alpha in (0.25, 0.5, 0.75):
                out.append((f"framepost_n{last_n}_k{int(tail_k*100)}_a{int(alpha*100)}",
                            _r_frame_plus_post_tail,
                            dict(last_n=last_n, tail_k=tail_k, alpha=alpha)))

    return out


# --- Plumbing -------------------------------------------------------------

def _load_seed_records(heatmap_run_root: Path,
                       vbench_rows: dict[tuple[str, int], dict[str, object]]) -> list[dict[str, object]]:
    seed_records = []
    for seed_dir in _iter_seed_dirs(heatmap_run_root):
        f_path = seed_dir / FRAME_FILE
        p_path = seed_dir / POST_FILE
        if not (f_path.exists() and p_path.exists()):
            continue
        prompt_id = seed_dir.parent.name
        seed_idx = _seed_idx_from_name(seed_dir)
        frame = np.load(f_path)
        post = np.load(p_path)
        if (prompt_id, seed_idx) not in vbench_rows:
            continue
        record = dict(vbench_rows[(prompt_id, seed_idx)])
        record["prompt_id"] = prompt_id
        record["seed_idx"] = seed_idx
        record["frame"] = frame
        record["post"] = post
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
    return seed_records


def _alignment_stats(by_prompt_subset: dict[str, list[dict[str, object]]],
                     metric: str) -> dict[str, object]:
    prompt_spearmans: list[float] = []
    winner_match = 0
    top3_hit = 0
    top5_contains_best = 0
    top5_overlap_total = 0.0
    for _, group in by_prompt_subset.items():
        sv = np.array([-row["score_value"] for row in group], dtype=float)
        mv = np.array([-row[metric] for row in group], dtype=float)
        if np.std(sv) == 0 or np.std(mv) == 0:
            prompt_spearmans.append(float("nan"))
        else:
            prompt_spearmans.append(float(np.corrcoef(_rankdata(sv), _rankdata(mv))[0, 1]))
        score_sorted = sorted(group, key=lambda r: (-r["score_value"], int(r["seed_idx"])))
        metric_sorted = sorted(group, key=lambda r: (-r[metric], int(r["seed_idx"])))
        score_winner = score_sorted[0]
        score_top5 = score_sorted[:5]
        metric_best = metric_sorted[0]
        metric_top5 = metric_sorted[:5]
        metric_rank = next(idx + 1 for idx, row in enumerate(metric_sorted)
                           if int(row["seed_idx"]) == int(score_winner["seed_idx"]))
        if metric_rank == 1:
            winner_match += 1
        if metric_rank <= 3:
            top3_hit += 1
        if any(int(r["seed_idx"]) == int(metric_best["seed_idx"]) for r in score_top5):
            top5_contains_best += 1
        s5 = {int(r["seed_idx"]) for r in score_top5}
        m5 = {int(r["seed_idx"]) for r in metric_top5}
        top5_overlap_total += len(s5 & m5) / 5.0
    n = len(by_prompt_subset)
    return {
        "prompt_mean_spearman": float(np.nanmean(prompt_spearmans)) if prompt_spearmans else float("nan"),
        "winner_match": winner_match,
        "winner_match_rate": winner_match / n if n else float("nan"),
        "top3_hit": top3_hit,
        "top5_contains_best": top5_contains_best,
        "top5_overlap_rate": top5_overlap_total / n if n else float("nan"),
        "n_prompts": n,
    }


def _evaluate(seed_records: list[dict[str, object]],
              reduction_name: str,
              fn: Callable,
              kwargs: dict) -> dict[str, object]:
    by_prompt = defaultdict(list)
    cand = []
    for row in seed_records:
        score = fn(row["frame"], row["post"], **kwargs)
        c = {
            "prompt_id": row["prompt_id"],
            "seed_idx": row["seed_idx"],
            "score_value": float(score),
        }
        for metric in SUBSCORES + ["avg_vbench_z", "avg_vbench_raw"]:
            c[metric] = float(row[metric])
        cand.append(c)
        by_prompt[str(row["prompt_id"])].append(c)

    summary: dict[str, object] = {"score_name": reduction_name}
    score_values = np.array([c["score_value"] for c in cand], dtype=float)
    sizes = sorted({len(g) for g in by_prompt.values()})

    for metric in SUBSCORES + ["avg_vbench_z", "avg_vbench_raw"]:
        metric_values = np.array([c[metric] for c in cand], dtype=float)
        if np.std(score_values) == 0 or np.std(metric_values) == 0:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(np.corrcoef(score_values, metric_values)[0, 1])
            spearman = float(np.corrcoef(_rankdata(score_values), _rankdata(metric_values))[0, 1])
        prompt_spearmans = []
        winner_match = 0
        top3_hit = 0
        top5_contains_best = 0
        top5_overlap_total = 0.0
        for _, group in by_prompt.items():
            sv = np.array([-row["score_value"] for row in group], dtype=float)
            mv = np.array([-row[metric] for row in group], dtype=float)
            if np.std(sv) == 0 or np.std(mv) == 0:
                prompt_spearmans.append(float("nan"))
            else:
                prompt_spearmans.append(float(np.corrcoef(_rankdata(sv), _rankdata(mv))[0, 1]))
            score_sorted = sorted(group, key=lambda r: (-r["score_value"], int(r["seed_idx"])))
            metric_sorted = sorted(group, key=lambda r: (-r[metric], int(r["seed_idx"])))
            score_winner = score_sorted[0]
            score_top5 = score_sorted[:5]
            metric_best = metric_sorted[0]
            metric_top5 = metric_sorted[:5]
            metric_rank = next(idx + 1 for idx, row in enumerate(metric_sorted)
                               if int(row["seed_idx"]) == int(score_winner["seed_idx"]))
            if metric_rank == 1:
                winner_match += 1
            if metric_rank <= 3:
                top3_hit += 1
            if any(int(r["seed_idx"]) == int(metric_best["seed_idx"]) for r in score_top5):
                top5_contains_best += 1
            s5 = {int(r["seed_idx"]) for r in score_top5}
            m5 = {int(r["seed_idx"]) for r in metric_top5}
            top5_overlap_total += len(s5 & m5) / 5.0
        n_prompts = len(by_prompt)
        summary["n_prompts"] = n_prompts
        summary[f"{metric}_pearson"] = pearson
        summary[f"{metric}_spearman"] = spearman
        summary[f"{metric}_prompt_mean_spearman"] = float(np.nanmean(prompt_spearmans))
        summary[f"{metric}_winner_match"] = winner_match
        summary[f"{metric}_winner_match_rate"] = winner_match / n_prompts
        summary[f"{metric}_top3_hit"] = top3_hit
        summary[f"{metric}_top5_contains_best"] = top5_contains_best
        summary[f"{metric}_top5_overlap_rate"] = top5_overlap_total / n_prompts
        for sz in sizes:
            sub = {pid: g for pid, g in by_prompt.items() if len(g) == sz}
            b = _alignment_stats(sub, metric)
            summary[f"n_prompts_n{sz}"] = b["n_prompts"]
            summary[f"{metric}_prompt_mean_spearman_n{sz}"] = b["prompt_mean_spearman"]
            summary[f"{metric}_winner_match_n{sz}"] = b["winner_match"]
            summary[f"{metric}_winner_match_rate_n{sz}"] = b["winner_match_rate"]
            summary[f"{metric}_top3_hit_n{sz}"] = b["top3_hit"]
            summary[f"{metric}_top5_contains_best_n{sz}"] = b["top5_contains_best"]
            summary[f"{metric}_top5_overlap_rate_n{sz}"] = b["top5_overlap_rate"]
    return summary


def _print_progress_if_better(summary: dict[str, object],
                              best: dict[str, dict[str, object] | float]) -> None:
    sp = summary["avg_vbench_z_prompt_mean_spearman"]
    wm = summary["avg_vbench_z_winner_match"]
    overlap = summary["avg_vbench_z_top5_overlap_rate"]
    np_ = summary.get("n_prompts", "?")
    if sp > best["spearman_value"]:
        best["spearman_value"] = sp
        best["spearman_row"] = summary
        print(f"[NEW BEST Spearman] {summary['score_name']:<35} "
              f"sp={sp:.4f}  winner={wm}/{np_}  top5_overlap={overlap:.4f}", flush=True)
    if wm > best["winner_value"] or (wm == best["winner_value"] and sp > best["winner_tiebreak"]):
        best["winner_value"] = wm
        best["winner_tiebreak"] = sp
        best["winner_row"] = summary
        print(f"[NEW BEST Winner ] {summary['score_name']:<35} "
              f"winner={wm}/{np_}  sp={sp:.4f}  top5_overlap={overlap:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-run-root", type=Path, required=True)
    parser.add_argument("--vbench-long-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.heatmap_run_root / "_advanced_reductions")
    output_dir.mkdir(parents=True, exist_ok=True)

    vbench_rows = _load_vbench_rows(args.vbench_long_csv)
    seed_records = _load_seed_records(args.heatmap_run_root, vbench_rows)
    print(f"Loaded {len(seed_records)} seeds, "
          f"{len({r['prompt_id'] for r in seed_records})} prompts", flush=True)

    sweep = _build_sweep()
    print(f"Sweep size: {len(sweep)} reductions", flush=True)

    best = {
        "spearman_value": float("-inf"),
        "spearman_row": None,
        "winner_value": -1,
        "winner_tiebreak": float("-inf"),
        "winner_row": None,
    }
    rows = []
    for idx, (name, fn, kwargs) in enumerate(sweep, 1):
        summary = _evaluate(seed_records, name, fn, kwargs)
        rows.append(summary)
        _print_progress_if_better(summary, best)
        if idx % 20 == 0:
            print(f"  ...{idx}/{len(sweep)} evaluated", flush=True)

    fieldnames = list(rows[0].keys())
    with (output_dir / "advanced_reductions_table.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sp_row = best["spearman_row"]
    wm_row = best["winner_row"]
    n_prompts = rows[0].get("n_prompts", "?") if rows else "?"
    md_lines = [
        "# Advanced Reductions Sweep",
        "",
        f"Total reductions evaluated: {len(rows)}",
        f"Number of prompts: {n_prompts}",
        "",
        "## Best by avg_vbench_z prompt-mean Spearman",
        f"- name: `{sp_row['score_name']}`",
        f"- `avg_vbench_z_prompt_mean_spearman = {sp_row['avg_vbench_z_prompt_mean_spearman']:.4f}`",
        f"- `avg_vbench_z_winner_match = {sp_row['avg_vbench_z_winner_match']}/{n_prompts}`",
        f"- `avg_vbench_z_top5_overlap_rate = {sp_row['avg_vbench_z_top5_overlap_rate']:.4f}`",
        f"- `subject_consistency_prompt_mean_spearman = {sp_row['subject_consistency_prompt_mean_spearman']:.4f}`",
        "",
        "## Best by avg_vbench_z winner_match",
        f"- name: `{wm_row['score_name']}`",
        f"- `avg_vbench_z_winner_match = {wm_row['avg_vbench_z_winner_match']}/{n_prompts}`",
        f"- `avg_vbench_z_prompt_mean_spearman = {wm_row['avg_vbench_z_prompt_mean_spearman']:.4f}`",
        f"- `avg_vbench_z_top5_overlap_rate = {wm_row['avg_vbench_z_top5_overlap_rate']:.4f}`",
        "",
        "## Top 15 by avg_vbench_z prompt-mean Spearman",
        "",
        f"| name | spearman | winner /{n_prompts} | top5_overlap | subj_sp |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    top15 = sorted(rows, key=lambda r: r["avg_vbench_z_prompt_mean_spearman"], reverse=True)[:15]
    for r in top15:
        md_lines.append(
            f"| `{r['score_name']}` | {r['avg_vbench_z_prompt_mean_spearman']:.4f} | "
            f"{r['avg_vbench_z_winner_match']} | {r['avg_vbench_z_top5_overlap_rate']:.4f} | "
            f"{r['subject_consistency_prompt_mean_spearman']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Top 15 by avg_vbench_z winner_match")
    md_lines.append("")
    md_lines.append(f"| name | winner /{n_prompts} | spearman | top5_overlap | subj_sp |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: |")
    top15w = sorted(
        rows,
        key=lambda r: (r["avg_vbench_z_winner_match"], r["avg_vbench_z_prompt_mean_spearman"]),
        reverse=True,
    )[:15]
    for r in top15w:
        md_lines.append(
            f"| `{r['score_name']}` | {r['avg_vbench_z_winner_match']} | "
            f"{r['avg_vbench_z_prompt_mean_spearman']:.4f} | "
            f"{r['avg_vbench_z_top5_overlap_rate']:.4f} | "
            f"{r['subject_consistency_prompt_mean_spearman']:.4f} |"
        )
    (output_dir / "advanced_reductions_summary.md").write_text("\n".join(md_lines))
    print(f"Wrote {output_dir / 'advanced_reductions_table.csv'}")
    print(f"Wrote {output_dir / 'advanced_reductions_summary.md'}")


if __name__ == "__main__":
    main()
