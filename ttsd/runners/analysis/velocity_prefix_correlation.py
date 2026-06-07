"""Online prefix-to-current velocity-locking analysis (PLAN.md Experiment 2).

For each seed, compute the patch-mean DINOv2 inter-frame velocity per posterior
step, then form a 2-D grid of per-seed scalars

    L_bar_s^(s_0) = (1 / ((s - s_0) * (N - 1)))
                    * sum_{r=s_0}^{s-1} sum_k <v_{r,k}, v_{s,k}>

for every (s_0, s) with 0 <= s_0 < s <= T - 1. T = 11 posterior steps, so the
grid has C(11, 2) = 55 entries, named `L_s0_{s0:02d}_s_{s:02d}`.

At (s_0 = 0, s = T - 1) the statistic equals the existing
`finalpost_velcos_pm_mean` (see `feature_vbench_correlation._patch_metrics`):
this is the sanity row used to validate the implementation against
`runs/analysis/<run_id>/feature_vbench_correlation/correlation_report.txt`.

The text report mimics `feature_vbench_correlation.py` row-for-row (pooled table
sorted by |sp_within|, then per-bucket sub-tables in pooled order) and is written
to `ranking_report.txt` in the output dir. Figures land in the output dir too.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttsd.runners.utilities.ranking import _rankdata
from ttsd.runners.utilities.run_layout import resolve_run_id, stage_output_dir
from ttsd.runners.utilities.seed_vbench_loaders import (
    SUBSCORES,
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
)

PATCH_FILE = "posterior_mean_patch_features.npy"


def _within(score: np.ndarray, y: np.ndarray, prompt_to_idx: dict[str, list[int]]) -> float:
    rhos = []
    for ps in prompt_to_idx.values():
        if len(ps) < 2:
            continue
        sv = score[ps]; mv = y[ps]
        ra = _rankdata(sv); rb = _rankdata(mv)
        ra -= ra.mean(); rb -= rb.mean()
        denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
        if denom <= 0:
            continue
        rhos.append(float((ra * rb).sum() / denom))
    return float(np.mean(rhos)) if rhos else float("nan")


def _winner_match(score: np.ndarray, y: np.ndarray, prompt_to_idx: dict[str, list[int]]) -> int:
    return sum(int(score[ps].argmax() == y[np.array(ps)].argmax()) for ps in prompt_to_idx.values())


def _prefix_locking_scalars(F: np.ndarray) -> dict[str, float]:
    """F: (T, N, P, D), float16/32, rows L2-normalized per patch token.

    Returns 55 scalars keyed `L_s0_{s0:02d}_s_{s:02d}` for 0 <= s_0 < s <= T-1.
    """
    F = F.astype(np.float32)
    T = F.shape[0]

    # 1) Patch-mean (D-vector per (i, k))
    pm = F.mean(axis=2)
    pm = pm / (np.linalg.norm(pm, axis=-1, keepdims=True) + 1e-8)

    # 2) Inter-frame velocity, L2-normalized
    v = pm[:, 1:, :] - pm[:, :-1, :]                  # (T, N-1, D)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    out: dict[str, float] = {}
    for s in range(1, T):
        cos_to_s = (v[:s] * v[s][None]).sum(-1)       # (s, N-1)
        for s0 in range(s):
            block = cos_to_s[s0:]                     # (s - s0, N-1)
            out[f"L_s0_{s0:02d}_s_{s:02d}"] = float(block.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-run-root", type=Path, default=None)
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args()
    resolve_run_id(args, parser, needs=["patch_run_root", "vbench_long_csv"])
    args.output_dir = args.output_dir or stage_output_dir(args.patch_run_root, "analysis", __file__)

    vbench = _load_vbench_rows(args.vbench_long_csv)

    records: list[dict[str, object]] = []
    for sd in _iter_seed_dirs(args.patch_run_root):
        fp = sd / PATCH_FILE
        if not fp.exists():
            continue
        prompt_id = sd.parent.name
        seed_idx = _seed_idx_from_name(sd)
        if (prompt_id, seed_idx) not in vbench:
            continue
        F = np.load(fp)
        scalars = _prefix_locking_scalars(F)
        rec = dict(vbench[(prompt_id, seed_idx)])
        rec["prompt_id"] = prompt_id
        rec["seed_idx"] = seed_idx
        rec.update(scalars)
        records.append(rec)
        del F
    print(f"[velocity_prefix_correlation] loaded {len(records)} seeds", file=sys.stderr)

    # Per-prompt z-normalize the 6 subscores -> avg_vbench_z
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        grouped[str(r["prompt_id"])].append(r)
    for group in grouped.values():
        for metric in SUBSCORES:
            vs = np.array([float(g[metric]) for g in group])
            mu, sd = vs.mean(), vs.std(ddof=0)
            zs = np.zeros_like(vs) if sd == 0 else (vs - mu) / sd
            for g, z in zip(group, zs):
                g[f"z_{metric}"] = float(z)
        for g in group:
            g["avg_vbench_z"] = float(np.mean([g[f"z_{m}"] for m in SUBSCORES]))

    y = np.array([r["avg_vbench_z"] for r in records])
    prompt_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        prompt_to_idx[str(r["prompt_id"])].append(i)

    n_prompts = len(prompt_to_idx)
    metric_keys = sorted(k for k in records[0].keys() if k.startswith("L_s0_"))
    rows: list[tuple[str, int, float, int]] = []
    for k in metric_keys:
        s = np.array([r[k] for r in records], float)
        for sign in (1, -1):
            ss = sign * s
            rows.append((k, sign, _within(ss, y, prompt_to_idx), _winner_match(ss, y, prompt_to_idx)))
    rows.sort(key=lambda r: -abs(r[2]))

    report: list[str] = []
    wm_w = len(str(n_prompts)) * 2 + 1
    report.append(f"{'metric':<48} {'sign':>4} {'sp_within':>10} {'WM':>{wm_w}}")
    report.append("-" * (66 + wm_w))
    for k, sign, sp, wm in rows:
        report.append(f"{k:<48} {sign:>4} {sp:>10.4f} {wm:>{wm_w - len(str(n_prompts)) - 1}}/{n_prompts}")

    # Bucket alignment stats (sign = +1) keyed by bucket label, then by (s0, s)
    bucket_sp: dict[str, dict[tuple[int, int], float]] = {"pooled": {}}
    for k, sign, sp, _ in rows:
        if sign == 1:
            parts = k.split("_")
            s0, ss = int(parts[2]), int(parts[4])
            bucket_sp["pooled"][(s0, ss)] = sp

    sizes = sorted({len(ps) for ps in prompt_to_idx.values()})
    if len(sizes) > 1:
        pooled_order = [(r[0], r[1]) for r in rows]
        for sz in sizes:
            bucket_recs = [r for r in records if len(prompt_to_idx[str(r["prompt_id"])]) == sz]
            pti_sub: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(bucket_recs):
                pti_sub[str(r["prompt_id"])].append(i)
            n_sub = len(pti_sub)
            y_sub = np.array([r["avg_vbench_z"] for r in bucket_recs])
            bucket_stats: dict[tuple[str, int], tuple[float, int]] = {}
            for k in metric_keys:
                s_sub = np.array([r[k] for r in bucket_recs], float)
                for sign in (1, -1):
                    ss_arr = sign * s_sub
                    bucket_stats[(k, sign)] = (
                        _within(ss_arr, y_sub, pti_sub),
                        _winner_match(ss_arr, y_sub, pti_sub),
                    )
            wm_w_b = len(str(n_sub)) * 2 + 1
            report.append("")
            report.append(f"--- bucket n={sz} ({n_sub} prompts) ---")
            report.append(f"{'metric':<48} {'sign':>4} {'sp_within':>10} {'WM':>{wm_w_b}}")
            report.append("-" * (66 + wm_w_b))
            for k, sign in pooled_order:
                sp, wm = bucket_stats[(k, sign)]
                report.append(f"{k:<48} {sign:>4} {sp:>10.4f} {wm:>{wm_w_b - len(str(n_sub)) - 1}}/{n_sub}")

            label = f"n{sz}"
            bucket_sp[label] = {}
            for k, (sp, _) in bucket_stats.items():
                if k[1] == 1:
                    parts = k[0].split("_")
                    s0, ss = int(parts[2]), int(parts[4])
                    bucket_sp[label][(s0, ss)] = sp

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ranking_report.txt").write_text("\n".join(report) + "\n")
    print("Wrote", args.output_dir / "ranking_report.txt", file=sys.stderr)

    # Figures: one subfolder per bucket, 10 per-s0 plots + 1 overlay each
    T = max(int(k.split("_s_")[1]) for k in metric_keys) + 1
    cmap = plt.get_cmap("viridis")

    for bucket_label, sp_map in bucket_sp.items():
        sub_dir = args.output_dir / bucket_label
        sub_dir.mkdir(parents=True, exist_ok=True)

        overlay_fig, overlay_ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        s0_values = sorted({s0 for (s0, _) in sp_map})
        for s0 in s0_values:
            s_vals = sorted(s for (sx, s) in sp_map if sx == s0)
            sp_vals = [sp_map[(s0, s)] for s in s_vals]

            fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            ax.plot(s_vals, sp_vals, "o-", color=cmap(s0 / max(1, T - 2)))
            ax.set_xlabel("posterior step  s")
            ax.set_ylabel(r"within-prompt Spearman$(\bar L_s,\ \mathrm{avg\_vbench\_z})$")
            ax.set_title(f"prefix-locking ({bucket_label}): $s_0$ = {s0}")
            ax.set_xticks(s_vals)
            ax.grid(alpha=0.25, lw=0.4)
            fig.savefig(sub_dir / f"fig_s0_{s0:02d}.png", dpi=140)
            plt.close(fig)

            overlay_ax.plot(s_vals, sp_vals, "o-",
                            color=cmap(s0 / max(1, T - 2)), label=f"$s_0$={s0}")

        overlay_ax.axhline(0, color="grey", lw=0.5, ls="--")
        overlay_ax.set_xlabel("posterior step  s")
        overlay_ax.set_ylabel(r"within-prompt Spearman$(\bar L_s,\ \mathrm{avg\_vbench\_z})$")
        overlay_ax.set_title(f"prefix-locking ({bucket_label}): all $s_0$ overlaid")
        overlay_ax.set_xticks(list(range(1, T)))
        overlay_ax.grid(alpha=0.25, lw=0.4)
        overlay_ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                          fontsize=8, frameon=False)
        overlay_fig.savefig(sub_dir / "fig_overlay.png", dpi=140)
        plt.close(overlay_fig)


if __name__ == "__main__":
    main()
