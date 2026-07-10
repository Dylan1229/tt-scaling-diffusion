"""Generate figures + summary numbers for the DINOv2 feature-property report.

One showcase (prompt, dynamic_degree) cell per stratum — the cell whose best-vs-worst
`vbench_quality` gap is widest — so the moving showcase never contrasts a frozen clip with a
moving one.

Outputs (under --output-dir):
  fig_frame_heatmap_<prompt>_dyn<0|1>.png
  fig_velcos_curve_<prompt>_dyn<0|1>.png
  fig_scalar_scatter.png     (all clips, coloured by dynamic_degree, showcase seeds circled)
  metrics_summary.json
"""
from __future__ import annotations

import argparse
import json
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
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
    annotate_vbench_targets,
    best_worst,
    cell_groups,
    prompt_strata,
)


def _patch_velcos_curve(F):
    F = F.astype(np.float32)
    pm = F.mean(axis=2)
    pm /= np.linalg.norm(pm, axis=-1, keepdims=True) + 1e-8
    v = pm[:, 1:] - pm[:, :-1]
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    v_final = v[-1]
    vc = (v[:-1] * v_final[None]).sum(-1).mean(axis=-1)   # (T-1,)
    return vc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap-run-root", type=Path, default=None)
    ap.add_argument("--patch-run-root", type=Path, default=None)
    ap.add_argument("--vbench-long-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--run-id", type=str, default=None,
                    help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = ap.parse_args()
    resolve_run_id(args, ap, needs=["heatmap_run_root", "patch_run_root", "vbench_long_csv"])
    args.output_dir = args.output_dir or stage_output_dir(args.heatmap_run_root, "report", __file__)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vb = _load_vbench_rows(args.vbench_long_csv)
    records = []
    for sd in _iter_seed_dirs(args.heatmap_run_root):
        prompt_id = sd.parent.name
        si = _seed_idx_from_name(sd)
        if (prompt_id, si) not in vb:
            continue
        rec = dict(vb[(prompt_id, si)])
        rec["prompt_id"] = prompt_id; rec["seed_idx"] = si
        rec["seed_dir"] = str(sd)
        rec["patch_dir"] = str(args.patch_run_root / prompt_id / sd.name)
        records.append(rec)
    annotate_vbench_targets(records)

    # Compute headline single scalars on every seed
    for r in records:
        sd = Path(r["seed_dir"])
        frame = np.load(sd / "posterior_mean_frame_neighbor_similarity.npy")
        # f_tail80_n11: mean of bottom 80% of cross-frame neighbor cos values
        flat = np.sort(frame.reshape(-1))
        n = max(1, int(np.ceil(flat.size * 0.80)))
        r["f_tail80"] = float(flat[:n].mean())
        F_cls = np.load(sd / "posterior_mean_features.npy").astype(np.float32)
        # vf_postcos_mean: cosine of consecutive-frame velocities across posterior steps
        vf = F_cls[:, 1:] - F_cls[:, :-1]
        vf /= np.linalg.norm(vf, axis=-1, keepdims=True) + 1e-8
        vfp = (vf[:-1] * vf[1:]).sum(-1)
        r["vf_postcos_mean"] = float(vfp.mean())
        # finalpost_velcos_pm_mean: needs patch features
        patch_path = Path(r["patch_dir"]) / "posterior_mean_patch_features.npy"
        F_p = np.load(patch_path)
        r["velcos_curve"] = _patch_velcos_curve(F_p)
        r["finalpost_velcos_pm_mean"] = float(r["velcos_curve"].mean())
        del F_cls, F_p

    # One showcase cell per dynamic_degree stratum, each the (prompt, stratum) whose best/worst
    # vbench_quality gap is widest. Keeping the two strata separate means the moving showcase
    # never contrasts a frozen clip with a moving one.
    by_p = defaultdict(list)
    for r in records:
        by_p[r["prompt_id"]].append(r)

    cells: dict[int, tuple[str, dict, dict, list[dict]]] = {}
    for prompt_id, group in by_p.items():
        for dyn, stratum in prompt_strata(group):
            g, b = best_worst(stratum)
            gap = g["vbench_quality"] - b["vbench_quality"]
            if dyn not in cells or gap > (cells[dyn][1]["vbench_quality"]
                                          - cells[dyn][2]["vbench_quality"]):
                cells[dyn] = (prompt_id, g, b, stratum)

    showcases = []
    for dyn in sorted(cells, reverse=True):
        pick, good, bad, stratum = cells[dyn]
        tag = f"dyn={dyn} n={len(stratum)}"
        showcases.append((dyn, pick, good, bad, tag))
        print(f"[fig] showcase dyn={dyn}: prompt={pick} [{tag}]  "
              f"good=seed{int(good['seed_idx']):04d} q={good['vbench_quality']:.3f}  "
              f"bad=seed{int(bad['seed_idx']):04d} q={bad['vbench_quality']:.3f}", file=sys.stderr)

    for dyn, pick, good, bad, stratum_tag in showcases:
        gi, bi = int(good["seed_idx"]), int(bad["seed_idx"])

        # ---------- Figure 1: frame-neighbor heatmaps good vs bad ----------
        fr_g = np.load(Path(good["seed_dir"]) / "posterior_mean_frame_neighbor_similarity.npy")
        fr_b = np.load(Path(bad["seed_dir"]) / "posterior_mean_frame_neighbor_similarity.npy")
        vmin = min(fr_g.min(), fr_b.min()); vmax = max(fr_g.max(), fr_b.max())
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        for ax, M, ttl in [(axes[0], fr_g, f"good {pick} seed{gi:04d} (q={good['vbench_quality']:.3f})"),
                           (axes[1], fr_b, f"bad {pick} seed{bi:04d} (q={bad['vbench_quality']:.3f})")]:
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_xlabel("frame pair k")
            ax.set_ylabel("posterior step i")
            ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="cos(F[i,k], F[i,k+1])")
        fig.suptitle(f"Cross-frame DINOv2 cosine — prompt {pick}  [{stratum_tag}]", fontsize=11)
        fig.savefig(args.output_dir / f"fig_frame_heatmap_{pick}_dyn{dyn}.png", dpi=140)
        plt.close(fig)

        # ---------- Figure 2: velcos convergence curve good vs bad ----------
        fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        xs = np.arange(len(good["velcos_curve"]))
        ax.plot(xs, good["velcos_curve"], "o-", color="C2",
                label=f"good seed{gi:04d} (q={good['vbench_quality']:.3f})")
        ax.plot(xs, bad["velcos_curve"], "s-", color="C3",
                label=f"bad seed{bi:04d} (q={bad['vbench_quality']:.3f})")
        ax.axhline(1.0, ls=":", color="grey", lw=0.8)
        ax.set_xlabel("posterior step i")
        ax.set_ylabel(r"$\overline{\cos(v_i,\,v_{T-1})}$")
        ax.set_title(f"Inter-frame velocity alignment to final\n"
                     f"prompt {pick}  [{stratum_tag}]", fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig(args.output_dir / f"fig_velcos_curve_{pick}_dyn{dyn}.png", dpi=140)
        plt.close(fig)

    # ---------- Figure 3: scatter of 3 headline scalars vs vbench_quality ----------
    # within-prompt residualized scalars to avoid prompt-bias dominating the eye
    def _wpr(key):
        out = np.array([r[key] for r in records], float)
        for grp in by_p.values():
            idx = [records.index(g) for g in grp]
            out[idx] -= out[idx].mean()
        return out

    prompt_size = {p: len(g) for p, g in by_p.items()}
    sizes = sorted(set(prompt_size.values()))
    bucket_idx = {sz: np.array([i for i, r in enumerate(records)
                                if prompt_size[r["prompt_id"]] == sz])
                  for sz in sizes}

    # Prompt-mean Spearman: per-group rank correlation, averaged across groups. Matches
    # similarity_reduction_gridsearch.py and feature_ridge_regression.py.
    prompt_ids = [r["prompt_id"] for r in records]

    def _rho_w_groups(x, y_, groups):
        rhos = []
        for g_idx in groups.values():
            if len(g_idx) < 2:
                continue
            xb = x[g_idx]; yb = y_[g_idx]
            ra = _rankdata(xb); rb = _rankdata(yb)
            ra -= ra.mean(); rb -= rb.mean()
            denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
            if denom <= 0:
                continue
            rhos.append(float((ra * rb).sum() / denom))
        return float(np.mean(rhos)) if rhos else float("nan")

    def _rho_w_subset(x, y_, idx):
        groups = defaultdict(list)
        for i in idx:
            groups[prompt_ids[int(i)]].append(int(i))
        return _rho_w_groups(x, y_, groups)

    idx_of = {id(r): i for i, r in enumerate(records)}
    prompt_to_idx = {p: [idx_of[id(r)] for r in g] for p, g in by_p.items()}
    # Within a (prompt, dynamic_degree) cell, motion is held fixed, so the correlation there is
    # quality signal rather than the stillness confound.
    dyn_groups = {d: cell_groups(records, d) for d in (1, 0)}
    has_dyn = "dynamic_degree" in records[0]
    dyn_arr = (np.array([int(float(r["dynamic_degree"])) for r in records])
               if has_dyn else np.zeros(len(records), int))

    y = _wpr("vbench_quality")
    metrics = [
        ("f_tail80", r"frame consistency  $f_\mathrm{tail80}$"),
        ("finalpost_velcos_pm_mean", r"velocity-to-final cos  $\bar c_\mathrm{vel\to T}$"),
    ]
    all_idx = np.arange(len(records))
    scatter_rhos: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (k, ttl) in zip(axes, metrics):
        x = _wpr(k)
        sp = _rho_w_subset(x, y, all_idx)
        bucket_sps = {sz: _rho_w_subset(x, y, bucket_idx[sz]) for sz in sizes}
        sp_dyn = {d: _rho_w_groups(x, y, dyn_groups[d]) for d in (1, 0)}
        scatter_rhos[k] = {"pooled": sp,
                           **{f"n{sz}": v for sz, v in bucket_sps.items()},
                           **{f"dyn{d}": v for d, v in sp_dyn.items()}}

        for d, color, lab in [(1, "C0", "moving"), (0, "C1", "static")]:
            m = dyn_arr == d
            if m.any():
                ax.scatter(x[m], y[m], s=14, alpha=0.55, edgecolor="none", color=color,
                           label=f"dyn={d} {lab} (n={int(m.sum())})")
        for _dyn, pick_s, good_s, bad_s, _tag in showcases:
            for rec, role in ((good_s, "good"), (bad_s, "bad")):
                i = idx_of[id(rec)]
                ax.scatter([x[i]], [y[i]], s=52, facecolors="none", edgecolors="k", lw=0.9, zorder=3)
                ax.annotate(f"{pick_s} seed{int(rec['seed_idx']):04d} {role}", (x[i], y[i]),
                            textcoords="offset points", xytext=(4, 3), fontsize=5)
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        dyn_str = (f"; {sp_dyn[1]:+.3f} dyn=1; {sp_dyn[0]:+.3f} dyn=0" if has_dyn else "")
        ax.set_title(f"{ttl}\n($\\rho_w$ = {sp:+.3f} all{dyn_str})", fontsize=7)
        ax.set_xlabel("within-prompt centered metric")
    axes[0].set_ylabel("within-prompt centered vbench_quality")
    axes[0].legend(fontsize=5, loc="best", framealpha=0.7)
    fig.suptitle(f"{len(records)} clips over {len(by_p)} prompts; circled = the good/bad seeds "
                 f"of the two showcase cells", fontsize=8)
    fig.savefig(args.output_dir / "fig_scalar_scatter.png", dpi=140)
    plt.close(fig)

    # summary stats
    out_json = {
        "showcases": [
            {
                "dynamic_degree": d,
                "prompt_used": cells[d][0],
                "stratum_n_seeds": len(cells[d][3]),
                "good_seed": int(cells[d][1]["seed_idx"]),
                "good_quality": cells[d][1]["vbench_quality"],
                "bad_seed": int(cells[d][2]["seed_idx"]),
                "bad_quality": cells[d][2]["vbench_quality"],
            }
            for d in sorted(cells, reverse=True)
        ],
        "n_seeds": len(records),
        "n_prompts": len(by_p),
        "n_cells": {f"dyn{d}": len(dyn_groups[d]) for d in (1, 0)},
        "rho_w": scatter_rhos,
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(out_json, indent=2))
    print(f"[fig] wrote figures to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
