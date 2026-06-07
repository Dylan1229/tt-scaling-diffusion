"""Generate figures + summary numbers for the DINOv2 feature-property report.

Outputs (under --output-dir):
  fig_frame_heatmap_good_vs_bad.png
  fig_velcos_curve_good_vs_bad.png
  fig_scalar_scatter.png
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
    SUBSCORES,
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
)


def _avg_z_records(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["prompt_id"]].append(r)
    for group in grouped.values():
        for m in SUBSCORES:
            v = np.array([float(g[m]) for g in group])
            mu, sd = v.mean(), v.std(ddof=0)
            z = np.zeros_like(v) if sd == 0 else (v - mu) / sd
            for g, zz in zip(group, z):
                g[f"z_{m}"] = float(zz)
        for g in group:
            g["avg_vbench_z"] = float(np.mean([g[f"z_{m}"] for m in SUBSCORES]))
    return records


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
    records = _avg_z_records(records)

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

    # Pick prompt with largest spread in avg_vbench_z (visible contrast)
    by_p = defaultdict(list)
    for r in records:
        by_p[r["prompt_id"]].append(r)
    spread = {p: max(g, key=lambda x: x["avg_vbench_z"])["avg_vbench_z"]
                  - min(g, key=lambda x: x["avg_vbench_z"])["avg_vbench_z"]
              for p, g in by_p.items()}
    pick = max(spread, key=spread.get)
    g = by_p[pick]
    good = max(g, key=lambda x: x["avg_vbench_z"])
    bad = min(g, key=lambda x: x["avg_vbench_z"])
    print(f"[fig] prompt={pick}  good=seed{good['seed_idx']:04d} z={good['avg_vbench_z']:.2f}"
          f"  bad=seed{bad['seed_idx']:04d} z={bad['avg_vbench_z']:.2f}", file=sys.stderr)

    # ---------- Figure 1: frame-neighbor heatmaps good vs bad ----------
    fr_g = np.load(Path(good["seed_dir"]) / "posterior_mean_frame_neighbor_similarity.npy")
    fr_b = np.load(Path(bad["seed_dir"]) / "posterior_mean_frame_neighbor_similarity.npy")
    vmin = min(fr_g.min(), fr_b.min()); vmax = max(fr_g.max(), fr_b.max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for ax, M, ttl in [(axes[0], fr_g, f"good seed (z={good['avg_vbench_z']:+.2f})"),
                       (axes[1], fr_b, f"bad seed (z={bad['avg_vbench_z']:+.2f})")]:
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xlabel("frame pair k")
        ax.set_ylabel("posterior step i")
        ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="cos(F[i,k], F[i,k+1])")
    fig.suptitle(f"Cross-frame DINOv2 cosine — prompt {pick}", fontsize=11)
    fig.savefig(args.output_dir / "fig_frame_heatmap_good_vs_bad.png", dpi=140)
    plt.close(fig)

    # ---------- Figure 2: velcos convergence curve good vs bad ----------
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    xs = np.arange(len(good["velcos_curve"]))
    ax.plot(xs, good["velcos_curve"], "o-", color="C2",
            label=f"good (z={good['avg_vbench_z']:+.2f})")
    ax.plot(xs, bad["velcos_curve"], "s-", color="C3",
            label=f"bad (z={bad['avg_vbench_z']:+.2f})")
    ax.axhline(1.0, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("posterior step i")
    ax.set_ylabel(r"$\overline{\cos(v_i,\,v_{T-1})}$")
    ax.set_title(f"Inter-frame velocity alignment to final\n"
                 f"prompt {pick}", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "fig_velcos_curve_good_vs_bad.png", dpi=140)
    plt.close(fig)

    # ---------- Figure 3: scatter of 3 headline scalars vs avg_vbench_z ----------
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

    # Prompt-mean Spearman: per-prompt rank correlation, averaged across prompts in
    # the subset. Matches similarity_reduction_gridsearch.py and feature_ridge_regression.py.
    prompt_ids = [r["prompt_id"] for r in records]

    def _rho_w_subset(x, y_, idx):
        by_p = defaultdict(list)
        for i in idx:
            by_p[prompt_ids[int(i)]].append(int(i))
        rhos = []
        for p_idx in by_p.values():
            if len(p_idx) < 2:
                continue
            xb = x[p_idx]; yb = y_[p_idx]
            ra = _rankdata(xb); rb = _rankdata(yb)
            ra -= ra.mean(); rb -= rb.mean()
            denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
            if denom <= 0:
                continue
            rhos.append(float((ra * rb).sum() / denom))
        return float(np.mean(rhos)) if rhos else float("nan")

    y = _wpr("avg_vbench_z")
    metrics = [
        ("f_tail80", r"frame consistency  $f_\mathrm{tail80}$"),
        ("finalpost_velcos_pm_mean", r"velocity-to-final cos  $\bar c_\mathrm{vel\to T}$"),
    ]
    all_idx = np.arange(len(records))
    scatter_rhos: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for ax, (k, ttl) in zip(axes, metrics):
        x = _wpr(k)
        sp = _rho_w_subset(x, y, all_idx)
        bucket_sps = {sz: _rho_w_subset(x, y, bucket_idx[sz]) for sz in sizes}
        scatter_rhos[k] = {"pooled": sp, **{f"n{sz}": v for sz, v in bucket_sps.items()}}
        ax.scatter(x, y, s=14, alpha=0.65, edgecolor="none")
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        sgn = "+" if sp > 0 else r"$-$"
        if len(sizes) > 1:
            bucket_str = "; " + " / ".join(f"{bucket_sps[sz]:+.3f} (n={sz})" for sz in sizes)
            ax.set_title(f"{ttl}\n($\\rho_w$ = {sp:+.3f} all{bucket_str}, {sgn} correlated)",
                         fontsize=7)
        else:
            ax.set_title(f"{ttl}\n($\\rho_w$ = {sp:+.3f}, {sgn} correlated)", fontsize=9)
        ax.set_xlabel("within-prompt centered metric")
    axes[0].set_ylabel("within-prompt avg VBench z")
    fig.savefig(args.output_dir / "fig_scalar_scatter.png", dpi=140)
    plt.close(fig)

    # summary stats
    out_json = {
        "prompt_used": pick,
        "good_seed": good["seed_idx"], "good_z": good["avg_vbench_z"],
        "bad_seed": bad["seed_idx"], "bad_z": bad["avg_vbench_z"],
        "n_seeds": len(records),
        "rho_w": scatter_rhos,
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(out_json, indent=2))
    print(f"[fig] wrote figures to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
