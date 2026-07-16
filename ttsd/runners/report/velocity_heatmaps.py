"""Good-vs-bad velocity-direction heatmaps in patch-mean space, per motion stratum.

For each (prompt, dynamic_degree) stratum, picks the best and worst seed by `vbench_quality`
and produces a 2x2 figure. A prompt whose seeds all move yields one figure; a prompt whose
seeds differ yields two. Best and worst are never drawn from different strata, because
`vbench_quality` rewards stillness and the contrast would then be motion, not quality.

    +------------------------------+------------------------------+
    | step-adj cos: good seed      | step-adj cos: bad seed       |
    |   D[s, k] = <v_{s-1,k}, v_{s,k}>
    +------------------------------+------------------------------+
    | prefix-lock: good seed       | prefix-lock: bad seed        |
    |   L[s, k] = (1/s) * sum_{r<s} <v_{r,k}, v_{s,k}>
    +------------------------------+------------------------------+

where v_{s,k} is the L2-normalized patch-mean inter-frame velocity
(patch features -> mean over P patches -> L2-normalize -> frame diff ->
L2-normalize). Color scale is shared per-row (good/bad share vmin/vmax),
matching the style of `runs/report/<run_id>/feature_property_figs/fig_frame_heatmap_good_vs_bad.png`.

It also writes a raw per-cell dump of the four matrices (good-D, good-L, bad-D,
bad-L) for every prompt to `velocity_heatmap_cells.txt` in the output dir.
Per-prompt log lines go to stderr.

Usage:
    python -m ttsd.runners.report.velocity_heatmaps --run-id <run_id>
    # → figures + velocity_heatmap_cells.txt in runs/report/<run_id>/velocity_heatmaps/
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

from ttsd.runners.utilities.run_layout import resolve_run_id, stage_output_dir
from ttsd.runners.utilities.seed_vbench_loaders import (
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
    annotate_vbench_targets,
    best_worst,
    prompt_strata,
)

PATCH_FILE = "posterior_mean_patch_features.npy"


def _patch_mean_velocity(F: np.ndarray) -> np.ndarray:
    """F: (T, N, P, D) patch tokens, L2-normalized per token.

    Returns v: (T, N-1, D), L2-normalized inter-frame patch-mean velocity.
    """
    F = F.astype(np.float32)
    pm = F.mean(axis=2)
    pm = pm / (np.linalg.norm(pm, axis=-1, keepdims=True) + 1e-8)
    v = pm[:, 1:, :] - pm[:, :-1, :]
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    return v


def _heatmaps_from_v(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v: (T, N-1, D), L2-normalized velocity.

    Returns (D, L), each shape (T-1, N-1):
      D[s-1, k] = <v[s-1, k], v[s, k]>             for s = 1..T-1
      L[s-1, k] = mean_{r < s} <v[r, k], v[s, k]>  for s = 1..T-1
    """
    T = v.shape[0]
    D = np.empty((T - 1, v.shape[1]), dtype=np.float32)
    L = np.empty((T - 1, v.shape[1]), dtype=np.float32)
    for s in range(1, T):
        cos_to_s = (v[:s] * v[s][None]).sum(-1)   # (s, N-1)
        D[s - 1] = cos_to_s[s - 1]
        L[s - 1] = cos_to_s.mean(axis=0)
    return D, L


def _imshow(ax, M, vmin, vmax, title):
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("frame pair k")
    ax.set_ylabel("posterior step s")
    ax.set_title(title, fontsize=10)
    n_steps, n_pairs = M.shape
    ax.set_yticks(np.arange(n_steps))
    ax.set_yticklabels([str(s) for s in range(1, n_steps + 1)])
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels([str(k) for k in range(n_pairs)], fontsize=7)
    return im


def _format_matrix(M: np.ndarray) -> str:
    """Render a (steps x frame-pairs) matrix as a fixed-width text block.

    Header row = frame-pair index k=0..N-1; row label = step s=1..steps.
    """
    n_pairs = M.shape[1]
    label = "step\\k"
    lines = [f"{label:>6}" + "".join(f"{k:>9d}" for k in range(n_pairs))]
    for s in range(M.shape[0]):
        lines.append(f"{s + 1:>6d}" + "".join(f"{M[s, k]:>+9.4f}" for k in range(n_pairs)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-run-root", type=Path, default=None)
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args()
    resolve_run_id(args, parser, needs=["patch_run_root", "vbench_long_csv"])
    args.output_dir = args.output_dir or stage_output_dir(args.patch_run_root, "report", __file__)

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
        v = _patch_mean_velocity(F)
        rec = dict(vbench[(prompt_id, seed_idx)])
        rec["prompt_id"] = prompt_id
        rec["seed_idx"] = seed_idx
        rec["v"] = v
        records.append(rec)
        del F

    annotate_vbench_targets(records)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        grouped[str(r["prompt_id"])].append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cell_report: list[str] = []
    cell_report.append("[velocity_heatmap_cells] one block per (prompt, dynamic_degree) stratum | "
                       "good+bad seed drawn from within that stratum | D and L | 10 steps x 20 frame pairs")
    cell_report.append("[velocity_heatmap_cells] D[s,k] = <v_{s-1,k}, v_{s,k}>")
    cell_report.append("[velocity_heatmap_cells] L[s,k] = (1/s) sum_{r=0}^{s-1} <v_{r,k}, v_{s,k}>")
    cell_report.append("")

    n_figs = 0
    for prompt_id in sorted(grouped.keys()):
        strata = prompt_strata(grouped[prompt_id])
        if not strata:
            print(f"[velocity_heatmaps] SKIP {prompt_id}: no stratum with 2 seeds", file=sys.stderr)
            continue

        # One figure per (prompt, dynamic_degree) stratum: a prompt whose seeds all move gets
        # one, a prompt whose seeds differ gets two. Best and worst are never drawn from
        # different strata, so the contrast is quality rather than motion.
        for dyn, stratum in strata:
            good, bad = best_worst(stratum)
            stratum_tag = f"dyn={dyn} n={len(stratum)}"
            gi, bi = int(good["seed_idx"]), int(bad["seed_idx"])
            n_figs += 1

            D_good, L_good = _heatmaps_from_v(good["v"])
            D_bad, L_bad = _heatmaps_from_v(bad["v"])
            d_vmin = float(min(D_good.min(), D_bad.min()))
            d_vmax = float(max(D_good.max(), D_bad.max()))
            l_vmin = float(min(L_good.min(), L_bad.min()))
            l_vmax = float(max(L_good.max(), L_bad.max()))

            fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)
            good_tag = f"good {prompt_id} seed{gi:04d} (q={good['vbench_quality']:.3f})"
            bad_tag = f"bad {prompt_id} seed{bi:04d} (q={bad['vbench_quality']:.3f})"

            im_d = _imshow(axes[0, 0], D_good, d_vmin, d_vmax, f"step-adj: {good_tag}")
            _imshow(axes[0, 1], D_bad, d_vmin, d_vmax, f"step-adj: {bad_tag}")
            im_l = _imshow(axes[1, 0], L_good, l_vmin, l_vmax, f"prefix-lock: {good_tag}")
            _imshow(axes[1, 1], L_bad, l_vmin, l_vmax, f"prefix-lock: {bad_tag}")

            fig.colorbar(im_d, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02,
                         label=r"$\langle v_{s-1,k},\ v_{s,k}\rangle$")
            fig.colorbar(im_l, ax=axes[1, :].tolist(), fraction=0.03, pad=0.02,
                         label=r"$\frac{1}{s}\sum_{r<s}\langle v_{r,k},\ v_{s,k}\rangle$")
            fig.suptitle(f"Patch-mean velocity direction — prompt {prompt_id}  [{stratum_tag}]  "
                         f"good seed{gi:04d} vs bad seed{bi:04d}", fontsize=11)
            fig.savefig(args.output_dir / f"fig_{prompt_id}_dyn{dyn}.png", dpi=140)
            plt.close(fig)

            cell_report.append(f"==== prompt {prompt_id} ({stratum_tag}) ====")
            cell_report.append(f"good = seed{gi:04d} (vbench_quality = {good['vbench_quality']:.4f})")
            cell_report.append(f"bad  = seed{bi:04d} (vbench_quality = {bad['vbench_quality']:.4f})")
            cell_report.append("")
            for tag, M in [(f"good {prompt_id} seed{gi:04d} | D step-adjacent", D_good),
                           (f"good {prompt_id} seed{gi:04d} | L prefix-lock", L_good),
                           (f"bad {prompt_id} seed{bi:04d} | D step-adjacent", D_bad),
                           (f"bad {prompt_id} seed{bi:04d} | L prefix-lock", L_bad)]:
                cell_report.append(f"-- {tag} --")
                cell_report.append(_format_matrix(M))
            cell_report.append("")

            print(f"[velocity_heatmaps] {prompt_id} [{stratum_tag}]: "
                  f"good=seed{gi:04d} q={good['vbench_quality']:.3f}  "
                  f"bad=seed{bi:04d} q={bad['vbench_quality']:.3f}", file=sys.stderr)

    print(f"[velocity_heatmaps] wrote {n_figs} figures over {len(grouped)} prompts", file=sys.stderr)
    (args.output_dir / "velocity_heatmap_cells.txt").write_text("\n".join(cell_report) + "\n")
    print("Wrote", args.output_dir / "velocity_heatmap_cells.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
