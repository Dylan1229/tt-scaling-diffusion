"""Per-prompt good-vs-bad velocity-direction heatmaps in patch-mean space.

For each prompt in sweep_v2, picks the best and worst seed by within-prompt
`avg_vbench_z` (per-prompt z-normalized mean of the 5 VBench subscores), then
produces a 2x2 figure:

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
matching the style of `runs/_report/fig_frame_heatmap_good_vs_bad.png`.

Usage:
    python -m ttsd.runners.posterior_mean_velocity_heatmaps \
        --patch-run-root runs/posterior_mean_patch_features/<rid> \
        --vbench-long-csv runs/vbench/<rid>/vbench_scores_long.csv \
        --output-dir runs/_report/velocity_heatmaps_per_prompt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttsd.runners.posterior_mean_tail_rank import (
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
)

PATCH_FILE = "posterior_mean_patch_features.npy"
SUBSCORES = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-run-root", required=True, type=Path)
    parser.add_argument("--vbench-long-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_id in sorted(grouped.keys()):
        group = grouped[prompt_id]
        good = max(group, key=lambda r: r["avg_vbench_z"])
        bad = min(group, key=lambda r: r["avg_vbench_z"])
        D_good, L_good = _heatmaps_from_v(good["v"])
        D_bad, L_bad = _heatmaps_from_v(bad["v"])

        d_vmin = float(min(D_good.min(), D_bad.min()))
        d_vmax = float(max(D_good.max(), D_bad.max()))
        l_vmin = float(min(L_good.min(), L_bad.min()))
        l_vmax = float(max(L_good.max(), L_bad.max()))

        fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)
        good_tag = f"good seed{int(good['seed_idx']):04d} (z={good['avg_vbench_z']:+.2f})"
        bad_tag = f"bad seed{int(bad['seed_idx']):04d} (z={bad['avg_vbench_z']:+.2f})"

        im_d = _imshow(axes[0, 0], D_good, d_vmin, d_vmax, f"step-adj: {good_tag}")
        _imshow(axes[0, 1], D_bad, d_vmin, d_vmax, f"step-adj: {bad_tag}")
        im_l = _imshow(axes[1, 0], L_good, l_vmin, l_vmax, f"prefix-lock: {good_tag}")
        _imshow(axes[1, 1], L_bad, l_vmin, l_vmax, f"prefix-lock: {bad_tag}")

        fig.colorbar(im_d, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02,
                     label=r"$\langle v_{s-1,k},\ v_{s,k}\rangle$")
        fig.colorbar(im_l, ax=axes[1, :].tolist(), fraction=0.03, pad=0.02,
                     label=r"$\frac{1}{s}\sum_{r<s}\langle v_{r,k},\ v_{s,k}\rangle$")
        fig.suptitle(f"Patch-mean velocity direction — prompt {prompt_id}", fontsize=11)
        fig.savefig(args.output_dir / f"fig_{prompt_id}.png", dpi=140)
        plt.close(fig)

        print(f"[velocity_heatmaps] {prompt_id}: "
              f"good=seed{int(good['seed_idx']):04d} z={good['avg_vbench_z']:+.2f}  "
              f"bad=seed{int(bad['seed_idx']):04d} z={bad['avg_vbench_z']:+.2f}")


if __name__ == "__main__":
    main()
