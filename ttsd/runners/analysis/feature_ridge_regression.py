"""Per-seed feature vector + ridge regression with leave-one-prompt-out CV.

For each seed we extract a small fixed feature vector from `frame_neighbor`
and `posterior_neighbor` matrices. Then for each held-out prompt we fit a
ridge regression of features -> per-seed prompt-centered vbench_quality on the
other prompts and predict the held-out prompt's seed scores. We collect those predictions
across all prompts and evaluate alignment with VBench exactly like the
existing grid search.

This evaluates whether richer joint features beat the best single
hand-tuned reduction (0.7907 prompt-mean Spearman, 12/15 winner_match).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttsd.runners.utilities.ranking import _rankdata
from ttsd.runners.utilities.run_layout import resolve_run_id, stage_output_dir
from ttsd.runners.utilities.seed_vbench_loaders import (
    SUBSCORES,
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
    annotate_vbench_targets,
    cell_groups_rows,
)

FRAME_FILE = "posterior_mean_frame_neighbor_similarity.npy"
POST_FILE = "posterior_mean_posterior_neighbor_similarity.npy"
FEATURES_FILE = "posterior_mean_features.npy"
PATCH_FILE = "posterior_mean_patch_features.npy"


def _tail_mean_sorted_low(values: np.ndarray, tail_fraction: float) -> float:
    """Mean of the smallest `tail_fraction` of `values` (after flatten)."""
    flat = np.sort(values.reshape(-1))
    n = max(1, math.ceil(flat.size * tail_fraction))
    return float(flat[:n].mean())


def _velocity_features(F: np.ndarray) -> tuple[list[tuple[str, float]], None]:
    """Compute velocity / direction-consistency features from F ∈ R^{T×N×D}.

    F rows are L2-normalized DINOv2 CLS features.  We measure how the per-frame
    feature changes within a posterior step (frame-axis velocity) and how the
    per-frame feature changes across posterior steps (posterior-axis
    velocity), and how consistent the *direction* of those velocities is.
    """
    F = F.astype(np.float32, copy=False)
    eps = 1e-8

    # Frame-axis velocity: vf[i, k] = F[i, k+1] - F[i, k], shape (T, N-1, D)
    vf = F[:, 1:, :] - F[:, :-1, :]
    vf_mag = np.linalg.norm(vf, axis=-1)  # (T, N-1)
    vf_unit = vf / (vf_mag[..., None] + eps)
    # direction cosine between consecutive frame velocities, shape (T, N-2)
    vf_dircos = np.sum(vf_unit[:, :-1, :] * vf_unit[:, 1:, :], axis=-1)

    # Posterior-axis velocity: vp[k, j] = F[k+1, j] - F[k, j], shape (T-1, N, D)
    vp = F[1:, :, :] - F[:-1, :, :]
    vp_mag = np.linalg.norm(vp, axis=-1)  # (T-1, N)
    vp_unit = vp / (vp_mag[..., None] + eps)
    # direction cosine between consecutive posterior velocities, shape (T-2, N)
    vp_dircos = np.sum(vp_unit[:-1, :, :] * vp_unit[1:, :, :], axis=-1)

    feats: list[tuple[str, float]] = []
    # Frame velocity magnitude (large = jumpy intra-step motion)
    feats.append(("v_frame_mag_mean", float(vf_mag.mean())))
    feats.append(("v_frame_mag_std", float(vf_mag.std())))
    feats.append(("v_frame_mag_p90", float(np.quantile(vf_mag, 0.90))))
    feats.append(("v_frame_mag_last3_mean", float(vf_mag[-3:, :].mean())))
    feats.append(("v_frame_mag_late_minus_early",
                  float(vf_mag[-3:, :].mean() - vf_mag[:3, :].mean())))

    # Frame direction-cosine (high = smooth coherent motion through frames)
    feats.append(("v_frame_dircos_mean", float(vf_dircos.mean())))
    feats.append(("v_frame_dircos_tail50_low",
                  _tail_mean_sorted_low(vf_dircos, 0.50)))
    feats.append(("v_frame_dircos_p10",
                  float(np.quantile(vf_dircos, 0.10))))
    feats.append(("v_frame_dircos_last3_mean",
                  float(vf_dircos[-3:, :].mean())))
    feats.append(("v_frame_dircos_late_minus_early",
                  float(vf_dircos[-3:, :].mean() - vf_dircos[:3, :].mean())))

    # Posterior velocity magnitude (large = denoising still moving)
    feats.append(("v_post_mag_mean", float(vp_mag.mean())))
    feats.append(("v_post_mag_last3_mean", float(vp_mag[-3:, :].mean())))
    feats.append(("v_post_mag_late_minus_early",
                  float(vp_mag[-3:, :].mean() - vp_mag[:3, :].mean())))

    # Posterior direction-cosine (high = denoising moves in a consistent
    # direction = converging)
    feats.append(("v_post_dircos_mean", float(vp_dircos.mean())))
    feats.append(("v_post_dircos_last3_mean",
                  float(vp_dircos[-3:, :].mean())))
    feats.append(("v_post_dircos_late_minus_early",
                  float(vp_dircos[-3:, :].mean() - vp_dircos[:3, :].mean())))

    # --- Ablation: features that use ONLY the final posterior mean ---
    # F[-1, :, :] is the "fully-denoised" trajectory in feature space, which
    # is what every single-trajectory baseline would have access to.  These
    # let us measure how much the multi-posterior-step axis is actually
    # contributing on top of a velocity analysis of just one trajectory.
    vf_last = vf[-1, :, :]            # (N-1, D)
    vf_last_mag = vf_mag[-1, :]       # (N-1,)
    vf_last_dircos = vf_dircos[-1, :] # (N-2,)
    feats.append(("v_lastpost_mag_mean", float(vf_last_mag.mean())))
    feats.append(("v_lastpost_mag_p90", float(np.quantile(vf_last_mag, 0.90))))
    feats.append(("v_lastpost_mag_std", float(vf_last_mag.std())))
    feats.append(("v_lastpost_dircos_mean", float(vf_last_dircos.mean())))
    feats.append(("v_lastpost_dircos_p10",
                  float(np.quantile(vf_last_dircos, 0.10))))
    feats.append(("v_lastpost_dircos_tail50_low",
                  _tail_mean_sorted_low(vf_last_dircos, 0.50)))

    # --- Cross-axis features: GENUINELY require multi-posterior structure ---
    # These cannot be computed from F[-1, :, :] alone.  They measure how the
    # *per-frame motion direction* changes across denoising steps.  If the
    # video is converging cleanly, frame-to-frame motion at step k should be
    # similar to frame-to-frame motion at step k+1; if denoising is still
    # rearranging content, the per-frame velocity flips around.
    # vf has shape (T, N-1, D). cos(vf[i,k], vf[i+1,k]) -> (T-1, N-1).
    vf_postcos = np.sum(vf_unit[:-1, :, :] * vf_unit[1:, :, :], axis=-1)
    feats.append(("vf_postcos_mean", float(vf_postcos.mean())))
    feats.append(("vf_postcos_p10", float(np.quantile(vf_postcos, 0.10))))
    feats.append(("vf_postcos_late_mean", float(vf_postcos[-3:, :].mean())))
    feats.append(("vf_postcos_late_minus_early",
                  float(vf_postcos[-3:, :].mean() - vf_postcos[:3, :].mean())))

    # Per-posterior-step cross-frame consistency curve: c_i = mean_k cos(F[i,k],F[i,k+1]).
    # Already implicitly captured as `f_row_means_*` from `frame`, but here
    # using F directly so the same code path that has v_post_* can also have
    # the trend-based signals computed from the same tensor.  Trend features:
    # how does cross-frame consistency improve with denoising progress?
    # frame_cos[i, k] = cos(F[i,k], F[i,k+1]); shape (T, N-1).
    frame_cos = np.sum(F[:, :-1, :] * F[:, 1:, :], axis=-1)
    c_curve = frame_cos.mean(axis=1)        # (T,)
    feats.append(("c_curve_late3_mean", float(c_curve[-3:].mean())))
    feats.append(("c_curve_late_minus_early",
                  float(c_curve[-3:].mean() - c_curve[:3].mean())))
    feats.append(("c_curve_std", float(c_curve.std())))
    feats.append(("c_curve_monotone_frac",
                  float(np.mean(np.diff(c_curve) >= 0))))

    return feats, None


def _finalpost_features(F_patch: np.ndarray | None,
                        F_cls: np.ndarray | None) -> list[tuple[str, float]]:
    """Genuinely-multi-posterior single scalars derived from convergence
    of intermediate posterior means to the final one along the FRAME axis.
    """
    feats: list[tuple[str, float]] = []
    eps = 1e-8

    if F_patch is not None:
        F = F_patch.astype(np.float32)                # (T, N, P, D)
        T, N, P, D = F.shape
        pm = F.mean(axis=2)                           # (T, N, D)
        pm = pm / (np.linalg.norm(pm, axis=-1, keepdims=True) + eps)
        # Frame-velocity direction convergence (patch-mean space)
        v = pm[:, 1:] - pm[:, :-1]                   # (T, N-1, D)
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)
        v_final = v[-1]
        vc = (v[:-1] * v_final[None]).sum(-1)        # (T-1, N-1)
        feats.append(("finalpost_velcos_pm_mean", float(vc.mean())))
        feats.append(("finalpost_velcos_pm_first", float(vc[0].mean())))
        feats.append(("finalpost_velcos_pm_late_minus_early",
                      float(vc[-3:].mean() - vc[:3].mean())))
        # Same-position cosine to final (averaged over N, P)
        final = F[-1]
        cos_curve = np.empty(T - 1)
        for i in range(T - 1):
            cos_curve[i] = float((F[i] * final).sum(-1).mean())
        feats.append(("finalpost_cos_mean", float(cos_curve.mean())))
        feats.append(("finalpost_cos_auc",
                      float(np.trapz(cos_curve) / (T - 2 + 1e-12))))

    if F_cls is not None:
        Fc = F_cls.astype(np.float32)
        Fc = Fc / (np.linalg.norm(Fc, axis=-1, keepdims=True) + eps)
        Tc, Nc, _ = Fc.shape
        v = Fc[:, 1:] - Fc[:, :-1]
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)
        v_final = v[-1]
        vc = (v[:-1] * v_final[None]).sum(-1)
        feats.append(("finalpost_cls_velcos_mean", float(vc.mean())))
        feats.append(("finalpost_cls_velcos_late_minus_early",
                      float(vc[-3:].mean() - vc[:3].mean())))
        final = Fc[-1]
        cos_curve = (Fc[:-1] * final[None]).sum(-1).mean(axis=-1)
        feats.append(("finalpost_cls_cos_mean", float(cos_curve.mean())))
        feats.append(("finalpost_cls_cos_auc",
                      float(np.trapz(cos_curve) / (Tc - 2 + 1e-12))))

    return feats


def _tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    flat = np.sort(values.reshape(-1))
    n = max(1, math.ceil(flat.size * tail_fraction))
    return float(flat[:n].mean())


def _feature_vector(frame: np.ndarray, post: np.ndarray,
                    features_tensor: np.ndarray | None = None,
                    patch_tensor: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    feats: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        feats.append(float(value))
        names.append(name)

    # frame_neighbor scalar features
    add("f_mean", frame.mean())
    add("f_std", frame.std())
    add("f_min", frame.min())
    add("f_tail80_n11", _tail_mean(frame, 0.80))
    add("f_tail50_n11", _tail_mean(frame, 0.50))
    add("f_tail20_n11", _tail_mean(frame, 0.20))
    add("f_tail80_n5", _tail_mean(frame[-5:, :], 0.80))
    add("f_tail50_n5", _tail_mean(frame[-5:, :], 0.50))
    add("f_tail50_n2", _tail_mean(frame[-2:, :], 0.50))
    add("f_last_row_mean", frame[-1, :].mean())
    add("f_last_row_min", frame[-1, :].min())
    add("f_lastpost_tail80", _tail_mean(frame[-1:, :], 0.80))
    add("f_first_row_mean", frame[0, :].mean())
    add("f_late_minus_early", frame[-3:, :].mean() - frame[:3, :].mean())
    # column statistics across rows
    col_min = frame.min(axis=0)
    col_std = frame.std(axis=0)
    add("f_col_min_mean", col_min.mean())
    add("f_col_min_tail50", _tail_mean(col_min, 0.50))
    add("f_col_min_worst", col_min.min())
    add("f_col_std_mean", col_std.mean())
    add("f_col_std_tail50_high", -_tail_mean(-col_std, 0.50))  # mean of largest 50% stds
    # row statistics across cols
    row_min = frame.min(axis=1)
    add("f_row_min_mean", row_min.mean())
    add("f_row_min_worst", row_min.min())
    # row-trajectory features
    row_means = frame.mean(axis=1)
    add("f_row_means_slope", float(np.polyfit(np.arange(len(row_means)), row_means, 1)[0]))
    add("f_row_means_last_minus_mid", row_means[-1] - row_means[len(row_means) // 2])
    add("f_row_means_monotonic_frac",
        float(np.mean(np.diff(row_means) >= 0)))
    add("f_late3_min", frame[-3:, :].min())
    add("f_late3_p10", float(np.quantile(frame[-3:, :], 0.10)))
    add("f_late3_p25", float(np.quantile(frame[-3:, :], 0.25)))

    # posterior_neighbor scalar features
    add("p_mean", post.mean())
    add("p_std", post.std())
    add("p_min", post.min())
    add("p_tail50_n11", _tail_mean(post, 0.50))
    add("p_last_row_mean", post[-1, :].mean())
    add("p_first_row_mean", post[0, :].mean())
    add("p_late_minus_early", post[-3:, :].mean() - post[:3, :].mean())
    p_col_min = post.min(axis=0)
    add("p_col_min_mean", p_col_min.mean())
    add("p_col_min_worst", p_col_min.min())
    p_row_means = post.mean(axis=1)
    add("p_row_means_slope", float(np.polyfit(np.arange(len(p_row_means)), p_row_means, 1)[0]))

    if features_tensor is not None:
        for name, value in _velocity_features(features_tensor)[0]:
            add(name, value)

    if patch_tensor is not None or features_tensor is not None:
        for name, value in _finalpost_features(patch_tensor, features_tensor):
            add(name, value)

    return np.asarray(feats, dtype=np.float64), names


def _load_seed_records(heatmap_run_root: Path,
                       vbench_rows: dict[tuple[str, int], dict[str, object]],
                       patch_run_root: Path | None = None) -> list[dict[str, object]]:
    seed_records = []
    for seed_dir in _iter_seed_dirs(heatmap_run_root):
        f_path = seed_dir / FRAME_FILE
        p_path = seed_dir / POST_FILE
        if not (f_path.exists() and p_path.exists()):
            continue
        prompt_id = seed_dir.parent.name
        seed_idx = _seed_idx_from_name(seed_dir)
        if (prompt_id, seed_idx) not in vbench_rows:
            continue
        frame = np.load(f_path)
        post = np.load(p_path)
        feat_path = seed_dir / FEATURES_FILE
        feat_tensor = np.load(feat_path).astype(np.float32) if feat_path.exists() else None
        patch_tensor = None
        if patch_run_root is not None:
            patch_path = patch_run_root / prompt_id / seed_dir.name / PATCH_FILE
            if patch_path.exists():
                patch_tensor = np.load(patch_path)
        record = dict(vbench_rows[(prompt_id, seed_idx)])
        record["prompt_id"] = prompt_id
        record["seed_idx"] = seed_idx
        feats, names = _feature_vector(frame, post, feat_tensor, patch_tensor)
        record["features"] = feats
        record["feature_names"] = names
        seed_records.append(record)

    annotate_vbench_targets(seed_records)

    # Subtract each prompt's mean quality. Roughly two thirds of vbench_quality's
    # variance is prompt difficulty, which the features cannot explain on a held-out
    # prompt, so fitting the raw score wastes capacity on it. This is a modeling choice
    # local to the regression — it keeps absolute units and does not divide by a
    # standard deviation, so it is not the retired avg_vbench_z. The reported metrics
    # are rank-based within prompt and so are unaffected either way.
    grouped = defaultdict(list)
    for record in seed_records:
        grouped[str(record["prompt_id"])].append(record)
    for group in grouped.values():
        mean = float(np.mean([row["vbench_quality"] for row in group]))
        for row in group:
            row["vbench_quality_centered"] = float(row["vbench_quality"]) - mean
    return seed_records


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # X shape (n, d). Solve (X^T X + alpha I) w = X^T y.
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


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


def _evaluate_predictions(seed_records: list[dict[str, object]],
                          predictions: list[float]) -> dict[str, object]:
    by_prompt = defaultdict(list)
    cand = []
    for row, pred in zip(seed_records, predictions):
        c = {
            "prompt_id": row["prompt_id"],
            "seed_idx": row["seed_idx"],
            "score_value": float(pred),
        }
        for metric in SUBSCORES + ["vbench_quality", "vbench_quality_centered"]:
            c[metric] = float(row[metric])
        if "dynamic_degree" in row:
            c["dynamic_degree"] = float(row["dynamic_degree"])
        cand.append(c)
        by_prompt[str(row["prompt_id"])].append(c)

    summary: dict[str, object] = {}
    score_values = np.array([c["score_value"] for c in cand], dtype=float)
    # Regroup the SAME leave-one-prompt-out predictions by (prompt, dynamic_degree). The fit is
    # untouched: LOPO holds out whole prompts, and a cell is a subset of one.
    stratify = all("dynamic_degree" in c for c in cand)
    cells_by_dyn = {d: cell_groups_rows(cand, d) for d in (0, 1)} if stratify else {}
    sizes = sorted({len(g) for g in by_prompt.values()})
    for metric in SUBSCORES + ["vbench_quality", "vbench_quality_centered"]:
        metric_values = np.array([c[metric] for c in cand], dtype=float)
        if np.std(score_values) == 0 or np.std(metric_values) == 0:
            pearson = spearman = float("nan")
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
        for d in (0, 1):
            st = _alignment_stats(cells_by_dyn[d], metric) if stratify else {}
            summary[f"n_cells_dyn{d}"] = st.get("n_prompts", 0)
            for stat in ("prompt_mean_spearman", "winner_match", "winner_match_rate",
                         "top3_hit", "top5_contains_best", "top5_overlap_rate"):
                summary[f"{metric}_{stat}_dyn{d}"] = st.get(stat, float("nan"))
    return summary


def _run_loocv(seed_records: list[dict[str, object]], alpha: float,
               feature_idx: list[int] | None = None,
               target: str = "vbench_quality_centered",
               prompt_residualize: bool = False) -> list[float]:
    prompts = sorted({r["prompt_id"] for r in seed_records})
    predictions = [0.0] * len(seed_records)
    X_all = np.stack([r["features"] for r in seed_records])
    y_all = np.array([float(r[target]) for r in seed_records])
    prompt_of = np.array([r["prompt_id"] for r in seed_records])

    if feature_idx is not None:
        X_all = X_all[:, feature_idx]

    if prompt_residualize:
        # subtract per-prompt mean and divide by per-prompt std (in-sample
        # since it only uses the seed-side info, not the targets); this puts
        # all prompts on the same within-prompt scale.
        X_res = X_all.copy()
        for p in prompts:
            mask = prompt_of == p
            block = X_all[mask]
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            sd[sd == 0] = 1.0
            X_res[mask] = (block - mu) / sd
        X_for_fit = X_res
    else:
        X_for_fit = X_all

    for held in prompts:
        train_mask = prompt_of != held
        test_mask = ~train_mask
        Xt = X_for_fit[train_mask]
        yt = y_all[train_mask]
        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0)
        sd[sd == 0] = 1.0
        Xt_n = (Xt - mu) / sd
        Xt_b = np.hstack([Xt_n, np.ones((Xt_n.shape[0], 1))])
        w = _ridge_fit(Xt_b, yt, alpha)
        Xte_n = (X_for_fit[test_mask] - mu) / sd
        Xte_b = np.hstack([Xte_n, np.ones((Xte_n.shape[0], 1))])
        preds = Xte_b @ w
        for idx, p in zip(np.where(test_mask)[0], preds):
            predictions[int(idx)] = float(p)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-run-root", type=Path, default=None)
    parser.add_argument("--patch-run-root", type=Path, default=None,
                        help="Optional root with posterior_mean_patch_features.npy per seed.")
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alphas", type=str, default="0.03,0.1,0.3,1,3,10,30,100")
    parser.add_argument("--targets", type=str,
                        default="vbench_quality_centered,subject_consistency")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args()
    resolve_run_id(args, parser, needs=["heatmap_run_root", "vbench_long_csv"])

    output_dir = args.output_dir or stage_output_dir(args.heatmap_run_root, "analysis", __file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    vbench_rows = _load_vbench_rows(args.vbench_long_csv)
    seed_records = _load_seed_records(args.heatmap_run_root, vbench_rows,
                                      patch_run_root=args.patch_run_root)
    print(f"Loaded {len(seed_records)} seeds, {len({r['prompt_id'] for r in seed_records})} prompts", flush=True, file=sys.stderr)
    feature_names = seed_records[0]["feature_names"]
    print(f"Feature dim: {len(feature_names)}", flush=True, file=sys.stderr)

    alphas = [float(x) for x in args.alphas.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]

    best = {"sp": float("-inf"), "row": None, "wm": -1, "wm_tb": float("-inf"), "wm_row": None}
    rows = []

    # Sweep 1: full feature set, multiple alphas, each training target, both
    # plain and prompt-residualized features
    for residualize in (False, True):
        res_tag = "presid" if residualize else "plain"
        for target in targets:
            for alpha in alphas:
                preds = _run_loocv(seed_records, alpha, feature_idx=None,
                                   target=target, prompt_residualize=residualize)
                summary = _evaluate_predictions(seed_records, preds)
                tag = f"ridge_full_{res_tag}_a{alpha}_train_{target}"
                summary["score_name"] = tag
                rows.append(summary)
                sp = summary["vbench_quality_centered_prompt_mean_spearman"]
                wm = summary["vbench_quality_centered_winner_match"]
                ov = summary["vbench_quality_centered_top5_overlap_rate"]
                np_ = summary.get("n_prompts", "?")
                if sp > best["sp"]:
                    best["sp"] = sp; best["row"] = summary
                    print(f"[NEW BEST Spearman] {tag:<55} sp={sp:.4f}  winner={wm}/{np_}  top5={ov:.4f}", flush=True, file=sys.stderr)
                if wm > best["wm"] or (wm == best["wm"] and sp > best["wm_tb"]):
                    best["wm"] = wm; best["wm_tb"] = sp; best["wm_row"] = summary
                    print(f"[NEW BEST Winner ] {tag:<55} winner={wm}/{np_}  sp={sp:.4f}  top5={ov:.4f}", flush=True, file=sys.stderr)

    # Sweep 2: small per-target feature subsets to reduce overfitting
    # use ranking of features by abs correlation with training target on full data first (lazy: just
    # try a few hand-picked subsets)
    subsets = {
        "core5": ["f_tail80_n11", "f_col_min_mean", "f_late_minus_early", "f_last_row_mean", "p_tail50_n11"],
        "core7": ["f_tail80_n11", "f_tail50_n5", "f_col_min_mean", "f_col_min_tail50",
                  "f_late_minus_early", "f_last_row_mean", "p_late_minus_early"],
        "core10": ["f_mean", "f_std", "f_tail80_n11", "f_tail50_n5", "f_col_min_mean",
                   "f_col_min_tail50", "f_col_std_mean", "f_late_minus_early",
                   "f_last_row_mean", "p_tail50_n11"],
        "core10_traj": ["f_tail80_n11", "f_tail50_n5", "f_col_min_mean", "f_col_min_tail50",
                        "f_late_minus_early", "f_last_row_mean", "f_row_means_slope",
                        "f_late3_p10", "p_tail50_n11", "p_late_minus_early"],
        "core13": ["f_mean", "f_std", "f_tail80_n11", "f_tail50_n5", "f_col_min_mean",
                   "f_col_min_tail50", "f_col_std_mean", "f_late_minus_early",
                   "f_last_row_mean", "f_row_means_slope", "f_late3_p10",
                   "p_tail50_n11", "p_col_min_mean"],
        "frame_only_5": ["f_tail80_n11", "f_col_min_mean", "f_late_minus_early",
                         "f_last_row_mean", "f_col_std_mean"],
        "tail_traj_4": ["f_tail80_n11", "f_late3_p10", "f_late_minus_early", "f_row_means_slope"],
    }
    # If velocity features were extracted (raw F tensor present), add subsets
    # that combine the prior winners with velocity / direction-consistency
    # features.
    have_vel = "v_frame_dircos_mean" in feature_names
    if have_vel:
        vel_core6 = [
            "v_frame_dircos_mean",
            "v_frame_dircos_p10",
            "v_frame_dircos_last3_mean",
            "v_post_dircos_mean",
            "v_post_dircos_last3_mean",
            "v_post_mag_late_minus_early",
        ]
        vel_all = [n for n in feature_names if n.startswith("v_") and not n.startswith("v_lastpost")]
        subsets["vel_only_6"] = vel_core6
        subsets["vel_only_all"] = vel_all
        subsets["core10_vel6"] = subsets["core10"] + vel_core6
        subsets["core10_traj_vel6"] = subsets["core10_traj"] + vel_core6
        subsets["core13_vel6"] = subsets["core13"] + vel_core6
        subsets["core13_velall"] = subsets["core13"] + vel_all
        subsets["traj4_vel6"] = subsets["tail_traj_4"] + vel_core6

        # --- Minimal-feature ablations: how few velocity features suffice? ---
        subsets["vel_min2"] = ["v_frame_dircos_mean", "v_post_dircos_mean"]
        subsets["vel_min3"] = subsets["vel_min2"] + ["v_post_mag_late_minus_early"]
        subsets["vel_min4"] = subsets["vel_min3"] + ["v_frame_dircos_p10"]
        # baseline + minimal velocity (just add one velocity feature to the
        # single best legacy summary)
        subsets["tail80_plus_vel1"] = ["f_tail80_n11", "v_frame_dircos_mean"]
        subsets["tail80_plus_vel2"] = ["f_tail80_n11", "v_frame_dircos_mean", "v_post_dircos_mean"]

        # --- Posterior-axis ablations: only the FINAL posterior mean ---
        # If these match `vel_only_6`/`core10_vel6`, then the multi-posterior
        # axis isn't really helping; if much worse, then it is.
        lastpost_only = [
            "v_lastpost_mag_mean",
            "v_lastpost_mag_p90",
            "v_lastpost_dircos_mean",
            "v_lastpost_dircos_p10",
            "v_lastpost_dircos_tail50_low",
        ]
        subsets["vel_lastpost_only_5"] = lastpost_only
        subsets["vel_lastpost_only_2"] = [
            "v_lastpost_mag_mean", "v_lastpost_dircos_mean",
        ]
        subsets["core10_lastpost_only"] = subsets["core10"] + lastpost_only
        # frame-only velocity (uses all posterior steps but ignores posterior
        # axis as a derivative direction): isolates "what does the posterior
        # AXIS curvature add?"
        vel_frame_only_5 = [
            "v_frame_mag_mean",
            "v_frame_dircos_mean",
            "v_frame_dircos_p10",
            "v_frame_dircos_last3_mean",
            "v_frame_dircos_late_minus_early",
        ]
        subsets["vel_frame_only_5"] = vel_frame_only_5
        subsets["core10_velframe_only"] = subsets["core10"] + vel_frame_only_5

        # --- Head-to-head clean metrics: single-trajectory vs multi-posterior ---
        # Single trajectory = use only F[-1, :, :] (the final, fully-denoised
        # posterior mean) -> mimics a vanilla cross-frame consistency check.
        # Multi-posterior = same definitions but averaged over all 11 posterior
        # means.  Apples-to-apples comparison; the gap between them is the
        # marginal value of the posterior-mean axis.
        subsets["clean_single_1"] = ["f_lastpost_tail80"]
        subsets["clean_single_2"] = ["f_lastpost_tail80", "v_lastpost_dircos_mean"]
        subsets["clean_single_3"] = [
            "f_lastpost_tail80", "v_lastpost_dircos_mean", "v_lastpost_mag_mean",
        ]
        subsets["clean_multi_1"] = ["f_tail80_n11"]
        subsets["clean_multi_2"] = ["f_tail80_n11", "v_frame_dircos_mean"]
        subsets["clean_multi_3"] = [
            "f_tail80_n11", "v_frame_dircos_mean", "v_post_dircos_mean",
        ]
        subsets["clean_multi_4"] = [
            "f_tail80_n11", "v_frame_dircos_mean",
            "v_post_dircos_mean", "v_post_mag_late_minus_early",
        ]

        # --- True multi-posterior features (cannot be reduced to F[-1]) ---
        # vf_postcos_*, c_curve_* all probe how things *change* across the
        # posterior axis -> 0 information loss collapsing this onto a single
        # trajectory; not a re-skin of cross-frame consistency.
        subsets["xpost_only_1"] = ["vf_postcos_mean"]
        subsets["xpost_only_2"] = ["vf_postcos_mean", "c_curve_late_minus_early"]
        subsets["xpost_only_3"] = [
            "vf_postcos_mean", "c_curve_late_minus_early", "c_curve_late3_mean",
        ]
        # Drop-in replacements for clean_multi_2 with cross-axis term in place
        # of the simple-average dircos.
        subsets["xpost_v2"] = ["f_tail80_n11", "vf_postcos_mean"]
        subsets["xpost_v2b"] = ["f_tail80_n11", "vf_postcos_late_mean"]
        subsets["xpost_v2c"] = ["f_tail80_n11", "c_curve_late_minus_early"]
        # Add cross-axis on top of the current 2-feature champion.
        subsets["xpost_v3"] = [
            "f_tail80_n11", "v_frame_dircos_mean", "vf_postcos_mean",
        ]
        subsets["xpost_v3b"] = [
            "f_tail80_n11", "v_frame_dircos_mean", "vf_postcos_late_mean",
        ]
        subsets["xpost_v3c"] = [
            "f_tail80_n11", "v_frame_dircos_mean", "c_curve_late_minus_early",
        ]
        subsets["xpost_v3d"] = [
            "f_tail80_n11", "v_frame_dircos_mean", "c_curve_late3_mean",
        ]
        subsets["xpost_v4"] = [
            "f_tail80_n11", "v_frame_dircos_mean",
            "vf_postcos_mean", "c_curve_late_minus_early",
        ]
        # Single-traj counterparts that share *no* cross-axis term -> isolates
        # the lift the cross-axis brings on top of the same anchor.
        subsets["xpost_singletraj_1"] = ["f_lastpost_tail80"]
        subsets["xpost_singletraj_2"] = ["f_lastpost_tail80", "v_lastpost_dircos_mean"]

        # --- Final-posterior convergence (NEW: needs patch features). Sign of
        # finalpost_velcos_pm_mean is NEGATIVE (low cos => high quality), but
        # ridge handles signs automatically.
        if "finalpost_velcos_pm_mean" in feature_names:
            fp_core = [
                "finalpost_velcos_pm_mean",
                "finalpost_velcos_pm_late_minus_early",
                "finalpost_cos_mean",
            ]
            fp_full = [n for n in feature_names if n.startswith("finalpost_")]
            subsets["finalpost_only_3"] = fp_core
            subsets["finalpost_only_all"] = fp_full
            # combinations with the 1-feature legacy anchor
            subsets["tail80_plus_velcospm"] = ["f_tail80_n11", "finalpost_velcos_pm_mean"]
            subsets["tail80_plus_fp_core"] = ["f_tail80_n11"] + fp_core
            # combinations with the 2-feature champion (xpost_v2)
            subsets["xpost_v2_plus_velcospm"] = [
                "f_tail80_n11", "vf_postcos_mean", "finalpost_velcos_pm_mean",
            ]
            subsets["xpost_v2_plus_fp_core"] = [
                "f_tail80_n11", "vf_postcos_mean",
            ] + fp_core
            # combinations with the 15/15 winner ridge subset (core10_vel6)
            subsets["core10_vel6_plus_velcospm"] = subsets["core10_vel6"] + [
                "finalpost_velcos_pm_mean",
            ]
            subsets["core10_vel6_plus_fp_core"] = subsets["core10_vel6"] + fp_core
            subsets["core10_vel6_plus_fp_all"] = subsets["core10_vel6"] + fp_full
    for name, fs in subsets.items():
        idx = [feature_names.index(x) for x in fs]
        for residualize in (False, True):
            res_tag = "presid" if residualize else "plain"
            for target in targets:
                for alpha in alphas:
                    preds = _run_loocv(seed_records, alpha, feature_idx=idx,
                                       target=target, prompt_residualize=residualize)
                    summary = _evaluate_predictions(seed_records, preds)
                    tag = f"ridge_{name}_{res_tag}_a{alpha}_train_{target}"
                    summary["score_name"] = tag
                    rows.append(summary)
                    sp = summary["vbench_quality_centered_prompt_mean_spearman"]
                    wm = summary["vbench_quality_centered_winner_match"]
                    ov = summary["vbench_quality_centered_top5_overlap_rate"]
                    np_ = summary.get("n_prompts", "?")
                    if sp > best["sp"]:
                        best["sp"] = sp; best["row"] = summary
                        print(f"[NEW BEST Spearman] {tag:<55} sp={sp:.4f}  winner={wm}/{np_}  top5={ov:.4f}", flush=True, file=sys.stderr)
                    if wm > best["wm"] or (wm == best["wm"] and sp > best["wm_tb"]):
                        best["wm"] = wm; best["wm_tb"] = sp; best["wm_row"] = summary
                        print(f"[NEW BEST Winner ] {tag:<55} winner={wm}/{np_}  sp={sp:.4f}  top5={ov:.4f}", flush=True, file=sys.stderr)

    fieldnames = list(rows[0].keys())
    with (output_dir / "ridge_feature_results.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sp_row = best["row"]; wm_row = best["wm_row"]
    n_prompts = rows[0].get("n_prompts", "?") if rows else "?"
    md_lines = [
        "# Ridge feature model (leave-one-prompt-out)",
        "",
        f"Feature dim: {len(feature_names)}",
        f"Total configs evaluated: {len(rows)}",
        f"Number of prompts: {n_prompts}",
        "",
        "## Best by vbench_quality_centered prompt-mean Spearman",
        f"- name: `{sp_row['score_name']}`",
        f"- spearman = {sp_row['vbench_quality_centered_prompt_mean_spearman']:.4f}",
        f"- winner   = {sp_row['vbench_quality_centered_winner_match']}/{n_prompts}",
        f"- top5_overlap = {sp_row['vbench_quality_centered_top5_overlap_rate']:.4f}",
        f"- subject prompt-mean spearman = {sp_row['subject_consistency_prompt_mean_spearman']:.4f}",
        "",
        "## Best by vbench_quality_centered winner_match",
        f"- name: `{wm_row['score_name']}`",
        f"- winner   = {wm_row['vbench_quality_centered_winner_match']}/{n_prompts}",
        f"- spearman = {wm_row['vbench_quality_centered_prompt_mean_spearman']:.4f}",
        f"- top5_overlap = {wm_row['vbench_quality_centered_top5_overlap_rate']:.4f}",
        "",
        "## Top 15 by vbench_quality_centered prompt-mean Spearman",
        "",
        "| name | spearman | winner | top5_overlap | subj_sp |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    top15 = sorted(rows, key=lambda r: r["vbench_quality_centered_prompt_mean_spearman"], reverse=True)[:15]
    for r in top15:
        md_lines.append(
            f"| `{r['score_name']}` | {r['vbench_quality_centered_prompt_mean_spearman']:.4f} | "
            f"{r['vbench_quality_centered_winner_match']} | {r['vbench_quality_centered_top5_overlap_rate']:.4f} | "
            f"{r['subject_consistency_prompt_mean_spearman']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Top 15 by vbench_quality_centered winner_match")
    md_lines.append("")
    md_lines.append("| name | winner | spearman | top5_overlap | subj_sp |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: |")
    top15w = sorted(rows, key=lambda r: (r["vbench_quality_centered_winner_match"], r["vbench_quality_centered_prompt_mean_spearman"]),
                    reverse=True)[:15]
    for r in top15w:
        md_lines.append(
            f"| `{r['score_name']}` | {r['vbench_quality_centered_winner_match']} | "
            f"{r['vbench_quality_centered_prompt_mean_spearman']:.4f} | "
            f"{r['vbench_quality_centered_top5_overlap_rate']:.4f} | "
            f"{r['subject_consistency_prompt_mean_spearman']:.4f} |"
        )
    (output_dir / "ridge_feature_summary.md").write_text("\n".join(md_lines))
    print("Wrote", output_dir / "ridge_feature_results.csv", file=sys.stderr)
    print("Wrote", output_dir / "ridge_feature_summary.md", file=sys.stderr)


if __name__ == "__main__":
    main()
