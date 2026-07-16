"""Single-scalar DINOv2 feature diagnostics (patch + CLS) for each seed.

Given per-seed patch features in shape (T, N_sub, P, D), compute a small
collection of conceptually-clean single-number properties and report
within-prompt Spearman / winner-match against vbench_quality, dynamic_degree, and each
individual VBench subscore.

Properties probed (each one is a single scalar):
  - PATCH-MEAN-COS   : mean cos(patch_mean[k], patch_mean[k+1]) over (i, k)
                       (patch-pooled equivalent of CLS cross-frame consistency)
  - PATCH-BESTMATCH  : for each patch p in frame k, cos(F[i,k,p], F[i,k+1,bestmatch(p)])
                       averaged over (i, k, p).  Motion-robust frame coherence.
  - PATCH-IDENTMATCH : cos(F[i,k,p], F[i,k+1,p]) at the same spatial location
                       (no motion compensation; tests how stationary patches are)
  - PATCH-SELFRANK   : how often the diagonal patch is its own best match
                       (high = stationary scene; low = moving content)
  - PATCH-TRAJ-DIR   : per-patch top-SVD direction across the 11 posterior
                       steps, then mean |cos| across patches & frames.
                       Genuinely multi-posterior; cannot be computed from F[-1].
  - PATCH-EFFRANK    : effective rank of the per-frame patch matrix F[i,k] in
                       R^{P x D}, averaged over (i, k).  Spatial diversity.
  - PATCH-EFFRANK-LATE-MINUS-EARLY : trend across denoising steps.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttsd.runners.utilities.ranking import _rankdata
from ttsd.runners.utilities.run_layout import resolve_run_id, stage_output_dir
from ttsd.runners.utilities.seed_vbench_loaders import (
    MIN_STRATUM_SEEDS,
    SUBSCORES,
    _iter_seed_dirs,
    _load_vbench_rows,
    _seed_idx_from_name,
    annotate_vbench_targets,
    cell_groups,
    load_legacy_avg_vbench_z,
)

PATCH_FILE = "posterior_mean_patch_features.npy"
CLS_FILE = "posterior_mean_features.npy"

# Compact column headers for the per-target report table.
SHORT_LABEL = {
    "vbench_quality": "quality",
    "dynamic_degree": "dyn",
    "avg_vbench_z": "avgZ",
    "subject_consistency": "subj",
    "background_consistency": "bg",
    "motion_smoothness": "motion",
    "aesthetic_quality": "aesth",
    "imaging_quality": "imag",
    "overall_consistency": "overall",
}

# Targets whose within-prompt argmax is meaningful. `dynamic_degree` is boolean, so its
# argmax picks an arbitrary member of the winning tie.
_NO_WINNER_MATCH = {"dynamic_degree"}


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


def _patch_metrics(F: np.ndarray) -> dict[str, float]:
    """F: (T, N, P, D), float16/32, rows L2-normalized."""
    F = F.astype(np.float32)
    T, N, P, D = F.shape
    out: dict[str, float] = {}

    # 1) Patch-mean cosine cross-frame consistency
    pm = F.mean(axis=2)                      # (T, N, D)
    pm = pm / (np.linalg.norm(pm, axis=-1, keepdims=True) + 1e-8)
    out["patch_mean_cos"] = float((pm[:, :-1, :] * pm[:, 1:, :]).sum(axis=-1).mean())

    # 2/3) Patch best-match and identity-match cross-frame
    # Process per (i, k) to avoid an enormous tensor.
    bestmatch_vals = []
    ident_vals = []
    selfrank_vals = []
    for i in range(T):
        for k in range(N - 1):
            A = F[i, k]      # (P, D)
            B = F[i, k + 1]  # (P, D)
            sim = A @ B.T    # (P, P), already L2-normalized -> cosine
            bestmatch_vals.append(sim.max(axis=1).mean())
            ident_vals.append(np.diag(sim).mean())
            selfrank_vals.append((sim.argmax(axis=1) == np.arange(P)).mean())
    out["patch_bestmatch_cos"] = float(np.mean(bestmatch_vals))
    out["patch_identmatch_cos"] = float(np.mean(ident_vals))
    out["patch_selfrank"] = float(np.mean(selfrank_vals))

    # Same metrics restricted to last 3 posterior steps (closer to final video)
    # Reuse already-computed lists by re-computing on subset for clarity.
    bestmatch_late = []
    for i in range(max(0, T - 3), T):
        for k in range(N - 1):
            A = F[i, k]; B = F[i, k + 1]
            bestmatch_late.append((A @ B.T).max(axis=1).mean())
    out["patch_bestmatch_cos_late3"] = float(np.mean(bestmatch_late))

    # Same on F[-1] only (single-trajectory baseline)
    bestmatch_end = []
    for k in range(N - 1):
        A = F[-1, k]; B = F[-1, k + 1]
        bestmatch_end.append((A @ B.T).max(axis=1).mean())
    out["patch_bestmatch_cos_end"] = float(np.mean(bestmatch_end))

    # 4) Patch trajectory direction alignment (genuinely multi-posterior).
    # For each (frame k, patch p), compute the top SVD direction of the
    # trajectory matrix F[:, k, p, :] in R^{T x D}.  Average |cos| of those
    # direction vectors across (k, p).
    # Center across i; SVD in batch mode.
    M = F - F.mean(axis=0, keepdims=True)             # (T, N, P, D)
    M = M.transpose(1, 2, 0, 3).reshape(N * P, T, D)  # (N*P, T, D)
    # batched SVD
    try:
        _, _, Vh = np.linalg.svd(M, full_matrices=False)  # (N*P, min(T,D), D)
    except np.linalg.LinAlgError:
        out["patch_traj_dir_align"] = float("nan")
        out["patch_traj_dir_align_late_minus_early"] = float("nan")
        out["patch_eff_rank_mean"] = float("nan")
        out["patch_eff_rank_late_minus_early"] = float("nan")
        return out
    top_dir = Vh[:, 0, :]                             # (N*P, D)
    top_dir /= np.linalg.norm(top_dir, axis=-1, keepdims=True) + 1e-8
    # Mean |cos| between all pairs is too expensive (N*P ~ 5k); subsample.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(top_dir.shape[0], size=min(1024, top_dir.shape[0]), replace=False)
    sub = top_dir[sample_idx]
    cos_pairs = np.abs(sub @ sub.T)
    iu = np.triu_indices(sub.shape[0], k=1)
    out["patch_traj_dir_align"] = float(cos_pairs[iu].mean())

    # 5) Effective rank of per-frame patch matrix F[i, k] in R^{P x D}
    er_curve = np.empty(T)
    for i in range(T):
        # Average ER over frames at this step
        ers_per_frame = np.empty(N)
        for k in range(N):
            G = F[i, k] @ F[i, k].T  # (P, P)
            w = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
            s = w / (w.sum() + 1e-12)
            s = s[s > 1e-12]
            ers_per_frame[k] = float(np.exp(-(s * np.log(s)).sum()))
        er_curve[i] = ers_per_frame.mean()
    out["patch_eff_rank_mean"] = float(er_curve.mean())
    out["patch_eff_rank_late_minus_early"] = float(er_curve[-3:].mean() - er_curve[:3].mean())

    # 6) Convergence to the final posterior mean F[-1].
    # Per-step cos(F[i, k, p], F[-1, k, p]) at the same (frame, patch) position.
    final = F[-1]                                     # (N, P, D)
    # cos curve: shape (T-1,)  (drop trivial i=T-1 self-similarity)
    cos_curve = np.empty(T - 1)
    for i in range(T - 1):
        cos_curve[i] = float((F[i] * final).sum(-1).mean())   # mean over (N, P)
    out["finalpost_cos_mean"] = float(cos_curve.mean())
    out["finalpost_cos_first"] = float(cos_curve[0])
    out["finalpost_cos_last_minus_first"] = float(cos_curve[-1] - cos_curve[0])
    # Area-under-curve normalized: shape of approach to the final state.
    out["finalpost_cos_auc"] = float(np.trapz(cos_curve) / (T - 2 + 1e-12))
    # Half-life-style: index of first step that exceeds 0.9 of final-cos[-1].
    target = 0.9 * cos_curve[-1] + 0.1 * cos_curve[0]
    above = np.where(cos_curve >= target)[0]
    out["finalpost_cos_t90"] = float(above[0]) if len(above) else float(T - 1)

    # Same metric using best-match across patches (motion-robust convergence).
    bm_curve = np.empty(T - 1)
    for i in range(T - 1):
        v = []
        for k in range(N):
            v.append((F[i, k] @ final[k].T).max(axis=1).mean())
        bm_curve[i] = float(np.mean(v))
    out["finalpost_bm_mean"] = float(bm_curve.mean())
    out["finalpost_bm_first"] = float(bm_curve[0])
    out["finalpost_bm_last_minus_first"] = float(bm_curve[-1] - bm_curve[0])

    # Patch-mean (D-vector per frame) cosine to final.
    pm_final = pm[-1]                                # (N, D), already normalized
    pm_cos_curve = np.empty(T - 1)
    for i in range(T - 1):
        pm_cos_curve[i] = float((pm[i] * pm_final).sum(-1).mean())
    out["finalpost_patchmean_cos_mean"] = float(pm_cos_curve.mean())
    out["finalpost_patchmean_cos_first"] = float(pm_cos_curve[0])

    # 7) Frame-axis structure convergence: compare cross-frame structure at
    # step i to that of step -1.
    # 7a) Adjacent-frame cosine vector r_i = (cos(pm[i,k], pm[i,k+1]))_k in R^{N-1}
    # Spearman/pearson correlation with r_{-1} averaged over i.
    r_curve = np.empty((T, N - 1))
    for i in range(T):
        r_curve[i] = (pm[i, :-1] * pm[i, 1:]).sum(-1)
    r_final = r_curve[-1]
    rf_centered = r_final - r_final.mean()
    rf_norm = np.linalg.norm(rf_centered) + 1e-8
    corr_curve = np.empty(T - 1)
    for i in range(T - 1):
        ri = r_curve[i] - r_curve[i].mean()
        corr_curve[i] = float((ri * rf_centered).sum() / (np.linalg.norm(ri) + 1e-8) / rf_norm)
    out["finalpost_neighborpat_corr_mean"] = float(corr_curve.mean())
    out["finalpost_neighborpat_corr_first"] = float(corr_curve[0])

    # 7b) Frame-velocity direction convergence (patch-mean space).
    # v[i, k] = pm[i, k+1] - pm[i, k]  (D,);  cos(v[i,k], v[-1,k]) averaged over (i<T-1, k).
    v = pm[:, 1:] - pm[:, :-1]                       # (T, N-1, D)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    v_final = v[-1]                                  # (N-1, D)
    vc = (v[:-1] * v_final[None]).sum(-1)            # (T-1, N-1)
    out["finalpost_velcos_pm_mean"] = float(vc.mean())
    out["finalpost_velcos_pm_first"] = float(vc[0].mean())
    out["finalpost_velcos_pm_late_minus_early"] = float(vc[-3:].mean() - vc[:3].mean())

    return out


def _cls_metrics(F_cls: np.ndarray) -> dict[str, float]:
    """F_cls: (T, N, D), L2-normalized CLS tokens."""
    F_cls = F_cls.astype(np.float32)
    F_cls = F_cls / (np.linalg.norm(F_cls, axis=-1, keepdims=True) + 1e-8)
    T, N, D = F_cls.shape
    out: dict[str, float] = {}

    # Same-(frame) cosine of CLS to final
    final = F_cls[-1]
    cos_curve = (F_cls[:-1] * final[None]).sum(-1).mean(axis=-1)   # (T-1,)
    out["finalpost_cls_cos_mean"] = float(cos_curve.mean())
    out["finalpost_cls_cos_first"] = float(cos_curve[0])
    out["finalpost_cls_cos_auc"] = float(np.trapz(cos_curve) / (T - 2 + 1e-12))
    out["finalpost_cls_cos_last_minus_first"] = float(cos_curve[-1] - cos_curve[0])

    # Adjacent-frame cosine vector convergence (CLS)
    r_curve = np.empty((T, N - 1))
    for i in range(T):
        r_curve[i] = (F_cls[i, :-1] * F_cls[i, 1:]).sum(-1)
    r_final = r_curve[-1]
    rf_c = r_final - r_final.mean()
    rf_n = np.linalg.norm(rf_c) + 1e-8
    corr_curve = np.empty(T - 1)
    for i in range(T - 1):
        ri = r_curve[i] - r_curve[i].mean()
        corr_curve[i] = float((ri * rf_c).sum() / (np.linalg.norm(ri) + 1e-8) / rf_n)
    out["finalpost_cls_neighborpat_corr_mean"] = float(corr_curve.mean())
    out["finalpost_cls_neighborpat_corr_first"] = float(corr_curve[0])

    # Frame-velocity direction convergence (CLS)
    v = F_cls[:, 1:] - F_cls[:, :-1]
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    v_final = v[-1]
    vc = (v[:-1] * v_final[None]).sum(-1)            # (T-1, N-1)
    out["finalpost_cls_velcos_mean"] = float(vc.mean())
    out["finalpost_cls_velcos_first"] = float(vc[0].mean())
    out["finalpost_cls_velcos_late_minus_early"] = float(vc[-3:].mean() - vc[:3].mean())

    # Plain CLS cross-frame consistency at step -1 (single-trajectory baseline)
    out["finalpost_cls_neighborcos_end"] = float(r_curve[-1].mean())

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-run-root", type=Path, default=None)
    parser.add_argument("--cls-run-root", type=Path, default=None)
    parser.add_argument("--vbench-long-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Fill unset input roots from runs/<stage>/<run-id>.")
    args = parser.parse_args()
    resolve_run_id(args, parser, needs=["patch_run_root", "cls_run_root", "vbench_long_csv"])
    args.output_dir = args.output_dir or stage_output_dir(args.cls_run_root, "analysis", __file__)

    vbench = _load_vbench_rows(args.vbench_long_csv)

    records = []
    for sd in _iter_seed_dirs(args.patch_run_root):
        fp = sd / PATCH_FILE
        if not fp.exists():
            continue
        prompt_id = sd.parent.name
        seed_idx = _seed_idx_from_name(sd)
        if (prompt_id, seed_idx) not in vbench:
            continue
        F = np.load(fp).astype(np.float32)
        m = _patch_metrics(F)
        rec = dict(vbench[(prompt_id, seed_idx)])
        rec["prompt_id"] = prompt_id
        rec["seed_idx"] = seed_idx
        rec.update(m)
        cls_path = args.cls_run_root / prompt_id / sd.name / CLS_FILE
        if cls_path.exists():
            F_cls = np.load(cls_path)
            rec.update(_cls_metrics(F_cls))
            del F_cls
        records.append(rec)
        del F
    print(f"[feature_vbench_correlation] loaded {len(records)} seeds", file=sys.stderr)

    annotate_vbench_targets(records)

    prompt_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        prompt_to_idx[r["prompt_id"]].append(i)
    n_prompts = len(prompt_to_idx)

    # Align each metric against vbench_quality AND every individual subscore. Within-prompt
    # Spearman is rank-based, so a target's raw scale never matters here.
    targets = ["vbench_quality"] + SUBSCORES
    have_dyn = all("dynamic_degree" in r for r in records)
    if have_dyn:
        targets.append("dynamic_degree")
    else:
        print("[feature_vbench_correlation] no dynamic_degree scores; skipping that target",
              file=sys.stderr)

    legacy_z = load_legacy_avg_vbench_z(args.vbench_long_csv)
    if legacy_z:
        for r in records:
            r["avg_vbench_z"] = legacy_z[(r["prompt_id"], r["seed_idx"])]
        targets.append("avg_vbench_z")

    target_arr = {t: np.array([float(r[t]) for r in records]) for t in targets}

    # Re-group by (prompt, dynamic_degree) so motion is held fixed. `dynamic_degree` is constant
    # inside a cell, so it drops out as a target there.
    strat_targets = [t for t in targets if t != "dynamic_degree"]
    pti_dyn = {f"dyn{d}": cell_groups(records, d) for d in (0, 1)} if have_dyn else {}
    dyn_total = ({d: sum(1 for r in records if int(float(r["dynamic_degree"])) == d)
                  for d in (0, 1)} if have_dyn else {})

    metric_keys = [k for k in records[0].keys() if k.startswith(("patch_", "finalpost_"))]
    metric_col = {k: np.array([r[k] for r in records], float) for k in metric_keys}

    def stats_for(metric_signed, arrs, pti, tgts):
        return {t: (_within(metric_signed, arrs[t], pti),
                    None if t in _NO_WINNER_MATCH else _winner_match(metric_signed, arrs[t], pti))
                for t in tgts}

    rows = []  # each: (metric, sign, {target: (sp_within, winner_match)})
    for k in metric_keys:
        for sign in (1, -1):
            rows.append((k, sign, stats_for(sign * metric_col[k], target_arr, prompt_to_idx, targets)))
    rows.sort(key=lambda r: -abs(r[2]["vbench_quality"][0]))

    strat_stats = {
        lbl: {(k, sign): stats_for(sign * metric_col[k], target_arr, pti, strat_targets)
              for k in metric_keys for sign in (1, -1)}
        for lbl, pti in pti_dyn.items()
    }

    def render_table(table_rows, n, tgts):
        header = (f"{'metric':<44} {'sgn':>3} "
                  + " ".join(f"{SHORT_LABEL[t]:>9}" for t in tgts)
                  + f"  {'WM_qual':>8}")
        lines = [header, "-" * len(header)]
        for k, sign, st in table_rows:
            sps = " ".join(f"{st[t][0]:>+9.4f}" for t in tgts)
            wm = st["vbench_quality"][1]
            lines.append(f"{k:<44} {sign:>3} {sps}  {f'{wm}/{n}':>8}")
        return lines

    report: list[str] = []
    report.append("within-prompt Spearman of each metric vs each target "
                  "(columns are signed; WM_qual = winner-match vs vbench_quality)")
    report.extend(render_table(rows, n_prompts, targets))

    # Same statistic, recomputed inside (prompt, dynamic_degree) cells. vbench_quality rewards
    # stillness, so an all-seeds correlation mixes "this seed is better" with "it moves less".
    pretty = {"dyn0": "static", "dyn1": "moving"}
    for lbl, pti in pti_dyn.items():
        n_cells = len(pti)
        n_kept = sum(len(v) for v in pti.values())
        n_dropped = dyn_total[int(lbl[-1])] - n_kept
        strat_rows = [(k, sign, strat_stats[lbl][(k, sign)]) for k, sign, _ in rows]
        report.append("")
        report.append(f"--- stratum {lbl} ({pretty[lbl]}): {n_cells} cells, {n_kept} clips "
                      f"(dropped {n_dropped} in cells with <{MIN_STRATUM_SEEDS} seeds; "
                      f"dynamic_degree is constant here so it is not a target) ---")
        report.extend(render_table(strat_rows, n_cells, strat_targets))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "correlation_table.csv").open("w", newline="") as h:
        cols = ["metric", "sign", "n_prompts"]
        for t in targets:
            cols += [f"sp_{t}", f"wm_{t}"]
        if have_dyn:
            cols += ["n_cells_dyn0", "n_cells_dyn1"]
            for t in strat_targets:
                cols += [f"sp_{t}_dyn0", f"wm_{t}_dyn0", f"sp_{t}_dyn1", f"wm_{t}_dyn1"]
        h.write(",".join(cols) + "\n")
        n_cells = {lbl: str(len(pti)) for lbl, pti in pti_dyn.items()}
        for k, sign, st in rows:
            vals = [k, str(sign), str(n_prompts)]
            for t in targets:
                sp, wm = st[t]
                vals += [f"{sp:.6f}", "" if wm is None else str(wm)]
            if have_dyn:
                vals += [n_cells["dyn0"], n_cells["dyn1"]]
                for t in strat_targets:
                    sp0, wm0 = strat_stats["dyn0"][(k, sign)][t]
                    sp1, wm1 = strat_stats["dyn1"][(k, sign)][t]
                    vals += [f"{sp0:.6f}", str(wm0), f"{sp1:.6f}", str(wm1)]
            h.write(",".join(vals) + "\n")

    sizes = sorted({len(ps) for ps in prompt_to_idx.values()})
    if len(sizes) > 1:
        pooled_order = [(r[0], r[1]) for r in rows]
        for sz in sizes:
            bucket_recs = [r for r in records if len(prompt_to_idx[r["prompt_id"]]) == sz]
            pti_sub: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(bucket_recs):
                pti_sub[r["prompt_id"]].append(i)
            n_sub = len(pti_sub)
            arr_sub = {t: np.array([float(r[t]) for r in bucket_recs]) for t in targets}
            metric_sub = {k: np.array([r[k] for r in bucket_recs], float) for k in metric_keys}
            bucket_rows = [(k, sign, stats_for(sign * metric_sub[k], arr_sub, pti_sub, targets))
                           for k, sign in pooled_order]
            report.append("")
            report.append(f"--- bucket n={sz} ({n_sub} prompts) ---")
            report.extend(render_table(bucket_rows, n_sub, targets))

    # How the targets relate to each other. If vbench_quality tracks dynamic_degree
    # negatively, a higher score means a stiller video, not a better one.
    diag = [t for t in ("vbench_quality", "dynamic_degree", "overall_consistency", "avg_vbench_z")
            if t in target_arr]
    report.append("")
    report.append("target-vs-target within-prompt Spearman")
    if "dynamic_degree" in target_arr:
        n_dyn = sum(len({records[i]["dynamic_degree"] for i in ps}) > 1
                    for ps in prompt_to_idx.values() if len(ps) >= 2)
        report.append(f"(dynamic_degree varies within {n_dyn}/{n_prompts} prompts; "
                      "prompts where it is constant are skipped)")
    report.append(f"{'':<10}" + " ".join(f"{SHORT_LABEL[t]:>9}" for t in diag))
    for a in diag:
        cells = " ".join(f"{_within(target_arr[a], target_arr[b], prompt_to_idx):>+9.4f}"
                         for b in diag)
        report.append(f"{SHORT_LABEL[a]:<10}{cells}")

    (args.output_dir / "correlation_report.txt").write_text("\n".join(report) + "\n")
    print("Wrote", args.output_dir / "correlation_table.csv", file=sys.stderr)
    print("Wrote", args.output_dir / "correlation_report.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
