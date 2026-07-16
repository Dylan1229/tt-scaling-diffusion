"""Analyze paired Euler baseline vs fixed step-10 Renoise + AddSteps results."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SIX_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
]
ALL_DIMS = [*SIX_DIMS[:3], "dynamic_degree", *SIX_DIMS[3:]]


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4 * total)) / total)
        / denom
    )
    return center - radius, center + radius


def _bootstrap_ci(
    values: np.ndarray,
    *,
    statistic=np.mean,
    n_boot: int = 10000,
    seed: int = 20260716,
) -> tuple[float, float]:
    arr = np.asarray(values)
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    if statistic is np.mean:
        stats = arr[indices].mean(axis=1)
    else:
        stats = np.asarray([statistic(arr[index]) for index in indices])
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _cluster_bootstrap_difference(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    value_col: str,
    n_boot: int = 10000,
    seed: int = 20260716,
) -> tuple[float, float]:
    prompts = frame["prompt_id"].drop_duplicates().to_numpy()
    selected_ids = set(frame.loc[mask, "sample_id"])
    selected_sum = []
    selected_count = []
    rest_sum = []
    rest_count = []
    for prompt in prompts:
        group = frame[frame["prompt_id"] == prompt]
        group_mask = group["sample_id"].isin(selected_ids)
        selected_sum.append(group.loc[group_mask, value_col].sum())
        selected_count.append(int(group_mask.sum()))
        rest_sum.append(group.loc[~group_mask, value_col].sum())
        rest_count.append(int((~group_mask).sum()))

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(prompts),
        np.full(len(prompts), 1.0 / len(prompts)),
        size=n_boot,
    )
    selected_sum_arr = weights @ np.asarray(selected_sum, dtype=np.float64)
    selected_count_arr = weights @ np.asarray(selected_count, dtype=np.float64)
    rest_sum_arr = weights @ np.asarray(rest_sum, dtype=np.float64)
    rest_count_arr = weights @ np.asarray(rest_count, dtype=np.float64)
    valid = (selected_count_arr > 0) & (rest_count_arr > 0)
    if not valid.any():
        return math.nan, math.nan
    values = (
        selected_sum_arr[valid] / selected_count_arr[valid]
        - rest_sum_arr[valid] / rest_count_arr[valid]
    )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n_boot: int = 10000,
    seed: int = 20260716,
) -> tuple[float, float]:
    grouped = frame.groupby("prompt_id")[value_col].agg(["sum", "count"])
    if grouped.empty:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(grouped),
        np.full(len(grouped), 1.0 / len(grouped)),
        size=n_boot,
    )
    sums = weights @ grouped["sum"].to_numpy(dtype=np.float64)
    counts = weights @ grouped["count"].to_numpy(dtype=np.float64)
    values = sums / counts
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _load_paired(base_csv: Path, renoise_csv: Path) -> pd.DataFrame:
    key = ["prompt_id", "prompt_text", "axis", "seed_idx", "dimension"]
    base = pd.read_csv(base_csv)
    renoise = pd.read_csv(renoise_csv)
    merged = base.merge(
        renoise,
        on=key,
        suffixes=("_base", "_renoise"),
        validate="one_to_one",
    )
    merged["delta"] = merged["score_renoise"] - merged["score_base"]
    wide = merged.pivot_table(
        index=["prompt_id", "prompt_text", "axis", "seed_idx"],
        columns="dimension",
        values=["score_base", "score_renoise", "delta"],
        aggfunc="first",
    )
    wide.columns = [f"{kind}_{dimension}" for kind, dimension in wide.columns]
    wide = wide.reset_index()
    wide["sample_id"] = (
        wide["prompt_id"] + "_seed" + wide["seed_idx"].astype(int).map(lambda x: f"{x:04d}")
    )

    missing = [
        f"{kind}_{dimension}"
        for kind in ("score_base", "score_renoise", "delta")
        for dimension in ALL_DIMS
        if f"{kind}_{dimension}" not in wide.columns
    ]
    if missing:
        raise ValueError(f"Missing paired VBench columns: {missing}")

    for prefix in ("score_base", "score_renoise", "delta"):
        wide[f"{prefix}_no_dynamic6"] = wide[
            [f"{prefix}_{dimension}" for dimension in SIX_DIMS]
        ].mean(axis=1)
        wide[f"{prefix}_all7"] = wide[
            [f"{prefix}_{dimension}" for dimension in ALL_DIMS]
        ].mean(axis=1)

    wide["six_components_up"] = (
        wide[[f"delta_{dimension}" for dimension in SIX_DIMS]] > 0
    ).sum(axis=1)
    wide["win"] = wide["delta_no_dynamic6"] > 0
    wide["robust_win"] = (
        (wide["delta_no_dynamic6"] > 0.002)
        & (wide["six_components_up"] >= 4)
    )
    return wide.sort_values(["prompt_id", "seed_idx"]).reset_index(drop=True)


def _group_row(name: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    wins = int(frame["win"].sum())
    robust = int(frame["robust_win"].sum())
    win_lo, win_hi = _wilson(wins, len(frame))
    delta_lo, delta_hi = _bootstrap_ci(frame["delta_no_dynamic6"].to_numpy())
    prompt_delta_lo, prompt_delta_hi = _cluster_bootstrap_mean(
        frame,
        value_col="delta_no_dynamic6",
    )
    return {
        "group": name,
        "n": len(frame),
        "baseline_mean": frame["score_base_no_dynamic6"].mean(),
        "renoise_mean": frame["score_renoise_no_dynamic6"].mean(),
        "mean_delta": frame["delta_no_dynamic6"].mean(),
        "median_delta": frame["delta_no_dynamic6"].median(),
        "mean_delta_ci_low": delta_lo,
        "mean_delta_ci_high": delta_hi,
        "mean_delta_prompt_ci_low": prompt_delta_lo,
        "mean_delta_prompt_ci_high": prompt_delta_hi,
        "win_rate": wins / len(frame) if len(frame) else math.nan,
        "win_ci_low": win_lo,
        "win_ci_high": win_hi,
        "robust_win_rate": robust / len(frame) if len(frame) else math.nan,
        "mean_components_up": frame["six_components_up"].mean(),
        "mean_iq_delta": frame["delta_imaging_quality"].mean(),
        "mean_dynamic_delta": frame["delta_dynamic_degree"].mean(),
        "dynamic_loss_rate": (frame["delta_dynamic_degree"] < 0).mean(),
        "dynamic_gain_rate": (frame["delta_dynamic_degree"] > 0).mean(),
    }


def _dimension_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in [*SIX_DIMS, "dynamic_degree", "no_dynamic6", "all7"]:
        delta = frame[f"delta_{dimension}"]
        wins = int((delta > 0).sum())
        lo, hi = _bootstrap_ci(delta.to_numpy())
        win_lo, win_hi = _wilson(wins, len(delta))
        rows.append(
            {
                "dimension": dimension,
                "n": len(delta),
                "baseline_mean": frame[f"score_base_{dimension}"].mean(),
                "renoise_mean": frame[f"score_renoise_{dimension}"].mean(),
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "mean_delta_ci_low": lo,
                "mean_delta_ci_high": hi,
                "win_rate": wins / len(delta),
                "win_ci_low": win_lo,
                "win_ci_high": win_hi,
            }
        )
    return pd.DataFrame(rows)


def _badness_groups(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = [_group_row("all", frame)]
    rank = frame["score_base_no_dynamic6"].rank(method="first")
    for fraction in (0.10, 0.20, 0.30, 0.50):
        count = int(round(len(frame) * fraction))
        mask = rank <= count
        selected = frame[mask]
        row = _group_row(f"global_bottom_{int(fraction * 100)}pct", selected)
        diff = (
            selected["delta_no_dynamic6"].mean()
            - frame.loc[~mask, "delta_no_dynamic6"].mean()
        )
        diff_lo, diff_hi = _cluster_bootstrap_difference(
            frame,
            mask,
            value_col="delta_no_dynamic6",
        )
        row.update(
            {
                "delta_vs_rest": diff,
                "delta_vs_rest_ci_low": diff_lo,
                "delta_vs_rest_ci_high": diff_hi,
            }
        )
        rows.append(row)

    for seeds_per_prompt in (1, 2, 3, 5):
        local_rank = frame.groupby("prompt_id")["score_base_no_dynamic6"].rank(
            method="first"
        )
        mask = local_rank <= seeds_per_prompt
        row = _group_row(f"within_prompt_bottom_{seeds_per_prompt}of10", frame[mask])
        diff = (
            frame.loc[mask, "delta_no_dynamic6"].mean()
            - frame.loc[~mask, "delta_no_dynamic6"].mean()
        )
        diff_lo, diff_hi = _cluster_bootstrap_difference(
            frame,
            mask,
            value_col="delta_no_dynamic6",
        )
        row.update(
            {
                "delta_vs_rest": diff,
                "delta_vs_rest_ci_low": diff_lo,
                "delta_vs_rest_ci_high": diff_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_dimension_badness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    count = int(round(len(frame) * 0.30))
    for selector_dims in itertools.combinations(SIX_DIMS, 3):
        outcome_dims = [dimension for dimension in SIX_DIMS if dimension not in selector_dims]
        selector = frame[
            [f"score_base_{dimension}" for dimension in selector_dims]
        ].mean(axis=1)
        outcome_delta = frame[
            [f"delta_{dimension}" for dimension in outcome_dims]
        ].mean(axis=1)
        mask = selector.rank(method="first") <= count
        rows.append(
            {
                "selector_dims": "|".join(selector_dims),
                "outcome_dims": "|".join(outcome_dims),
                "n_bottom": int(mask.sum()),
                "bottom_mean_outcome_delta": outcome_delta[mask].mean(),
                "rest_mean_outcome_delta": outcome_delta[~mask].mean(),
                "delta_vs_rest": outcome_delta[mask].mean() - outcome_delta[~mask].mean(),
                "bottom_outcome_win_rate": float((outcome_delta[mask] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--renoise-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = _load_paired(args.baseline_csv, args.renoise_csv)
    if len(paired) != 150:
        raise ValueError(f"Expected 150 paired videos, got {len(paired)}")

    dimensions = _dimension_summary(paired)
    badness = _badness_groups(paired)
    cross_dim = _cross_dimension_badness(paired)
    axis = pd.DataFrame(
        [_group_row(axis_name, group) | {"axis": axis_name} for axis_name, group in paired.groupby("axis")]
    ).drop(columns="group")
    prompt = pd.DataFrame(
        [
            _group_row(prompt_id, group)
            | {
                "prompt_id": prompt_id,
                "prompt_text": group["prompt_text"].iloc[0],
                "axis": group["axis"].iloc[0],
            }
            for prompt_id, group in paired.groupby("prompt_id")
        ]
    ).drop(columns="group")

    paired.to_csv(args.output_dir / "paired_video_results.csv", index=False)
    dimensions.to_csv(args.output_dir / "dimension_summary.csv", index=False)
    badness.to_csv(args.output_dir / "bad_video_summary.csv", index=False)
    cross_dim.to_csv(args.output_dir / "cross_dimension_badness.csv", index=False)
    axis.to_csv(args.output_dir / "axis_summary.csv", index=False)
    prompt.to_csv(args.output_dir / "prompt_summary.csv", index=False)
    paired.nlargest(20, "delta_no_dynamic6").to_csv(
        args.output_dir / "top20_gains.csv", index=False
    )
    paired.nsmallest(20, "delta_no_dynamic6").to_csv(
        args.output_dir / "top20_losses.csv", index=False
    )

    corr, corr_p = spearmanr(
        paired["score_base_no_dynamic6"],
        paired["delta_no_dynamic6"],
    )
    summary = {
        "n": len(paired),
        "global": _group_row("all", paired),
        "baseline_delta_spearman": float(corr),
        "baseline_delta_spearman_p": float(corr_p),
        "cross_dimension_bottom30_positive_fraction": float(
            (cross_dim["delta_vs_rest"] > 0).mean()
        ),
        "cross_dimension_bottom30_mean_delta_vs_rest": float(
            cross_dim["delta_vs_rest"].mean()
        ),
        "dynamic_losses": int((paired["delta_dynamic_degree"] < 0).sum()),
        "dynamic_gains": int((paired["delta_dynamic_degree"] > 0).sum()),
        "primary_metric": "mean of six all-video VBench dimensions, excluding dynamic_degree",
        "robust_win": "delta_no_dynamic6 > 0.002 and at least 4 of 6 dimensions improve",
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
