"""Nested Best-of-M and prompt-held-out online-verifier analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-12
IDENTITY_COLUMNS = {
    "prompt_id",
    "prompt_text",
    "root_seed",
    "candidate_index",
    "candidate_seed",
    "branch_kind",
}


def _read_metas(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run.glob("*/seed*/meta.json")):
        row = json.loads(path.read_text())
        row["video_path"] = str(path.parent / "video.mp4")
        rows.append(row)
    if not rows:
        raise ValueError(f"no candidate metadata under {run}")
    return pd.DataFrame(rows)


def _prompt_features(text: str) -> dict[str, float]:
    words = text.lower().split()
    motion_words = {
        "swimming",
        "drinking",
        "running",
        "walking",
        "dancing",
        "playing",
        "riding",
        "flying",
    }
    scene_words = {"ocean", "cafe", "bedroom", "street", "beach", "room"}
    return {
        "prompt_word_count": float(len(words)),
        "prompt_char_count": float(len(text)),
        "prompt_has_person": float("person" in words or "human" in words),
        "prompt_has_and": float("and" in words),
        "prompt_has_motion_verb": float(bool(set(words) & motion_words)),
        "prompt_has_scene_word": float(bool(set(words) & scene_words)),
    }


def _load_run(
    run: Path,
    targets_path: Path,
    features_path: Path,
    baseline_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    metas = _read_metas(run)
    targets = pd.read_csv(targets_path).rename(columns={"seed_idx": "candidate_seed"})
    features = pd.read_csv(features_path)
    baseline = pd.read_csv(baseline_path).rename(columns={"seed_idx": "root_seed"})

    merged = metas.merge(
        targets,
        left_on=["prompt_id", "seed"],
        right_on=["prompt_id", "candidate_seed"],
        validate="one_to_one",
        suffixes=("_meta", ""),
    )
    merged = merged.merge(
        features,
        on=[
            "prompt_id",
            "prompt_text",
            "root_seed",
            "candidate_index",
            "candidate_seed",
            "branch_kind",
        ],
        validate="one_to_one",
    )
    baseline_columns = [
        "prompt_id",
        "root_seed",
        "vbench_quality",
        "dynamic_degree",
        "overall_consistency",
    ]
    merged = merged.merge(
        baseline[baseline_columns],
        on=["prompt_id", "root_seed"],
        validate="many_to_one",
        suffixes=("", "_baseline"),
    )
    if len(merged) != len(metas):
        raise ValueError("candidate metrics or online features are incomplete")

    raw_features = [
        column
        for column in features.columns
        if column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    controls = merged[merged["branch_kind"] == "batched_control"][
        ["prompt_id", "root_seed", *raw_features]
    ].copy()
    controls = controls.rename(
        columns={column: f"control_{column}" for column in raw_features}
    )
    noises = merged[merged["branch_kind"] == "noise"].copy()
    noises = noises.merge(
        controls, on=["prompt_id", "root_seed"], validate="many_to_one"
    )
    for column in raw_features:
        noises[f"delta_{column}"] = noises[column] - noises[f"control_{column}"]
    prompt_rows = pd.DataFrame(
        [
            {"prompt_text": text, **_prompt_features(text)}
            for text in noises["prompt_text"].unique()
        ]
    )
    noises = noises.merge(prompt_rows, on="prompt_text", validate="many_to_one")

    noises["quality_delta"] = (
        noises["vbench_quality"] - noises["vbench_quality_baseline"]
    )
    noises["dynamic_delta"] = (
        noises["dynamic_degree"] - noises["dynamic_degree_baseline"]
    )
    noises["overall_delta"] = (
        noises["overall_consistency"] - noises["overall_consistency_baseline"]
    )
    noises["quality_win"] = noises["quality_delta"] > EPSILON
    noises["safe_win"] = (
        noises["quality_win"]
        & (noises["dynamic_delta"] >= -EPSILON)
        & (noises["overall_delta"] >= -EPSILON)
    )
    model_features = [
        *raw_features,
        *(f"control_{column}" for column in raw_features),
        *(f"delta_{column}" for column in raw_features),
        *prompt_rows.columns.drop("prompt_text").tolist(),
    ]
    return noises, model_features


def _bootstrap_ci(values: np.ndarray, seed: int = 20260723) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(3000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _nested_curve(
    candidates: pd.DataFrame, m_values: list[int], scope: str, run_name: str
) -> pd.DataFrame:
    rows = []
    for m in m_values:
        root_rows = []
        subset = candidates[candidates["candidate_index"] <= m]
        for _, group in subset.groupby(["prompt_id", "root_seed"], sort=False):
            quality_gain = max(0.0, float(group["quality_delta"].max()))
            safe = group[group["safe_win"]]
            safe_gain = max(0.0, float(safe["quality_delta"].max())) if len(safe) else 0.0
            root_rows.append(
                {
                    "quality_win": float(quality_gain > EPSILON),
                    "safe_win": float(safe_gain > EPSILON),
                    "quality_gain": quality_gain,
                    "safe_gain": safe_gain,
                    "random_quality_win": float(group["quality_win"].mean()),
                    "random_safe_win": float(group["safe_win"].mean()),
                }
            )
        root_frame = pd.DataFrame(root_rows)
        row = {
            "run": run_name,
            "scope": scope,
            "m": m,
            "n_roots": len(root_frame),
        }
        for column in root_frame:
            values = root_frame[column].to_numpy(float)
            low, high = _bootstrap_ci(values, seed=20260723 + m)
            row[column] = float(values.mean())
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _models() -> dict[str, object]:
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                max_iter=3000,
                class_weight="balanced",
                random_state=20260723,
            ),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=6,
            max_features=0.5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260723,
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=9,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=20260723,
        ),
    }


def _inner_oof_probabilities(
    frame: pd.DataFrame, features: list[str], estimator
) -> np.ndarray:
    groups = frame["prompt_id"].to_numpy()
    splits = min(5, frame["prompt_id"].nunique())
    oof = np.full(len(frame), np.nan)
    for train_index, valid_index in GroupKFold(n_splits=splits).split(
        frame, frame["safe_win"], groups
    ):
        model = clone(estimator)
        model.fit(
            frame.iloc[train_index][features],
            frame.iloc[train_index]["safe_win"],
        )
        oof[valid_index] = model.predict_proba(
            frame.iloc[valid_index][features]
        )[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("inner OOF prediction is incomplete")
    return oof


def _top_per_root(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    indices = frame.groupby(["prompt_id", "root_seed"])[probability_column].idxmax()
    return frame.loc[indices].copy()


def _training_threshold(
    top_rows: pd.DataFrame, target_precision: float, minimum_accepts: int = 8
) -> float:
    ordered = top_rows.sort_values("probability", ascending=False).reset_index(drop=True)
    wins = ordered["safe_win"].astype(float).cumsum()
    counts = np.arange(1, len(ordered) + 1)
    precision = wins / counts
    valid = np.flatnonzero(
        (counts >= min(minimum_accepts, len(ordered)))
        & (precision >= target_precision)
    )
    if len(valid) == 0:
        return float("inf")
    accepted_count = int(valid[-1] + 1)
    if accepted_count == len(ordered):
        return float("-inf")
    return float(
        (
            ordered.loc[accepted_count - 1, "probability"]
            + ordered.loc[accepted_count, "probability"]
        )
        / 2
    )


def _selection_rows(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    run_name: str,
    model_name: str,
    outer_prompt: str,
    thresholds: dict[str, float],
) -> list[dict]:
    scored = test.copy()
    scored["probability"] = probabilities
    top = _top_per_root(scored, "probability")
    rows = []
    all_thresholds = {"always_select": float("-inf"), **thresholds}
    for gate, threshold in all_thresholds.items():
        for _, selected in top.iterrows():
            accepted = bool(float(selected["probability"]) >= threshold)
            rows.append(
                {
                    "run": run_name,
                    "model": model_name,
                    "gate": gate,
                    "outer_prompt": outer_prompt,
                    "prompt_id": selected["prompt_id"],
                    "root_seed": int(selected["root_seed"]),
                    "candidate_index": int(selected["candidate_index"]),
                    "candidate_seed": int(selected["candidate_seed"]),
                    "probability": float(selected["probability"]),
                    "threshold": threshold,
                    "accepted": accepted,
                    "quality_win": bool(selected["quality_win"]) if accepted else False,
                    "safe_win": bool(selected["safe_win"]) if accepted else False,
                    "quality_delta": (
                        float(selected["quality_delta"]) if accepted else 0.0
                    ),
                    "dynamic_delta": (
                        float(selected["dynamic_delta"]) if accepted else 0.0
                    ),
                    "overall_delta": (
                        float(selected["overall_delta"]) if accepted else 0.0
                    ),
                    "video_path": selected["video_path"],
                }
            )
    return rows


def _evaluate_verifiers(
    train_source: pd.DataFrame,
    secondary: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for outer_prompt in sorted(train_source["prompt_id"].unique()):
        train = train_source[train_source["prompt_id"] != outer_prompt].copy()
        test_m8 = train_source[train_source["prompt_id"] == outer_prompt].copy()
        test_m16 = secondary[secondary["prompt_id"] == outer_prompt].copy()
        for model_name, estimator in _models().items():
            train = train.copy()
            train["probability"] = _inner_oof_probabilities(
                train, features, estimator
            )
            training_top = _top_per_root(train, "probability")
            thresholds = {
                "train_precision_80": _training_threshold(training_top, 0.80),
                "train_precision_90": _training_threshold(training_top, 0.90),
            }
            fitted = clone(estimator).fit(train[features], train["safe_win"])
            rows.extend(
                _selection_rows(
                    test_m8,
                    fitted.predict_proba(test_m8[features])[:, 1],
                    run_name="M8_full",
                    model_name=model_name,
                    outer_prompt=outer_prompt,
                    thresholds=thresholds,
                )
            )
            if len(test_m16):
                rows.extend(
                    _selection_rows(
                        test_m16,
                        fitted.predict_proba(test_m16[features])[:, 1],
                        run_name="M16_representative",
                        model_name=model_name,
                        outer_prompt=outer_prompt,
                        thresholds=thresholds,
                    )
                )
    return pd.DataFrame(rows)


def _summarize_verifiers(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["run", "model", "gate"], sort=True):
        accepted = group[group["accepted"]]
        rows.append(
            {
                "run": keys[0],
                "model": keys[1],
                "gate": keys[2],
                "n_roots": len(group),
                "accepted": len(accepted),
                "coverage": len(accepted) / len(group),
                "quality_win_rate": (
                    float(accepted["quality_win"].mean()) if len(accepted) else np.nan
                ),
                "safe_win_rate": (
                    float(accepted["safe_win"].mean()) if len(accepted) else np.nan
                ),
                "mean_quality_delta": (
                    float(accepted["quality_delta"].mean()) if len(accepted) else np.nan
                ),
                "mean_dynamic_delta": (
                    float(accepted["dynamic_delta"].mean()) if len(accepted) else np.nan
                ),
                "mean_overall_delta": (
                    float(accepted["overall_delta"].mean()) if len(accepted) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m8-run", required=True, type=Path)
    parser.add_argument("--m8-targets", required=True, type=Path)
    parser.add_argument("--m8-features", required=True, type=Path)
    parser.add_argument("--m16-run", required=True, type=Path)
    parser.add_argument("--m16-targets", required=True, type=Path)
    parser.add_argument("--m16-features", required=True, type=Path)
    parser.add_argument("--baseline-targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    m8, features = _load_run(
        args.m8_run, args.m8_targets, args.m8_features, args.baseline_targets
    )
    m16, m16_features = _load_run(
        args.m16_run, args.m16_targets, args.m16_features, args.baseline_targets
    )
    if features != m16_features:
        raise ValueError("M8 and M16 online feature schemas differ")

    bottom_roots = (
        m8[["prompt_id", "root_seed", "vbench_quality_baseline"]]
        .drop_duplicates()
        .nsmallest(15, "vbench_quality_baseline")[["prompt_id", "root_seed"]]
    )
    bottom_m8 = m8.merge(bottom_roots, on=["prompt_id", "root_seed"])
    curves = pd.concat(
        [
            _nested_curve(m8, [1, 2, 4, 8], "all_150", "M8_full"),
            _nested_curve(bottom_m8, [1, 2, 4, 8], "bottom_15", "M8_full"),
            _nested_curve(
                m16, [1, 2, 4, 8, 16], "representative_45", "M16_representative"
            ),
        ],
        ignore_index=True,
    )

    predictions = _evaluate_verifiers(m8, m16, features)
    verifier_summary = _summarize_verifiers(predictions)

    args.output.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output / "nested_oracle_curves.csv", index=False)
    predictions.to_csv(args.output / "online_verifier_predictions.csv", index=False)
    verifier_summary.to_csv(
        args.output / "online_verifier_summary.csv", index=False
    )
    m8.to_csv(args.output / "m8_candidates.csv", index=False)
    m16.to_csv(args.output / "m16_candidates.csv", index=False)

    summary = {
        "scope": (
            "Step-35 branching only. M8 curves use all 150 roots; M16 curves use "
            "45 roots stratified low/median/high within every prompt."
        ),
        "oracle_warning": (
            "Oracle curves use final updated VBench targets and are opportunity "
            "upper bounds, not online selection performance."
        ),
        "verifier_protocol": (
            "Posterior-mean latent features from global denoising steps 36/38/40; "
            "outer leave-one-prompt-out evaluation; gate thresholds selected only "
            "from inner prompt-grouped OOF training predictions."
        ),
        "n_m8_candidates": len(m8),
        "n_m16_candidates": len(m16),
        "n_online_features": len(features),
    }
    (args.output / "analysis_protocol.json").write_text(json.dumps(summary, indent=2))
    print(f"[nested-analysis] wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
