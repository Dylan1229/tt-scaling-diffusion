"""Evaluate causal step-10 Renoise gates with prompt-held-out predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return math.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4 * total)) / total)
        / denom
    )
    return center - radius


def _prompt_bootstrap_precision_ci(
    labels: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 20260716,
) -> tuple[float, float]:
    unique_groups = np.unique(groups)
    successes = []
    counts = []
    for group in unique_groups:
        mask = (groups == group) & accepted
        successes.append(int(labels[mask].sum()))
        counts.append(int(mask.sum()))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(unique_groups),
        np.full(len(unique_groups), 1.0 / len(unique_groups)),
        size=n_boot,
    )
    success_total = weights @ np.asarray(successes, dtype=np.float64)
    count_total = weights @ np.asarray(counts, dtype=np.float64)
    valid = count_total > 0
    if not valid.any():
        return math.nan, math.nan
    precision = success_total[valid] / count_total[valid]
    return float(np.quantile(precision, 0.025)), float(np.quantile(precision, 0.975))


def _load_features(feature_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    raw: list[np.ndarray] = []
    for path in sorted(feature_root.glob("p*/seed*/online_features.npz")):
        with np.load(path) as data:
            names = [str(name) for name in data["scalar_names"]]
            values = data["scalar_values"].astype(np.float32)
            row = dict(zip(names, values, strict=True))
            prompt_id = str(data["prompt_id"])
            seed_idx = int(data["seed_idx"])
            row.update(
                {
                    "sample_id": f"{prompt_id}_seed{seed_idx:04d}",
                    "prompt_id": prompt_id,
                    "prompt_text": str(data["prompt_text"]),
                    "axis": str(data["axis"]),
                    "seed_idx": seed_idx,
                }
            )
            cls = data["cls"].astype(np.float32)
            pieces = []
            for step in cls:
                pieces.extend(
                    [
                        np.quantile(step, 0.10, axis=0),
                        np.quantile(step, 0.50, axis=0),
                        np.quantile(step, 0.90, axis=0),
                        step.mean(axis=0),
                        step.std(axis=0),
                    ]
                )
            pieces.extend(
                [
                    (cls[-1] - cls[0]).mean(axis=0),
                    (cls[-1] - cls[0]).std(axis=0),
                ]
            )
            rows.append(row)
            raw.append(np.concatenate(pieces).astype(np.float32))
    if not rows:
        raise FileNotFoundError(f"No online_features.npz files under {feature_root}")
    return pd.DataFrame(rows), np.stack(raw)


def _threshold_from_inner_oof(
    probabilities: np.ndarray,
    labels: np.ndarray,
    deltas: np.ndarray,
    *,
    min_accept: int,
) -> float:
    candidates = np.unique(probabilities)
    best: tuple[float, float, float, float] | None = None
    best_threshold = 1.1
    for threshold in candidates:
        accepted = probabilities >= threshold
        count = int(accepted.sum())
        if count < min_accept:
            continue
        successes = int(labels[accepted].sum())
        precision = successes / count
        lower = _wilson_lower(successes, count)
        mean_delta = float(deltas[accepted].mean())
        key = (lower, precision, mean_delta, threshold)
        if best is None or key > best:
            best = key
            best_threshold = float(threshold)
    return best_threshold


def _model_specs(
    scalar_columns: list[str],
    dino_columns: list[str],
    pixel_columns: list[str],
    trajectory_columns: list[str],
    scalar_axis_columns: list[str],
    raw_dim: int,
) -> dict[str, tuple[object, list[str] | None, bool]]:
    def logreg_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=20260716,
                    ),
                ),
            ]
        )

    scalar_logreg = logreg_pipeline()
    scalar_axis = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "scalar",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            scalar_columns,
                        ),
                        (
                            "axis",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["axis"],
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20260716,
                ),
            ),
        ]
    )
    prompt_only = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "axis",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["axis"],
                        )
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20260716,
                ),
            ),
        ]
    )
    hist = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=2,
                    max_iter=150,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=20260716,
                ),
            ),
        ]
    )
    trees = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=3,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=20260716,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    shallow_tree = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=20260716,
                ),
            ),
        ]
    )
    raw_pca = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(12, raw_dim), whiten=True, random_state=20260716)),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20260716,
                ),
            ),
        ]
    )
    return {
        "prompt_axis_only": (prompt_only, ["axis"], False),
        "online_dino_logreg": (logreg_pipeline(), dino_columns, False),
        "online_pixel_motion_logreg": (logreg_pipeline(), pixel_columns, False),
        "online_trajectory_logreg": (logreg_pipeline(), trajectory_columns, False),
        "online_dino_pixel_logreg": (scalar_logreg, scalar_columns, False),
        "online_dino_pixel_axis_logreg": (scalar_axis, scalar_axis_columns, False),
        "online_dino_pixel_hist": (hist, scalar_columns, False),
        "online_shallow_tree": (shallow_tree, scalar_columns, False),
        "online_dino_pixel_trees": (trees, scalar_columns, False),
        "online_raw_dino_pca": (raw_pca, None, True),
    }


def _predict_probability(model, x_train, y_train, x_test) -> np.ndarray:
    fitted = clone(model).fit(x_train, y_train)
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(x_test)[:, 1]
    decision = fitted.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def _evaluate_model(
    name: str,
    model,
    features: pd.DataFrame,
    raw_features: np.ndarray,
    labels: np.ndarray,
    deltas: np.ndarray,
    robust_labels: np.ndarray,
    dynamic_deltas: np.ndarray,
    groups: np.ndarray,
    columns: list[str] | None,
    use_raw: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    oof_probability = np.full(len(labels), np.nan)
    accepted_by_min = {
        min_accept: np.zeros(len(labels), dtype=bool)
        for min_accept in (5, 10, 20, 30)
    }
    thresholds_by_min = {
        min_accept: np.full(len(labels), np.nan)
        for min_accept in accepted_by_min
    }

    for train_idx, test_idx in logo.split(np.zeros(len(labels)), labels, groups):
        train_groups = groups[train_idx]
        unique_train_groups = np.unique(train_groups)
        inner = GroupKFold(n_splits=min(5, len(unique_train_groups)))
        x_all = raw_features if use_raw else features[columns]
        x_train = x_all[train_idx] if use_raw else x_all.iloc[train_idx]
        x_test = x_all[test_idx] if use_raw else x_all.iloc[test_idx]
        inner_probability = cross_val_predict(
            clone(model),
            x_train,
            labels[train_idx],
            groups=train_groups,
            cv=inner,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        test_probability = _predict_probability(
            model,
            x_train,
            labels[train_idx],
            x_test,
        )
        oof_probability[test_idx] = test_probability
        for min_accept in accepted_by_min:
            threshold = _threshold_from_inner_oof(
                inner_probability,
                labels[train_idx],
                deltas[train_idx],
                min_accept=min_accept,
            )
            accepted_by_min[min_accept][test_idx] = test_probability >= threshold
            thresholds_by_min[min_accept][test_idx] = threshold

    prediction_rows = features[
        ["sample_id", "prompt_id", "prompt_text", "axis", "seed_idx"]
    ].copy()
    prediction_rows["model"] = name
    prediction_rows["oof_probability"] = oof_probability
    prediction_rows["win"] = labels
    prediction_rows["robust_win"] = robust_labels
    prediction_rows["delta_no_dynamic6"] = deltas
    prediction_rows["delta_dynamic_degree"] = dynamic_deltas
    for min_accept, accepted in accepted_by_min.items():
        prediction_rows[f"accepted_min{min_accept}"] = accepted
        prediction_rows[f"threshold_min{min_accept}"] = thresholds_by_min[min_accept]

    rows = []
    try:
        auc = roc_auc_score(labels, oof_probability)
    except ValueError:
        auc = math.nan
    ap = average_precision_score(labels, oof_probability)
    for min_accept, accepted in accepted_by_min.items():
        count = int(accepted.sum())
        successes = int(labels[accepted].sum()) if count else 0
        robust_successes = int(robust_labels[accepted].sum()) if count else 0
        prompt_low, prompt_high = _prompt_bootstrap_precision_ci(
            labels,
            accepted,
            groups,
        )
        wilson_low = _wilson_lower(successes, count)
        rows.append(
            {
                "model": name,
                "inner_min_accept": min_accept,
                "accepted": count,
                "coverage": count / len(labels),
                "win_rate": successes / count if count else math.nan,
                "win_rate_wilson_low": wilson_low,
                "win_rate_prompt_boot_low": prompt_low,
                "win_rate_prompt_boot_high": prompt_high,
                "win_rate_conservative_low": min(wilson_low, prompt_low),
                "accepted_prompts": int(np.unique(groups[accepted]).size),
                "robust_win_rate": robust_successes / count if count else math.nan,
                "mean_delta": deltas[accepted].mean() if count else math.nan,
                "median_delta": np.median(deltas[accepted]) if count else math.nan,
                "mean_dynamic_delta": dynamic_deltas[accepted].mean() if count else math.nan,
                "dynamic_loss_rate": (
                    (dynamic_deltas[accepted] < 0).mean() if count else math.nan
                ),
                "oof_roc_auc": auc,
                "oof_average_precision": ap,
                "base_win_rate": labels.mean(),
            }
        )
    return pd.DataFrame(rows), prediction_rows


def _posthoc_topk_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize the OOF ranking ceiling without treating it as a validated gate."""
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        ranked = group.sort_values(
            ["oof_probability", "sample_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
        for top_k in range(1, len(ranked) + 1):
            accepted = ranked.iloc[:top_k]
            successes = int(accepted["win"].sum())
            robust_successes = int(accepted["robust_win"].sum())
            rows.append(
                {
                    "model": model,
                    "top_k": top_k,
                    "accepted_prompts": int(accepted["prompt_id"].nunique()),
                    "win_rate": successes / top_k,
                    "win_rate_wilson_low": _wilson_lower(successes, top_k),
                    "robust_win_rate": robust_successes / top_k,
                    "mean_delta": float(accepted["delta_no_dynamic6"].mean()),
                    "median_delta": float(accepted["delta_no_dynamic6"].median()),
                    "mean_dynamic_delta": float(
                        accepted["delta_dynamic_degree"].mean()
                    ),
                    "dynamic_loss_rate": float(
                        (accepted["delta_dynamic_degree"] < 0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--paired-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, raw_features = _load_features(args.features_dir)
    paired = pd.read_csv(args.paired_csv)
    data = features.merge(
        paired[
            [
                "sample_id",
                "delta_no_dynamic6",
                "win",
                "robust_win",
                "delta_dynamic_degree",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    ).sort_values(["prompt_id", "seed_idx"]).reset_index(drop=True)
    raw_order = features["sample_id"].tolist()
    raw_lookup = {sample_id: index for index, sample_id in enumerate(raw_order)}
    raw_features = np.stack([raw_features[raw_lookup[sample_id]] for sample_id in data["sample_id"]])
    if len(data) != 150:
        raise ValueError(f"Expected 150 feature/target pairs, got {len(data)}")

    metadata = {"sample_id", "prompt_id", "prompt_text", "axis", "seed_idx"}
    scalar_columns = sorted(
        column
        for column in features.columns
        if column not in metadata
    )
    specs = _model_specs(
        scalar_columns,
        [
            column
            for column in scalar_columns
            if "dino" in column
            or column.startswith("trajectory_step5_to_step10")
        ],
        [
            column
            for column in scalar_columns
            if "dino" not in column
            and not column.startswith("trajectory_step5_to_step10")
        ],
        [column for column in scalar_columns if column.startswith("trajectory_")],
        [*scalar_columns, "axis"],
        raw_features.shape[1],
    )
    labels = data["win"].astype(int).to_numpy()
    robust_labels = data["robust_win"].astype(int).to_numpy()
    deltas = data["delta_no_dynamic6"].to_numpy(dtype=np.float64)
    dynamic_deltas = data["delta_dynamic_degree"].to_numpy(dtype=np.float64)
    groups = data["prompt_id"].to_numpy()

    all_results = []
    all_predictions = []
    for name, (model, columns, use_raw) in specs.items():
        print(f"[renoise_verifier] model={name}")
        results, predictions = _evaluate_model(
            name,
            model,
            data,
            raw_features,
            labels,
            deltas,
            robust_labels,
            dynamic_deltas,
            groups,
            columns,
            use_raw,
        )
        all_results.append(results)
        all_predictions.append(predictions)

    results = pd.concat(all_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    results = results.sort_values(
        ["win_rate_conservative_low", "win_rate", "accepted"],
        ascending=[False, False, False],
    )
    results.to_csv(args.output_dir / "verifier_results.csv", index=False)
    predictions.to_csv(args.output_dir / "verifier_oof_predictions.csv", index=False)
    posthoc_topk = _posthoc_topk_curve(predictions)
    posthoc_topk.to_csv(
        args.output_dir / "verifier_posthoc_topk.csv",
        index=False,
    )

    interpretable = clone(specs["online_shallow_tree"][0]).fit(
        data[scalar_columns],
        labels,
    )
    tree_model = interpretable.named_steps["model"]
    imputer = interpretable.named_steps["impute"]
    rule_text = export_text(
        tree_model,
        feature_names=list(imputer.get_feature_names_out(scalar_columns)),
        decimals=5,
    )
    (args.output_dir / "shallow_tree_full_data_rule.txt").write_text(
        "Exploratory deployable candidate fitted on all 150 labels.\n"
        "Use prompt-held-out rows in verifier_results.csv for performance claims.\n\n"
        + rule_text
    )

    best = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in results.iloc[0].to_dict().items()
    }
    summary = {
        "n": len(data),
        "base_win_rate": float(labels.mean()),
        "best_prompt_heldout_gate": best,
        "validation": (
            "Outer leave-one-prompt-out predictions. Each fold selects its gate "
            "threshold using only inner GroupKFold predictions from training prompts."
        ),
        "online_inputs": (
            "Prompt axis plus step-5/step-10 decoded posterior DINO trajectories "
            "and lightweight pixel/motion statistics. No final video, VBench, "
            "or intervention output is used."
        ),
        "posthoc_topk_warning": (
            "verifier_posthoc_topk.csv ranks outer-fold OOF predictions but selects "
            "top-k after observing all OOF labels. It is an exploratory signal ceiling, "
            "not a threshold-validated deployment claim."
        ),
    }
    (args.output_dir / "verifier_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
