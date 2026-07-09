"""Evaluate root-normalized reward fusions on saved branch leaves.

This module assumes reward score CSVs have already been produced for the saved
branch run.  It does not generate videos and does not run VBench.  It maps one
selected path per root back to existing VBench-Long path scores.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMPARE_DIR = Path("runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1")
DEFAULT_OUTPUT_SUBDIR = "root_normalized_reward_fusions"

DEFAULT_FORMULAS: dict[str, dict[str, float]] = {
    "root_norm_clipiqa_ta": {
        "clipiqa_mean": 1.0,
        "vr8_ta": 0.5,
    },
    "root_norm_clipiqa_ta_openclip_mq": {
        "clipiqa_mean": 1.0,
        "vr8_ta": 1.25,
        "openclip_min": 0.5,
        "vr8_mq": -0.25,
    },
    "root_norm_quality_ensemble_ta_openclip_mq": {
        "clipiqa_mean": 0.4,
        "clipiqaplus_rn_robust": 0.2,
        "liqe_mean": 0.2,
        "nima_koniq_mean": 0.2,
        "vr8_ta": 0.25,
        "openclip_min": 0.25,
        "vr8_mq": -0.25,
    },
    "root_norm_clipiqa_ta_openclip_mq_temporal": {
        "clipiqa_mean": 1.0,
        "vr8_ta": 1.25,
        "openclip_min": 0.5,
        "vr8_mq": -0.25,
        "clip_img_adj_mean": 0.25,
        "pixel_diff_mean": 0.25,
    },
    "root_norm_clean_quality_alignment_temporal": {
        "clipiqa_mean": 1.0,
        "vr8_ta": 0.75,
        "openclip_min": 0.5,
        "clip_img_adj_mean": 0.25,
        "pixel_diff_mean": 0.25,
        "nima_koniq_score_mean": 0.25,
    },
    "root_norm_positive_beam_no_mq": {
        "clipiqa_max": 0.125,
        "vr8_ta": 0.25,
        "openclip_min": 0.125,
        "clipiqaplus_score_mean": 0.375,
        "pixel_diff_mean": 0.375,
        "pixel_diff_min": 0.25,
    },
    "root_norm_fast_beam_video_quality": {
        "clipiqa_mean": 0.875,
        "vr8_ta": 1.0,
        "vr8_mq": -0.75,
        "clip_img_adj_min": 0.375,
        "clip_img_all_min": 0.75,
        "pixel_diff_mean": 0.75,
        "nima_koniq_score_mean": 0.5,
        "niqe_score_mean": 0.25,
    },
}

VBENCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_mean",
)


@dataclass(frozen=True)
class Config:
    compare_dir: Path
    output_dir: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else float("nan")


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _feature_key(row: dict[str, str]) -> tuple[str, int, str]:
    return (row["prompt_id"], int(row["root_seed"]), row["path_bits"])


def _load_path_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    rows = _read_csv(compare_dir / "chunk_branch_by_path.csv")
    out: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[_feature_key(row)][row["dimension"]] = float(row["score"])
    return out


def _load_root_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    return {
        (row["prompt_id"], int(row["root_seed"]), row["dimension"]): row
        for row in _read_csv(compare_dir / "chunk_branch_by_root.csv")
    }


def _load_features(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    features: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    frozen_dir = compare_dir / "frozen_reward_selectors"
    pyiqa_dir = frozen_dir / "pyiqa_sweep"

    clipiqa_path = frozen_dir / "clipiqa_scores.csv"
    if clipiqa_path.exists():
        for row in _read_csv(clipiqa_path):
            key = _feature_key(row)
            for feature in (
                "clipiqa_mean",
                "clipiqa_min",
                "clipiqa_max",
                "clipiqa_std",
                "clipiqa_mean_minus_std",
            ):
                if feature in row and row[feature] != "":
                    features[key][feature] = float(row[feature])

    videoreward_path = frozen_dir / "video_reward_scores.csv"
    if videoreward_path.exists():
        for row in _read_csv(videoreward_path):
            key = _feature_key(row)
            features[key]["vr8_ta"] = float(row["TA"])
            features[key]["vr8_mq"] = float(row["MQ"])
            features[key]["vr8_overall"] = float(row["Overall"])

    openclip_path = frozen_dir / "openclip_scores.csv"
    if openclip_path.exists():
        for row in _read_csv(openclip_path):
            key = _feature_key(row)
            features[key]["openclip_min"] = float(row["clip_min"])
            features[key]["openclip_mean"] = float(row["clip_mean"])

    dover_path = frozen_dir / "dover_mobile" / "dover_mobile_scores.csv"
    if dover_path.exists():
        for row in _read_csv(dover_path):
            key = _feature_key(row)
            features[key]["dover_technical"] = float(row["dover_technical"])
            features[key]["dover_aesthetic"] = float(row["dover_aesthetic"])
            features[key]["dover_fused"] = float(row["dover_fused"])

    temporal_path = frozen_dir / "temporal_clip" / "temporal_clip_scores.csv"
    temporal_features = (
        "clip_img_adj_mean",
        "clip_img_adj_min",
        "clip_img_first_last",
        "clip_img_all_mean",
        "clip_img_all_min",
        "clip_img_adj_std",
        "pixel_diff_mean",
        "pixel_diff_std",
        "pixel_diff_min",
        "pixel_diff_max",
    )
    if temporal_path.exists():
        for row in _read_csv(temporal_path):
            key = _feature_key(row)
            for feature in temporal_features:
                if feature in row and row[feature] != "":
                    features[key][feature] = float(row[feature])

    sweep_files = {
        "clipiqaplus": ("clipiqaplus_scores.csv", "clipiqaplus"),
        "clipiqaplus_rn": ("clipiqaplus_rn50_512_scores.csv", "clipiqaplus_rn50_512"),
        "liqe": ("liqe_scores.csv", "liqe"),
        "liqe_mix": ("liqe_mix_scores.csv", "liqe_mix"),
        "topiq": ("topiq_nr_scores.csv", "topiq_nr"),
        "dbcnn": ("dbcnn_scores.csv", "dbcnn"),
        "hyperiqa": ("hyperiqa_scores.csv", "hyperiqa"),
        "nima": ("nima_scores.csv", "nima"),
        "nima_koniq": ("nima_koniq_scores.csv", "nima_koniq"),
        "paq2piq": ("paq2piq_scores.csv", "paq2piq"),
        "brisque": ("brisque_scores.csv", "brisque"),
        "niqe": ("niqe_scores.csv", "niqe"),
    }
    score_fields = (
        "score_mean",
        "score_min",
        "score_max",
        "score_std",
        "score_mean_minus_std",
        "score_mean_plus_std",
        "select_mean",
        "select_min",
        "select_robust",
    )
    for prefix, (filename, raw_prefix) in sweep_files.items():
        path = pyiqa_dir / filename
        if not path.exists():
            continue
        for row in _read_csv(path):
            key = _feature_key(row)
            features[key][f"{prefix}_mean"] = float(row["select_mean"])
            features[key][f"{prefix}_min"] = float(row["select_min"])
            features[key][f"{prefix}_robust"] = float(row["select_robust"])
            for field in score_fields:
                if field not in row or row[field] == "":
                    continue
                features[key][f"{raw_prefix}_{field}"] = float(row[field])

    return features


def _roots(path_scores: dict[tuple[str, int, str], dict[str, float]]) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = defaultdict(list)
    for prompt_id, root_seed, path_bits in path_scores:
        out[(prompt_id, root_seed)].append(path_bits)
    for path_bits in out.values():
        path_bits.sort()
    return dict(sorted(out.items()))


def _zscore(
    features: dict[tuple[str, int, str], dict[str, float]],
    root_paths: dict[tuple[str, int], list[str]],
    root: tuple[str, int],
    path_bits: str,
    feature: str,
) -> float:
    values = [
        features.get((root[0], root[1], bits), {}).get(feature)
        for bits in root_paths[root]
    ]
    vals = [value for value in values if value is not None]
    key = (root[0], root[1], path_bits)
    if not vals or feature not in features.get(key, {}):
        return 0.0
    mu = _mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if sd <= 1e-12:
        return 0.0
    return (features[key][feature] - mu) / sd


def _select_paths(
    formula_name: str,
    weights: dict[str, float],
    root_paths: dict[tuple[str, int], list[str]],
    features: dict[tuple[str, int, str], dict[str, float]],
) -> list[dict[str, str]]:
    rows = []
    for root, paths in root_paths.items():
        scored = []
        for path_bits in paths:
            score = sum(
                weight * _zscore(features, root_paths, root, path_bits, feature)
                for feature, weight in weights.items()
            )
            scored.append((score, path_bits))
        selected_score, selected_bits = max(scored, key=lambda item: (item[0], -int(item[1], 2)))
        rows.append(
            {
                "formula": formula_name,
                "prompt_id": root[0],
                "root_seed": str(root[1]),
                "selected_path_bits": selected_bits,
                "fusion_score": _fmt(selected_score),
            }
        )
    return rows


def _evaluate(cfg: Config) -> None:
    path_scores = _load_path_scores(cfg.compare_dir)
    root_scores = _load_root_scores(cfg.compare_dir)
    features = _load_features(cfg.compare_dir)
    root_paths = _roots(path_scores)

    by_root_rows = []
    selection_rows = []
    for formula_name, weights in DEFAULT_FORMULAS.items():
        selected = _select_paths(formula_name, weights, root_paths, features)
        selection_rows.extend(selected)
        for row in selected:
            root = (row["prompt_id"], int(row["root_seed"]))
            path_bits = row["selected_path_bits"]
            key = (root[0], root[1], path_bits)
            for dim in VBENCH_DIMS:
                root_row = root_scores[(root[0], root[1], dim)]
                value = path_scores[key][dim]
                independent = float(root_row["independent_concat"])
                all_zero = float(root_row["all_zero"])
                branch_mean = float(root_row["branch_mean"])
                oracle = float(root_row["branch_max"])
                by_root_rows.append(
                    {
                        "formula": formula_name,
                        "prompt_id": root[0],
                        "root_seed": str(root[1]),
                        "selected_path_bits": path_bits,
                        "fusion_score": row["fusion_score"],
                        "dimension": dim,
                        "selected": _fmt(value),
                        "independent_concat": _fmt(independent),
                        "all_zero": _fmt(all_zero),
                        "branch_mean": _fmt(branch_mean),
                        "oracle_best_of_32": _fmt(oracle),
                        "selected_minus_independent": _fmt(value - independent),
                        "selected_minus_all_zero": _fmt(value - all_zero),
                        "selected_minus_branch_mean": _fmt(value - branch_mean),
                        "gap_to_oracle": _fmt(oracle - value),
                    }
                )

    summary_rows = []
    for formula_name in DEFAULT_FORMULAS:
        for dim in VBENCH_DIMS:
            rows = [
                row
                for row in by_root_rows
                if row["formula"] == formula_name and row["dimension"] == dim
            ]
            selected = [float(row["selected"]) for row in rows]
            independent = [float(row["independent_concat"]) for row in rows]
            all_zero = [float(row["all_zero"]) for row in rows]
            branch_mean = [float(row["branch_mean"]) for row in rows]
            oracle = [float(row["oracle_best_of_32"]) for row in rows]
            summary_rows.append(
                {
                    "formula": formula_name,
                    "dimension": dim,
                    "num_roots": str(len(rows)),
                    "selected_mean": _fmt(_mean(selected)),
                    "independent_mean": _fmt(_mean(independent)),
                    "all_zero_mean": _fmt(_mean(all_zero)),
                    "branch_mean": _fmt(_mean(branch_mean)),
                    "oracle_best_of_32_mean": _fmt(_mean(oracle)),
                    "selected_minus_independent": _fmt(_mean(a - b for a, b in zip(selected, independent))),
                    "selected_minus_all_zero": _fmt(_mean(a - b for a, b in zip(selected, all_zero))),
                    "selected_minus_branch_mean": _fmt(_mean(a - b for a, b in zip(selected, branch_mean))),
                    "gap_to_oracle": _fmt(_mean(a - b for a, b in zip(oracle, selected))),
                }
            )

    _write_csv(
        cfg.output_dir / "fusion_selected_paths.csv",
        selection_rows,
        ["formula", "prompt_id", "root_seed", "selected_path_bits", "fusion_score"],
    )
    _write_csv(
        cfg.output_dir / "fusion_by_root.csv",
        by_root_rows,
        [
            "formula",
            "prompt_id",
            "root_seed",
            "selected_path_bits",
            "fusion_score",
            "dimension",
            "selected",
            "independent_concat",
            "all_zero",
            "branch_mean",
            "oracle_best_of_32",
            "selected_minus_independent",
            "selected_minus_all_zero",
            "selected_minus_branch_mean",
            "gap_to_oracle",
        ],
    )
    _write_csv(
        cfg.output_dir / "fusion_summary.csv",
        summary_rows,
        [
            "formula",
            "dimension",
            "num_roots",
            "selected_mean",
            "independent_mean",
            "all_zero_mean",
            "branch_mean",
            "oracle_best_of_32_mean",
            "selected_minus_independent",
            "selected_minus_all_zero",
            "selected_minus_branch_mean",
            "gap_to_oracle",
        ],
    )
    print(f"wrote {cfg.output_dir / 'fusion_summary.csv'}")


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.compare_dir / DEFAULT_OUTPUT_SUBDIR
    return Config(compare_dir=args.compare_dir, output_dir=output_dir)


def main() -> None:
    _evaluate(_parse_args())


if __name__ == "__main__":
    main()
