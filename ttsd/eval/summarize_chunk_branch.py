"""Summarize chunk-branch VBench-Long results against independent concat."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _manifest(path: Path) -> dict[tuple[str, int], dict]:
    rows = _read_csv(path)
    return {(row["prompt_id"], int(row["seed_idx"])): row for row in rows}


def _branch_scores(score_csv: Path, manifest_csv: Path) -> dict:
    manifest = _manifest(manifest_csv)
    scores: dict[tuple[str, int, str, str], dict[str, float]] = defaultdict(dict)
    info: dict[tuple[str, int, str, str], dict] = {}
    for row in _read_csv(score_csv):
        key = (row["prompt_id"], int(row["seed_idx"]))
        meta = manifest.get(key)
        if meta is None:
            raise KeyError(f"No manifest row for prompt={row['prompt_id']} seed_idx={row['seed_idx']}")
        path_key = (
            row["prompt_id"],
            int(meta["root_seed"]),
            str(meta["path_bits"]),
            str(meta["path_id"]),
        )
        scores[path_key][row["dimension"]] = float(row["score"])
        info[path_key] = {
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "root_seed": int(meta["root_seed"]),
            "path_bits": str(meta["path_bits"]),
            "path_id": int(meta["path_id"]),
            "seed_idx": int(row["seed_idx"]),
        }

    for path_key, dim_scores in scores.items():
        present = [dim_scores[dim] for dim in DEFAULT_DIMS if dim in dim_scores]
        if present:
            dim_scores["overall_mean"] = statistics.fmean(present)
    return {"scores": scores, "info": info}


def _independent_scores(score_csv: Path) -> dict[tuple[str, int, str], float]:
    by_prompt_seed: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in _read_csv(score_csv):
        prompt_id = row["prompt_id"]
        seed = int(row["seed_idx"])
        dim = row["dimension"]
        by_prompt_seed[(prompt_id, seed)][dim] = float(row["score"])

    out: dict[tuple[str, int, str], float] = {}
    for (prompt_id, seed), dim_scores in by_prompt_seed.items():
        for dim, score in dim_scores.items():
            out[(prompt_id, seed, dim)] = score
        present = [dim_scores[dim] for dim in DEFAULT_DIMS if dim in dim_scores]
        if present:
            out[(prompt_id, seed, "overall_mean")] = statistics.fmean(present)
    return out


def _best_path(path_scores: dict[str, dict[str, float]], dimension: str) -> tuple[str, float]:
    return max(
        ((bits, scores[dimension]) for bits, scores in path_scores.items() if dimension in scores),
        key=lambda item: (item[1], item[0]),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-run", required=True, type=Path)
    parser.add_argument("--branch-scores", required=True, type=Path)
    parser.add_argument("--independent-scores", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vbench_long_compare/chunk_branch_i2v_p01_p02_s0_1"))
    args = parser.parse_args(argv)

    manifest_csv = args.branch_run / "branch_manifest.csv"
    if not manifest_csv.exists():
        raise SystemExit(f"Missing branch manifest: {manifest_csv}")

    branch = _branch_scores(args.branch_scores / "vbench_long_scores_long.csv", manifest_csv)
    independent = _independent_scores(args.independent_scores / "vbench_long_scores_long.csv")

    dims = list(DEFAULT_DIMS) + ["overall_mean"]
    by_root: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    root_info: dict[tuple[str, int], dict] = {}
    by_path_rows = []
    for path_key, dim_scores in branch["scores"].items():
        prompt_id, root_seed, path_bits, _path_id = path_key
        info = branch["info"][path_key]
        root_key = (prompt_id, root_seed)
        by_root[root_key][path_bits] = dim_scores
        root_info[root_key] = info
        for dim in dims:
            if dim not in dim_scores:
                continue
            independent_value = independent.get((prompt_id, root_seed, dim))
            by_path_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": info["prompt_text"],
                    "root_seed": root_seed,
                    "seed_idx": info["seed_idx"],
                    "path_bits": path_bits,
                    "path_id": info["path_id"],
                    "dimension": dim,
                    "score": _fmt(dim_scores[dim]),
                    "is_all_zero": str(path_bits == "00000").lower(),
                    "independent_concat": _fmt(independent_value),
                    "score_minus_independent": _fmt(
                        dim_scores[dim] - independent_value if independent_value is not None else None
                    ),
                }
            )

    root_rows = []
    for root_key, path_scores in sorted(by_root.items()):
        prompt_id, root_seed = root_key
        info = root_info[root_key]
        overall_bits, overall_best = _best_path(path_scores, "overall_mean")
        for dim in dims:
            values = [scores[dim] for scores in path_scores.values() if dim in scores]
            if not values:
                continue
            per_metric_bits, per_metric_best = _best_path(path_scores, dim)
            all_zero = path_scores.get("00000", {}).get(dim)
            overall_best_value = path_scores[overall_bits].get(dim)
            independent_value = independent.get((prompt_id, root_seed, dim))
            root_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": info["prompt_text"],
                    "root_seed": root_seed,
                    "dimension": dim,
                    "branch_min": _fmt(min(values)),
                    "branch_mean": _fmt(statistics.fmean(values)),
                    "branch_max": _fmt(max(values)),
                    "branch_std": _fmt(_stdev(values)),
                    "all_zero": _fmt(all_zero),
                    "overall_best_path_value": _fmt(overall_best_value),
                    "overall_best_path_bits": overall_bits,
                    "overall_best_overall_mean": _fmt(overall_best),
                    "per_metric_best_value": _fmt(per_metric_best),
                    "per_metric_best_path_bits": per_metric_bits,
                    "independent_concat": _fmt(independent_value),
                    "max_minus_independent": _fmt(
                        per_metric_best - independent_value if independent_value is not None else None
                    ),
                    "overall_best_minus_independent": _fmt(
                        overall_best_value - independent_value
                        if independent_value is not None and overall_best_value is not None
                        else None
                    ),
                    "all_zero_minus_independent": _fmt(
                        all_zero - independent_value
                        if independent_value is not None and all_zero is not None
                        else None
                    ),
                    "mean_minus_independent": _fmt(
                        statistics.fmean(values) - independent_value if independent_value is not None else None
                    ),
                }
            )

    summary_rows = []
    for dim in dims:
        rows = [row for row in root_rows if row["dimension"] == dim and row["independent_concat"] != ""]
        if not rows:
            continue
        max_deltas = [float(row["max_minus_independent"]) for row in rows]
        overall_deltas = [float(row["overall_best_minus_independent"]) for row in rows]
        all_zero_deltas = [float(row["all_zero_minus_independent"]) for row in rows]
        mean_deltas = [float(row["mean_minus_independent"]) for row in rows]
        summary_rows.append(
            {
                "dimension": dim,
                "n_roots": len(rows),
                "branch_max_beats_independent_frac": _fmt(sum(delta > 0 for delta in max_deltas) / len(rows)),
                "overall_best_beats_independent_frac": _fmt(
                    sum(delta > 0 for delta in overall_deltas) / len(rows)
                ),
                "all_zero_beats_independent_frac": _fmt(sum(delta > 0 for delta in all_zero_deltas) / len(rows)),
                "avg_max_minus_independent": _fmt(statistics.fmean(max_deltas)),
                "avg_overall_best_minus_independent": _fmt(statistics.fmean(overall_deltas)),
                "avg_all_zero_minus_independent": _fmt(statistics.fmean(all_zero_deltas)),
                "avg_mean_minus_independent": _fmt(statistics.fmean(mean_deltas)),
            }
        )

    out = args.output_dir
    _write_csv(
        out / "chunk_branch_by_path.csv",
        sorted(by_path_rows, key=lambda r: (r["prompt_id"], int(r["root_seed"]), int(r["path_id"]), r["dimension"])),
        [
            "prompt_id",
            "prompt_text",
            "root_seed",
            "seed_idx",
            "path_bits",
            "path_id",
            "dimension",
            "score",
            "is_all_zero",
            "independent_concat",
            "score_minus_independent",
        ],
    )
    _write_csv(
        out / "chunk_branch_by_root.csv",
        root_rows,
        [
            "prompt_id",
            "prompt_text",
            "root_seed",
            "dimension",
            "branch_min",
            "branch_mean",
            "branch_max",
            "branch_std",
            "all_zero",
            "overall_best_path_value",
            "overall_best_path_bits",
            "overall_best_overall_mean",
            "per_metric_best_value",
            "per_metric_best_path_bits",
            "independent_concat",
            "max_minus_independent",
            "overall_best_minus_independent",
            "all_zero_minus_independent",
            "mean_minus_independent",
        ],
    )
    _write_csv(
        out / "chunk_branch_summary.csv",
        summary_rows,
        [
            "dimension",
            "n_roots",
            "branch_max_beats_independent_frac",
            "overall_best_beats_independent_frac",
            "all_zero_beats_independent_frac",
            "avg_max_minus_independent",
            "avg_overall_best_minus_independent",
            "avg_all_zero_minus_independent",
            "avg_mean_minus_independent",
        ],
    )
    print(f"[chunk-branch-summary] wrote {out}")


if __name__ == "__main__":
    main()
