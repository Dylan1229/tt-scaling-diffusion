"""Shared loaders over the per-seed run-artifact tree and the VBench long-CSV.

Imported by the analysis and report runners under ``ttsd/runners/``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ttsd.eval.vbench import vbench_quality

# VBench dimensions reported as individual alignment targets. Every dimension listed
# here must be present for every clip in vbench_scores_long.csv (run VBench with these
# dimensions before using them).
SUBSCORES = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
]


def annotate_vbench_targets(records: list[dict]) -> None:
    """Set `vbench_quality` on each record, in place.

    Purely per-video: no statistic of the other clips enters, which is what makes the
    score comparable across prompts, seeds and runs. The two composites that used to
    live here — `avg_vbench_z` and `avg_vbench_raw` — are retired.
    """
    for record in records:
        record["vbench_quality"] = vbench_quality(record)


# A 2-seed Spearman is always +/-1 regardless of the data, so cells that small are noise.
MIN_STRATUM_SEEDS = 3

# A best-vs-worst figure only needs two clips to contrast.
MIN_STRATUM_FIGURE_SEEDS = 2


def cell_groups(records: list[dict], dyn: int, min_seeds: int = MIN_STRATUM_SEEDS
                ) -> dict[str, list[int]]:
    """Positions into `records`, grouped by prompt, restricted to one dynamic_degree stratum.

    Keys look like "p04|dyn1". Cells with fewer than `min_seeds` clips are dropped. Returns {}
    when the records carry no dynamic_degree (a run scored before it was added).

    `vbench_quality` rewards stillness (within-prompt Spearman vs dynamic_degree = -0.43), so a
    statistic computed over a whole prompt mixes "this seed is better" with "this seed moves
    less". Recomputing it inside a stratum holds motion fixed.
    """
    if not records or "dynamic_degree" not in records[0]:
        return {}

    groups: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        if int(float(record["dynamic_degree"])) == dyn:
            groups[f"{record['prompt_id']}|dyn{dyn}"].append(i)
    return {k: v for k, v in groups.items() if len(v) >= min_seeds}


def cell_groups_rows(records: list[dict], dyn: int, min_seeds: int = MIN_STRATUM_SEEDS
                     ) -> dict[str, list[dict]]:
    """Row-dict form of `cell_groups`, for the runners whose stats take groups of rows."""
    return {key: [records[i] for i in idx]
            for key, idx in cell_groups(records, dyn, min_seeds).items()}


def prompt_strata(clips: list[dict], min_seeds: int = MIN_STRATUM_FIGURE_SEEDS
                  ) -> list[tuple[int, list[dict]]]:
    """The dynamic_degree strata of one prompt, moving first, each with enough clips to rank.

    A prompt whose seeds all move (8 of 30 here) yields one stratum; one whose seeds differ
    yields two. Strata too small to have both a best and a worst clip are dropped.
    """
    if not clips or "dynamic_degree" not in clips[0]:
        return [(-1, clips)]

    out = []
    for dyn in (1, 0):
        stratum = [c for c in clips if int(float(c["dynamic_degree"])) == dyn]
        if len(stratum) >= min_seeds:
            out.append((dyn, stratum))
    return out


def best_worst(stratum: list[dict]) -> tuple[dict, dict]:
    """Highest- and lowest-`vbench_quality` clip of one stratum."""
    return (max(stratum, key=lambda c: c["vbench_quality"]),
            min(stratum, key=lambda c: c["vbench_quality"]))


def load_legacy_avg_vbench_z(vbench_long_csv: Path) -> dict[tuple[str, int], float]:
    """Read the deprecated `avg_vbench_z` column from the run's vbench_targets.csv.

    Returns {} when that file or column is absent, which is the case for every run
    scored without `--legacy-avg-vbench-z`. Nothing here recomputes it.
    """
    targets_csv = vbench_long_csv.parent / "vbench_targets.csv"
    if not targets_csv.exists():
        return {}

    with targets_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "avg_vbench_z" not in (reader.fieldnames or []):
            return {}
        return {
            (row["prompt_id"], int(row["seed_idx"])): float(row["avg_vbench_z"])
            for row in reader
        }


def _load_vbench_rows(vbench_long_csv: Path) -> dict[tuple[str, int], dict[str, object]]:
    rows_by_seed: dict[tuple[str, int], dict[str, object]] = {}

    with vbench_long_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt_id = row["prompt_id"]
            prompt_text = row["prompt_text"]
            seed_idx = int(row["seed_idx"])
            dimension = row["dimension"]
            score = float(row["score"])
            key = (prompt_id, seed_idx)

            if key not in rows_by_seed:
                rows_by_seed[key] = {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed_idx": seed_idx,
                }

            rows_by_seed[key][dimension] = score

    return rows_by_seed


def _iter_seed_dirs(heatmap_run_root: Path) -> list[Path]:
    return sorted(path for path in heatmap_run_root.glob("p*/seed*") if path.is_dir())


def _seed_idx_from_name(seed_dir: Path) -> int:
    return int(seed_dir.name.replace("seed", ""))
