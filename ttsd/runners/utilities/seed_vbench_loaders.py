"""Shared loaders over the per-seed run-artifact tree and the VBench long-CSV.

Imported by the analysis and report runners under ``ttsd/runners/``.
"""

from __future__ import annotations

import csv
from pathlib import Path

# VBench dimensions averaged (per-prompt z-normalized) into the avg_vbench_z alignment
# target. Every dimension listed here must be present for every clip in
# vbench_scores_long.csv (run VBench with these dimensions before using them).
SUBSCORES = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
]


def _load_vbench_rows(vbench_long_csv: Path) -> dict[tuple[str, int], dict[str, object]]:
    rows_by_seed: dict[tuple[str, int], dict[str, object]] = {}

    with vbench_long_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt_id = row["prompt_id"]
            prompt_text = row["prompt_text"]
            axis = row["axis"]
            seed_idx = int(row["seed_idx"])
            dimension = row["dimension"]
            score = float(row["score"])
            key = (prompt_id, seed_idx)

            if key not in rows_by_seed:
                rows_by_seed[key] = {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "axis": axis,
                    "seed_idx": seed_idx,
                }

            rows_by_seed[key][dimension] = score

    return rows_by_seed


def _iter_seed_dirs(heatmap_run_root: Path) -> list[Path]:
    return sorted(path for path in heatmap_run_root.glob("p*/seed*") if path.is_dir())


def _seed_idx_from_name(seed_dir: Path) -> int:
    return int(seed_dir.name.replace("seed", ""))
