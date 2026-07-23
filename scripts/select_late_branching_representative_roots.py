#!/usr/bin/env python3
"""Select low/median/high baseline-quality roots within every prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    targets = pd.read_csv(args.targets).sort_values(
        ["prompt_id", "vbench_quality", "seed_idx"]
    )
    rows: list[dict] = []
    ranks = ("low", "median", "high")
    for prompt_id, group in targets.groupby("prompt_id", sort=True):
        group = group.reset_index(drop=True)
        indices = (0, len(group) // 2, len(group) - 1)
        for rank, index in zip(ranks, indices, strict=True):
            row = group.iloc[index]
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "root_seed": int(row["seed_idx"]),
                    "baseline_quality_stratum": rank,
                    "baseline_vbench_quality": float(row["vbench_quality"]),
                    "baseline_dynamic_degree": float(row["dynamic_degree"]),
                    "baseline_overall_consistency": float(
                        row["overall_consistency"]
                    ),
                }
            )

    selected = pd.DataFrame(rows)
    if len(selected) != 45 or selected["prompt_id"].nunique() != 15:
        raise RuntimeError(
            f"expected 45 roots across 15 prompts, got {len(selected)} rows"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print(f"wrote {len(selected)} roots to {args.output}")


if __name__ == "__main__":
    main()
