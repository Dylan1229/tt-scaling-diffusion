"""Combine dimension-parallel VBench outputs into one validated long CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-videos", type=int, default=150)
    args = parser.parse_args()

    paths = sorted(args.parts_root.glob("*/vbench_scores_long.csv"))
    if not paths:
        raise SystemExit(f"No part CSVs found under {args.parts_root}")
    frames = [pd.read_csv(path) for path in paths]
    long = pd.concat(frames, ignore_index=True)
    key = ["prompt_id", "prompt_text", "axis", "seed_idx", "dimension"]
    duplicates = long.duplicated(key, keep=False)
    if duplicates.any():
        duplicate_rows = long.loc[duplicates, key].drop_duplicates()
        raise ValueError(f"Duplicate VBench rows:\n{duplicate_rows.to_string(index=False)}")

    counts = long.groupby("dimension").size()
    incomplete = counts[counts != args.expected_videos]
    if not incomplete.empty:
        raise ValueError(
            "Incomplete VBench dimensions:\n"
            + incomplete.to_string()
            + f"\nExpected {args.expected_videos} rows each."
        )

    summary = (
        long.groupby(["prompt_id", "prompt_text", "dimension"])["score"]
        .agg(n_seeds="size", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.sort_values(["prompt_id", "dimension", "seed_idx"]).to_csv(
        args.output_dir / "vbench_scores_long.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "vbench_scores_summary.csv", index=False)
    print(counts.to_string())
    print(f"[combine_vbench_parts] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
