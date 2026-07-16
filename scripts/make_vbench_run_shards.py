"""Create symlink-only run views for dimension-parallel VBench evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _samples(run_root: Path) -> list[Path]:
    return sorted(
        seed_dir
        for seed_dir in run_root.glob("p*/seed*")
        if (seed_dir / "meta.json").exists() and (seed_dir / "video.mp4").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    args = parser.parse_args()

    samples = _samples(args.run)
    if not samples:
        raise SystemExit(f"No completed samples under {args.run}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, seed_dir in enumerate(samples):
        shard_index = index % args.num_shards
        prompt_id = seed_dir.parent.name
        seed_name = seed_dir.name
        shard_root = args.output_root / f"shard{shard_index}"
        target = shard_root / prompt_id / seed_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(seed_dir.resolve(), target_is_directory=True)
        manifest.append(
            {
                "shard": shard_index,
                "prompt_id": prompt_id,
                "seed": seed_name,
                "source": str(seed_dir.resolve()),
            }
        )

    counts = {
        shard_index: sum(row["shard"] == shard_index for row in manifest)
        for shard_index in range(args.num_shards)
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps({"counts": counts, "samples": manifest}, indent=2)
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
