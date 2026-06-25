"""Combine VBench-Long CSV outputs across long-video methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _parse_method(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected METHOD=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty method label in {spec!r}")
    return label, Path(path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[compare-vbench-long] WARN missing {path}")
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_method_rows(rows: list[dict], method: str, source_dir: Path) -> list[dict]:
    out = []
    for row in rows:
        out.append({"method": method, "source_dir": str(source_dir), **row})
    return out


def _with_seed_overall(rows: list[dict]) -> list[dict]:
    out = list(rows)
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        dimension = row.get("dimension", "")
        if dimension == "overall_mean":
            continue
        seed_idx = row.get("seed_idx", "")
        try:
            score = float(row.get("score", ""))
        except ValueError:
            continue
        by_seed.setdefault(seed_idx, []).append(score)

    for seed_idx, scores in sorted(by_seed.items(), key=lambda item: int(item[0])):
        if not scores:
            continue
        out.append(
            {
                "seed_idx": seed_idx,
                "dimension": "overall_mean",
                "score": f"{sum(scores) / len(scores):.6f}",
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--stage", required=True, choices=["pilot", "full"])
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method label and VBench-Long output dir, e.g. direct=runs/vbench_long/run_id.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or Path("runs/vbench_long_compare") / args.comparison_id

    overall_rows: list[dict] = []
    by_seed_rows: list[dict] = []
    prompt_rows: list[dict] = []
    for method_spec in args.method:
        method, score_dir = _parse_method(method_spec)
        overall_rows.extend(
            _append_method_rows(
                _read_csv(score_dir / "vbench_long_scores_overall.csv"),
                method,
                score_dir,
            )
        )
        by_seed_rows.extend(
            _append_method_rows(
                _with_seed_overall(_read_csv(score_dir / "vbench_long_scores_by_seed.csv")),
                method,
                score_dir,
            )
        )
        prompt_rows.extend(
            _append_method_rows(
                _read_csv(score_dir / "vbench_long_scores_summary.csv"),
                method,
                score_dir,
            )
        )

    prefix = output_dir / f"{args.stage}_method"
    _write_csv(
        prefix.with_name(f"{prefix.name}_summary.csv"),
        overall_rows,
        ["method", "source_dir", "dimension", "n_seeds", "mean", "std", "min", "max"],
    )
    _write_csv(
        prefix.with_name(f"{prefix.name}_by_seed.csv"),
        by_seed_rows,
        ["method", "source_dir", "seed_idx", "dimension", "score"],
    )
    _write_csv(
        prefix.with_name(f"{prefix.name}_prompt_summary.csv"),
        prompt_rows,
        ["method", "source_dir", "prompt_id", "prompt_text", "dimension", "n_seeds", "mean", "std", "min", "max"],
    )

    print(f"[compare-vbench-long] wrote {output_dir}")


if __name__ == "__main__":
    main()
