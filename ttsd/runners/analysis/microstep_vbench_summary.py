"""Summarize VBench results across microstep-grid variants."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


NORMALIZED_BY_100_DIMENSIONS = {"imaging_quality"}


def _format_float(value: object, digits: int = 6) -> object:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in columns})


def _relative_path(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=from_dir.resolve())


def _default_vbench_root(grid_run_root: Path) -> Path:
    # runs/microstep_grid/<run_id> -> runs/vbench_microstep_grid/<run_id>
    if grid_run_root.parent.name == "microstep_grid":
        return grid_run_root.parent.parent / "vbench_microstep_grid" / grid_run_root.name
    return Path("runs/vbench_microstep_grid") / grid_run_root.name


def _discover_variants(grid_run_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in grid_run_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and any(p.glob("*/seed*/meta.json"))
    )


def _normalize_score(dimension: str, score: float) -> float:
    if dimension in NORMALIZED_BY_100_DIMENSIONS and score > 1.0:
        return score / 100.0
    return score


def _load_dimension_scores(long_csv: Path) -> dict[str, list[float]]:
    by_dim: dict[str, list[float]] = defaultdict(list)
    with long_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dimension = row["dimension"]
            by_dim[dimension].append(_normalize_score(dimension, float(row["score"])))
    return dict(by_dim)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _dimension_means(by_dim: dict[str, list[float]]) -> dict[str, tuple[float, int]]:
    return {dim: (statistics.fmean(scores), len(scores)) for dim, scores in by_dim.items() if scores}


def _safe_delta(value: float, baseline: float | None) -> tuple[float | None, float | None]:
    if baseline is None:
        return None, None
    delta = value - baseline
    rel = delta / baseline if baseline != 0 else None
    return delta, rel


def summarize(
    *,
    grid_run_root: Path,
    vbench_root: Path,
    output_dir: Path,
    baseline_variant: str,
) -> tuple[Path, Path, Path, Path]:
    variants = _discover_variants(grid_run_root)
    if baseline_variant in variants:
        variants = [baseline_variant] + [v for v in variants if v != baseline_variant]

    means_by_variant: dict[str, dict[str, tuple[float, int]]] = {}
    missing: list[str] = []
    for variant in variants:
        long_csv = vbench_root / variant / "vbench_scores_long.csv"
        if not long_csv.exists():
            missing.append(variant)
            continue
        means_by_variant[variant] = _dimension_means(_load_dimension_scores(long_csv))

    if baseline_variant not in means_by_variant:
        raise FileNotFoundError(
            f"Missing baseline VBench CSV: {vbench_root / baseline_variant / 'vbench_scores_long.csv'}"
        )

    baseline = means_by_variant[baseline_variant]
    all_dims = sorted({dim for means in means_by_variant.values() for dim in means})

    dimension_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    for variant, means in means_by_variant.items():
        wide_row: dict[str, object] = {"variant": variant}
        for dim in all_dims:
            if dim not in means:
                continue
            mean, n = means[dim]
            base_mean = baseline.get(dim, (None, 0))[0]
            delta, relative_delta = _safe_delta(mean, base_mean)
            wide_row[f"{dim}_score"] = mean
            wide_row[f"{dim}_baseline"] = base_mean
            wide_row[f"{dim}_delta"] = delta
            wide_row[f"{dim}_relative_delta"] = relative_delta
            dimension_rows.append(
                {
                    "variant": variant,
                    "dimension": dim,
                    "n": n,
                    "mean_score": mean,
                    "baseline_mean_score": base_mean,
                    "delta_vs_baseline": delta,
                    "relative_delta_vs_baseline": relative_delta,
                }
            )
        wide_rows.append(wide_row)

    common_dims = sorted(set.intersection(*(set(m) for m in means_by_variant.values())))
    baseline_aggregate = statistics.fmean(baseline[dim][0] for dim in common_dims)
    variant_rows: list[dict[str, object]] = []
    for variant, means in means_by_variant.items():
        aggregate = statistics.fmean(means[dim][0] for dim in common_dims)
        delta, relative_delta = _safe_delta(aggregate, baseline_aggregate)
        dimension_deltas = [means[dim][0] - baseline[dim][0] for dim in common_dims]
        dimension_relative_deltas = [
            (means[dim][0] - baseline[dim][0]) / baseline[dim][0]
            for dim in common_dims
            if baseline[dim][0] != 0
        ]
        row: dict[str, object] = {
            "variant": variant,
            "n_common_dimensions": len(common_dims),
            "mean_common_dimensions": aggregate,
            "baseline_mean_common_dimensions": baseline_aggregate,
            "delta_vs_baseline": delta,
            "relative_delta_vs_baseline": relative_delta,
            "mean_dimension_delta_vs_baseline": statistics.fmean(dimension_deltas),
            "mean_dimension_relative_delta_vs_baseline": (
                statistics.fmean(dimension_relative_deltas) if dimension_relative_deltas else None
            ),
        }
        if "overall_consistency" in means:
            overall = means["overall_consistency"][0]
            base_overall = baseline.get("overall_consistency", (None, 0))[0]
            overall_delta, overall_relative_delta = _safe_delta(overall, base_overall)
            row.update(
                {
                    "overall_consistency_mean": overall,
                    "baseline_overall_consistency_mean": base_overall,
                    "overall_consistency_delta_vs_baseline": overall_delta,
                    "overall_consistency_relative_delta_vs_baseline": overall_relative_delta,
                }
            )
        variant_rows.append(row)

    if missing:
        _write_csv(output_dir / "missing_vbench.csv", [{"variant": name} for name in missing])

    dim_csv = output_dir / "dimension_delta.csv"
    variant_csv = output_dir / "variant_delta.csv"
    wide_csv = output_dir / "component_scores_wide.csv"
    gallery_html = output_dir / "video_gallery.html"
    _write_csv(dim_csv, dimension_rows)
    _write_csv(variant_csv, variant_rows)
    _write_csv(wide_csv, wide_rows)
    _write_gallery_html(
        path=gallery_html,
        grid_run_root=grid_run_root,
        output_dir=output_dir,
        variants=list(means_by_variant),
        all_dims=all_dims,
        means_by_variant=means_by_variant,
        baseline=baseline,
    )
    return dim_csv, variant_csv, wide_csv, gallery_html


def _component_table_html(
    *,
    variant: str,
    all_dims: list[str],
    means_by_variant: dict[str, dict[str, tuple[float, int]]],
    baseline: dict[str, tuple[float, int]],
) -> str:
    means = means_by_variant.get(variant, {})
    rows = []
    for dim in all_dims:
        if dim not in means:
            continue
        score = means[dim][0]
        base = baseline[dim][0]
        delta, relative_delta = _safe_delta(score, base)
        rows.append(
            "<tr>"
            f"<td>{html.escape(dim)}</td>"
            f"<td>{base:.6f}</td>"
            f"<td>{score:.6f}</td>"
            f"<td>{delta:.6f}</td>"
            f"<td>{relative_delta:.6f}</td>"
            "</tr>"
        )
    return (
        "<table class=\"scores\"><thead><tr>"
        "<th>component</th><th>baseline</th><th>current</th><th>delta</th><th>relative delta</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _write_gallery_html(
    *,
    path: Path,
    grid_run_root: Path,
    output_dir: Path,
    variants: list[str],
    all_dims: list[str],
    means_by_variant: dict[str, dict[str, tuple[float, int]]],
    baseline: dict[str, tuple[float, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for variant in variants:
        for video in sorted((grid_run_root / variant).glob("*/seed*/video.mp4")):
            meta = _read_json(video.parent / "meta.json")
            rel_video = _relative_path(output_dir, video)
            prompt_id = str(meta.get("prompt_id") or video.parent.parent.name)
            seed = str(meta.get("seed") or video.parent.name.replace("seed", ""))
            prompt_text = str(meta.get("prompt_text") or "")
            scores = _component_table_html(
                variant=variant,
                all_dims=all_dims,
                means_by_variant=means_by_variant,
                baseline=baseline,
            )
            cards.append(
                "<section class=\"card\">"
                f"<h2>{html.escape(variant)} / {html.escape(prompt_id)} / seed{html.escape(seed)}</h2>"
                f"<video controls preload=\"metadata\" src=\"{html.escape(rel_video)}\"></video>"
                f"<p>{html.escape(prompt_text)}</p>"
                f"{scores}"
                "</section>"
            )

    doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Microstep VBench Video Gallery</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }
    h1 { font-size: 24px; margin-bottom: 16px; }
    .note { color: #57606a; font-size: 13px; margin: -6px 0 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; }
    h2 { font-size: 15px; margin: 0 0 10px; }
    video { width: 100%; background: #000; border-radius: 4px; }
    p { font-size: 12px; line-height: 1.35; min-height: 34px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th, td { border-top: 1px solid #d8dee4; padding: 4px 5px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
  </style>
</head>
<body>
  <h1>Microstep VBench Video Gallery</h1>
  <p class="note">Scores use normalized VBench component values; imaging_quality is divided by 100 when raw per-video MUSIQ scores are present.</p>
  <div class="grid">
"""
    doc += "\n".join(cards)
    doc += """
  </div>
</body>
</html>
"""
    path.write_text(doc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-run-root", required=True, type=Path)
    parser.add_argument("--vbench-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-variant", default="baseline")
    args = parser.parse_args(argv)

    grid_run_root = args.grid_run_root
    vbench_root = args.vbench_root or _default_vbench_root(grid_run_root)
    output_dir = args.output_dir or (Path("runs/analysis") / grid_run_root.name / "microstep_vbench_summary")

    dim_csv, variant_csv, wide_csv, gallery_html = summarize(
        grid_run_root=grid_run_root,
        vbench_root=vbench_root,
        output_dir=output_dir,
        baseline_variant=args.baseline_variant,
    )
    print(f"[microstep_vbench_summary] dimension CSV: {dim_csv}")
    print(f"[microstep_vbench_summary] variant CSV: {variant_csv}")
    print(f"[microstep_vbench_summary] component wide CSV: {wide_csv}")
    print(f"[microstep_vbench_summary] video gallery HTML: {gallery_html}")


if __name__ == "__main__":
    main()
