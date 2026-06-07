"""Canonical ``runs/<stage>/<run_id>`` input paths — single source of truth for the run layout.

``resolve_run_id`` lets the analysis/report runners accept one ``--run-id`` that fills any
unset input root with its canonical path:

    --heatmap-run-root / --cls-run-root  ->  runs/cls_features/<run_id>
    --patch-run-root                     ->  runs/patch_features/<run_id>
    --vbench-long-csv                    ->  runs/vbench/<run_id>/vbench_scores_long.csv

Explicitly-passed roots always win. After filling, it errors via ``parser`` if any root the
runner declared as required (listed in ``needs``) is still unset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_run_id(
    args: argparse.Namespace, parser: argparse.ArgumentParser, needs: list[str]
) -> None:
    rid = getattr(args, "run_id", None)
    if rid:
        if hasattr(args, "heatmap_run_root") and args.heatmap_run_root is None:
            args.heatmap_run_root = Path("runs/cls_features") / rid
        if hasattr(args, "cls_run_root") and args.cls_run_root is None:
            args.cls_run_root = Path("runs/cls_features") / rid
        if hasattr(args, "patch_run_root") and args.patch_run_root is None:
            args.patch_run_root = Path("runs/patch_features") / rid
        if hasattr(args, "vbench_long_csv") and args.vbench_long_csv is None:
            args.vbench_long_csv = Path("runs/vbench") / rid / "vbench_scores_long.csv"
    for name in needs:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required (or pass --run-id <run_id>)")


def stage_output_dir(run_root: Path, stage: str, module_file: str) -> Path:
    """Default per-script output dir for an analysis/report runner, keyed by run-id.

    ``run_root`` is an input root under ``runs/<input_stage>/<run_id>`` (e.g. the heatmap
    or patch root); ``module_file`` is the calling runner's ``__file__``. The subfolder is
    named after that runner's module, so each runner lands in
    ``runs/<stage>/<run_id>/<runner_module_name>/`` (analysis or report) — for relative or
    absolute input paths alike.
    """
    return run_root.parent.parent / stage / run_root.name / Path(module_file).stem
