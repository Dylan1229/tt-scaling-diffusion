# report

Figures for the DINOv2 feature-property report. Each reads a feature root plus
`vbench_scores_long.csv`.

| module | reads | writes |
|---|---|---|
| `feature_property_figs.py` | `cls_features/` + `patch_features/` | good-vs-bad frame-similarity heatmaps, velocity-to-final cosine curve, scalar scatter, `metrics_summary.json` |
| `velocity_heatmaps.py` | `patch_features/` | per-prompt good-vs-bad patch-velocity heatmaps (step-adjacent + prefix-locked) + `velocity_heatmap_cells.txt` raw dump |

```
python -m ttsd.runners.report.velocity_heatmaps --run-id <run_id>
# → figures + velocity_heatmap_cells.txt in runs/report/<run_id>/velocity_heatmaps/  (override with --output-dir)
```

With `--run-id` (and no `--output-dir`), each report runner writes to `runs/report/<run_id>/<module>/` —
the subfolder is named after the runner's own module (`feature_property_figs/`, `velocity_heatmaps/`).
Figures, summary JSON, and the per-cell text dump all land inside that subfolder; nothing is written to
stdout.
