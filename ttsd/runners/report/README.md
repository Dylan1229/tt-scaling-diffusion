# report

Figures for the DINOv2 feature-property report. Each reads a feature root plus
`vbench_scores_long.csv`.

Each prompt is split into its `dynamic_degree` strata — moving (`dyn=1`) and static (`dyn=0`) —
and the best and worst seed are taken from within one stratum, never across. `vbench_quality`
rewards stillness (within-prompt Spearman against `dynamic_degree` = −0.43), so an unstratified
best/worst pair contrasts a frozen clip with a moving one rather than a good clip with a bad one.
A stratum needs at least two seeds to have both a best and a worst; smaller ones are skipped.

`velocity_heatmaps` emits one figure per `(prompt, stratum)`: a prompt whose seeds all move gets
one, a prompt whose seeds differ gets two. `feature_property_figs` emits one showcase per stratum
— the cell whose best-vs-worst gap is widest. Every figure title carries `dyn=<0|1> n=<seeds>`
plus the prompt and seed ids.

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
