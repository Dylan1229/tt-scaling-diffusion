# analysis

Per-seed scalar metrics aligned against VBench. Each reads a feature root plus
`vbench_scores_long.csv`, z-normalizes the 6 VBench subscores within each prompt
(`avg_vbench_z`), and reports correlation / winner-match.

| module | reads | metric |
|---|---|---|
| `similarity_tail_ranking.py` | `cls_features/` (diagonal similarity) | LateTailMean_q per-prompt seed ranking |
| `similarity_reduction_gridsearch.py` | `cls_features/` (similarity matrices) | grid-search of (last_n, tail_k) reductions vs VBench |
| `similarity_reduction_sweep.py` | `cls_features/` (frame/posterior similarity) | alternative reduction family (col-min, softmin, harmonic, …) vs VBench |
| `feature_ridge_regression.py` | `cls_features/` (+ optional `patch_features/`) | leave-one-prompt-out ridge regression → avg_vbench_z |
| `feature_vbench_correlation.py` | `patch_features/` + `cls_features/` | per-seed DINOv2 feature scalars (mean-cos, bestmatch, …) vs VBench |
| `velocity_prefix_correlation.py` | `patch_features/` | prefix-to-current patch-velocity locking scalars vs VBench |
| `microstep_vbench_summary.py` | `runs/vbench_microstep_grid/<run_id>/<variant>/vbench_scores_long.csv` | per-dimension and aggregate VBench deltas for local microstep grid variants |
| `renoise_online_features.py` | an Euler baseline run with saved `posterior_means/` | causal step-5/step-10 DINO CLS trajectories plus pixel/motion scalars for a Renoise intervention gate |

```
python -m ttsd.runners.analysis.similarity_tail_ranking --run-id <run_id>
python -m ttsd.runners.analysis.microstep_vbench_summary --grid-run-root runs/microstep_grid/<run_id>
python -m ttsd.runners.analysis.renoise_online_features \
  --baseline-run runs/baseline/all150_euler_online_001 \
  --output-dir runs/renoise_online_features/all150_euler_s10_001
# --run-id fills the input roots from runs/<stage>/<run_id> (cls_features, patch_features,
# vbench). Pass --heatmap-run-root / --patch-run-root / --cls-run-root / --vbench-long-csv
# explicitly to override. Every analysis/report runner accepts --run-id.
```

With `--run-id` (and no `--output-dir`), each runner writes to `runs/analysis/<run_id>/<module>/` — the
subfolder is named after the runner's own module: `similarity_tail_ranking/`,
`similarity_reduction_gridsearch/`, `similarity_reduction_sweep/`, `feature_ridge_regression/`,
`feature_vbench_correlation/`, `velocity_prefix_correlation/`. The full text report plus any
CSVs/figures land inside that subfolder; nothing is written to stdout.

`similarity_reduction_gridsearch` nests one more level per `matrix_type`/`direction` variant, e.g.
`similarity_reduction_gridsearch/diagonal_rows/`, so different variants never overwrite each other.
`similarity_tail_ranking` instead disambiguates its `--score-type`/`--tail-fraction`/`--late-rows`
variants by prefixing every output file with the metric stem (e.g. `late_tail20_last2_*`).

Shared loaders, ranking, the `--run-id` resolver, and the output-path helper come from `../utilities`.
