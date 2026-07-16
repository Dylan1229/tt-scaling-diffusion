# analysis

Per-seed scalar metrics aligned against VBench. Each reads a feature root plus
`vbench_scores_long.csv`, derives the per-video `vbench_quality` score from it, and
reports correlation / winner-match.

`vbench_quality` applies VBench's own `NORMALIZE_DIC` bounds and `DIM_WEIGHT` weights to
one clip's five prompt-agnostic quality subscores. Because that map is affine, averaging
it over any set of clips reproduces exactly what `external/VBench/scripts/cal_final_score.py`
computes for Quality from the dimension averages. `dynamic_degree` (boolean per clip) and
`overall_consistency` are reported alongside it rather than folded in.

## Motion stratification

`vbench_quality` rewards stillness: its within-prompt Spearman against `dynamic_degree` is
−0.43, and a seed picked purely on quality is a frozen clip in 19 of the 22 prompts whose seeds
differ in motion. Every statistic is therefore reported twice — once over whole prompts, and
once recomputed inside `(prompt, dynamic_degree)` cells, where motion is held fixed.

Stratified numbers carry a `_dyn0` (static) or `_dyn1` (moving) suffix, alongside `n_cells_dyn0`
/ `n_cells_dyn1`. Cells with fewer than `MIN_STRATUM_SEEDS = 3` seeds are dropped, because a
2-seed Spearman is ±1 regardless of the data. On `sweep_v2_20260604_072609` that keeps 436 of
450 clips across 26 moving and 17 static cells. `dynamic_degree` is constant inside a cell, so
it is not a stratified target.

`similarity_tail_ranking` instead ranks seeds a second time within each cell, adding
`dynamic_degree`, `stratum_size` and `stratum_rank` columns plus a `*_stratum_winners.csv`.

| module | reads | metric |
|---|---|---|
| `similarity_tail_ranking.py` | `cls_features/` (diagonal similarity) | LateTailMean_q per-prompt seed ranking |
| `similarity_reduction_gridsearch.py` | `cls_features/` (similarity matrices) | grid-search of (last_n, tail_k) reductions vs VBench |
| `similarity_reduction_sweep.py` | `cls_features/` (frame/posterior similarity) | alternative reduction family (col-min, softmin, harmonic, …) vs VBench |
| `feature_ridge_regression.py` | `cls_features/` (+ optional `patch_features/`) | leave-one-prompt-out ridge regression → prompt-centered `vbench_quality` |
| `feature_vbench_correlation.py` | `patch_features/` + `cls_features/` | per-seed DINOv2 feature scalars (mean-cos, bestmatch, …) vs VBench |
| `velocity_prefix_correlation.py` | `patch_features/` | prefix-to-current patch-velocity locking scalars vs VBench |

```
python -m ttsd.runners.analysis.similarity_tail_ranking --run-id <run_id>
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
