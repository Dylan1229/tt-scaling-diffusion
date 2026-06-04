# utilities

Shared helpers imported by the `analysis/` and `report/` runners, so those
runners no longer reach into one another's modules.

| module | provides | used by |
|---|---|---|
| `seed_vbench_loaders.py` | `_iter_seed_dirs`, `_seed_idx_from_name` (walk `p*/seed*`), `_load_vbench_rows` (parse `vbench_scores_long.csv`) | every `analysis/` + `report/` runner |
| `ranking.py` | `_rankdata` — tie-aware rank transform (Spearman / within-prompt ranking) | `feature_ridge_regression`, `feature_vbench_correlation`, `velocity_prefix_correlation`, `feature_property_figs` |
