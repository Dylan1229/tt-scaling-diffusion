# generate

Sweep generation and VAE decoding.

| module | reads | writes |
|---|---|---|
| `baseline.py` | a config YAML; resolves `prompts.source` to a prompt list | `runs/baseline/<run_id>/<prompt_id>/seed<NNNN>/` — `video.mp4`, `latents/`, `posterior_means/` (x0_hat), `meta.json`, `DONE`. Defines `_save_video`. |
| `microstep_grid.py` | `configs/microstep_grid_wan22_480p.yaml` | `runs/microstep_grid/<run_id>/<variant>/<prompt_id>/seed<NNNN>/` — one VBench-compatible run per local microstep variant. |
| `decode_latents.py` | a dir of `step_*.pt` latents (`--latents-dir`) | one `step_*.mp4` per latent. Generic latent→pixel decoder; the pipeline points it at `posterior_means/` to produce DINOv2 input frames. |

```
python -m ttsd.runners.generate.baseline --config configs/sweep_v2_wan22_480p.yaml --capture-posterior-means
python -m ttsd.runners.generate.microstep_grid --config configs/microstep_grid_wan22_480p.yaml --list-variants
python -m ttsd.runners.generate.microstep_grid --config configs/microstep_grid_wan22_480p.yaml --smoke
python -m ttsd.runners.generate.decode_latents --latents-dir <seed>/posterior_means --output-dir <out>
```

For microstep-grid VBench:

```
scripts/run_microstep_vbench_variants.sh --run-root runs/microstep_grid/<run_id>
python -m ttsd.runners.analysis.microstep_vbench_summary --grid-run-root runs/microstep_grid/<run_id>
```

Multi-GPU wrappers: `scripts/generate_sweep_v2_4gpu.sh`, `scripts/prepare_dino_inputs_batch.sh`.
