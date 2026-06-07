# generate

Sweep generation and VAE decoding.

| module | reads | writes |
|---|---|---|
| `baseline.py` | a config YAML; resolves `prompts.source` to a prompt list | `runs/baseline/<run_id>/<prompt_id>/seed<NNNN>/` — `video.mp4`, `latents/`, `posterior_means/` (x0_hat), `meta.json`, `DONE`. Defines `_save_video`. |
| `decode_latents.py` | a dir of `step_*.pt` latents (`--latents-dir`) | one `step_*.mp4` per latent. Generic latent→pixel decoder; the pipeline points it at `posterior_means/` to produce DINOv2 input frames. |

```
python -m ttsd.runners.generate.baseline --config configs/sweep_v2_wan22_480p.yaml --capture-posterior-means
python -m ttsd.runners.generate.decode_latents --latents-dir <seed>/posterior_means --output-dir <out>
```

Multi-GPU wrappers: `scripts/generate_sweep_v2_4gpu.sh`, `scripts/prepare_dino_inputs_batch.sh`.
