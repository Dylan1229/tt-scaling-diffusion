# generate

Sweep generation and VAE decoding.

| module | reads | writes |
|---|---|---|
| `baseline.py` | a config YAML; resolves `prompts.source` to a prompt list | `runs/baseline/<run_id>/<prompt_id>/seed<NNNN>/` — `video.mp4`, `latents/`, `posterior_means/` (x0_hat), `meta.json`, `DONE`. Defines `_save_video`. |
| `late_branching.py` | a late-branch config and prompt list | One unperturbed batched control plus M perturbed suffixes per root seed under `runs/late_branching/<run_id>/`. All candidates share the prefix through the configured 1-based branch step. |
| `decode_latents.py` | a dir of `step_*.pt` latents (`--latents-dir`) | one `step_*.mp4` per latent. Generic latent→pixel decoder; the pipeline points it at `posterior_means/` to produce DINOv2 input frames. |

```
python -m ttsd.runners.generate.baseline --config configs/sweep_v2_wan22_480p.yaml --capture-posterior-means
python -m ttsd.runners.generate.decode_latents --latents-dir <seed>/posterior_means --output-dir <out>
```

Multi-GPU wrappers: `scripts/generate_sweep_v2_4gpu.sh`, `scripts/prepare_dino_inputs_batch.sh`.

## Late-stage branching feasibility test

The default config forks **after denoising step 35** (1-based), producing one
unperturbed batched control and four noisy candidates. Because merely expanding
the suffix batch causes small numerical drift, intervention claims use a separate
batch-one baseline target file. This is a Best-of-M oracle probe, not an online
selection policy.

```bash
# One root video, with only one noisy candidate for a fast implementation smoke.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m ttsd.runners.generate.late_branching \
  --config configs/late_branching_s35_wan22_480p.yaml \
  --run-id late_branch_s35_smoke \
  --smoke --num-noise-branches 1

# Full 15-prompt x 10-seed run on four GPUs.
RUN_ID=late_branch_s35_m4_001 scripts/generate_late_branching_4gpu.sh

# Score the three separate targets from PR #3's updated protocol.
.venv/bin/python -m ttsd.eval.vbench \
  --run runs/late_branching/late_branch_s35_m4_001 \
  --output runs/vbench/late_branch_s35_m4_001

# Offline upper bound: overall, bottom 15 baselines, and bottom quartile.
.venv/bin/python -m ttsd.eval.late_branch_oracle \
  --run runs/late_branching/late_branch_s35_m4_001 \
  --targets runs/vbench/late_branch_s35_m4_001/vbench_targets.csv \
  --baseline-targets runs/baseline/20260511_224405/vbench/vbench_targets.csv
```

With 50 denoising steps, step 35, and five total outputs, each root costs 110
denoising-step equivalents: `35 + (50 - 35) * 5`, or 2.2x one baseline.
