# Frozen reward selector comparison, 2026-07-08

Saved branch run:

```text
runs/baseline_long/chunk_branch_i2v_all_prompts_s0_1
```

Saved VBench-Long comparison:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1
```

## Setup used for VideoReward

VideoReward requires the official VideoAlign code path and is not compatible
with the project venv's `transformers==5.7.0`.  I used a thin temporary
compatibility overlay that keeps the project Torch/TorchVision stack intact:

```bash
.venv/bin/python -m pip install --target /tmp/videoreward_compat --no-deps --upgrade --no-cache-dir \
  transformers==4.45.2 tokenizers==0.20.3 peft==0.10.0 accelerate==0.34.0 trl==0.8.6
.venv/bin/python -m pip install --target /tmp/videoreward_compat --no-deps --upgrade --no-cache-dir \
  huggingface-hub==0.23.2
```

VideoReward checkpoint:

```text
/tmp/VideoReward_ckpt
```

VideoAlign repo:

```text
/tmp/VideoAlign
```

## Commands

VideoReward, 8 frames:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python ttsd/eval/frozen_reward_selector_compare.py \
  --batch-size 4 \
  --num-frames 8 \
  --force
```

VideoReward, 16 frames:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python ttsd/eval/frozen_reward_selector_compare.py \
  --batch-size 8 \
  --num-frames 16 \
  --output-dir runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors_16f
```

OpenCLIP was run as a one-off frozen verifier over 8 sampled frames using
`ViT-B-32/laion2b_s34b_b79k`.  Outputs were written under:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors
```

## Key VBench overall_mean results

All rows are averaged over 30 roots.  Higher is better.

```text
selector                         selected   vs independent   vs all_zero   gap to best-of-32
video_reward_8f_ta               0.750364   -0.001843        +0.010860     0.027910
video_reward_8f_overall          0.747537   -0.004670        +0.008033     0.030737
video_reward_16f_ta              0.746905   -0.005301        +0.007401     0.031369
video_reward_16f_overall         0.744798   -0.007409        +0.005294     0.033476
openclip_8f_clip_min             0.744179   -0.008027        +0.004676     0.034094
all_zero baseline                0.739504   -0.012702        +0.000000     0.038770
independent concat baseline      0.752206   +0.000000        +0.012702     0.026068
oracle best-of-32                0.778274   +0.026068        +0.038770     0.000000
```

Interpretation:

```text
Best frozen reward selector found: VideoReward TA with 8 sampled frames.
It improves over the all-zero branch by +0.010860 VBench overall_mean,
but still trails independent concat by -0.001843.
```

VideoReward 16-frame correlations with VBench overall_mean:

```text
TA       pearson 0.227105   spearman 0.245082
Overall  pearson 0.115953   spearman 0.122058
VQ       pearson -0.027576  spearman -0.150207
MQ       pearson -0.234785  spearman -0.218161
```

Conclusion:

```text
Frozen reward models are useful enough to beat the naive all-zero branch, but
the best result still does not beat independent concat.  The current evidence
points away from single-step greedy frozen reward selection and toward either
beam/lookahead selection or a verifier that explicitly scores cross-chunk
continuity and long-horizon consistency.
```
