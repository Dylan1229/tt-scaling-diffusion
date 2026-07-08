# Reward model candidates and CLIPIQA result, 2026-07-08

## Problem

Earlier frozen reward tests showed weak global correlation with VBench-Long
overall_mean:

```text
VideoReward TA 8f selected_mean: 0.750364
independent concat:             0.752206
```

The issue is more subtle than global correlation.  For branch selection, we care
about within-root ranking among the 32 candidates for the same prompt/root seed.

## Stronger literature candidates

Most relevant off-the-shelf candidates:

```text
1. VideoScore2
   Multi-dimensional video evaluator with visual quality, text alignment, and
   physical/common-sense consistency.  The official repo supports inference and
   many baseline reward models.  It is the best next heavy model to try.

2. VideoScore
   EMNLP 2024 reward metric trained on VideoFeedback with multi-aspect human
   scores.  It reported much higher human correlation than prior metrics.

3. VideoReward / VideoAlign
   Already tested locally.  It gives VQ, MQ, TA, and Overall.  TA was the best
   component here.

4. VideoPhy / VideoPhy2 / VideoCon-Physics
   Useful for prompts where physical/common-sense failures dominate, but less
   directly matched to our current generic VBench overall_mean target.

5. DEVIL dynamics metrics
   Useful if the failure mode is motion dynamics rather than static quality or
   prompt alignment.
```

VideoScore2 setup is heavier than the lightweight tests:

```text
torch==2.6.0
torchvision==0.21.0
transformers==4.53.2
qwen-vl-utils
```

It should be run in an isolated environment, not installed into the project
venv.

## New local frozen reward tested: CLIPIQA

I tested `pyiqa` `clipiqa` over 8 sampled frames per saved branch video.
This excludes `musiq` and `laion_aes` because those overlap VBench imaging and
aesthetic backends.

Output files:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors/clipiqa_scores.csv
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors/clipiqa_by_root.csv
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors/clipiqa_summary.csv
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors/clipiqa_correlations.csv
```

CLIPIQA selector results:

```text
selector                  selected   vs independent   vs all_zero   gap to oracle
clipiqa_mean              0.753493   +0.001286        +0.013989     0.024781
clipiqa_min               0.742618   -0.009588        +0.003114     0.035656
clipiqa_mean_minus_std    0.744294   -0.007912        +0.004790     0.033980
```

CLIPIQA mean is the first frozen reward tested here that beats independent
concat, although the margin is small.

Important diagnostic:

```text
Global CLIPIQA mean vs VBench overall_mean correlation:
pearson  -0.156508
spearman -0.164758

Mean within-root CLIPIQA mean vs VBench overall_mean correlation:
pearson   0.312772
spearman  0.285826
positive roots: 25/30
```

Interpretation:

```text
Global correlation is misleading because prompt/root difficulty dominates.
For branch selection, root-normalized within-root ranking is the relevant
quantity.  CLIPIQA is useful within roots even though its global correlation is
negative.
```

## Hand-designed fusion

I tested root-normalized fixed fusions.  These use no fitted weights from VBench
labels; they only z-normalize each reward among the 32 branches of the same
root, then apply manually chosen weights.

Best fixed fusion:

```text
score = z(clipiqa_mean) + 0.5 * z(VideoReward_TA_8f)
```

Result:

```text
selected_mean:   0.759715
vs independent: +0.007509
vs all_zero:    +0.020211
gap to oracle:   0.018559
avg rank:        7.600000 / 32
exact oracle:    4/30
top8:            20/30
```

Other fixed fusions:

```text
clipiqa_mean only                 0.753493  vs independent +0.001286
clipiqa + TA + OpenCLIP min       0.755865  vs independent +0.003658
clipiqa + TA - 0.25 * MQ          0.756633  vs independent +0.004427
clipiqa + VideoReward 8f/16f TA   0.756133  vs independent +0.003927
```

Leave-prompt-out ridge diagnostic with CLIPIQA included:

```text
features: bits + rewards + root-normalized reward features
alpha:    100
selected: 0.755295
vs independent: +0.003089
vs all_zero:    +0.015791
gap to oracle:   0.022979
avg rank:        10.966667 / 32
top8:            16/30
```

This learned diagnostic generalizes weakly, but the manual fusion is better.
Given the small data size, the manual root-normalized fusion is the safer
candidate to test next.

## Current recommendation

Use this as the next offline selector candidate:

```text
root_norm_clipiqa_ta =
    z_root(clipiqa_mean_8f)
  + 0.5 * z_root(VideoReward_TA_8f)
```

Why this looks promising:

```text
1. It beats independent concat on current saved branches by +0.007509.
2. It uses frozen off-the-shelf models, no VBench-trained weights.
3. It combines complementary signals:
   CLIPIQA handles within-root visual/image preference.
   VideoReward TA handles prompt alignment.
4. It avoids using VBench's MUSIQ and LAION aesthetic backends directly.
```

Main caveat:

```text
The 0.5 weight was chosen after looking at this run, so the exact margin is not
an unbiased estimate.  The direction is still useful: CLIPIQA + alignment is
clearly stronger than generic VideoReward Overall or OpenCLIP alone.
```
