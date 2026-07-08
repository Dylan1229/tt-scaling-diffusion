# Aligned reward metric search, 2026-07-08

## Goal

Find an offline verifier/reward that ranks saved branch leaves in a way that is
aligned with VBench-Long `overall_mean`, without directly using VBench metric
implementations as reward.

Branch set:

```text
runs/baseline_long/chunk_branch_i2v_all_prompts_s0_1
```

Comparison target:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/chunk_branch_by_path.csv
```

Baselines:

```text
all_zero branch:       0.739504
branch mean:           0.741002
independent concat:    0.752206
oracle best-of-32:     0.778274
```

## Metrics swept

Already tested:

```text
VideoReward 8f and 16f: VQ, MQ, TA, Overall
OpenCLIP ViT-B-32 frame text similarity
CLIPIQA
```

Additional `pyiqa` metrics tested in this pass:

```text
clipiqa+
clipiqa+_rn50_512
liqe
liqe_mix
topiq_nr
dbcnn
hyperiqa
nima
nima-koniq
paq2piq
brisque
niqe
```

Excluded:

```text
musiq / musiq-*      VBench imaging-quality backend
laion_aes            VBench aesthetic-quality backend
clipscore            too close to direct CLIP text-image scoring already covered by OpenCLIP
maniqa               OOMed in this environment; model expanded to roughly 90 GB VRAM
```

Sweep outputs:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/frozen_reward_selectors/pyiqa_sweep
```

## Best single metrics

None of the extra single metrics beat plain CLIPIQA.

Best single frozen visual metrics by VBench `overall_mean`:

```text
metric/select rule              selected   vs independent   vs all_zero
clipiqa_mean                    0.753493   +0.001286        +0.013989
clipiqa+_rn50_512 robust         0.752171   -0.000036        +0.012667
clipiqa+ mean                   0.751075   -0.001131        +0.011571
liqe mean                       0.749682   -0.002525        +0.010178
nima-koniq mean                 0.748775   -0.003432        +0.009271
topiq_nr min                    0.748118   -0.004088        +0.008614
```

This reinforces the earlier conclusion:

```text
CLIPIQA is the strongest standalone frozen visual selector found so far.
```

## Best aligned fusion

The best fixed root-normalized fusion found in this pass:

```text
score =
    1.00 * z_root(clipiqa_mean)
  + 1.25 * z_root(VideoReward_TA_8f)
  + 0.50 * z_root(OpenCLIP_min_8f)
  - 0.25 * z_root(VideoReward_MQ_8f)
```

Result:

```text
selected_mean:         0.761468
vs independent concat: +0.009262
vs all_zero:           +0.021964
gap to oracle:          0.016805
avg rank:               8.233333 / 32
exact oracle:           5/30
top8:                   18/30
```

The simpler previous fusion remains a strong low-complexity option:

```text
score = z_root(clipiqa_mean) + 0.5 * z_root(VideoReward_TA_8f)

selected_mean:         0.759715
vs independent concat: +0.007509
vs all_zero:           +0.020211
gap to oracle:          0.018559
top8:                  20/30
```

The best quality-ensemble variant:

```text
score =
    0.40 * z_root(clipiqa_mean)
  + 0.20 * z_root(clipiqa+_rn50_512_robust)
  + 0.20 * z_root(liqe_mean)
  + 0.20 * z_root(nima_koniq_mean)
  + 0.25 * z_root(VideoReward_TA_8f)
  + 0.25 * z_root(OpenCLIP_min_8f)
  - 0.25 * z_root(VideoReward_MQ_8f)

selected_mean:         0.760332
vs independent concat: +0.008125
top8:                  20/30
```

## Interpretation

The aligned metric is not a single global reward.  It is a root-normalized
within-branch ranker.

Why this works:

```text
1. CLIPIQA gives the strongest within-root visual preference signal.
2. VideoReward TA adds prompt-alignment signal.
3. OpenCLIP worst-frame alignment helps reject locally off-prompt frames.
4. VideoReward MQ is negatively useful here, so a small penalty improves
   VBench overall.  This likely means MQ rewards motion that VBench does not
   always value for this branch set.
```

Important caveat:

```text
The weights above were selected after looking at this saved run.  Treat the
exact +0.009262 margin as an offline-discovery result, not an unbiased estimate.
The robust conclusion is that root-normalized CLIPIQA + text-alignment signals
are meaningfully more aligned than VideoReward Overall, OpenCLIP alone, or
generic NR-IQA alone.
```

## Reproduction helper

Reusable evaluator added:

```bash
.venv/bin/python ttsd/eval/root_normalized_reward_fusion.py
```

It reads the existing reward-score CSVs and writes:

```text
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/root_normalized_reward_fusions/fusion_summary.csv
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/root_normalized_reward_fusions/fusion_by_root.csv
runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/root_normalized_reward_fusions/fusion_selected_paths.csv
```

## Recommended next selector

Use this as the current best offline verifier:

```text
root_norm_clipiqa_ta_openclip_mq
```

Formula:

```text
z_root(clipiqa_mean)
+ 1.25 * z_root(VideoReward_TA_8f)
+ 0.50 * z_root(OpenCLIP_min_8f)
- 0.25 * z_root(VideoReward_MQ_8f)
```

Next research step:

```text
Evaluate this formula on a newly generated branch set or leave-prompt-out
branches before trusting the exact weights.  If it holds, use it as the
chunkwise/beam verifier score.
```
