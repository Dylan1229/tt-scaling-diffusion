# Positive semantic metric search

Date: 2026-07-09

## Motivation

The previous best formula used a negative coefficient on VideoReward `MQ`. That is not a defensible metric interpretation because VideoReward defines `MQ` as a positive motion-quality axis: dynamic stability, dynamic reasonableness, naturalness, and dynamic degree.

Sources:

- VideoReward model card: https://huggingface.co/KlingTeam/VideoReward
- VideoReward repository: https://github.com/KlingAIResearch/VideoAlign
- VideoScore paper: https://arxiv.org/abs/2406.15252
- Q-Align paper: https://arxiv.org/abs/2312.17090

## Result

Positive `MQ` alone is weak on the saved branch-selection task:

```text
MQ_positive selected_mean: 0.743197
vs independent_concat:   -0.009009
```

VideoReward `Overall` is also weak:

```text
Overall_positive selected_mean: 0.747537
vs independent_concat:        -0.004670
```

This means the previous negative `MQ` coefficient was compensating for target mismatch on this saved VBench subset. It should not be described as a general motion-quality principle.

## Positive-only search

The clean search excluded `MQ` and VideoReward `Overall`, allowed only nonnegative weights, and used direction-coded features where higher means better.

Best positive-only searched formula with up to 6 terms:

```text
0.125 * z(clipiqa_max)
+ 0.250 * z(VideoReward_TA)
+ 0.125 * z(OpenCLIP_min)
+ 0.375 * z(CLIP-IQA+_mean)
+ 0.375 * z(pixel_diff_mean)
+ 0.250 * z(pixel_diff_min)
```

Saved as `root_norm_positive_beam_no_mq`.

Result:

```text
selected_mean:        0.766364
vs independent_concat:+0.014158
vs all_zero:          +0.026860
gap_to_oracle:         0.011910
exact_oracle roots:    8/30
top-8 roots:          23/30
```

This is close to the old negative-`MQ` best:

```text
old negative-MQ best: 0.767457
positive no-MQ best: 0.766364
difference:          0.001093
```

## Cleaner manually interpretable formula

For a more defensible paper/story formula:

```text
1.000 * z(clipiqa_mean)
+ 0.750 * z(VideoReward_TA)
+ 0.500 * z(OpenCLIP_min)
+ 0.250 * z(CLIP adjacent-frame mean similarity)
+ 0.250 * z(pixel_diff_mean)
+ 0.250 * z(NIMA-Koniq mean)
```

Saved as `root_norm_clean_quality_alignment_temporal`.

Result:

```text
selected_mean:        0.759661
vs independent_concat:+0.007455
vs all_zero:          +0.020157
gap_to_oracle:         0.018613
```

## Recommendation

Do not use the negative-`MQ` formula as the main metric. Use:

- `root_norm_clean_quality_alignment_temporal` for a defensible, interpretable selector.
- `root_norm_positive_beam_no_mq` if the priority is stronger offline performance while keeping all weights semantically positive.

The next real research step should be a proper video reward model such as VideoScore2 or Q-Align/OneAlign video scoring, evaluated on heldout roots or human preferences rather than tuning directly on VBench outputs.
