# Video-analysis metric search for chunk selection

Date: 2026-07-08

## Goal

Find a stronger offline verifier for saved branch leaves in:

`runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1`

The target is still one selected path per root, scored against existing VBench-Long `overall_mean`.

## Literature signals searched

- DOVER/DOVER++ separates UGC video quality into aesthetic and technical perspectives, which motivated trying explicit video-quality features. Paper: https://arxiv.org/abs/2211.04894, code: https://github.com/VQAssessment/DOVER
- FAST-VQA/FasterVQA use fragment sampling for efficient end-to-end VQA, motivating quality-retained spatiotemporal sampling as a future scorer. Papers: https://arxiv.org/abs/2207.02595 and https://arxiv.org/abs/2210.05357, code: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
- Q-Align/OneAlign exposes a pretrained video quality scorer and treats scoring as discrete text-defined levels. Code: https://github.com/q-future/q-align
- T2VQA reports that pure text-video alignment or pure fidelity is insufficient; video fidelity matters, but a single fidelity perspective cannot solve T2V quality when prompt mismatch occurs. Paper: https://jhc.sjtu.edu.cn/~xiaohongliu/papers/24T2VQA.pdf
- VideoScore and VideoScore2 are the strongest next heavy candidates because they are trained on generated-video feedback with multi-aspect annotations. Paper/code: https://arxiv.org/abs/2406.15252 and https://github.com/TIGER-AI-Lab/VideoScore; VideoScore2 code: https://github.com/TIGER-AI-Lab/VideoScore2
- LMM-VQA supports the same direction: explicitly modeling temporal tokens improves blind VQA generalization. Paper: https://arxiv.org/abs/2408.14008

## Local experiments

Already cached signals:

- `clipiqa_scores.csv`
- `video_reward_scores.csv`
- `openclip_scores.csv`
- `pyiqa_sweep/*_scores.csv`
- `dover_mobile/dover_mobile_scores.csv`
- `temporal_clip/temporal_clip_scores.csv`

New broad search outputs:

- `frozen_reward_selectors/fusion_fast_beam_search.csv`
- `frozen_reward_selectors/fusion_fast_beam_formula_table.csv`
- `frozen_reward_selectors/fusion_fast_beam_leave_one_root.csv`

## Baselines

- all-zero path: `0.739504`
- branch mean: `0.741002`
- independent concat: `0.752206`
- oracle best-of-32: `0.778274`
- CLIPIQA mean standalone: `0.753493`
- previous 6-signal temporal fusion: `0.762891`

## New best offline formula

Root-normalized score:

```text
0.875 * z(clipiqa_mean)
+ 1.000 * z(VideoReward_TA)
- 0.750 * z(VideoReward_MQ)
+ 0.375 * z(clip_img_adj_min)
+ 0.750 * z(clip_img_all_min)
+ 0.750 * z(pixel_diff_mean)
+ 0.500 * z(nima_koniq_score_mean)
+ 0.250 * z(niqe_score_mean)
```

Saved as `root_norm_fast_beam_video_quality` in `ttsd/eval/root_normalized_reward_fusion.py`.

Result on the saved 30-root set:

- selected mean: `0.767457`
- vs independent concat: `+0.015250`
- vs all-zero: `+0.027953`
- gap to oracle: `0.010817`
- average rank: `4.866667`
- exact oracle roots: `9/30`
- top-8 roots: `25/30`

Leave-one-root formula selection among discovered formulas:

- held-root mean: `0.763907`
- vs independent concat: `+0.011701`
- vs all-zero: `+0.024403`
- previous 6-signal formula on same roots: `0.762891`
- CLIPIQA standalone: `0.753493`

## Interpretation

The improvement is real on this saved set, but the 8-signal formula is offline-discovered and has higher overfit risk. The more defensible pattern is:

- CLIPIQA handles local visual preference within a root.
- VideoReward text-alignment helps, but VideoReward motion-quality is negatively weighted on this VBench target.
- Temporal image consistency/motion descriptors help more than DOVER-Mobile standalone.
- DOVER-Mobile standalone did not transfer here, even though DOVER is a strong UGC-VQA model.
- The next serious candidate should be VideoScore/VideoScore2 or Q-Align video scoring, not another handcrafted image-quality-only metric.

## Recommendation

Use `root_norm_clipiqa_ta_openclip_mq_temporal` as the conservative selector and `root_norm_fast_beam_video_quality` as the aggressive offline selector. If compute permits, the next iteration should score a subset with VideoScore2 or Q-Align before launching another full 960-video pass.
