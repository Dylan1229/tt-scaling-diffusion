# Wan Long-Video Generation Findings

This report summarizes our current long-video experiments with Wan2.2 TI2V-5B on the VBench-Long 30s setting. The setup uses 481-frame videos at 16 fps, equivalent to six 81-frame chunks with one-frame overlap. We evaluated three practical ways to use Wan for 30s generation, then tested whether chunk-level test-time search can recover quality when autoregressive I2V chaining is weak.

## Methods Compared

We evaluated three long-video generation options:

1. **Direct long T2V**
   Generate the full 481-frame video in one Wan T2V call.

2. **Independent T2V chunk concat**
   Generate six independent 81-frame T2V chunks and concatenate them. This aligns better with the short chunk length used by the model, but the chunks have no visual conditioning between them.

3. **T2V first chunk + last-frame I2V continuations**
   Generate chunk 0 with T2V, then generate chunks 1-5 using last-frame I2V conditioning from the previous chunk. This should improve cross-chunk continuity, but it can accumulate errors and may lock into a weak continuation trajectory.

We also tested a fourth variant for analysis: **chunk-branch search**. It uses option 3 as the base, but branches each continuation with two seeds over five I2V chunks, producing 32 complete paths per prompt/seed root. We then select the best path by VBench-Long score. This is an oracle-style test-time scaling experiment, not yet a deployable selection policy.

## Full 150-Video Baseline Results

Full comparison used 15 prompts x 10 seeds = 150 videos.

| Metric | Independent concat | Direct 481 T2V | Last-frame I2V |
|---|---:|---:|---:|
| overall_mean | 0.751928 | 0.656386 | 0.731502 |
| subject_consistency | 0.920453 | 0.989193 | 0.965741 |
| background_consistency | 0.942586 | 0.978709 | 0.959281 |
| motion_smoothness | 0.967251 | 0.995089 | 0.976204 |
| dynamic_degree | 0.586250 | 0.096250 | 0.442500 |
| aesthetic_quality | 0.480973 | 0.436986 | 0.465619 |
| imaging_quality | 0.614057 | 0.442089 | 0.579668 |

The strongest no-search baseline is **independent T2V chunk concat**. Direct 481-frame T2V has very high subject/background/motion smoothness scores, but collapses on dynamic degree and imaging quality. The metric pattern suggests direct long generation is overly stable: it preserves identity and background, but produces low-motion, low-detail videos. This is consistent with a long-context mismatch where asking Wan to generate 481 frames directly pushes it outside the more reliable 81-frame sampling regime.

Last-frame I2V also underperforms independent concat overall by about `-0.0204`. It improves consistency over independent chunking, but loses on dynamic degree, aesthetic quality, and imaging quality. The likely reason is that autoregressive I2V conditions strongly on the previous final frame, so later chunks can become conservative and inherit earlier errors. The chain improves continuity, but without search it often chooses a continuation path that is less dynamic or visually worse than independent T2V chunks.

## Chunk-Branch Search Results

We then tested all 15 prompts with root seeds 0 and 1, giving 30 prompt/seed roots. For each root, chunk 0 was fixed and chunks 1-5 branched two ways, producing 32 paths per root and 960 videos total.

| Setting | overall_mean | Delta vs independent |
|---|---:|---:|
| independent concat | 0.752206 | 0 |
| all-zero/default I2V path | 0.739504 | -0.012702 |
| random-path mean over 32 | 0.741002 | -0.011204 |
| best-of-32 path | 0.778274 | +0.026068 |

This confirms the important point: **without search, T2V + I2V does not beat independent concat on average**. The default path and average random branch are both worse than independent. However, **search over continuation paths changes the result**: best-of-32 beats independent concat by `+0.0261` overall and wins on 24/30 prompt/seed roots.

The clearest view is the searched distribution. For each prompt/seed root and each metric, we have 32 complete I2V continuation paths. The table below averages the worst path, mean path, and best path across the 30 roots.

| Metric | Independent | Search min | Search mean | Search max | Mean delta | Max delta |
|---|---:|---:|---:|---:|---:|---:|
| overall_mean | 0.752206 | 0.707818 | 0.741002 | 0.778274 | -0.011204 | +0.026068 |
| subject_consistency | 0.917340 | 0.943367 | 0.961601 | 0.972639 | +0.044261 | +0.055299 |
| background_consistency | 0.941914 | 0.950216 | 0.957438 | 0.962843 | +0.015524 | +0.020929 |
| motion_smoothness | 0.964316 | 0.958046 | 0.972931 | 0.980230 | +0.008615 | +0.015914 |
| dynamic_degree | 0.587500 | 0.335417 | 0.495247 | 0.695833 | -0.092253 | +0.108333 |
| aesthetic_quality | 0.483186 | 0.420958 | 0.459473 | 0.498473 | -0.023714 | +0.015286 |
| imaging_quality | 0.618983 | 0.516772 | 0.599323 | 0.664446 | -0.019660 | +0.045463 |

This table shows why random I2V is not enough. The mean path is below independent overall, and is especially worse on dynamic degree, aesthetic quality, and imaging quality. But the best path recovers or improves every listed metric, including a large `+0.1083` gain on dynamic degree and a `+0.0455` gain on imaging quality. The main exception is when selecting a **single path by overall score** rather than an oracle per-metric path: that overall-best path still improves overall but remains slightly worse on aesthetic quality (`-0.0118`).

The search gain is mainly from recovering dynamic degree and imaging quality while retaining the consistency benefits of I2V. This supports the hypothesis that I2V continuation has high variance: many paths are weak, but some paths preserve continuity and still produce enough motion and detail. Selecting among branches avoids committing to a poor continuation trajectory.

## Interpretation

The three no-search options trade off different failure modes:

- **Direct 481-frame T2V** is too static. It gets excellent consistency and smoothness, but the dynamic degree score is extremely low. This makes it a poor default for VBench-Long.
- **Independent chunk concat** is the best no-search VBench baseline. It preserves the local 81-frame generation regime and keeps motion/imaging stronger, but visual continuity across chunks can be poor because chunks are independent.
- **T2V + I2V chaining** is conceptually better for continuity, but without search it underperforms. The chain often becomes conservative or accumulates artifacts, reducing dynamic degree, aesthetics, and imaging quality.

Search helps because the bad result is not that I2V continuation cannot work; it is that a single sampled path is unreliable. Branching the continuation seeds exposes multiple futures from the same first chunk. The best path often keeps the I2V continuity advantage while avoiding low-motion or degraded continuations.

## Recommendation

For no-search generation, use **independent T2V chunk concat** as the VBench-Long baseline. Do not use direct 481-frame T2V as the default unless the goal is static consistency rather than dynamic long video.

For test-time scaling, use **T2V first chunk + branched I2V continuations**. Current evidence shows that best-of-32 path selection can beat independent concat on all-prompts/2-seed evaluation. The remaining open problem is selection: our current best-of-32 result uses VBench-Long as an oracle. A practical pipeline needs a cheaper proxy ranker or automatic scoring rule that correlates with VBench and can select the good continuation path without using the final benchmark metric directly.

Relevant result files:

- Three-way full comparison: `runs/vbench_long_compare/three_way_full_30s/full_method_summary.csv`
- Branch all-prompts summary: `runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/chunk_branch_summary.csv`
- Branch root details: `runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/chunk_branch_by_root.csv`
- Branch path details: `runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/chunk_branch_by_path.csv`
