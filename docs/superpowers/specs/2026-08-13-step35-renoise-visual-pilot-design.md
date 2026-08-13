# Step-35 RENOISE Visual Pilot Design

## Goal

Test whether a large fixed-noise-level RENOISE rotation still produces visible structural or motion variation when branching late in a 50-step Wan 2.2 UniPC trajectory.

## Experiment

- Model and generation settings: reuse the Step-2 pilot settings unchanged.
- Prompts:
  - `p01`: `a person swimming in ocean`
  - `p03`: `a bird and a cat`
  - `p05`: `a bicycle on the left of a car, front view`
- Root seed: `0`.
- Branch immediately after one-based denoising Step 35.
- Amplitudes, displayed in ascending order: `[0.0, 0.4, 0.6, 0.8, 1.0]`.
- All amplitudes for one prompt reuse the same fresh-noise direction.
- Do not generate another independent-seed reference; the completed Step-2 pilot already provides that qualitative reference.

At the measured post-Step-35 scheduler noise level `sigma`, construct each branch as

\[
\epsilon_\alpha=\sqrt{1-\alpha^2}\,\hat\epsilon+\alpha\epsilon_{new},
\qquad
x_t^{(\alpha)}=(1-\sigma)\hat{x}_0+\sigma\epsilon_\alpha.
\]

`alpha=1.0` fully replaces the noise direction while preserving the Step-35 noise level. It does not replace the whole latent with pure noise and does not move the scheduler to an earlier timestep.

## Execution

1. Reuse the existing configurable RENOISE adapter and pilot runner rather than add another generation path.
2. Add only a Step-35 experiment configuration and any minimal runner generalization required for labels/output naming.
3. Run `p01` first as a GPU smoke test, then resume for all three prompts.
4. Record the actual post-Step-35 sigma in metadata.

## Outputs

Under a distinct Step-35 run directory, produce:

- 15 raw MP4 files: 3 prompts × 5 amplitudes;
- per-video metadata and a manifest;
- a labeled synchronized 3×5 HTML comparison grid;
- a concise manual-review report in Chinese.

## Evaluation

This is a visual-only theory check. Watch every video and report, per prompt:

- whether each nonzero amplitude visibly differs from `alpha=0`;
- the first amplitude with clear structural or motion change;
- whether `alpha=1.0` remains coherent;
- whether late branching mostly changes structure/motion or only local appearance/detail.

No DINO, VBench, quantitative threshold, or claim of distributional equivalence is required.

## Success Condition

The run is complete when all 15 videos decode fully, the synchronized grid is usable, and manual review can answer whether large fixed-sigma RENOISE at Step 35 produces meaningful late-stage variation.
