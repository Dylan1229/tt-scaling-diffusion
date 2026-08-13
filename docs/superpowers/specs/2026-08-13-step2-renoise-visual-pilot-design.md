# Step-2 RENOISE Visual Pilot Design

## Goal

Test whether a partial noise perturbation immediately after denoising Step 2 can cause a visually significant final-video change, approaching the qualitative diversity of an independent-seed sample without fully replacing the trajectory noise.

This is a fast qualitative pilot, not the full amplitude trade-off study.

## Fixed experiment

- Model: existing Wan 2.2 TI2V-5B setup
- Sampler: UniPC
- Inference steps: 50
- Branch point: immediately after Step 2
- Root seed: 0
- Perturbation amplitudes: `0.0`, `0.2`, `0.4`, `0.8`
- Independent visual reference: an ordinary generation with seed 1
- No amplitude `1.0`
- No DINO, VBench, or other quantitative evaluation

The three prompts are existing reproducible dev prompts:

| ID | Axis | Text |
|---|---|---|
| `p01` | subject consistency | `a person swimming in ocean` |
| `p03` | multiple objects | `a bird and a cat` |
| `p05` | spatial relationship | `a bicycle on the left of a car, front view` |

## RENOISE operation

At the branch point, use the model's Step-2 prediction to decompose the post-step latent according to the flow-matching parameterization:

\[
x_t = (1-\sigma_t)\hat{x}_0 + \sigma_t\hat{\epsilon}.
\]

Recover the trajectory's implied noise component:

\[
\hat{\epsilon} = \frac{x_t-(1-\sigma_t)\hat{x}_0}{\sigma_t}.
\]

For amplitude \(\alpha\), rotate that noise component toward fresh Gaussian noise while preserving its expected variance:

\[
\hat{\epsilon}_\alpha = \sqrt{1-\alpha^2}\hat{\epsilon} + \alpha\epsilon_{new},
\qquad \epsilon_{new}\sim\mathcal{N}(0,I).
\]

Reconstruct the branch latent:

\[
x_t^{(\alpha)} = (1-\sigma_t)\hat{x}_0 + \sigma_t\hat{\epsilon}_\alpha.
\]

All amplitudes for one prompt share the same root trajectory and the same `epsilon_new`, so differences across columns are attributable to perturbation amplitude rather than direction. `alpha=0.0` must reconstruct the unperturbed post-Step-2 latent, within numerical tolerance.

This operation is explicit in project code. The installed Diffusers scheduler does not automatically rescale a latent modified by the callback.

## Outputs

Keep each generated MP4 and machine-readable metadata recording prompt, root seed, branch step, sigma, amplitude, and perturbation seed.

Create one synchronized looping HTML comparison with three prompt rows and five labeled columns:

1. `alpha=0.0`
2. `alpha=0.2`
3. `alpha=0.4`
4. `alpha=0.8`
5. `independent seed=1`

Videos should autoplay muted, loop, use controls, and begin together when the page loads. The page is a local experiment artifact, not a production UI.

## Execution order

1. Run a `p01` smoke generation and verify all four branch videos decode and `alpha=0.0` matches the unperturbed branch.
2. Run the complete three-prompt pilot on `yukelab` GPUs.
3. Build the comparison page.
4. Inspect the videos manually, including motion over time rather than only first frames.
5. Report per-prompt observations and an overall answer to whether Step-2 partial RENOISE creates significant structural variation, and at which tested amplitude it first becomes obvious.

## Manual evaluation rubric

For each amplitude and prompt, compare against `alpha=0.0` on:

- scene layout and camera framing;
- subject identity, count, and spatial arrangement;
- action or motion trajectory;
- background and appearance details;
- visible artifacts or collapse.

Use the independent seed only as a qualitative scale reference. The pilot supports the hypothesis if one or more partial amplitudes produce unmistakable structural or motion changes while remaining coherent. It does not claim distributional equivalence to Best-of-N.

## Out of scope

- Automated similarity metrics or thresholds
- Statistical claims from three prompts
- Step 1, Step 5, or Step 10
- A denser amplitude sweep
- Prompt rewriting or prompt weighting
- Full Best-of-N quality evaluation
