# Step-2 Six-Amplitude RENOISE Visual Pilot Design

## Goal

Complete the Step-2 visual comparison by adding `alpha=0.6` and `alpha=1.0`, while removing the independent-seed column from the new comparison.

## Experiment

Reuse the existing Wan 2.2 Step-2 fixed-noise-level RENOISE implementation and generation settings:

- Model: Wan 2.2 TI2V 5B.
- Scheduler: UniPC, 50 inference steps.
- Branch immediately after one-based denoising Step 2.
- Root seed: `0`.
- Fresh-noise seed: `10000000`.
- Prompts:
  - `p01`: `a person swimming in ocean`
  - `p03`: `a bird and a cat`
  - `p05`: `a bicycle on the left of a car, front view`
- Amplitudes, displayed in ascending order: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- All six amplitudes for one prompt are generated in one batched suffix and reuse the same fresh-noise direction.
- Do not generate or display `independent seed=1` in v2.

Generating all six amplitudes together avoids comparing suffixes produced with different batch sizes and preserves the shared-noise-direction contract.

## Artifact Versioning

- Keep the completed v1 run unchanged for provenance.
- Write the new run to `runs/step2_renoise_visual_pilot/step2_renoise_visual_v2/`.
- Produce 18 MP4 files: 3 prompts × 6 amplitudes.
- Produce per-video metadata, manifest, and a synchronized labeled 3×6 HTML comparison page.
- Produce a concise Chinese manual review.

## Runner Changes

Reuse the existing runner and adapter; do not add another generation path. Update the approved Step-2 profile to the six amplitudes with `independent_seed: null`. The Step-35 profile and completed artifacts must remain unchanged.

The comparison HTML grid must derive its column count from the manifest instead of assuming five columns, so both the Step-2 3×6 page and Step-35 3×5 page render correctly.

## Execution

1. Add tests for the six-amplitude Step-2 profile, absence of an independent-seed column, dynamic six-column HTML, and unchanged Step-35 behavior.
2. Run `p01` as a GPU smoke test in the v2 directory.
3. Resume for all three prompts.
4. Fully decode all 18 videos and verify 81 frames, 832×480, 16 fps, Step 2 metadata, one branch sigma, and one shared fresh-noise seed.
5. Watch all videos and update the qualitative conclusion, especially the progression from `alpha=0.6` to `alpha=1.0`.
6. Rebuild the local mentor ZIP using Step-2 v2 and the existing Step-35 run; omit Step-2 v1 and independent-seed videos from the new ZIP.

## Success Condition

The pilot is complete when the Step-2 v2 page contains exactly six amplitude columns and 18 fully decodable videos, Step-35 remains valid, the manual review is written, and the refreshed mentor ZIP opens offline with both comparison pages and all referenced MP4 files.
