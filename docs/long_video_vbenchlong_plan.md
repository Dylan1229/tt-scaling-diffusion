# Minimal Long-Video Generation + VBench-Long Scoring

## Summary

Use 30s long videos because recent VBench-Long papers commonly report 30s as the main setting, with longer variants at 60s/120s/240s. For this repo, `81` frames are saved at `16 fps`, so one chunk is `81 / 16 = 5.06s`. A 30s video is therefore 6 chunks.

Default stitching:

- Chunk count: 6
- Frames per chunk: 81
- Overlap/drop: drop the first frame of chunks 2-6
- Final frame count: `81 + 5 * 80 = 481`
- Final duration at 16 fps: `30.06s`

## Key Changes

- Add a minimal long-generation runner:
  - CLI: `python -m ttsd.runners.generate.long_video --config configs/long_wan22_480p.yaml`
  - Output: `runs/baseline_long/<run_id>/<prompt_id>/seedXXXX/video.mp4`
  - Metadata extends the current run metadata with long-video fields.
- Reuse current prompt/seed logic:
  - Prompt source: `ttsd.prompts.dev_set:DEV_PROMPTS`
  - Seeds: `0..9`
  - This matches the existing 15 prompt x 10 seed result set.
- Keep v1 chunking simple:
  - Generate chunks independently with the same prompt.
  - Use deterministic chunk seeds: `seed + chunk_idx * 1000000`.
  - Chunk 0 keeps the original seed.
  - Concatenate frames after dropping the first frame from chunks after chunk 0.

## VBench-Long Evaluation

- Add an evaluator mirroring `ttsd.eval.vbench`, but using VBench-Long:
  - CLI: `python -m ttsd.eval.vbench_long --run runs/baseline_long/<run_id>`
  - Output: `runs/vbench_long/<run_id>/vbench_long_scores_long.csv`
  - Output: `runs/vbench_long/<run_id>/vbench_long_scores_summary.csv`
- Use VBench-Long `long_custom_input` for the same custom prompt subset.
- Score the custom-input dimensions supported by VBench-Long docs:
  - `subject_consistency`
  - `background_consistency`
  - `motion_smoothness`
  - `dynamic_degree`
  - `aesthetic_quality`
  - `imaging_quality`
- Stage one seed at a time to preserve prompt filenames exactly for VBench-Long:
  - `_staging/<dimension>/seed0000/<prompt_text>.mp4`
  - Run VBench-Long.
  - Aggregate results back to `(prompt_id, prompt_text, seed, dimension)`.

## Test Plan

- Smoke generation:
  - Run one prompt x one seed with `num_chunks=2`.
  - Verify `video.mp4` has `161` frames and valid `meta.json`.
- Full generation:
  - Run 15 prompts x 10 seeds x 6 chunks.
  - Verify `find runs/baseline_long/<run_id> -name DONE | wc -l == 150`.
- VBench-Long smoke:
  - Score one seed on `subject_consistency`.
  - Verify raw VBench-Long JSON exists and CSV rows include original `prompt_id` and `seed`.
- Full scoring:
  - Score all six VBench-Long custom dimensions over all 150 videos.
  - Report summary mean/std per prompt and dimension, plus overall mean across the six dimensions.

## Assumptions

- "Current model" means the existing Wan2.2 TI2V-5B adapter/config.
- "Exactly same prompt as current results" means `DEV_PROMPTS` and seeds `0..9`.
- Minimal v1 does not implement video continuation or cross-chunk conditioning.
- The default target is 30s, implemented as 6 stitched 81-frame chunks at 16 fps.
