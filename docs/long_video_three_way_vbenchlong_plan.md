# Three-Way Wan 30s VBench-Long Comparison

## Summary

Compare three no-training 30s settings with identical prompts, seeds, resolution,
fps, steps, guidance, and VBench-Long dimensions:

1. `independent_t2v_chunks`: existing `6 x 81` T2V chunks, stitched with 1-frame overlap.
2. `direct_t2v`: one Wan T2V call with `num_frames=481`.
3. `last_frame_i2v_chunks`: chunk 0 T2V, chunks 1-5 I2V from the previous chunk's last frame.

Run order:

- Pilot: `p01` (`a person swimming in ocean`) x seeds `0-4`.
- Full: all 15 dev prompts x seeds `0-9` = 150 videos per method.

## Minimal-Change Policy

- Keep the current T2V adapter, independent-concat behavior, output layout, and
  VBench-Long evaluator structure intact.
- Add direct 481-frame T2V by reusing the existing `Wan22Adapter.generate(...)` path.
- Keep I2V separate where practical. The I2V helper may wrap the already-loaded
  T2V pipeline components to avoid a second 5B model copy on the same GPU, but
  it should not change existing T2V behavior.
- If I2V cannot produce the same `480x832` / 481-frame setting, mark that method
  blocked rather than mixing resolutions or models.

## Implementation Changes

- Add generation modes to the long-video runner:
  - `independent_t2v_chunks`
  - `direct_t2v`
  - `last_frame_i2v_chunks`
- Preserve output layout:
  - `runs/baseline_long/<run_id>/<prompt_id>/seedXXXX/video.mp4`
  - `meta.json`
  - `DONE`
- Add method metadata: `generation_mode`, `target_num_frames`, and `conditioning`.
- Add VBench-Long prompt filtering with `--prompt-ids`, while keeping seed filtering.
- Add a CSV comparison summarizer under `runs/vbench_long_compare/<comparison_id>/`.

## Execution Plan

- Feasibility smoke:
  - direct T2V: `p01`, seed `0`, `num_frames=481`
  - I2V continuation: `p01`, seed `0`, 2 chunks first if needed, then 6 chunks
  - verify each output is 481 frames at 16 fps before scoring
- Pilot:
  - independent concat: reuse `runs/baseline_long/long_vbench_30s_20260618_190202`
    with `--prompt-ids p01 --seeds 0-4`
  - direct T2V: generate `p01 x seeds 0-4`
  - I2V chunks: generate `p01 x seeds 0-4`
- Full:
  - independent concat: reuse existing 150-video run and score
  - direct T2V: all 15 prompts x 10 seeds
  - I2V chunks: all 15 prompts x 10 seeds

## Checks

- Static checks:
  - `python -m compileall` on touched Python files
  - `bash -n` on touched shell scripts
- Generation checks:
  - pilot methods each have 5 `DONE` markers
  - full direct/I2V methods each have 150 `DONE` markers
  - every generated output has exactly 481 frames
- Evaluation checks:
  - pilot score rows: `1 prompt x 5 seeds x 6 dims = 30`
  - full score rows: `15 prompts x 10 seeds x 6 dims = 900`
  - comparison includes per-dimension means and `overall_mean`
