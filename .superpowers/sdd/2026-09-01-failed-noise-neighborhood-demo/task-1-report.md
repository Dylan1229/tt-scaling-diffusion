# Task 1 Report

## Status
DONE

## What I implemented or attempted
- Built `ttsd/runners/generate/noise_neighborhood_demo.py` with deterministic neighbor construction, parent-latent capture/reinjection, atomic artifact writes, resume-safe sample skipping, sharding, and a lazy CLI.
- Added `tests/test_noise_neighborhood_demo.py` with CPU-only checks for exact noise math, shard partitioning, and contact-sheet sizing.
- Saved the explicit parent control artifacts under `parent_control/` and the 32 neighbor tensors under `noise/` with a manifest.

## Tests and exact results
- Focused red test on remote:
  - `python -m pytest tests/test_noise_neighborhood_demo.py -q`
  - Result: 3 failed with `ModuleNotFoundError: No module named 'ttsd.runners.generate.noise_neighborhood_demo'`
- Focused green test on remote:
  - `python -m pytest tests/test_noise_neighborhood_demo.py -q`
  - Result: `3 passed in 0.74s`
- CLI lazy-load check on remote:
  - `python -m ttsd.runners.generate.noise_neighborhood_demo --help >/dev/null`
  - Result: passed
- Full suite on remote:
  - `python -m pytest -q`
  - Result: `11 passed in 0.75s`
- Ruff on remote:
  - `python -m pip install ruff`
  - `python -m ruff check ttsd/runners/generate/noise_neighborhood_demo.py tests/test_noise_neighborhood_demo.py`
  - Result: `All checks passed!`

## TDD RED command
- `python -m pytest tests/test_noise_neighborhood_demo.py -q`
- Expected failing output: import failure for `ttsd.runners.generate.noise_neighborhood_demo`.
- Why it failed: the runner module did not exist yet.

## TDD GREEN command
- `python -m pytest tests/test_noise_neighborhood_demo.py -q`
- Passing output: `3 passed in 0.74s`

## Files changed
- `ttsd/runners/generate/noise_neighborhood_demo.py`
- `tests/test_noise_neighborhood_demo.py`

## Self-review findings
- The test import is runtime-only, so pytest fails in the test body rather than during collection.
- The runner stays lazy: `--help` works without loading Diffusers or the model stack.
- Parent reinjection is validated before control artifacts are saved.
- The contact sheet now uses a safe default font and atomic JPEG writes.

## Issues or concerns
- Remote verification needed `python -m pip install ruff` because Ruff was not preinstalled in the shared venv.
- Commit: `2bdc965 Tighten noise neighborhood prep metadata`

## Fix Round 1
- Changed behavior: preparation now finishes only after `prepare/DONE` exists and the full bundle is present (`parent_noise.pt`, `manifest.json`, `parent_control/{video.mp4,all_frames.jpg,meta.json,DONE}`); `fps` is recorded in parent metadata, neighbor metadata, and `manifest.json`.
- Focused RED command: `python -m pytest tests/test_noise_neighborhood_demo.py -q`
- Focused RED output: `....F` / `KeyError: 'fps'` in `test_metadata_records_fixed_fps_everywhere`.
- Focused GREEN command: `python -m pytest tests/test_noise_neighborhood_demo.py -q`
- Focused GREEN output: `..... [100%] 5 passed in 1.58s`
- Full-suite command: `python -m pytest -q`
- Full-suite output: `............. [100%] 13 passed in 1.63s`
- Ruff command: `python -m ruff check ttsd/runners/generate/noise_neighborhood_demo.py tests/test_noise_neighborhood_demo.py`
- Ruff output: `All checks passed!`
- Note: the remote worktree briefly had stray root-level duplicate files from the sync path; those were removed before the final passing suite.
