# Failed-Noise Neighborhood Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and visually review 32 variance-preserving Gaussian neighbors around the known failed Wan 2.2 seed 0 initial noise.

**Architecture:** Add one self-contained experiment runner that captures the parent noise from the unchanged Diffusers pipeline, verifies explicit reinjection, materializes deterministic neighbor tensors, and generates resume-safe videos plus all-frame contact sheets. Keep all semantic judgments manual and write one concise results report after reviewing every output.

**Tech Stack:** Python 3.10+, PyTorch, Diffusers 0.38 on the experiment host, Pillow, NumPy, pytest, Wan 2.2 TI2V-5B, UniPC.

**Spec:** `docs/superpowers/specs/2026-09-01-failed-noise-neighborhood-demo-design.md`

## Global Constraints

- Run all Python, tests, and model workloads on `yukelab` in `/data/datasets/peihao/tt-scaling-diffusion`; use the local checkout only for editing and orchestration.
- Fix generation to the existing red-ball setup: Wan 2.2 TI2V-5B, seed 0 parent, 480×832, 81 frames, 50 steps, guidance 5.0, and 24 fps.
- Keep the prompt, input image, model, scheduler, timesteps, and conditioning identical; only initial latent noise may vary.
- Use exactly eight neighbors at each `a` in `(0.02, 0.05, 0.10, 0.20)`, with `z = sqrt(1-a²) * z_fail + a * e` and independent standard-Gaussian `e`.
- Do not add an automatic semantic scorer. Review every frame manually and label success, failure, or ambiguous.
- Success means the ball clearly enters the blue goal at least once; it need not remain there.
- Treat counts as a feasibility result, not a probability estimate.
- Keep this as one experiment runner and one CPU-only test file; do not add a reusable search framework.
- Generated tensors, videos, sheets, logs, and manifests remain under ignored `runs/`; only source, tests, plan, and final report are committed.

## File Structure

- Create `ttsd/runners/generate/noise_neighborhood_demo.py`: fixed experiment constants, deterministic neighbor construction, parent capture/reinjection, generation CLI, atomic artifacts, sharding, and all-frame contact sheets.
- Create `tests/test_noise_neighborhood_demo.py`: CPU-only checks for neighborhood construction, metrics, deterministic sample enumeration/sharding, and contact-sheet coverage.
- Create after review `docs/failed_noise_neighborhood_demo_2026-09-01.md`: methodology, all 32 manual labels, successful/ambiguous examples, closest success, and limitations.
- Generate without committing `runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/`: parent/noise artifacts, videos, contact sheets, metadata, manifest, completion markers, and review labels.

---

### Task 1: Build the deterministic, resume-safe experiment runner

**Files:**
- Create: `ttsd/runners/generate/noise_neighborhood_demo.py`
- Create: `tests/test_noise_neighborhood_demo.py`

**Interfaces:**
- Produces: `neighbor_specs() -> list[dict[str, int | float | str]]`
- Produces: `specs_for_shard(shard_index: int, num_shards: int) -> list[dict[str, int | float | str]]`
- Produces: `make_neighbor(parent: torch.Tensor, alpha: float, perturb_seed: int) -> torch.Tensor`
- Produces: `noise_metrics(parent: torch.Tensor, neighbor: torch.Tensor) -> dict[str, float]`
- Produces: `save_contact_sheet(frames, path: Path, columns: int = 9, thumb_size: tuple[int, int] = (208, 120)) -> None`
- Produces CLI: `python -m ttsd.runners.generate.noise_neighborhood_demo [--prepare-only] [--indices N ...] [--shard-index N] [--num-shards N] [--output-root PATH]`

- [ ] **Step 1: Write the failing CPU tests**

Create `tests/test_noise_neighborhood_demo.py` with these focused behaviors:

```python
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from ttsd.runners.generate.noise_neighborhood_demo import (
    ALPHAS,
    make_neighbor,
    neighbor_specs,
    noise_metrics,
    save_contact_sheet,
    specs_for_shard,
)


def test_neighbor_construction_is_exact_and_reproducible() -> None:
    parent = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)

    torch.testing.assert_close(make_neighbor(parent, 0.0, 7), parent, rtol=0, atol=0)
    first = make_neighbor(parent, 0.1, 7)
    second = make_neighbor(parent, 0.1, 7)
    other = make_neighbor(parent, 0.1, 8)

    generator = torch.Generator(device="cpu").manual_seed(7)
    epsilon = torch.randn(parent.shape, generator=generator, dtype=torch.float32)
    expected = math.sqrt(1.0 - 0.1**2) * parent + 0.1 * epsilon

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
    assert first.shape == parent.shape
    assert first.dtype == torch.float32

    metrics = noise_metrics(parent, first)
    assert set(metrics) == {"rms_distance", "cosine_similarity", "norm_ratio"}
    assert metrics["rms_distance"] > 0
    assert -1 <= metrics["cosine_similarity"] <= 1
    assert metrics["norm_ratio"] > 0


def test_specs_are_32_unique_reproducible_neighbors_partitioned_once() -> None:
    specs = neighbor_specs()
    assert ALPHAS == (0.02, 0.05, 0.10, 0.20)
    assert len(specs) == 32
    assert len({spec["sample_id"] for spec in specs}) == 32
    assert [sum(spec["alpha"] == alpha for spec in specs) for alpha in ALPHAS] == [8, 8, 8, 8]

    shards = [specs_for_shard(i, 4) for i in range(4)]
    assert all(len(shard) == 8 for shard in shards)
    assert sorted(spec["index"] for shard in shards for spec in shard) == list(range(32))


def test_contact_sheet_contains_every_frame(tmp_path) -> None:
    frames = [np.full((4, 6, 3), value / 4, dtype=np.float32) for value in range(5)]
    output = tmp_path / "sheet.jpg"

    save_contact_sheet(frames, output, columns=3, thumb_size=(6, 4))

    with Image.open(output) as sheet:
        assert sheet.size == (18, 8)
```

- [ ] **Step 2: Sync the failing test to the remote checkout and verify the red state**

Run from the local worktree root:

```bash
rsync -a tests/test_noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/tests/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && pytest tests/test_noise_neighborhood_demo.py -q'
```

Expected: failure during collection because `ttsd.runners.generate.noise_neighborhood_demo` does not exist.

- [ ] **Step 3: Implement the pure neighborhood and review helpers minimally**

Start `ttsd/runners/generate/noise_neighborhood_demo.py` with fixed constants and lazy Diffusers imports so the CPU tests do not load the model stack:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

MODEL = Path(
    "/data/datasets/fanjiang/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/"
    "b8fff7315c768468a5333511427288870b2e9635"
)
INPUT = Path("runs/toy_red_ball_i2v_v2/input.png")
INPUT_SHA256 = "b1aad4e150009199e5a59c2f7867e32d9ae229d5f923dc53ffc857f14f95a8c9"
OUTPUT_ROOT = Path("runs/toy_red_ball_i2v_v2/noise_neighborhood_v1")
PROMPT = (
    "Static camera. A red ball moves in a straight horizontal line from left to right, "
    "enters through the open left side of a stationary blue box, and stops inside. "
    "The ball stays at the same height. The box does not move or change shape."
)
ALPHAS = (0.02, 0.05, 0.10, 0.20)
NEIGHBORS_PER_ALPHA = 8
PARENT_SEED = 0
HEIGHT, WIDTH, NUM_FRAMES, STEPS = 480, 832, 81, 50
GUIDANCE_SCALE, FPS = 5.0, 24


def neighbor_specs() -> list[dict[str, int | float | str]]:
    return [
        {
            "index": index,
            "alpha": alpha,
            "perturb_seed": 10_000 + index,
            "sample_id": f"n{index:02d}_a{round(alpha * 100):03d}",
        }
        for alpha_index, alpha in enumerate(ALPHAS)
        for local_index in range(NEIGHBORS_PER_ALPHA)
        for index in [alpha_index * NEIGHBORS_PER_ALPHA + local_index]
    ]


def specs_for_shard(shard_index: int, num_shards: int) -> list[dict[str, int | float | str]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [spec for spec in neighbor_specs() if int(spec["index"]) % num_shards == shard_index]


def make_neighbor(parent: torch.Tensor, alpha: float, perturb_seed: int) -> torch.Tensor:
    if not 0 <= alpha < 1:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    parent = parent.detach().to(device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(perturb_seed)
    epsilon = torch.randn(parent.shape, generator=generator, dtype=torch.float32)
    return math.sqrt(1.0 - alpha**2) * parent + alpha * epsilon


def noise_metrics(parent: torch.Tensor, neighbor: torch.Tensor) -> dict[str, float]:
    parent = parent.float().flatten()
    neighbor = neighbor.float().flatten()
    delta = neighbor - parent
    return {
        "rms_distance": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(parent, neighbor, dim=0)),
        "norm_ratio": float(neighbor.norm() / parent.norm()),
    }
```

Implement `save_contact_sheet` by converting PIL or NumPy frames to RGB, fitting every frame into a fixed tile, placing tiles left-to-right then top-to-bottom, writing each frame index in its tile, and atomically renaming a temporary JPEG. Its output dimensions must be `columns * thumb_width` by `ceil(frame_count / columns) * thumb_height`.

- [ ] **Step 4: Add exact parent capture, reinjection, and atomic artifacts**

Keep heavy imports inside `load_pipeline()` and preserve the existing setup exactly:

```python
def load_pipeline():
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(
        MODEL, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    if type(pipe.scheduler).__name__ != "UniPCMultistepScheduler":
        raise RuntimeError(f"expected UniPCMultistepScheduler, got {type(pipe.scheduler).__name__}")
    return pipe


def run_pipeline(pipe, image: Image.Image, *, seed: int | None = None, latents: torch.Tensor | None = None):
    kwargs = {
        "image": image,
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    if latents is None:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        kwargs["latents"] = latents.clone()
    return pipe(**kwargs).frames[0]
```

Capture the normal seed 0 latent by temporarily wrapping `pipe.prepare_latents`, cloning `outputs[0]` to CPU float32, and always restoring the original method in `finally`. Run the pipeline a second time with that explicit latent and require the two decoded frame arrays to agree within absolute tolerance `1e-5`; save only the explicit parent video and sheet as the control artifacts.

Add atomic helpers for JSON, Torch tensors, videos, and `DONE` markers. A neighbor is complete only when its `video.mp4`, `all_frames.jpg`, `meta.json`, and `DONE` marker exist. Metadata must include the sample spec, noise metrics, fixed generation values, prompt, input checksum, model path, scheduler class, Diffusers version, Torch version, elapsed seconds, and peak GPU memory.

During preparation:

1. Require `INPUT` to exist and match `INPUT_SHA256`.
2. Capture and atomically save `parent_noise.pt`.
3. Save the explicit parent control video, all-frame sheet, and metadata.
4. Materialize all 32 neighbor tensors under `noise/` and write `manifest.json` with their specs and metrics.

During generation, load only the selected neighbor tensors, run each at batch size one, write artifacts atomically, and skip complete samples.

- [ ] **Step 5: Add the minimal CLI and shard behavior**

Use these arguments only:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser
```

For serial use, prepare automatically when `parent_noise.pt` is absent, then generate all selected samples. For multi-GPU use, require preparation to finish first, filter with `specs_for_shard`, then intersect with `--indices` when supplied. Reject indices outside `0..31`, reject an incomplete preparation manifest, and print one concise JSON metadata line per completed sample.

- [ ] **Step 6: Sync implementation and verify the green state remotely**

```bash
rsync -a ttsd/runners/generate/noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/ttsd/runners/generate/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && pytest tests/test_noise_neighborhood_demo.py -q'
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && python -m ttsd.runners.generate.noise_neighborhood_demo --help >/dev/null'
```

Expected: all new tests pass and the CLI exits successfully without loading the model.

- [ ] **Step 7: Run focused regression checks**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && pytest -q'
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && ruff check ttsd/runners/generate/noise_neighborhood_demo.py tests/test_noise_neighborhood_demo.py'
```

Expected: the complete suite passes and Ruff reports no errors.

- [ ] **Step 8: Commit the tested runner**

```bash
git add ttsd/runners/generate/noise_neighborhood_demo.py tests/test_noise_neighborhood_demo.py
git commit -m "Add failed-noise neighborhood demo"
```

---

### Task 2: Run the parent validation, pilot, and 32-neighbor generation

**Files:**
- Generate: `runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/`

**Interfaces:**
- Consumes: Task 1 CLI and the exact input image checksum.
- Produces: one validated parent control, 32 complete neighbor directories, `manifest.json`, and all-frame sheets for manual review.

- [ ] **Step 1: Verify remote access, input, environment, and idle GPUs**

Use context-mode to run and summarize:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && sha256sum runs/toy_red_ball_i2v_v2/input.png && source .venv/bin/activate && python -c "import diffusers, torch; print(diffusers.__version__, torch.__version__)" && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

Require the input checksum from the global constraints, Diffusers 0.38-compatible explicit-latent support, and enough idle GPUs for one model replica each. If fewer than four GPUs are suitable, reduce `num_shards` rather than changing generation settings.

- [ ] **Step 2: Prepare and validate the failed parent on one GPU**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=4 python -u -m ttsd.runners.generate.noise_neighborhood_demo --prepare-only 2>&1 | tee runs/toy_red_ball_i2v_v2/noise_neighborhood_v1_prepare.log'
```

Copy `parent_control/all_frames.jpg` locally and inspect all 81 frames. Continue only if it reproduces the known failure: the ball does not clearly enter the goal. Also verify `manifest.json` contains 32 unique samples, eight per perturbation level, and all noise files exist.

- [ ] **Step 3: Run one end-to-end neighbor pilot**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=4 python -u -m ttsd.runners.generate.noise_neighborhood_demo --indices 0 2>&1 | tee runs/toy_red_ball_i2v_v2/noise_neighborhood_v1_pilot.log'
```

Require the pilot directory to contain `video.mp4`, `all_frames.jpg`, `meta.json`, and `DONE`. Inspect the all-frame sheet for chronological coverage and readable image quality; the pilot itself may succeed or fail semantically.

- [ ] **Step 4: Launch the full resume-safe generation**

For four suitable GPUs, launch one detached process per shard, using GPUs selected from the availability check:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && for pair in 4:0 5:1 6:2 7:3; do gpu=${pair%%:*}; shard=${pair##*:}; tmux new-session -d -s noise_nb_gpu${gpu} "cd /data/datasets/peihao/tt-scaling-diffusion && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=${gpu} python -u -m ttsd.runners.generate.noise_neighborhood_demo --shard-index ${shard} --num-shards 4 2>&1 | tee runs/toy_red_ball_i2v_v2/noise_neighborhood_v1_gpu${gpu}.log"; done'
```

If only one GPU is available, run the same module without shard arguments. Do not launch two model processes on one GPU.

- [ ] **Step 5: Monitor completion and investigate failures before retrying**

Periodically summarize logs and artifact counts with context-mode. Completion requires exactly 32 neighbor `DONE` markers. For any failed process, inspect its first error and artifact state, apply systematic debugging, and relaunch the same shard; completed neighbors must be skipped.

- [ ] **Step 6: Validate the completed artifact set**

Run one remote Python check that asserts:

- 32 manifest entries and 32 unique sample IDs
- eight entries for each alpha
- 32 `DONE` markers
- every video, sheet, metadata file, and saved noise tensor is non-empty
- metadata generation constants match the fixed setup
- metadata metrics match the manifest

Print only a concise pass/fail summary and any mismatched sample IDs.

---

### Task 3: Review every video and report the feasibility result

**Files:**
- Create: `docs/failed_noise_neighborhood_demo_2026-09-01.md`
- Generate without committing: `runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/review_labels.json`
- Generate without committing: `runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/comparison_montage.jpg`

**Interfaces:**
- Consumes: Task 2 all-frame sheets, videos, manifest, and metadata.
- Produces: one label per neighbor and a user-facing answer stating whether any local perturbation succeeded.

- [ ] **Step 1: Copy review-sized artifacts locally**

Copy the 32 all-frame sheets, parent sheet, manifest, and metadata into the ignored local run directory. Do not copy latent tensors unless needed for diagnosing a mismatch.

- [ ] **Step 2: Label all 32 neighbors manually**

Inspect each all-frame sheet in index order and open the corresponding video whenever a sheet is unclear. Assign exactly one label:

- `success`: the ball clearly enters the goal at least once
- `failure`: no clear entry occurs
- `ambiguous`: image quality or object behavior prevents a confident judgment

Write `review_labels.json` containing sample ID, index, alpha, label, and one short observation. Do not infer labels from alpha, metrics, or the known seed 4 result.

- [ ] **Step 3: Verify review completeness and summarize results**

Run a small Python check that joins `review_labels.json` to `manifest.json`, rejects missing or duplicate labels, and prints counts by alpha plus the successful sample with highest cosine similarity to the parent. If there is no success, report that none of these 32 tested neighbors succeeded rather than claiming the neighborhood contains no success.

- [ ] **Step 4: Build a compact comparison montage**

Use Pillow to combine the parent failure, every success, every ambiguous sample, and at most one representative failure per alpha. Label each panel with sample ID, alpha, cosine similarity, and manual outcome. Keep the full 81-frame sheets as the source of truth.

- [ ] **Step 5: Write the results report**

Create `docs/failed_noise_neighborhood_demo_2026-09-01.md` with:

1. Fixed generation setup and perturbation formula
2. Parent reinjection validation result
3. A 32-row table of sample ID, alpha, perturbation seed, cosine similarity, RMS distance, and manual label
4. Counts by alpha
5. Closest successful neighbor, or the precise statement that no tested neighbor succeeded
6. Links or paths to the comparison montage and raw run directory
7. Limitations: manual semantic judgment, eight samples per level, and no probability claim

- [ ] **Step 6: Verify and commit the report**

Check that the table has exactly 32 unique rows and its totals match `review_labels.json`, then run:

```bash
git add docs/failed_noise_neighborhood_demo_2026-09-01.md
git commit -m "Report failed-noise neighborhood demo"
```

- [ ] **Step 7: Final verification before claiming completion**

Freshly verify the remote artifact check, the full pytest suite, the report totals, and repository status. Report the observed outcome, closest successful perturbation if present, ambiguous cases, and the compute/access limitation if any part could not run.
