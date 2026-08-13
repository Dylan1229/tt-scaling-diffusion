# Step-2 Six-Amplitude RENOISE Visual Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and review a Step-2 v2 comparison with six amplitudes `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, no independent-seed column, and a refreshed offline mentor ZIP.

**Architecture:** Reuse `Wan22Adapter.generate_with_renoise_branches` and the existing fixed-pilot runner. Change only the approved Step-2 profile and make the HTML grid use the manifest's actual column count; preserve the Step-35 profile and both prior run directories.

**Tech Stack:** Python 3.11, PyTorch, Diffusers 0.38.0, Wan 2.2 TI2V 5B, UniPC, pytest, imageio/FFmpeg, static HTML, Python stdlib `zipfile`.

## Global Constraints

- Work only on branch `codex/late-stage-recovery`.
- Run GPU generation on `yukelab:/data/datasets/peihao/tt-scaling-diffusion` using `.venv`.
- Use UniPC with 50 steps; branch immediately after one-based Step 2.
- Use root seed `0`, noise seed `10000000`, and amplitudes `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- Generate all six amplitudes together for each prompt so they share one fresh-noise direction and one suffix batch size.
- Do not generate or display an independent-seed reference in Step-2 v2.
- Preserve Step-2 v1 and Step-35 source artifacts unchanged.
- Write v2 to `runs/step2_renoise_visual_pilot/step2_renoise_visual_v2/`.
- Evaluate visually only; do not add DINO, VBench, or quantitative thresholds.

---

### Task 1: Update the Fixed Step-2 Profile and Dynamic Grid

**Files:**
- Modify: `tests/generate/test_step2_renoise_pilot.py`
- Modify: `ttsd/runners/generate/step2_renoise_pilot.py`
- Modify: `configs/step2_renoise_visual_pilot_wan22_480p.yaml`

**Interfaces:**
- Consumes: `build_comparison_html(manifest: dict) -> str`, `_row_from_artifacts(...) -> dict`, and `validate_config(cfg: dict) -> None`.
- Produces: an approved Step-2 profile with six amplitudes and no independent seed; HTML whose CSS grid has the same number of video columns as the manifest.

- [ ] **Step 1: Write failing profile and HTML tests**

Change the Step-2 test fixture to:

```python
"renoise": {
    "branch_step": 2,
    "amplitudes": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "root_seed": 0,
    "independent_seed": None,
    "noise_seed": 10_000_000,
}
```

Replace the old Step-2 3×5 test with a 3×6 manifest containing only six amplitude videos. Assert:

```python
assert html.count("<video") == 18
assert "repeat(6, minmax(260px, 1fr))" in html
assert "independent seed" not in html
for label in ("alpha=0.0", "alpha=0.2", "alpha=0.4", "alpha=0.6", "alpha=0.8", "alpha=1.0"):
    assert label in html
```

Keep the Step-35 five-amplitude test and add:

```python
assert "repeat(5, minmax(260px, 1fr))" in comparison
```

Update the invalid Step-2 independent-seed case to expect `independent_seed must be None`.

- [ ] **Step 2: Verify RED remotely**

Synchronize only the changed test file to the remote checkout and run:

```bash
.venv/bin/pytest -q tests/generate/test_step2_renoise_pilot.py
```

Expected: failures because the production profile still expects four amplitudes plus seed 1 and the HTML CSS still uses `repeat(5, ...)`.

- [ ] **Step 3: Implement the minimum production change**

Change the Step-2 profile to:

```python
EXPECTED_PILOTS = {
    2: ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], None),
    35: ([0.0, 0.4, 0.6, 0.8, 1.0], None),
}
```

In `build_comparison_html`, calculate `column_count = len(column_labels)` and interpolate it into:

```css
grid-template-columns: 180px repeat({column_count}, minmax(260px, 1fr));
```

Set the minimum width from the same count so six-column pages do not overlap:

```python
minimum_width = 180 + column_count * 270
```

Update the Step-2 YAML amplitudes and set `independent_seed: null`. Do not change the Step-35 profile.

- [ ] **Step 4: Verify GREEN and regression safety**

Run remotely:

```bash
.venv/bin/pytest -q tests/generate/test_step2_renoise_pilot.py
.venv/bin/pytest -q
```

Expected: all focused and full tests pass. Also run a lightweight real-`main()` integration check with a fake adapter to assert:

- Step 2 calls branching three times with six amplitudes and never calls independent generation;
- a second run makes no new generation calls;
- Step 35 still emits five amplitude columns.

- [ ] **Step 5: Commit and synchronize**

```bash
git add tests/generate/test_step2_renoise_pilot.py \
  ttsd/runners/generate/step2_renoise_pilot.py \
  configs/step2_renoise_visual_pilot_wan22_480p.yaml
git commit -m "feat: complete Step-2 renoise amplitude sweep"
git push origin codex/late-stage-recovery
```

Fast-forward the `yukelab` checkout and verify local, origin, and remote HEADs match with clean worktrees.

---

### Task 2: Generate Step-2 v2 on GPU

**Files:**
- Produce: `runs/step2_renoise_visual_pilot/step2_renoise_visual_v2/`

**Interfaces:**
- Consumes: the updated fixed-pilot runner and YAML.
- Produces: 18 MP4s, 18 metadata files, config snapshot, manifest, and synchronized 3×6 HTML.

- [ ] **Step 1: Run the remote suite and inspect GPU availability**

Run `.venv/bin/pytest -q` and `nvidia-smi`; do not start generation unless tests pass and a suitable GPU is available.

- [ ] **Step 2: Run the p01 smoke**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m ttsd.runners.generate.step2_renoise_pilot \
  --config configs/step2_renoise_visual_pilot_wan22_480p.yaml \
  --run-id step2_renoise_visual_v2 --prompt-ids p01
```

Verify six `p01` videos fully decode to 81 frames at 832×480 and share one `branch_sigma` and `noise_seed=10000000`.

- [ ] **Step 3: Generate p03 and p05**

Run the same command for `p03` and `p05` on separate free GPUs if available. After both finish, run the CLI once without `--prompt-ids` to rebuild the final manifest through the resume path.

Expected final labels:

```text
alpha=0.0, alpha=0.2, alpha=0.4, alpha=0.6, alpha=0.8, alpha=1.0
```

---

### Task 3: Verify, Review, and Refresh the Mentor ZIP

**Files:**
- Produce: `runs/step2_renoise_visual_pilot/step2_renoise_visual_v2/manual_review.md`
- Replace locally: `/Users/peihaoying/Downloads/renoise-step2-step35.zip`

**Interfaces:**
- Consumes: Step-2 v2 and existing Step-35 artifacts.
- Produces: full-decode evidence, a Chinese qualitative review, and an offline ZIP with 33 MP4s total (18 Step-2 v2 + 15 Step-35).

- [ ] **Step 1: Strictly verify all v2 artifacts**

Fully decode every manifest video and assert:

```python
assert len(manifest["rows"]) == 3
assert len(paths) == 18
assert all(frame_count == 81 for frame_count in frame_counts)
assert all(resolution == (832, 480) for resolution in resolutions)
assert labels == ["alpha=0.0", "alpha=0.2", "alpha=0.4", "alpha=0.6", "alpha=0.8", "alpha=1.0"]
```

Also assert 16 fps, `branch_step == 2`, root seed 0, one branch sigma, one noise seed, and no independent-seed path or label.

- [ ] **Step 2: Synchronize and visually review**

Copy the v2 directory locally, open `comparison.html`, and play all 18 videos from beginning to end. Write a Chinese per-prompt review emphasizing whether `.6`, `.8`, and `1.0` introduce structural/motion changes and whether `1.0` stays coherent.

- [ ] **Step 3: Rebuild the offline mentor ZIP**

Use Python stdlib `zipfile` to create a staging folder containing:

```text
renoise-step2-step35/
  index.html
  README.txt
  step2/       # Step-2 v2 only: 18 MP4s and its page/report
  step35/      # Existing Step-35: 15 MP4s and its page/report
```

Do not include Step-2 v1 or any independent-seed video. Replace:

```text
/Users/peihaoying/Downloads/renoise-step2-step35.zip
```

- [ ] **Step 4: Final verification**

Run `ZipFile.testzip()`, extract to a temporary directory, and assert both pages resolve every relative MP4 path offline and the archive contains exactly 33 MP4s. Re-run the full remote test suite and confirm local/origin/SSH Git states are clean and synchronized.
