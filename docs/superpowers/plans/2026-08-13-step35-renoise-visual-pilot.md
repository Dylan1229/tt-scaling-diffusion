# Step-35 RENOISE Visual Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a three-prompt Step-35 fixed-noise-level RENOISE pilot at amplitudes `[0.0, 0.4, 0.6, 0.8, 1.0]` and produce a synchronized visual comparison with a Chinese manual review.

**Architecture:** Reuse `Wan22Adapter.generate_with_renoise_branches` unchanged. Generalize the existing fixed-pilot runner only enough to support two explicit profiles: the completed Step-2 pilot and the approved Step-35 pilot; optional independent-seed generation controls whether the fifth column is a baseline or another amplitude.

**Tech Stack:** Python 3.11, PyTorch, Diffusers `0.38.0`, Wan 2.2 TI2V 5B, UniPC, pytest, imageio/FFmpeg, static HTML.

## Global Constraints

- Work only on branch `codex/late-stage-recovery`.
- Run GPU generation on `yukelab:/data/datasets/peihao/tt-scaling-diffusion` using its `.venv`.
- Use 50 UniPC inference steps, root seed `0`, and branch immediately after one-based Step 35.
- Use amplitudes `[0.0, 0.4, 0.6, 0.8, 1.0]` in ascending display order.
- Reuse one fresh-noise direction across all amplitudes for each prompt.
- Generate no new independent-seed reference.
- Evaluate visually only; do not run DINO, VBench, or quantitative thresholds.
- Preserve all Step-2 pilot behavior and artifacts.

---

### Task 1: Generalize the Fixed RENOISE Pilot Runner

**Files:**
- Modify: `tests/generate/test_step2_renoise_pilot.py`
- Modify: `ttsd/runners/generate/step2_renoise_pilot.py`
- Create: `configs/step35_renoise_visual_pilot_wan22_480p.yaml`

**Interfaces:**
- Consumes: `Wan22Adapter.generate_with_renoise_branches(..., branch_step: int, amplitudes: list[float], noise_seed: int)`.
- Produces: the existing CLI `python -m ttsd.runners.generate.step2_renoise_pilot`, now accepting either approved fixed profile; `independent_seed: null` omits independent generation and comparison columns.

- [ ] **Step 1: Write failing Step-35 profile tests**

Add a fixture derived from the Step-2 fixture and tests equivalent to:

```python
@pytest.fixture
def step35_config(config: dict) -> dict:
    changed = deepcopy(config)
    changed["renoise"].update(
        branch_step=35,
        amplitudes=[0.0, 0.4, 0.6, 0.8, 1.0],
        independent_seed=None,
    )
    return changed


def test_validate_config_accepts_step35_pilot(step35_config: dict) -> None:
    validate_config(step35_config)


def test_step35_comparison_uses_five_amplitudes_without_independent_seed(
    tmp_path: Path,
) -> None:
    amplitudes = [0.0, 0.4, 0.6, 0.8, 1.0]
    for amplitude in amplitudes:
        path = tmp_path / "p01" / _amplitude_slug(amplitude) / "video.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
    row = _row_from_artifacts(
        tmp_path,
        {"id": "p01", "text": "a person swimming in ocean"},
        amplitudes,
        independent_seed=None,
    )
    assert [video["label"] for video in row["videos"]] == [
        "alpha=0.0", "alpha=0.4", "alpha=0.6", "alpha=0.8", "alpha=1.0"
    ]
```

Also add `branch_step: 35` to an HTML manifest and assert that the generated title contains `Step-35`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q tests/generate/test_step2_renoise_pilot.py
```

Expected: failures because Step 35 is rejected, `_row_from_artifacts` has no optional baseline argument, and the title is fixed to Step 2.

- [ ] **Step 3: Implement the minimum two-profile runner**

In `ttsd/runners/generate/step2_renoise_pilot.py`:

1. Define the two approved profiles:

```python
EXPECTED_PILOTS = {
    2: ([0.0, 0.2, 0.4, 0.8], 1),
    35: ([0.0, 0.4, 0.6, 0.8, 1.0], None),
}
```

2. Make `validate_config` select the expected amplitudes and optional independent seed by `branch_step`, while preserving the 50-step, root-seed-0, and prompt-ID checks.
3. Derive HTML title, experiment name, and log prefix from `branch_step`.
4. Change `_row_from_artifacts` to append an independent-seed video only when `independent_seed is not None`.
5. Wrap independent generation, completion checks, and manifest rows in the same optional condition.
6. Keep the existing Step-2 config and output behavior unchanged.

Create `configs/step35_renoise_visual_pilot_wan22_480p.yaml` with:

```yaml
renoise:
  branch_step: 35
  amplitudes: [0.0, 0.4, 0.6, 0.8, 1.0]
  root_seed: 0
  independent_seed: null
  noise_seed: 10000000

output:
  root: runs/step35_renoise_visual_pilot
  run_id: null
  fps: 16
```

Copy the model, generation, and prompt sections exactly from the Step-2 config.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
pytest -q tests/generate/test_step2_renoise_pilot.py
pytest -q
```

Expected: all tests pass, including the existing Step-2 assertions.

- [ ] **Step 5: Commit the runner and configuration**

```bash
git add tests/generate/test_step2_renoise_pilot.py \
  ttsd/runners/generate/step2_renoise_pilot.py \
  configs/step35_renoise_visual_pilot_wan22_480p.yaml
git commit -m "feat: add Step-35 renoise visual pilot"
```

---

### Task 2: Deploy and Run the GPU Pilot

**Files:**
- Produce remotely: `runs/step35_renoise_visual_pilot/step35_renoise_visual_v1/`

**Interfaces:**
- Consumes: the Task-1 CLI and Step-35 YAML.
- Produces: 15 MP4 files plus metadata, manifest, config snapshot, and synchronized HTML.

- [ ] **Step 1: Push and synchronize all three checkouts**

```bash
git push origin codex/late-stage-recovery
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && git pull --ff-only origin codex/late-stage-recovery'
```

Verify local, origin, and SSH HEADs match before generation.

- [ ] **Step 2: Run the remote test suite**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && .venv/bin/pytest -q'
```

Expected: all tests pass.

- [ ] **Step 3: Run the `p01` GPU smoke test**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && \
  .venv/bin/python -m ttsd.runners.generate.step2_renoise_pilot \
  --config configs/step35_renoise_visual_pilot_wan22_480p.yaml \
  --run-id step35_renoise_visual_v1 --prompt-ids p01'
```

Expected: five `p01` amplitude videos, each with metadata recording `branch_step: 35` and the same actual `branch_sigma`.

- [ ] **Step 4: Resume for all prompts**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && \
  .venv/bin/python -m ttsd.runners.generate.step2_renoise_pilot \
  --config configs/step35_renoise_visual_pilot_wan22_480p.yaml \
  --run-id step35_renoise_visual_v1'
```

Expected: `p01` is skipped; `p03` and `p05` are generated; manifest has three rows and five amplitude columns.

---

### Task 3: Verify and Review the Visual Artifacts

**Files:**
- Create remotely and synchronize locally: `runs/step35_renoise_visual_pilot/step35_renoise_visual_v1/manual_review.md`

**Interfaces:**
- Consumes: Task-2 manifest and MP4s.
- Produces: full-decode evidence and a per-prompt Chinese qualitative conclusion.

- [ ] **Step 1: Fully decode all videos**

Run an `imageio.v3.imiter(..., plugin="FFMPEG")` check over every manifest path and assert:

```python
assert len(manifest["rows"]) == 3
assert len(paths) == 15
assert all(frame_count == 81 for frame_count in frame_counts)
```

Also assert every video is 832×480 and metadata agrees on `branch_step == 35` and one `branch_sigma` value.

- [ ] **Step 2: Synchronize artifacts locally and watch all videos**

```bash
rsync -az --delete \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/runs/step35_renoise_visual_pilot/step35_renoise_visual_v1/ \
  runs/step35_renoise_visual_pilot/step35_renoise_visual_v1/
```

Open `comparison.html`, play all 15 clips from beginning to end, and compare each amplitude against `alpha=0.0` at synchronized times.

- [ ] **Step 3: Write the Chinese manual review**

For each prompt, record:

- visible differences at `alpha=0.4`, `0.6`, `0.8`, and `1.0`;
- the first amplitude with clear structural or motion variation;
- whether `alpha=1.0` remains coherent;
- whether differences are structural/motion-level or local appearance/detail only.

End with an overall answer to the Step-35 theory question without claiming quantitative equivalence to independent-seed Best-of-N.

- [ ] **Step 4: Final verification**

Re-run the full remote test suite and full video decode check. Confirm local, origin, and SSH checkouts are clean and resolve to the same commit.
