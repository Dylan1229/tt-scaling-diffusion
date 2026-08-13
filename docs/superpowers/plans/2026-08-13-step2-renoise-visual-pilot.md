# Step-2 RENOISE Visual Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and manually inspect a 3×5 visual comparison showing how partial variance-preserving Step-2 RENOISE changes Wan 2.2 videos.

**Architecture:** Add one pure tensor helper for noise-component rotation, then expose one focused adapter method that captures the Step-2 posterior estimate and forks four amplitudes in a shared suffix batch. A dedicated runner produces the three fixed prompt rows plus independent-seed controls and writes a static looping HTML comparison; it reuses the existing Wan adapter and video writer rather than extending the late-stage Best-of-M experiment schema.

**Tech Stack:** Python 3.12, PyTorch, Diffusers WanPipeline/UniPC, pytest, YAML, imageio, static HTML.

## Global Constraints

- Branch immediately after completing denoising Step 2 of 50 UniPC steps.
- Root seed is `0`; amplitudes are exactly `[0.0, 0.2, 0.4, 0.8]`.
- The four amplitudes for a prompt share one fresh Gaussian noise tensor.
- Prompts are exactly `p01`, `p03`, and `p05` from `ttsd.prompts.dev_set:DEV_PROMPTS`.
- Independent seed `1` is a visual reference only.
- Do not calculate DINO, VBench, automated thresholds, or amplitude `1.0`.
- Keep individual MP4s, metadata JSON, and a synchronized local HTML comparison.

---

### Task 1: Variance-Preserving RENOISE Primitive

**Files:**
- Modify: `ttsd/search/late_branching.py`
- Modify: `tests/search/test_late_branching.py`

**Interfaces:**
- Consumes: post-step `latents: torch.Tensor`, matching `posterior: torch.Tensor`, scalar `sigma: float`, ordered `amplitudes: tuple[float, ...]`, and `noise_seed: int`.
- Produces: `renoise_latents(...) -> torch.Tensor` with one batch item per amplitude, sharing one sampled noise direction.

- [ ] **Step 1: Write failing formula tests**

Add tests that construct known float32 tensors and assert:

```python
amplitudes = (0.0, 0.2, 0.4, 0.8)
branches = renoise_latents(
    latents,
    posterior=posterior,
    sigma=0.8,
    amplitudes=amplitudes,
    noise_seed=123,
)
assert branches.shape[0] == 4
assert torch.allclose(branches[0], latents, atol=1e-6, rtol=1e-6)
assert torch.equal(branches, renoise_latents(
    latents,
    posterior=posterior,
    sigma=0.8,
    amplitudes=amplitudes,
    noise_seed=123,
))
```

Also assert rejection of `sigma <= 0`, an empty amplitude list, and amplitudes outside `[0, 1]`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/search/test_late_branching.py -q`

Expected: collection fails because `renoise_latents` does not exist.

- [ ] **Step 3: Implement the minimal helper**

Implement:

```python
def renoise_latents(latents, *, posterior, sigma, amplitudes, noise_seed):
    if latents.shape != posterior.shape:
        raise ValueError(...)
    if sigma <= 0:
        raise ValueError(...)
    amplitudes = tuple(float(value) for value in amplitudes)
    if not amplitudes or any(not 0.0 <= value <= 1.0 for value in amplitudes):
        raise ValueError(...)
    implied_noise = (latents - (1.0 - sigma) * posterior) / sigma
    generator = torch.Generator(device=latents.device).manual_seed(noise_seed)
    fresh_noise = torch.randn(latents.shape, generator=generator,
                              device=latents.device, dtype=latents.dtype)
    return torch.cat([
        (1.0 - sigma) * posterior
        + sigma * ((1.0 - alpha**2) ** 0.5 * implied_noise + alpha * fresh_noise)
        for alpha in amplitudes
    ])
```

Do not change the existing additive `fork_latents`; prior late-stage experiments depend on it.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/pytest tests/search/test_late_branching.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add ttsd/search/late_branching.py tests/search/test_late_branching.py
git commit -m "feat: add variance-preserving renoise helper"
```

---

### Task 2: Step-2 Wan Adapter Path

**Files:**
- Modify: `ttsd/models/wan22_adapter.py`
- Modify: `tests/search/test_late_branching.py`

**Interfaces:**
- Consumes: `generate_with_renoise_branches(prompt, seed, amplitudes, branch_step=2, noise_seed=10_000_000, ...)`.
- Produces: `RenoiseGenerationOutput(frames_by_amplitude, amplitudes, branch_step, branch_sigma, noise_seed)`.
- Uses: `renoise_latents` from Task 1 and existing `_posterior_mean_from_step`, `sigma_after_step`, prompt-embedding repetition, and `_decode_latents_in_chunks`.

- [ ] **Step 1: Add a failing fake-pipeline adapter test**

Extend `_FakeScheduler` to expose a deterministic posterior and call the new method with amplitudes `(0.0, 0.2, 0.4, 0.8)`. Assert:

```python
assert fake_pipe.batch_sizes == [1, 4, 4, 4, 4]
assert fake_pipe.prompt_batch_sizes == [1, 4, 4, 4, 4]
assert result.amplitudes == (0.0, 0.2, 0.4, 0.8)
assert result.branch_step == 2
assert result.branch_sigma == pytest.approx(0.6)
assert len(result.frames_by_amplitude) == 4
```

The expected sigma follows `_FakeScheduler.sigmas[2]` after Step 2.

- [ ] **Step 2: Run the focused test to verify failure**

Run: `.venv/bin/pytest tests/search/test_late_branching.py -q`

Expected: failure because `generate_with_renoise_branches` is absent.

- [ ] **Step 3: Implement one adapter method**

Wrap `scheduler.step` only long enough to capture the model output and pre-step sample for Step 2, compute `posterior = _posterior_mean_from_step(...)`, and then in `callback_on_step_end`:

1. read post-step sigma with `sigma_after_step`;
2. call `renoise_latents` on callback `latents` and the captured posterior;
3. repeat conditional and negative prompt embeddings four times;
4. continue all suffixes in one batch;
5. decode final latents and return `RenoiseGenerationOutput`.

Always restore `scheduler.step` in `finally`. Reject branch steps outside `[1, num_inference_steps - 1]` and empty amplitudes through the helper.

- [ ] **Step 4: Run adapter and regression tests**

Run: `.venv/bin/pytest tests/search/test_late_branching.py -q`

Expected: all tests pass, including existing additive branching tests.

- [ ] **Step 5: Commit**

```bash
git add ttsd/models/wan22_adapter.py tests/search/test_late_branching.py
git commit -m "feat: branch Wan trajectories with Step-2 renoise"
```

---

### Task 3: Fixed Pilot Runner and Visual Comparison

**Files:**
- Create: `ttsd/runners/generate/step2_renoise_pilot.py`
- Create: `tests/generate/test_step2_renoise_pilot.py`
- Create: `configs/step2_renoise_visual_pilot_wan22_480p.yaml`

**Interfaces:**
- Consumes: `python -m ttsd.runners.generate.step2_renoise_pilot --config ... [--prompt-ids ...] [--run-id ...] [--device ...]`.
- Produces: `<output.root>/<run-id>/<prompt-id>/alpha_{0p0,0p2,0p4,0p8}/video.mp4`, `<prompt-id>/independent_seed_1/video.mp4`, per-video `meta.json`, `manifest.json`, and `comparison.html`.
- Uses: Task 2 adapter method, existing `_save_video`, and ordinary `adapter.generate` for seed 1.

- [ ] **Step 1: Write failing runner utility tests**

Test a pure `build_comparison_html(manifest)` helper using paths containing the three prompt IDs. Assert the HTML contains all five column labels, 15 `<video` tags, `autoplay`, `muted`, `loop`, `controls`, and JavaScript that resets/plays every video together.

Test config validation rejects any amplitudes other than `[0.0, 0.2, 0.4, 0.8]`, branch step other than `2`, root seed other than `0`, or independent seed other than `1`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/pytest tests/generate/test_step2_renoise_pilot.py -q`

Expected: import failure because the runner module is absent.

- [ ] **Step 3: Implement the dedicated runner**

Keep the runner direct:

1. load YAML and validate the fixed pilot values;
2. select `p01,p03,p05` from the configured prompt source;
3. for each prompt, call `generate_with_renoise_branches` once and save four videos/metadata files;
4. call `adapter.generate(..., seed=1)` once and save the visual reference;
5. append relative paths and labels to `manifest.json`;
6. write `comparison.html` with a CSS grid and native looping `<video>` elements;
7. skip an item when both its `video.mp4` and `meta.json` already exist, so interrupted GPU runs can resume.

Use noise seed `10_000_000` for every prompt as approved: direction is shared across amplitudes within each prompt; tensor shape is identical across prompts.

- [ ] **Step 4: Add the exact YAML config**

Set:

```yaml
model:
  name: wan22_ti2v_5b
  path: /data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
  dtype: bf16
  device: cuda
  scheduler: unipc
generation:
  resolution: [480, 832]
  num_frames: 81
  num_inference_steps: 50
  guidance_scale: 5.0
renoise:
  branch_step: 2
  amplitudes: [0.0, 0.2, 0.4, 0.8]
  root_seed: 0
  independent_seed: 1
  noise_seed: 10000000
prompts:
  source: ttsd.prompts.dev_set:DEV_PROMPTS
  ids: [p01, p03, p05]
output:
  root: runs/step2_renoise_visual_pilot
  run_id: null
  fps: 16
```

- [ ] **Step 5: Run runner tests and full CPU suite**

Run: `.venv/bin/pytest tests/generate/test_step2_renoise_pilot.py tests/search/test_late_branching.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ttsd/runners/generate/step2_renoise_pilot.py tests/generate/test_step2_renoise_pilot.py configs/step2_renoise_visual_pilot_wan22_480p.yaml
git commit -m "feat: add Step-2 renoise visual pilot"
```

---

### Task 4: Remote Smoke, Full GPU Pilot, and Manual Review

**Files:**
- No tracked source files required.
- Remote artifacts: `/data/datasets/peihao/tt-scaling-diffusion/runs/step2_renoise_visual_pilot/<run-id>/`

**Interfaces:**
- Consumes: committed implementation synchronized through `origin/codex/late-stage-recovery`.
- Produces: final videos, manifest, HTML, screenshots/contact sheets for inspection, and a written per-prompt conclusion.

- [ ] **Step 1: Push and fast-forward `yukelab`**

```bash
git push origin codex/late-stage-recovery
ssh yukelab 'git -C /data/datasets/peihao/tt-scaling-diffusion pull --ff-only'
```

Verify local, origin, and SSH HEAD hashes match before generation.

- [ ] **Step 2: Run one-prompt smoke**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && \
  .venv/bin/python -m ttsd.runners.generate.step2_renoise_pilot \
  --config configs/step2_renoise_visual_pilot_wan22_480p.yaml \
  --prompt-ids p01 --run-id step2_renoise_smoke'
```

Expected: four amplitude MP4s, one independent-seed MP4, metadata, manifest, and HTML; process exits 0.

- [ ] **Step 3: Inspect smoke integrity**

Use `ffprobe` to verify all five videos are 832×480, have nonzero duration, and decode without errors. Read metadata to verify branch step, amplitudes, seeds, and sigma. If valid, continue.

- [ ] **Step 4: Run complete three-prompt pilot**

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion && \
  .venv/bin/python -m ttsd.runners.generate.step2_renoise_pilot \
  --config configs/step2_renoise_visual_pilot_wan22_480p.yaml \
  --run-id step2_renoise_visual_v1'
```

Expected: 15 valid MP4s plus metadata, manifest, and comparison HTML.

- [ ] **Step 5: Inspect every video manually**

Copy the comparison artifact locally or serve it from a local temporary HTTP server. Watch all videos through their full duration, replaying as needed. Record, for each prompt and amplitude, changes in layout/framing, subject identity/count/relations, action trajectory, background/details, and artifacts. Compare the scale of change with independent seed 1 without claiming statistical equivalence.

- [ ] **Step 6: Verify final artifacts and report**

Run a manifest/ffprobe check proving exactly 15 videos exist and decode. Report:

- whether partial Step-2 RENOISE caused unmistakable final-video changes;
- the first tested amplitude where changes became obvious for each prompt;
- whether `alpha=0.8` remained coherent;
- how its qualitative change compared with the independent seed;
- limitations from three prompts and one direction.
