# tt-scaling-diffusion

**Test-time scaling for text-to-video diffusion.** Detect failed video-diffusion generations *during* the intermediate denoising steps (not after the full decode), then intervene — early-stop, re-seed, re-noise a window — to save compute.

Base model: **Wan 2.2 TI2V-5B** (T2V mode, 480p). Evaluation: **VBench**.

---

## Repository layout

The repo has **two layers**: an *offline* data/analysis stack (generate clips →
score with VBench → extract DINO features → study what predicts quality) that
produces and validates verifiers, and an *online* **pipeline framework**
(`ttsd/pipeline/`) that uses those verifiers live during sampling to detect and
intervene on failing generations.

```
tt-scaling-diffusion/
├── README.md
├── LICENSE                         # Apache-2.0
├── pyproject.toml                  # package + dependency definition
├── constraints-generation.txt      # pinned cu128 wheel versions (Blackwell env)
├── docs/
│   └── e2e_framework_plan.md       # design doc for the pipeline framework (read this first)
│
├── ttsd/                           # ── first-party package ───────────────────
│   ├── models/wan22_adapter.py     #   diffusers WanPipeline wrapper (T2V) + posterior-mean / scheduler-swap
│   ├── prompts/                    #   dev_set.py, sweep_v2.py — VBench prompt lists
│   ├── eval/vbench.py              #   VBench in custom_input mode → CSVs
│   ├── features/dino_cls.py        #   reusable DINOv2 CLS feature math
│   │
│   ├── verifiers/                  #   ── quality signal: predict final score from intermediate state ──
│   │   ├── base.py                 #     Verifier ABC (score() + REQUIRES set)
│   │   ├── noop.py                 #     constant-score stub (smoke tests)
│   │   └── dino/                   #     DINO-based verifiers
│   │       ├── *.py                #       offline probes (frame-cos-mean, similarity-profile, PCA-ridge, max-z fusion)
│   │       └── online_adapter.py   #       live wrap: decode x0_hat → DINOv2 → score (used by the pipeline)
│   │
│   ├── pipeline/                   #   ── e2e inference backbone (the framework) ──
│   │   ├── orchestrator.py         #     conductor: load plugins → run strategy → emit RunResult
│   │   ├── registry.py             #     name → class plugin table (@register_model/verifier/policy/action/strategy)
│   │   ├── config.py               #     typed PipelineConfig dataclasses + YAML loader
│   │   ├── core.py                 #     shared dataclasses (TrajectoryState, StepState, ActionSpec, ...)
│   │   ├── model_adapter.py        #     ModelAdapter Protocol + WanModelAdapter (on_step hook, latent capture/inject)
│   │   ├── policy.py               #     DecisionPolicy: noop / fixed_threshold / dynamic_sliding_window / best_of_n
│   │   ├── actions.py              #     Action: continue / stop_and_fail / single_frame_anchor_inject (T1) / refine_prompt_vlm (T2)
│   │   ├── vlm.py                  #     pluggable VLMClient for prompt refinement (provider left open)
│   │   ├── budget.py               #     wall-clock / GPU-seconds / VLM-token tracking
│   │   └── logger.py               #     JSON-lines event telemetry
│   │
│   ├── search/                     #   ── SearchStrategy: the outer loop ──
│   │   ├── base.py                 #     SearchStrategy ABC + RunContext
│   │   ├── sequential.py           #     SequentialTrialSearch (EFD&I tiered escalation)
│   │   └── parallel.py             #     ParallelCandidateSearch (naive best-of-N)
│   │
│   └── runners/                    #   ── CLI entry points: python -m ttsd.runners.<sub>.<module> ──
│       ├── generate/               #     baseline sweep, latent decode
│       ├── features/               #     DINOv2 CLS + patch feature extraction
│       ├── analysis/               #     VBench-alignment analyses
│       ├── report/                 #     figure generation
│       ├── utilities/              #     shared seed/VBench loaders + ranking helper
│       └── pipeline/               #     run_pipeline (single) · sweep (grid) · pareto_plot (compare)
│
├── configs/
│   ├── *.yaml                      # offline generation sweep configs
│   └── pipeline/                   # pipeline configs: efdi_dino, bon_dino, *_smoke + sweeps/
├── scripts/                        # multi-GPU tmux launchers / batch wrappers
├── external/                       # vendored upstreams (gitignored): t2v-search, VBench
└── runs/                           # ALL outputs (gitignored — regenerate from config snapshots)
    ├── baseline/<run_id>/<prompt_id>/seed<NNNN>/   # video.mp4 + latents/ + posterior_means/ + meta.json
    ├── dino_input_frames/ · cls_features/ · patch_features/   # offline feature stages
    ├── vbench/ · analysis/ · report/               # offline scoring + analysis
    ├── pipeline/<run_id>/                          # single pipeline run: video.mp4 + events.jsonl + result.json
    └── pipeline_sweeps/<sweep_id>/<strategy>/<prompt_id>__seed<NNNN>/   # sweep grid + _pareto/
```

**Three-tier code separation:**
- `ttsd/` — first-party. The only place we put real implementation work.
- `external/` — vendored upstreams, read-only references. Never edited (except the one-line VBench setup-guard patch, applied by `setup_external.sh`).
- `runs/` — outputs. Reproducible from the `config.snapshot.{yaml,json}` written into each run dir; gitignored.

**Pipeline extension model:** add a backbone / verifier / policy / action / strategy by writing one class, registering it with the matching `@register_*` decorator, and naming it (`kind:`) in a YAML config — no orchestrator edits. See [`docs/e2e_framework_plan.md`](./docs/e2e_framework_plan.md).

---

## Setup

```bash
git clone <this repo> tt-scaling-diffusion
cd tt-scaling-diffusion

# 1) Python env (uv recommended; conda/venv works too)
uv venv .venv --python 3.11
source .venv/bin/activate

# 2) Our package + base deps
uv pip install -e ".[eval,dev]"

# 3) Pin cu128 torch wheels for Blackwell / RTX PRO 6000 (sm_120).
#    constraints-generation.txt records the exact known-good versions.
uv pip install -c constraints-generation.txt torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu128

# 4) Vendored deps — clone shim0114 + VBench, patch VBench's CUDA guard
./scripts/setup_external.sh

# 5) Install VBench editable (needs --no-build-isolation so it sees our torch)
uv pip install --no-build-isolation -e external/VBench

# 6) Wan 2.2 5B weights — download to your HF cache if not already present.
#    Default path:
#      /data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
#    Override with:  export WAN22_MODEL_PATH=/your/path
```

**GPU note.** Wan 2.2 5B uses bf16 → needs Ampere (A100 / A6000 / RTX 3090 / 4090) or newer. Verified on Blackwell sm_120 (RTX PRO 6000) with cu128.

---

## Usage

The pipeline runs in stages; each stage reads the previous stage's `runs/` directory.

**1. Generate a sweep** (4 GPUs, sharded, resumable):
```bash
scripts/generate_sweep_v2_4gpu.sh        # GPUs 4,5,6,7 → runs/baseline/<run_id>/
```
Re-running skips any (prompt, seed) with a `DONE` marker. Each clip saves `video.mp4`, `latents/`, and `posterior_means/` (x0_hat) every 5 steps.

**2. Score with VBench:**
```bash
python -m ttsd.eval.vbench --run runs/baseline/<run_id>
# → runs/vbench/<run_id>/vbench_scores_long.csv  (+ summary)
```
First run downloads ~10–20 GB of backbones (CLIP / ViCLIP / RAFT / DINO / AMT) into `$HF_HOME`.

**3. Decode posterior means to DINOv2 input frames:**
```bash
scripts/prepare_dino_inputs_batch.sh \
  --source-run-root runs/baseline/<run_id> \
  --output-run-root runs/dino_input_frames/<run_id>
```
VAE-decodes each `posterior_means/step_NNN.pt` latent into pixel frames. These frames are DINOv2 inputs, not a deliverable video.

**4. Extract DINOv2 features:**
```bash
scripts/extract_cls_similarity_batch.sh \
  --decoded-run-root runs/dino_input_frames/<run_id> \
  --output-run-root runs/cls_features/<run_id>

scripts/extract_patch_features_batch.sh \
  --decoded-run-root runs/dino_input_frames/<run_id> \
  --output-run-root runs/patch_features/<run_id>
```

**5. Align features with VBench:**
```bash
python -m ttsd.runners.analysis.similarity_tail_ranking --run-id <run_id>
```
`--run-id` fills the input roots from `runs/<stage>/<run_id>`; pass `--heatmap-run-root` /
`--vbench-long-csv` to override. Further analyses live under `ttsd.runners.analysis.*` and figures
under `ttsd.runners.report.*` — see [`ttsd/runners/README.md`](./ttsd/runners/README.md).

---

## End-to-end test-time-scaling pipeline

`ttsd/pipeline/` is the inference-time backbone: it drives a diffusion model through
denoising, queries a verifier on intermediate state, and lets a decision policy trigger
search/intervention strategies (EFD&I-style tiered intervention, naive best-of-N, …).
Everything is plug-and-play via a registry — add a model/verifier/policy/action/strategy
by registering it and naming it in a config; no orchestrator edits. Full design:
[`docs/e2e_framework_plan.md`](./docs/e2e_framework_plan.md).

**Single run** (one prompt, one config):
```bash
# EFD&I-style tiered intervention (base → anchor-inject → prompt-refine)
python -m ttsd.runners.pipeline.run_pipeline \
  --config configs/pipeline/efdi_dino.yaml \
  --prompt "a person swimming in ocean" --seed 0 --run-id my_run

# Naive best-of-N (N=4 candidates, keep highest-scoring)
python -m ttsd.runners.pipeline.run_pipeline \
  --config configs/pipeline/bon_dino.yaml \
  --prompt "a person swimming in ocean" --seed 0 --run-id my_bon_run
```
Each run writes `runs/pipeline/<run_id>/`: `video.mp4`, `events.jsonl` (every verifier
call / policy decision / action), `result.json` (success, final score, per-trial cost),
and `config.snapshot.json`.

**Multi-GPU sweep** (prompt × seed × strategy grid, sharded, resumable):
```bash
SWEEP_ID=my_sweep GPUS=0,1,2,3 scripts/run_pipeline_sweep_4gpu.sh
# grid defined in configs/pipeline/sweeps/efdi_vs_bon_5x5.yaml
# → runs/pipeline_sweeps/<SWEEP_ID>/<strategy>/<prompt_id>__seed<NNNN>/
```
Re-launch the same `SWEEP_ID` to skip completed items. Quick test: prepend
`LIMIT_PROMPTS=1 LIMIT_SEEDS=1`.

**Cost-vs-score Pareto plot** (compare strategies):
```bash
python -m ttsd.runners.pipeline.pareto_plot \
  --runs runs/pipeline_sweeps/<SWEEP_ID>/*/* \
  --output-dir runs/pipeline_sweeps/<SWEEP_ID>/_pareto
# → pareto.png + pareto_data.csv  (gitignored — regenerate from the run dirs)
```

> Pipeline outputs (videos, `events.jsonl`, Pareto plots, CSVs) all live under `runs/`
> and are **gitignored** — regenerate them with the commands above. Only the code,
> configs, scripts, and docs are version-controlled.

---

## What gets committed to GitHub

The `.gitignore` is set up so that `git add -A` produces a clean upload. **Excluded:** `external/` (recover via `setup_external.sh`), `runs/` (all experiment + pipeline + analysis outputs — videos, latents, `events.jsonl`, scores, Pareto plots — regenerate from `config.snapshot.*`), weight files, video files, `.venv/`, `__pycache__/`, IDE configs, secrets.

**Included** (everything needed to set up and use the repo): `README.md`, `docs/`, `LICENSE`, `pyproject.toml`, `constraints-generation.txt`, `.gitignore`, `ttsd/`, `configs/`, `scripts/`, `external/.gitkeep`. ≈ a few hundred KB; clone + `setup_external.sh` + `pip install` reproduces the starting point.

---

## Citations / attribution

This repo vendors and depends on:
- **VBench** (Vchitect/VBench, Apache-2.0) — Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, CVPR 2024.
- **T2V-Diffusion-Search** (shim0114, Apache-2.0) — wraps the Wan T2V driver and provides a DLBS beam-search reference.
- **Wan 2.2** (Wan-AI) — base T2V model.

See `external/<repo>/LICENSE` for upstream notices. Our own code under `ttsd/` is Apache-2.0 (see [`LICENSE`](./LICENSE)).
