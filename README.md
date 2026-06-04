# tt-scaling-diffusion

**Test-time scaling for text-to-video diffusion.** Detect failed video-diffusion generations *during* the intermediate denoising steps (not after the full decode), then intervene — early-stop, re-seed, re-noise a window — to save compute.

Base model: **Wan 2.2 TI2V-5B** (T2V mode, 480p). Evaluation: **VBench**.

---

## Repository layout

```
tt-scaling-diffusion/
├── README.md
├── LICENSE                        # Apache-2.0
├── pyproject.toml                 # package + dependency definition
├── constraints-generation.txt     # pinned cu128 wheel versions (Blackwell generation env)
│
├── ttsd/                          # first-party package
│   ├── models/wan22_adapter.py    #   diffusers WanPipeline wrapper (T2V) + posterior-mean capture
│   ├── prompts/                   #   dev_set.py, sweep_v2.py — VBench prompt lists
│   ├── eval/vbench.py             #   VBench in custom_input mode → CSVs
│   ├── verifiers/base.py          #   Verifier interface (Phase 1)
│   ├── search/base.py             #   SearchPolicy interface (Phase 2)
│   └── runners/                   #   CLI entry points: python -m ttsd.runners.<sub>.<module>
│       ├── utilities/             #     shared seed/VBench loaders + ranking helper
│       ├── generate/              #     baseline sweep, latent decode
│       ├── features/              #     DINOv2 CLS + patch feature extraction
│       ├── analysis/              #     VBench-alignment analyses
│       └── report/                #     figure generation
│
├── configs/                       # sweep YAMLs
├── scripts/                       # multi-GPU launch / batch wrappers
├── external/                      # vendored upstreams (gitignored): t2v-search, VBench
└── runs/                          # experiment outputs (gitignored)
    ├── baseline/<run_id>/<prompt_id>/seed<NNNN>/
    │   ├── video.mp4
    │   ├── latents/step_NNN.pt          # noisy latent x_t snapshots
    │   ├── posterior_means/step_NNN.pt  # x0_hat snapshots
    │   ├── meta.json
    │   └── DONE                         # resume marker
    ├── dino_input_frames/<run_id>/...   # posterior-mean latents decoded to frames (DINOv2 input)
    ├── cls_features/<run_id>/...         # DINOv2 CLS features + similarity matrices
    ├── patch_features/<run_id>/...       # DINOv2 patch tokens
    ├── vbench/<run_id>/...               # VBench scores
    ├── analysis/                         # alignment CSVs / summaries
    ├── report/                           # figures
    └── logs/                             # run logs
```

**Three-tier code separation:**
- `ttsd/` — first-party. The only place we put real implementation work.
- `external/` — vendored upstreams, treated as read-only references. Never edited (except the one-line VBench setup-guard patch, applied by `setup_external.sh`).
- `runs/` — outputs. Reproducible from the `config.snapshot.yaml` written into each run dir.

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

## What gets committed to GitHub

The `.gitignore` is set up so that `git add -A` produces a clean upload. **Excluded:** `external/` (recover via `setup_external.sh`), `runs/` (regenerate from `config.snapshot.yaml`), weight files, video files, `.venv/`, `__pycache__/`, IDE configs, secrets.

**Included:** `README.md`, `LICENSE`, `pyproject.toml`, `constraints-generation.txt`, `.gitignore`, `ttsd/`, `configs/`, `scripts/`, `external/.gitkeep`. ≈ a few hundred KB; clone + `setup_external.sh` + `pip install` reproduces the starting point.

---

## Citations / attribution

This repo vendors and depends on:
- **VBench** (Vchitect/VBench, Apache-2.0) — Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, CVPR 2024.
- **T2V-Diffusion-Search** (shim0114, Apache-2.0) — wraps the Wan T2V driver and provides a DLBS beam-search reference.
- **Wan 2.2** (Wan-AI) — base T2V model.

See `external/<repo>/LICENSE` for upstream notices. Our own code under `ttsd/` is Apache-2.0 (see [`LICENSE`](./LICENSE)).
