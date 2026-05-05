# tt-scaling-diffusion

**Test-time scaling for text-to-video diffusion.** Detect failed video-diffusion generations *during* the intermediate denoising steps (not after the full decode), then intervene — early-stop, re-seed, re-noise a window — to save compute.

Base model: **Wan 2.2 TI2V-5B** (T2V mode, 480p). Evaluation: **VBench**. See [`ROADMAP.md`](./ROADMAP.md) for the full project plan and research wedge.

---

## Repository layout

```
tt-scaling-diffusion/
├── ROADMAP.md                 # living spec — what we're building and why
├── README.md                  # this file
├── LICENSE                    # Apache-2.0
├── pyproject.toml             # package + dependency definition
│
├── ttsd/                      # ── our package ────────────────────────────
│   ├── models/
│   │   └── wan22_adapter.py   #   diffusers WanPipeline wrapper, T2V mode
│   ├── verifiers/
│   │   └── base.py            #   Verifier interface (Phase 1)
│   ├── search/
│   │   └── base.py            #   SearchPolicy interface (Phase 2)
│   ├── prompts/
│   │   └── dev_set.py         #   15 hand-picked VBench prompts for Phase 0
│   ├── runners/
│   │   └── baseline.py        #   Phase 0 generation sweep, resumable
│   └── eval/
│       └── vbench.py          #   wraps VBench in custom_input mode → CSVs
│
├── configs/
│   └── baseline_wan22_480p.yaml   # Phase 0 sweep config (15 × 10 seeds)
│
├── scripts/
│   └── setup_external.sh      # one-shot: clone + patch upstream deps
│
├── external/                  # ── vendored upstream code (gitignored) ────
│   ├── t2v-search/            #   shim0114/T2V-Diffusion-Search (Apache-2.0)
│   └── VBench/                #   Vchitect/VBench (Apache-2.0)
│
├── paper-pdf/                 # ── references (PDFs gitignored) ───────────
│   └── *.md                   #   our worked-out theoretical notes
│
└── runs/                      # ── experiment outputs (gitignored) ────────
    └── baseline/<run_id>/
        └── <prompt_id>/<seed>/
            ├── video.mp4
            ├── latents/step_NNN.pt    # snapshots for verifier training
            ├── meta.json
            └── DONE                   # resume marker
```

**Three-tier code separation:**
- `ttsd/` — first-party. The only place we put real implementation work.
- `external/` — vendored upstreams, treated as read-only references. Never edited (except the one-line VBench setup-guard patch, applied by `setup_external.sh`).
- `runs/` — outputs. Always reproducible from the `config.snapshot.yaml` written into each run dir.

---

## Setup

```bash
git clone <this repo> tt-scaling-diffusion
cd tt-scaling-diffusion

# 1) Python env (uv recommended; conda/venv works too)
uv venv .venv --python 3.11
source .venv/bin/activate

# 2) Our package + base deps (torch is auto-resolved against your CUDA)
uv pip install -e ".[eval,dev]"

# 3) For Blackwell / RTX PRO 6000 (sm_120) on driver ≤ 12.8: pin cu128 wheel
uv pip uninstall torch torchvision torchaudio
uv pip install "torch==2.7.*" "torchvision==0.22.*" \
  --index-url https://download.pytorch.org/whl/cu128

# 4) Vendored deps — clone shim0114 + VBench, patch VBench's CUDA guard
./scripts/setup_external.sh

# 5) Install VBench editable (needs --no-build-isolation so it sees our torch)
uv pip install --no-build-isolation -e external/VBench

# 6) Wan 2.2 5B weights — download to your HF cache dir if not already present
#    Default expected path:
#      /data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
#    Override with:  export WAN22_MODEL_PATH=/your/path
```

**GPU note.** Wan 2.2 5B uses bf16 → needs Ampere (A100/A6000/RTX 3090/4090) or newer. Verified on Blackwell sm_120 (RTX PRO 6000) with cu128.

---

## Quick start

**Smoke test** (1 prompt × 1 seed, ~1 min on a single GPU):
```bash
python -m ttsd.runners.baseline --config configs/baseline_wan22_480p.yaml --smoke
```

**Phase 0 baseline sweep** (15 prompts × 10 seeds, ~3 h on one Blackwell GPU):
```bash
python -m ttsd.runners.baseline --config configs/baseline_wan22_480p.yaml
```
Resumable — re-running skips any (prompt, seed) with a `DONE` marker.

**VBench evaluation** of a finished run:
```bash
python -m ttsd.eval.vbench --run runs/baseline/<run_id>
```
Writes per-(prompt, seed, dim) scores to `<run_id>/vbench/vbench_scores_long.csv` and per-(prompt, dim) mean/std summary to `…/vbench_scores_summary.csv`. First run downloads ~10–20 GB of backbones (CLIP / ViCLIP / RAFT / DINO / AMT) into `$HF_HOME`.

---

## Phase 0 — what we're doing now

Prove that **test-time scaling is necessary** on Wan 2.2: show that VBench scores have non-trivial seed-to-seed variance for the same prompt. Without that variance, no downstream TTS work is justified. Headline artifact = per-prompt score histogram across seeds (see `ROADMAP.md` §3).

Secondary goal: every Phase-0 generation persists intermediate latents at every-5-steps. Those are the training set for the Phase-1 latent verifier.

---

## What gets committed to GitHub

The `.gitignore` is set up so that `git add -A` produces a clean upload. **Excluded:**

- `external/` — upstreams live in their own repos; recover via `scripts/setup_external.sh`
- `runs/` — experiment outputs (videos, latents, logs); regenerate from `config.snapshot.yaml`
- `paper-pdf/*.pdf` — copyrighted papers; only our `.md` notes are kept
- All weight files (`*.pt`, `*.safetensors`, `*.ckpt`, etc.) — model weights live in HF cache, never in repo
- All video files (`*.mp4`, `*.gif`)
- `.venv/`, `__pycache__/`, IDE configs, secrets

**Included** (everything you need to reproduce the project):
```
ROADMAP.md   README.md   LICENSE   pyproject.toml   .gitignore
ttsd/        configs/    scripts/   paper-pdf/*.md   external/.gitkeep
```
≈ a few hundred KB. Anyone cloning the repo runs `setup_external.sh` + `pip install` and is at the same starting point.

---

## Citations / attribution

This repo vendors and depends on:
- **VBench** (Vchitect/VBench, Apache-2.0) — Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, CVPR 2024.
- **T2V-Diffusion-Search** (shim0114, Apache-2.0) — wraps Wan T2V driver and provides DLBS beam search reference.
- **Wan 2.2** (Wan-AI) — base T2V model.

See `external/<repo>/LICENSE` for upstream notices. Our own code under `ttsd/` is Apache-2.0 (see [`LICENSE`](./LICENSE)).

---

## See also

- [`ROADMAP.md`](./ROADMAP.md) — full project plan: design axes, codebase decision, phase milestones, research extensions, deployment path.
- [`paper-pdf/early-failure-theoretical-analysis.md`](./paper-pdf/early-failure-theoretical-analysis.md) — rigorous theory for why early-step verification works (martingale + I-MMSE + Lipschitz).
