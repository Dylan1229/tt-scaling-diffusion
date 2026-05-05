# Test-Time Scaling for Video Diffusion — Project Roadmap

**Owner:** fy27@rice.edu  ·  **Started:** 2026-05-04  ·  **Repo:** `/data/datasets/fanjiang/repo/tt-scaling-diffusion`

---
## 1. Problem statement

Text-to-video (T2V) diffusion models like Wan 2.2, CogVideoX, HunyuanVideo, and Pyramid-Flow are *non-deterministic* and *expensive*. A single 480p–720p clip takes tens of seconds to minutes of GPU time, dominated by the multi-step denoising loop. Empirically, a non-trivial fraction of generations are **failures**: text–video misalignment, dropped objects, broken physics, incoherent motion, or simple aesthetic collapse. Today's standard practice is to wait until the full denoising loop finishes, decode to RGB, view the clip, decide it's bad, and start over from a new seed. That is the worst possible point in the pipeline to make the decision: all of the compute has already been spent.

The **test-time scaling (TTS)** thesis is that we can do better by treating sampling as a *search problem* with online feedback:

- **Detect** a likely failure as early in the denoising trajectory as possible (the latent already encodes most of the semantic content within the first ~10 of 50 steps — see `paper-pdf/early-failure-theoretical-analysis.md`, Theorem 1).
- **Intervene** before wasting the remaining compute: stop, re-seed, branch, swap noise, refine the prompt, or locally edit a region/frame.
- **Allocate** extra compute only when it pays off (rare/compositional prompts, ambiguous mid-trajectory states), and skip it when the trajectory is obviously fine.

Concretely the design space breaks into five axes (this is the lens we use for everything below):

| Axis | Question | Examples in literature |
|---|---|---|
| **Prompt** | How to refine/expand the user prompt before or during sampling? | EFD&I (paper 1) |
| **Seed / sampled noise** | Which initial noise / candidate noises are most promising? | BoN (papers 2, 3, 6-linear) |
| **Verifier** | How to predict final quality from intermediate state? Where does it operate — pixel preview, latent, terminal reward back-propagated? | RI preview (1), VHS latent head (2), NoisyCLIP (3), VisionReward (Video-T1), DTS terminal-back-prop (5) |
| **Intra-frame search** (within one trajectory) | Step-wise: which next-step candidate to keep? | Latent beam search (4), DTS MCTS (5), threshold-stop (1) |
| **Inter-frame search** (across frames/chunks of one video) | Per-chunk: which chunk to accept, which to redo? | Tree-of-Frames (Video-T1, 6), DLBS-LA (T2V-Diffusion-Search) |

**Scope decision (2026-05-04):** focus on **single-shot diffusion T2V** (Wan 2.2 5B, ~5-second 480p clips). Autoregressive long-video generators (FramePack, MAGI-1, StreamingT2V, etc.) are explicitly *out of scope* for the main project — they introduce a different infrastructure stack and a different literature of long-video coherence work. Long-form autoregressive extension is parked as a stretch goal in §6.8.

The **uncrowded sweet spot** under this scope: **per-window** early-failure detection *inside a single diffusion trajectory*, using a *latent-space* verifier and *partial-rollback* search — split the 81 frames of a Wan 2.2 generation into spatiotemporal windows (e.g., 3 × 27-frame chunks, or per-frame), score each window's latent independently, re-seed or re-noise only the worst window instead of restarting the whole clip. Of the six surveyed papers: EFD&I (1) decides at whole-video granularity; DLBS (4) and DTS (5) operate per-step but not per-window; Tree-of-Frames (6) does per-frame branching but only inside Pyramid-Flow's pyramid levels. **Per-window verification + partial-rollback inside a single diffusion pass** is the wedge.

References lived under `paper-pdf/`. Theoretical license for early stopping (martingale + I-MMSE + Lipschitz) is already worked out in `paper-pdf/early-failure-theoretical-analysis.md`.

---

## 2. Codebase decision

**Recommendation: clone-and-extend `shim0114/T2V-Diffusion-Search` (Apache-2.0), trimmed to its `Wan2.1/` subtree, then port to Wan 2.2 5B 480p.** Vendor the other four repos as read-only references.

Why this and not the other four:

| Repo | Stars | Wan support | VBench wired | Intra-search | Inter-search | Verdict |
|---|---|---|---|---|---|---|
| **shim0114/T2V-Diffusion-Search** | 16 | **Wan 2.1 1.3B + 14B** | **vendored as both reward and evaluator** | per-step BoN | **DLBS / DLBS-LA beam search w/ backtracking** | **Best fit. Port to Wan 2.2.** |
| THU-SI/Video-T1 | 312 | Pyramid-Flow only | no | image-CoT BoN | tree-of-frames branching | Tightly coupled to pyramid levels — won't transfer cleanly to Wan's flat flow-matching |
| aimagelab/VHS | 13 | T2I (Sana-Sprint) only | GeneVal | latent-MLLM verifier + early-stop BoN | n/a | Best *idea*: latent verifier on DiT hidden states. Borrow the design, not the code |
| Guhuary/ProbeSelect | 4 | T2I (SD3.5) only | ImageReward | early-features→quality probe | n/a | Borrow the probe recipe |
| vineetjain96/Diffusion-Tree-Sampling | 10 | toys + SD image | n/a | **MCTS over latents** | n/a | Cleanest MCTS-over-latents reference; image-only, no license — algorithm only |

Why not build entirely from scratch: the Wan T2V driver, FSDP/Ulysses parallelism, VBench-as-reward integration, and beam-search-with-backtracking in T2V-Diffusion-Search are real engineering we'd otherwise re-derive. The repo's per-model directory structure is modular enough that we can keep `Wan/` and drop the rest without entangling our work.

Why not Video-T1 despite its 312 stars: pyramid-level branching does not generalize to Wan's denoising schedule, and adapting it would be a near-rewrite. We'll borrow the *idea* of frame-level branching but implement it against Wan ourselves.

**Concrete plan:**
1. `git clone https://github.com/shim0114/T2V-Diffusion-Search.git external/t2v-search` (vendored, our edits in our own packages).
2. Strip to `Wan2.1/`, `verifiers/`, `my_rewards/`. Drop Latte, CogVideoX, Pyramid subtrees.
3. Build our own top-level package `ttsd/` (test-time-scaling-diffusion) that imports the vendored Wan driver and adds: model adapter for Wan 2.2 5B, our own verifier interface, our own search algorithms, our own experiment runner.
4. License: keep Apache-2.0 inheritance, add our own LICENSE at root.

I have **not** cloned anything yet — confirm the strategy and I'll execute the clone + scaffold.

---

## 3. Phase 0 — Baseline & seed-sensitivity probe (weeks 1–2)

**Primary goal of Phase 0:** *prove that test-time scaling is even necessary on Wan 2.2 5B.* Concretely: show that VBench scores have non-trivial variance across seeds for the same prompt, and that some seeds produce clearly failed clips. Without this, no downstream TTS work is justified. Secondary goal: persist a latent dataset for Phase 1.

- **Model:** Wan 2.2 **TI2V-5B** (Diffusers fork; used in T2V mode by passing no input image), 480p, default scheduler, single GPU. Decided 2026-05-04 — no 14B validation in v1.
- **Prompts:** **15 hand-picked prompts** spanning VBench-relevant axes (motion, multi-object, spatial relation, single-subject consistency, scene complexity). Not the full ~950-prompt suite — tight loop for dev.
- **Seeds:** **5–10 per prompt** → ~75–150 clips total. Sized so a single GPU can finish a sweep in a working day.
- **Reporting (the load-bearing artifact):** per-prompt VBench-score histogram across seeds, plus visual side-by-side of best vs. worst seed. The headline plot of Phase 0 is **"score variance across seeds, per prompt"** — this *is* the TTS-necessity argument.
- **Eval dimensions (start):** Imaging Quality, Aesthetic Quality, Subject Consistency, Background Consistency, Overall Consistency, Motion Smoothness, Dynamic Degree.
- **Storage layout:**
  ```
  runs/baseline/<run_id>/<prompt_id>/<seed>/
    ├── video.mp4
    ├── latents/step_{05,10,15,...,50}.pt   # for later early-detection work
    ├── meta.json                              # prompt, seed, scheduler, model rev
    └── vbench/<dim>.json
  ```
  Persist every-5-steps latents during the baseline run — they are the dataset for Phase 1's verifier. Do this *once*; sampling is the expensive part.
- **Deliverable:** a CSV per VBench dimension (mean, std across seeds) plus per-prompt failure-rate (clips below per-dimension τ). This grounds the "how often does Wan 2.2 fail?" denominator.

---

## 4. Phase 1 — Latent verifier (weeks 3–6)

Build a **latent-space failure detector** that consumes intermediate denoising latents (or cheap previews) and predicts the final VBench score per dimension.

Three verifier candidates, evaluated head-to-head:

1. **Cheap preview + ViCLIP (EFD&I-style):** decode latent at step *k* via the L2R quick decoder, score with ViCLIP / aesthetic predictors. Pixel-space, ~tens of ms.
2. **Latent MLLM head (VHS-style):** small Qwen2.5-0.5B / LLaVA-tiny on intermediate hidden states, no decode. Latent-space, sub-ms after the model is loaded.
3. **Lightweight probe (ProbeSelect-style):** PCA + MLP regressor from intermediate features → final VBench score. Fastest, smallest model, lowest ceiling.

Training data = the latents persisted in Phase 0 (~10 seeds × 30 prompts × 10 step snapshots = ~3K trajectories on the dev set; scale to the full VBench suite for the final verifier).

Evaluation:
- **Per-dimension MAE** between predicted-at-step-*k* score and final VBench score, as a function of *k* (recreate Fig. 10 of EFD&I for Wan 2.2).
- **AUROC for failure classification** at thresholds derived from VBench score distributions.
- **Compute overhead** per step.

Deliverable: a `Verifier` interface in `ttsd/verifiers/` with at least two implementations and a benchmark report. Pick the best for Phase 2.

---

## 5. Phase 2 — Search & intervention (weeks 7–12)

Once we trust the verifier, plug it into search procedures. Implement four, in order of difficulty:

1. **Threshold-gated early stop** (paper 1 baseline). Decide kill/keep at step *k*, regenerate from new seed if killed. The "minimum viable" use case for the cloud-serving pitch (§7).
2. **Intra-step best-of-N noise** at the bottleneck steps identified in Phase 1.
3. **Per-window beam search w/ partial rollback** (extends DLBS-LA): inside a *single* Wan 2.2 trajectory, partition the 81 frames into spatiotemporal windows (start with temporal-only: 3 × 27-frame windows; later add spatial tiling), score each window's latent at the checkpoint steps from §4, and re-noise / re-roll only the lowest-scoring window while preserving the rest. Splice via overlap-blend in latent space. *This is the uncrowded research direction.* Note this is intra-clip windowing — not autoregressive chunk concatenation, which is a separate problem (§6.8).
4. **MCTS over latent trajectories** (DTS-style) for the highest-budget setting; compare against beam search on VBench-vs-compute Pareto.

Reporting standard: every method gets reported as a curve **VBench score vs. wall-time / GPU-seconds**, against the Phase 0 baseline and a naive "regenerate from scratch on failure" upper-cost reference.

---

## 6. Research extensions (parallel tracks during Phases 1–2)

These are the user's existing ideas, slotted into the framework:

### 6.1 More detection metrics
Beyond Overall Consistency: physics-rule detector (object permanence, gravity sanity), spatial-relation detector (correctness of "left of / behind / in"), multi-object cardinality detector. Each becomes a verifier head; train against VBench's Spatial Relationship + Object Class subsets, plus a small custom physics test set.

### 6.2 Adaptive thresholds
The fixed τ in EFD&I leaves headroom. Two routes, in order of ambition:
- **Bayesian early stopping** (already worked out in `early-failure-theoretical-analysis.md`, §3): fit prompt → (s_∞, κ) prior, do Kalman update with on-line scores, stop when `Pr(s_∞ < τ | obs) > 1−α`. Inherits Chernoff (1959) optimal-stopping guarantees under Gaussian assumptions.
- **Prompt-complexity classifier:** entity count, action count, relational tokens, CLIP-text-embedding distance to corpus centroid → predicted κ and predicted prior failure probability. Drives a per-prompt τ.

### 6.3 RL for search policy
Cast intra/inter search as an MDP: state = (current latent summary, verifier score history, compute remaining), action = {continue / re-seed / branch / replace-noise / accept}. Train policy with PPO or DPO-style preference learning over rollouts. Reward = final VBench − λ·compute. Worth attempting only after the heuristic searches in §5 are saturated.

### 6.4 Fine-grained intervention
Most current work intervenes by re-seeding the whole trajectory. More targeted, all *inside a single Wan 2.2 diffusion pass*:
- **Single-step intervention:** replace noise on a localized spatial region or a single frame at a single step (closer to image editing than to regeneration).
- **Frame-window intervention:** re-roll only the temporal window where the verifier dipped, splice the rest with overlap-blend in latent space.
- **Spatial-tile intervention:** re-noise only a spatial tile of the latent (e.g., a quadrant) when the failure is localized (single bad object, broken physics in one corner). Requires verifier with per-tile scoring.
- All three are naturally exposed by the per-window beam search in §5.3.

### 6.5 Theory and trajectory analysis (already partially done)
- The theoretical license for early stopping is in `paper-pdf/early-failure-theoretical-analysis.md` — the next thing to do is the **toy linear-Gaussian sanity check** from §4 of that doc, to verify the predicted MMSE-decay curve quantitatively matches Wan 2.2's empirical Fig.-10-style curve.
- Build the **prompt → convergence-speed model** (§3 of the analysis doc): heteroscedastic 2-layer MLP from interpretable prompt features + CLIP embedding to (s_∞, κ, s_0). Feeds §6.2.

### 6.6 Compute-efficiency optimizations
- **L2R / verifier overlap with denoising:** the verifier currently runs serially after each step. Move it to a CUDA stream that overlaps with the next step's UNet/DiT forward. Hides latency.
- **Sparse checking / "key window":** the early-failure-theory result says the MMSE curve has its maximum information gain in a *narrow* window of denoising steps; check densely there, sparsely elsewhere. Pre-compute the per-model "key window" once.

### 6.7 Human-in-the-loop calibration
Collect a few hundred human "acceptable / unacceptable" judgments on early-step previews; fine-tune the verifier head with that as a margin loss. Aligns the operating point to perceptual rather than VBench-score boundaries.

### 6.8 Stretch goal — long autoregressive video (out of scope for v1)
Once §5.3's per-window method works on single-shot Wan 2.2 clips, the natural follow-up is to apply it across *autoregressive chunk boundaries* in long-video generators (FramePack, MAGI-1, StreamingT2V, Wan's own video-to-video chaining, etc.). Failure cascades there are worse — one bad chunk poisons every subsequent chunk via the conditioning — so the cost-saving story is bigger. Explicitly **deferred**: it requires a different model stack and a different literature survey (long-video coherence work) that is not on the v1 critical path. Treat as a follow-up paper, not a v1 deliverable.

---

## 7. Path to deployment / serving

The "research → serving" handoff is the part that most academic TTS papers skip. The minimal viable deployment story:

**Stage A — offline verifier training & calibration (research output):** an open-source verifier checkpoint that takes a Wan 2.2 latent at step *k* and a prompt, returns predicted VBench-score-per-dimension + uncertainty. *This is what we publish.*

**Stage B — serving-engine integration (engineering output, optional):** a single hook in the inference loop:
```python
for step in denoising_loop:
    z = unet(z, t)
    if step in CHECKPOINT_STEPS:
        score, sigma = verifier(z, prompt)
        if early_stop_rule(score, sigma, step):
            return EARLY_FAIL  # caller can re-seed or surface to user
```
~30 lines of code. The serving win is straightforward: if the failure rate is *p* and we kill failures at step *k* of *T*, expected compute saving is `p × (1 − k/T)` — at *p* = 0.2, *k/T* = 0.2, that's a **16% steady-state GPU-cost cut** on a workload that today does nothing about failures. Numbers like that are what makes this land at a serving team.

**Stage C — adaptive scheduling (longer-term):** if a failure is detected, instead of returning to the user, the system can auto-trigger §5.3 (chunk-rollback) with a small extra compute budget. Looks to the user like a transparent quality boost, not a retry.

The pitch we should be able to make at the end of this project: *"Drop this verifier into your video-gen serving stack; you save N% of GPU-seconds at no quality loss, or trade the same compute for measurably better VBench."*

---

## 8. Repo skeleton (proposed)

```
tt-scaling-diffusion/
├── ROADMAP.md                       # this file
├── README.md
├── LICENSE                          # Apache-2.0
├── pyproject.toml
├── ttsd/                            # our package
│   ├── models/
│   │   └── wan22_adapter.py         # wraps the vendored Wan driver
│   ├── verifiers/
│   │   ├── base.py                  # Verifier interface
│   │   ├── viclip_preview.py        # EFD&I-style
│   │   ├── latent_mllm.py           # VHS-style
│   │   └── probe.py                 # ProbeSelect-style
│   ├── search/
│   │   ├── threshold_stop.py
│   │   ├── intra_bon.py
│   │   ├── window_beam.py            # per-window partial rollback inside one clip
│   │   └── mcts.py
│   ├── prompts/
│   │   ├── vbench_loader.py
│   │   └── complexity.py            # prompt → κ predictor
│   └── runners/
│       ├── baseline.py
│       └── tts_experiment.py
├── external/
│   └── t2v-search/                  # vendored, trimmed shim0114 repo
├── configs/
│   ├── baseline_wan22_480p.yaml
│   └── search/{...}.yaml
├── runs/                            # gitignored — experiment outputs
├── paper-pdf/                       # already exists
└── notebooks/                       # exploratory, gitignored by default
```

---

## 9. Milestones and success criteria

| Phase | Weeks | Concrete deliverable | Success bar |
|---|---|---|---|
| 0. Baseline | 1–2 | VBench numbers + step-wise latents persisted | Reproducible to ±0.5% across re-runs |
| 1. Verifier | 3–6 | At least two verifier implementations + comparison report | One verifier reaches MAE-vs-final < 0.05 by step ~T/2 on Overall Consistency |
| 2. Search | 7–12 | Threshold-stop + intra-BoN + per-window-beam-search (intra-clip) with VBench-vs-compute curves | At equal compute, ≥ 5% VBench gain over baseline; or at equal VBench, ≥ 30% compute saving |
| 3. Extensions | parallel | Prompt-complexity κ-predictor + adaptive-threshold integration | Adaptive τ beats fixed τ on at least 3 VBench dimensions |
| 4. Serving demo | 13–14 | Minimal verifier-hooked inference script + cost-saving table | One-command demo + a page of "what would this save in production" |

---

## 10. Open questions to resolve before the first line of code

These are *decisions the user (you) should make*, not search problems:

1. **Wan 2.2 5B vs. 14B for the dev loop.** 5B fits the budget; 14B is what serving teams care about. Suggest: dev on 5B, validate the final pipeline on 14B near the end.
2. **480p vs. 720p.** 480p fits more seeds per GPU-hour. Suggest: 480p throughout, document that scaling to 720p is a re-evaluation, not a re-design.
3. **Single-GPU vs. FSDP-Ulysses from day one.** The vendored repo supports the latter; it adds engineering overhead. Suggest: single-GPU until Phase 2 chunk-search needs the throughput.
4. **VBench full suite vs. 1–2 prompts per dimension during dev.** The roadmap assumes the dev/full split above; confirm.
5. **External feedback channel.** Who's reviewing the verifier checkpoint and the serving pitch — advisor, lab group, an industry contact?

---

*Last updated: 2026-05-04. Edit freely; this file is the living spec.*
