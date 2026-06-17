# End-to-end framework for test-time scaling of T2V diffusion — PROPOSAL

**Branch:** `e2e-infra-dev`  ·  **Status:** draft for discussion, NO implementation yet  ·  **Author:** Dylan + Claude  ·  **Date:** 2026-05-23

**Scope (locked 2026-05-23):** v1 ships EFD&I-replication on our verifier + a naive best-of-N (BoN) baseline. The abstraction layer is shaped so beam search, MCTS, and RL strategies plug in later without rewriting the backbone. We do not build those in v1.

This document proposes the inference-time backbone that wires together our existing verifier (signal) and the search/intervention logic (action) into a live, model-agnostic, plug-and-play pipeline. EFD&I = the prior work we replicate; naive BoN = the baseline reviewers will demand. Same backbone, two plug-ins.

---

## 1. Goal & non-goals

**Goal.** Build the inference-time orchestrator that:

1. drives any latent diffusion T2V model through its denoising loop,
2. queries one or more verifiers on intermediate state to get a *failure signal*,
3. consults a *decision policy* that decides whether to continue, intervene, or abort,
4. executes the chosen *intervention* via a pluggable *search/refinement strategy*,
5. tracks budget, logs telemetry, and emits a final video.

**v1 must support, hands-on:**

- **EFD&I-style tiered intervention** — replicating the paper's Trial 0 / Trial 1 / Trial 2 / Trial 2→1 design, with our DINO verifier in place of L2R+ViCLIP.
- **Naive best-of-N (BoN)** — generate N candidates with different seeds, score each at one verifier checkpoint, return the highest-scoring one.

**v1 must NOT break for future:**

- Beam search with partial rollback, MCTS over diffusion trajectories, learned RL policies — none built in v1, but each must be a `SearchStrategy` plug-in away.

**Non-goals (v1):**

- Training new verifiers (we already have DINO ones).
- Multi-GPU model parallelism (orchestrator-level parallelism only).
- Production serving (we'll keep the design serving-friendly, but ship no server).
- An L2R-style tiny decoder for Wan (defer; v1 verifies at *checkpoint* steps only, where full VAE decode is tolerable).

---

## 2. Why a new backbone? — the gap

What we already have on `main`:


| Component                               | File                           | Status                                                                                                                              |
| --------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Wan 2.2 model adapter                   | `ttsd/models/wan22_adapter.py` | ✅ `generate()` + posterior-mean / raw-latent / model-output capture hooks. Pluggable scheduler (UniPC ODE / Euler ODE / Euler SDE). |
| Verifier interface                      | `ttsd/verifiers/base.py`       | ✅ ABC: `score(latent, prompt, step, total_steps) -> VerifierOutput`.                                                                |
| Concrete verifiers                      | `ttsd/verifiers/dino/*.py`     | ✅ 4 DINO-based + max-Z fusion. Currently consume **cached features**, not live state.                                               |
| Search interface                        | `ttsd/search/base.py`          | ✅ stub: `SearchPolicy.step(ctx) -> Decision` enum (continue / early-stop / branch / replace-noise).                                 |
| Concrete search policies                | —                              | ❌ none yet.                                                                                                                         |
| Decision policy (separable from search) | —                              | ❌ implicit in the `Decision` enum. No configurable policy.                                                                          |
| Action / intervention primitives        | —                              | ❌ none.                                                                                                                             |
| Live inference orchestrator             | —                              | ❌ none. The verifiers run **after** the sweep, against `runs/baseline/...` features.                                                |
| Budget / cost tracking                  | —                              | ❌ none.                                                                                                                             |
| Trajectory bookkeeping                  | —                              | ❌ none.                                                                                                                             |
| Telemetry / structured logs             | —                              | ❌ `print()` + ad-hoc CSVs.                                                                                                          |
| Registry / plugin loader                | —                              | ❌ none.                                                                                                                             |
| Config schema                           | YAMLs per sweep                | ⚠️ ad-hoc, no shared schema.                                                                                                        |


The pieces we're missing form a coherent block: a **runtime** that lets a verifier *influence* sampling instead of *labelling* it post-hoc.

The rest of this section spells out what each row does — what it provides, a concrete example, and why it matters. Skim if you already know; this is the "I want to understand each moving part" reference.

### 2.1 What we already have

#### `Wan22Adapter` — `ttsd/models/wan22_adapter.py`

**Role.** Wraps the diffusers `WanPipeline` into one object we control. Given `(prompt, seed, scheduler_kind)`, it runs the denoising loop and returns the video PLUS optional captures (intermediate `x_t` latents, `x0_hat` posterior means, raw model outputs, scheduler metadata) at any chosen step.
**Example.** `Wan22Adapter().generate_with_posterior_means(prompt, seed, snapshot_steps=[1,4,...,49])` — what the `sweep_v2` runs invoke.
**Why it matters.** Without it, every runner would re-implement scheduler-swap, callback hooks, and x0_hat capture. It's the *one* place that knows how Wan's diffusion loop works.

#### `Verifier` ABC — `ttsd/verifiers/base.py`

**Role.** The contract every verifier obeys: `score(latent, prompt, step) → VerifierOutput`. Doesn't do work itself — just defines the shape so different verifiers (DINO, ViCLIP, MLLM, learned probe) are interchangeable.
**Example.** `class Verifier(ABC): @abstractmethod def score(...): ...`
**Why it matters.** Lets the orchestrator call `verifier.score(...)` without caring which verifier is plugged in. Plugin discipline at the interface boundary.

#### Concrete DINO verifiers — `ttsd/verifiers/dino/*.py`

**Role.** The actual signal source. Each takes intermediate state (currently *cached* DINOv2 features), runs a probe (frame-cos-mean / similarity profile / quantile-PCA-ridge / max-Z fusion), returns a scalar predicting final VBench score.
**Example.** `CombinedDinoVerifier` is the production-quality combined version.
**Why it matters.** This is our "signal" — without it we have no way to judge "is this generation failing?" **Current limitation:** they consume *cached* features from a finished sweep, not live state during generation. v1 of the pipeline adds a thin online wrap.

#### `SearchPolicy` stub — `ttsd/search/base.py`

**Role.** Currently a placeholder defining `SearchPolicy.step(ctx) -> Decision` with a small `Decision` enum (`CONTINUE / EARLY_STOP_FAIL / EARLY_STOP_ACCEPT / BRANCH / REPLACE_NOISE`). No concrete implementation exists.
**Example.** None yet.
**Why it matters.** It's the slot the new framework fills in. In the new design we rename this layer to `SearchStrategy` (so it doesn't collide with the new `DecisionPolicy`), and the `Decision` enum is superseded by the richer `Action` system.

### 2.2 What's missing (what we'll build)

#### Concrete search strategies

**Role.** The actual outer-loop algorithms. EFD&I = "try Trial 0, if fail try Trial 1, if fail try Trial 2, ..." (sequential). BoN = "spawn N candidates, score them, keep the best" (parallel).
**Example.** `SequentialTrialSearch([T0, T1, T2, T2→1])`, `ParallelCandidateSearch(N=4)`.
**Why it matters.** This is where "test-time scaling" actually lives. Without concrete strategies we have a detector with nothing to do when it detects.

#### `DecisionPolicy` (separated from the search strategy)

**Role.** A *pure function* mapping `(VerifierOutput, history, budget) → ActionSpec`. The rule that turns a score into a decision. Examples: fixed threshold (`score < 0.22` → fail), sliding window (average last 3 scores), Bayesian (posterior over final score). Stateless and trivially unit-testable.
**Example.** `DynamicSlidingWindowPolicy(tau=0.22, window=3).decide(verifier_out, ctx)` returns either `Continue` or `SingleFrameAnchorInject`.
**Why it matters.** EFD&I bakes the rule into the search code. Separating them lets us A/B "same search, different rule" or "same rule, different search" — which is exactly what ablations need. Cost-wise this is one extra class; benefit is large.

#### `Action` primitives

**Role.** The smallest unit of intervention. Each Action knows how to mutate a trajectory in one specific way and reports its `estimated_cost`. v1 set: `Continue`, `StopAndFail`, `StopAndAccept`, `RestartWithNewSeed`, `SingleFrameAnchorInject` (EFD&I T1), `RefinePromptVLM` (EFD&I T2), `RolloutBoNCandidates`, `KeepBestCandidate`.
**Example.** `SingleFrameAnchorInject(steps=[0,1], k_img=8).apply(state, adapter, ctx)` knows how to fork a single-image rollout, capture its `x0_hat`, and splice it into the video init.
**Why it matters.** Without `Action`s as first-class objects, every search strategy re-implements "what does Trial 1 actually do." With them, "EFD&I" and "BoN" become *lists of actions* in YAML, not different code paths.

#### `Orchestrator`

**Role.** The conductor. Loads config → builds adapter + verifier + policy + strategy → runs them. ~150 LOC of glue, owns no domain logic itself.
**Example.** `Orchestrator(config).run(prompt="A cat sitting on a sofa", seed=42) → RunResult`.
**Why it matters.** It's the single entry point. Without it, users wire components by hand. With it, the CLI runner becomes ~50 LOC and the same code is embeddable in a notebook, a server, or a CI test.

#### `Budget` tracker

**Role.** Tracks wall-clock, GPU-seconds, VLM token cost in real time. Policies and Actions ask "do I have budget for this?" before doing expensive things.
**Example.** `if budget.remaining_s() < cost_estimate.wall_clock_s: return StopAndFail` — implements EFD&I's "even worst-case ≤ 56% of base regeneration cost" guarantee in three lines.
**Why it matters.** Without it we can't enforce time caps, can't compare strategies on a Pareto front, can't claim cost savings to a serving team.

#### `Trajectory` / `TrajectoryState`

**Role.** The state object that gets passed between steps. Holds: current latents, score history (so sliding-window policies can look back), action history (so we don't re-fire the same action), trial index, parent pointer (for branching / BoN), accumulated cost.
**Example.** `TrajectoryState(latent=..., step=10, score_history=[0.18, 0.19, 0.20], actions_taken=[Continue, Continue], trial=0)`.
**Why it matters.** Policies need context — "is this score 0.18 a one-off dip or a sustained low?" requires history. Without a state object, policies are stateless and can't see beyond the current step.

#### Telemetry / structured logs (JSON-lines)

**Role.** One JSON-line per event (verifier call, policy decision, action dispatch, budget update). Append-only, cheap to parse, ingestible by every analysis runner we already have.
**Example.** `{"event": "policy_decision", "step": 10, "verifier_score": 0.18, "decision": "trigger_T1", "trial": 0, "wall_clock_s": 12.4}`.
**Why it matters.** Today's stdout `print()` is for humans. The Pareto plot, the per-prompt trial-distribution histogram, and the cost breakdown table all need machine-readable logs.

#### Registry / plugin loader

**Role.** A `dict[str, class]` plus a decorator. `@register_verifier("dino_combined")` inserts a class. Config YAML writes `verifier.kind: dino_combined`; the orchestrator looks the class up by string — *never imports concrete classes directly*.
**Example.** HuggingFace's `AutoModel.from_pretrained("bert-base")` and gymnasium's `gym.make("CartPole-v1")` follow this exact pattern.
**Why it matters.** Without it, the orchestrator imports every possible verifier / policy / action at the top of the file → import cycles, can't add a verifier without editing the orchestrator. Registry = clean plugin architecture in ~30 LOC.

#### Config schema

**Role.** A typed dataclass hierarchy (`PipelineConfig` containing `ModelConfig`, `VerifierConfig`, `PolicyConfig`, `StrategyConfig`, `BudgetConfig`) plus a YAML loader. Every experiment is fully described by one YAML file.
**Example.** `configs/pipeline/efdi_style_dino.yaml` is the full reproducible spec of an EFD&I-style run; `configs/pipeline/bon_dino.yaml` is the BoN baseline. Diff between them = the experiment.
**Why it matters.** Today each sweep YAML has ad-hoc fields. A shared schema means (a) IDE autocomplete + type checking on configs, (b) trivially reproducible experiments, (c) configs can be archived next to results for forever-reproducibility.

### 2.3 TL;DR mapping


| Has               | Provides the …             | Missing           | Provides the …              |
| ----------------- | -------------------------- | ----------------- | --------------------------- |
| Wan22Adapter      | base model + state capture | Orchestrator      | conductor / lifecycle       |
| Verifier ABC      | signal contract            | Search Strategy   | outer-loop algorithm        |
| DINO verifiers    | signal                     | Decision Policy   | the rule (signal → choice)  |
| SearchPolicy stub | placeholder                | Action primitives | the units of intervention   |
|                   |                            | Budget            | cost gating                 |
|                   |                            | Trajectory        | per-rollout state + history |
|                   |                            | JSON-lines log    | machine-readable telemetry  |
|                   |                            | Registry          | plugin discovery            |
|                   |                            | Config schema     | one-file reproducibility    |


Of the 9 missing pieces, **Orchestrator + Registry + Config + Logger + Budget + Trajectory** are pure infrastructure (~600 LOC total, P1 work). **Decision Policy + Action + Search Strategy** are where the actual TTS algorithms live (the meat of P2–P4).

---

## 3. Mental model — three layers + cross-cutting registry

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Orchestrator (Pipeline driver)                                              │
│    drives one user request through a Trial sequence to a final video         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ for each Trial:
       ┌────────────────────┴────────────────────┐
       │                                         │
┌──────▼──────────┐                    ┌─────────▼─────────┐
│  Runtime loop   │                    │  Decision Policy  │
│  drives one     │  per-step state ▶  │  threshold /      │
│  diffusion      │                    │  sliding window / │
│  trajectory     │  ◀ apply Action    │  pick-best ▶      │
│                 │                    │       ActionSpec  │
└──┬─────┬────────┘                    └───────────────────┘
   │     │
   │     │ score(state)
   │     ▼
   │  ┌────────────┐
   │  │  Verifier  │  (DINO combined / online wrap of the existing classes)
   │  └────────────┘
   │
   │ generate / step / inject / decode
   ▼
┌────────────┐
│ Model      │  Wan22Adapter, (future: CogVideoXAdapter, ...)
│ Adapter    │  scheduler kind: unipc / euler / euler_sde
└────────────┘

Cross-cutting:
  • Registry — string ID → class (verifier / policy / action / search / model)
  • Config   — dataclasses; YAML loader; per-trial override block
  • Logger   — JSON-lines (one event per Decision / Action / Verifier call)
  • Budget   — wall-clock + GPU-seconds tracker; hard caps enforce a stop
```

The three layers map to **three abstractions you should be able to swap independently**:

- **Verifier** = what signal we read
- **Decision Policy** = how we turn signal → action choice
- **Search Strategy** + **Action Repertoire** = how the action actually mutates the trajectory

### v1 mappings

**EFD&I tiered intervention**:

- Verifier = `DinoOnlineVerifier(dino_combined)` (online wrap of our existing combined DINO verifier)
- Decision Policy = `DynamicSlidingWindowPolicy(tau=0.22, delta=0.05, window=3, escalation=[Continue, SingleFrameInject, RefinePromptVLM, SingleFrameInject])`
- Search Strategy = `SequentialTrialSearch` (run trials in order until one succeeds)
- Actions = `Continue`, `SingleFrameAnchorInject`, `RefinePromptVLM`

**Naive BoN (the baseline)**:

- Verifier = same online DINO verifier
- Decision Policy = `BestOfNPolicy(N=4, decide_at_step=10)`
- Search Strategy = `ParallelCandidateSearch(N=4)`
- Actions = `RolloutBoNCandidates`, `KeepBestCandidate`, `Continue`

**Beam search / MCTS / RL (post-v1):**
The same shape — different `SearchStrategy` + `Policy` + `Action` plugins. We will not implement these in v1, but the backbone is ready for them.

---

## 4. Core abstractions (proposed APIs)

These are the **interfaces** — implementations come later. Each has a short rationale.

### 4.1 `ModelAdapter` (refines what `Wan22Adapter` already does)

```python
class ModelAdapter(Protocol):
    """Pluggable diffusion backbone. Generates a trajectory; supports hooks
    for stepwise state capture AND mid-stream Action injection."""

    def generate(
        self,
        request: GenerationRequest,
        on_step_end: Callable[[StepState], StepDirective] | None = None,
    ) -> GenerationOutput: ...

    def decode_latent_preview(self, latent: torch.Tensor) -> torch.Tensor: ...
    # For RI-style L2R previews; lightweight per-step decode if available.
    # v1: this falls back to full VAE on Wan since L2R-for-Wan is deferred.
```

- `on_step_end` is the **hot path** for the orchestrator. The callback receives the current latent state and returns a `StepDirective` (continue / replace_latent / abort_with_reason). This is the *one extension point* that turns offline verification into online intervention.
- `Wan22Adapter` already implements a similar callback for `generate()` and `generate_with_posterior_means()`; we'll harmonize the two paths into one.

### 4.2 `Verifier` (extend the existing ABC)

```python
class Verifier(ABC):
    @abstractmethod
    def score(self, state: TrajectoryState, prompt: str) -> VerifierOutput: ...

    # New: declare what the verifier needs so the orchestrator caches efficiently.
    REQUIRES: ClassVar[set[str]]  # e.g. {"posterior_mean", "prompt_embed"}
```

- `TrajectoryState` is the snapshot the orchestrator builds (latent, posterior mean, decoded preview, prior scores). Each verifier reads what it needs.
- `REQUIRES` lets the orchestrator only run expensive captures (VAE decode, DINOv2 forward) when **some registered verifier asks for them**. Zero waste.
- `VerifierOutput` gains an optional `final_score_estimate: float | None` — the headline scalar that policies read. The existing `score: dict[str, float]` stays.

### 4.3 `DecisionPolicy` (NEW)

```python
class DecisionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        verifier_out: VerifierOutput,
        ctx: DecisionContext,
    ) -> ActionSpec | None: ...
    # None means "no action this step — just continue".
```

`DecisionContext` carries: step index, denoising total, verifier history (sliding window), budget remaining, trial number, prior actions in this rollout, prompt. Policies are *pure* — no side effects — so they're trivially testable.

**v1 policies:**

- `FixedThresholdPolicy(tau, decide_at_steps)` — single-shot threshold. The simplest possible policy; useful as a baseline and unit-test target.
- `DynamicSlidingWindowPolicy(tau, delta, window, escalation)` — EFD&I's actual detector. The `escalation` field is the ordered list of `ActionSpec`s to try when the policy returns "fail."
- `BestOfNPolicy(N, decide_at_step)` — for BoN. Scores all N candidates at one checkpoint, emits `KeepBestCandidate`.

**Post-v1 (not built):** `BayesianStoppingPolicy` (needs the prompt→κ predictor from `paper-pdf/early-failure-theoretical-analysis.md` §3), `BeamPolicy`, learned RL policy.

### 4.4 `Action` (NEW)

```python
class Action(ABC):
    @abstractmethod
    def apply(
        self,
        state: TrajectoryState,
        adapter: ModelAdapter,
        ctx: ApplyContext,
    ) -> ActionResult: ...

    @property
    @abstractmethod
    def estimated_cost(self) -> CostEstimate: ...  # for budget gating
```

**v1 action repertoire (each is one small file):**


| Action                                     | What it does                                                                  | Maps to                     |
| ------------------------------------------ | ----------------------------------------------------------------------------- | --------------------------- |
| `Continue`                                 | no-op (let denoising proceed)                                                 | EFD&I Trial 0; default      |
| `StopAndFail`                              | hard early exit; return failure to caller                                     | aggressive budget gating    |
| `StopAndAccept`                            | early commit to current preview as "good enough"                              | useful for cheap prompts    |
| `RestartWithNewSeed(seed)`                 | drop trajectory, start over                                                   | naive regeneration baseline |
| `RefinePromptVLM(vlm_spec)`                | call VLM with current preview + original prompt → new prompt; restart         | EFD&I Trial 2               |
| `SingleFrameAnchorInject(steps_to_inject)` | generate single-frame preview, inject its `x0_hat` at first k denoising steps | EFD&I Trial 1               |
| `RolloutBoNCandidates(N, until_step)`      | spawn N parallel candidates with different seeds                              | BoN setup                   |
| `KeepBestCandidate(scores)`                | from a set of candidates, keep the one with the highest verifier score        | BoN selection               |


Each action declares `estimated_cost` so the policy can gate by budget. **Beam-specific actions are deferred.**

### 4.5 `SearchStrategy` (NEW)

```python
class SearchStrategy(ABC):
    @abstractmethod
    def run(self, request: GenerationRequest, ctx: RunContext) -> GenerationOutput: ...
```

The search strategy is the *outer loop*. It owns:

- the set of active trajectories (1 for sequential, N for BoN),
- the trial sequence (e.g. EFD&I's [T0, T1, T2, T2→1]),
- the orchestration of `Action` results (e.g. "Trial 1 succeeded → return its output").

**v1 strategies:**

- `SequentialTrialSearch(trials)` — EFD&I-style. Run trials in order until one's verifier says "accept" or budget exhausted.
- `ParallelCandidateSearch(N)` — vanilla BoN.

**Post-v1 (not built):** `WindowedBeamSearch`, `MCTSSearch`, `RLPolicyRollout`.

### 4.6 `Orchestrator` (NEW)

```python
class Orchestrator:
    def __init__(self, config: PipelineConfig, registry: Registry):
        self.adapter = registry.build_model(config.model)
        self.verifier = registry.build_verifier(config.verifier)
        self.policy = registry.build_policy(config.policy)
        self.strategy = registry.build_strategy(config.strategy)
        self.budget = Budget.from_config(config.budget)
        self.logger = JsonlLogger(config.log_path)

    def run(self, prompt: str, seed: int | None) -> RunResult: ...
```

The orchestrator is intentionally thin (~150 LOC). It delegates everything to the plugged-in components and just owns the lifecycle, the budget, and the log.

### 4.7 Cross-cutting

- **Registry** — decorator-based plugin table. `@register_verifier("dino_combined")`. The orchestrator never imports concrete classes; it asks the registry by name from the config. *This is the key extensibility hook.*
- **Config** — `dataclass`-based. Hierarchical: `PipelineConfig` contains `ModelConfig`, `VerifierConfig`, `PolicyConfig`, `StrategyConfig`, `BudgetConfig`. YAML loader. Each subconfig has `kind: str` (the registry key) + free-form `params: dict` consumed by the concrete impl.
- **Budget** — tracks wall-clock + GPU-seconds (+ token budget if VLM in play). Soft warnings + hard cap. Actions' `estimated_cost` is consulted *before* dispatch.
- **Logger** — JSON-lines, one event per Verifier call / Policy decision / Action dispatch / Trajectory step. Cheap to parse; mirrors our existing `runs/_logs/*.log` pattern.
- **Trajectory** — opaque-to-callers state object the strategy passes around. Holds current latents, score history, action history, accumulated cost, trial index, parent (for branching).

---

## 5. Proposed file layout

Add a new top-level package `ttsd/pipeline/`. Existing packages stay; we augment.

```
ttsd/
├── models/              # existing — add ModelAdapter Protocol; Wan22 already there
├── verifiers/           # existing — extend Verifier ABC with REQUIRES set
│   └── dino/
│       └── online_adapter.py    ◀ NEW — thin live-wrap of existing offline DINO verifier
│                                  (VAE-decode posterior_mean → DINOv2 features → existing probe)
├── search/              # existing — add concrete strategies here
│   ├── base.py                  ◀ sharpen SearchStrategy ABC
│   ├── sequential.py            ◀ NEW — SequentialTrialSearch (EFD&I)
│   └── parallel.py              ◀ NEW — ParallelCandidateSearch (BoN)
│
├── pipeline/            ◀ NEW package
│   ├── __init__.py
│   ├── orchestrator.py          ◀ Orchestrator + RunResult
│   ├── trajectory.py            ◀ TrajectoryState + StepState dataclasses
│   ├── policy/
│   │   ├── base.py              ◀ DecisionPolicy ABC + DecisionContext + ActionSpec
│   │   ├── threshold.py         ◀ FixedThresholdPolicy, DynamicSlidingWindowPolicy
│   │   └── bon.py               ◀ BestOfNPolicy
│   ├── actions/
│   │   ├── base.py              ◀ Action ABC + CostEstimate + ActionResult
│   │   ├── basic.py             ◀ Continue, StopAndFail, StopAndAccept, RestartWithNewSeed
│   │   ├── inject.py            ◀ SingleFrameAnchorInject (EFD&I Trial 1)
│   │   ├── prompt.py            ◀ RefinePromptVLM (EFD&I Trial 2)
│   │   └── parallel.py          ◀ RolloutBoNCandidates, KeepBestCandidate
│   ├── budget.py                ◀ Budget tracker
│   ├── registry.py              ◀ Registry with decorator pattern
│   ├── config.py                ◀ dataclass configs + YAML loader
│   └── logger.py                ◀ JSON-lines telemetry
│
├── runners/             # existing — add ONE new runner:
│   └── pipeline/
│       └── run_pipeline.py      ◀ CLI: python -m ttsd.runners.pipeline.run_pipeline --config <yaml>
│
└── eval/                # existing — unchanged
```

The orchestrator is **library code**, not a script. It will be embeddable in a server, a notebook, or a CI test. The CLI runner is a thin wrapper around it (≤50 LOC), in keeping with what we already do for `baseline.py`.

---

## 6. Sample configs (illustrative — pseudo-YAML)

Two configs, one schema. `strategy.kind` picks EFD&I-style vs BoN.

```yaml
# configs/pipeline/efdi_style_dino.yaml — replicates EFD&I tiered intervention
# using OUR DINO verifier in place of L2R+ViCLIP.

model:
  kind: wan22_ti2v_5b
  params:
    dtype: bf16
    scheduler: euler_sde

verifier:
  kind: dino_combined_online       # online wrap of CombinedDinoVerifier
  params:
    artifacts_dir: runs/verifiers/dino_combined/
    decide_at_steps: [10]          # like EFD&I; score at step 10/50

policy:
  kind: dynamic_sliding_window
  params:
    tau: 0.22                      # paper's threshold (we'll re-calibrate for our scorer)
    delta: 0.05                    # margin for Trial 1 trigger (single-frame > video by this much)
    window: 3                      # average 3 most-recent scores
    escalation:
      - { kind: continue }
      - { kind: single_frame_anchor_inject, params: { inject_steps: [0, 1], k_img: 8 } }
      - { kind: refine_prompt_vlm, params: { vlm: claude_haiku_45 } }
      - { kind: single_frame_anchor_inject, params: { inject_steps: [0, 1], k_img: 8 } }

strategy:
  kind: sequential_trial

budget:
  wall_clock_s: 180
  gpu_seconds: 240
  vlm_tokens: 4000

log:
  path: runs/pipeline/{run_id}/events.jsonl
```

```yaml
# configs/pipeline/bon_dino.yaml — naive best-of-N baseline

model: { kind: wan22_ti2v_5b, params: { scheduler: unipc } }

verifier:
  kind: dino_combined_online
  params:
    artifacts_dir: runs/verifiers/dino_combined/
    decide_at_steps: [10]

policy:
  kind: best_of_n
  params: { N: 4, decide_at_step: 10 }

strategy:
  kind: parallel_candidates
  params: { N: 4 }

budget:
  wall_clock_s: 400

log:
  path: runs/pipeline/{run_id}/events.jsonl
```

Two completely different test-time-scaling designs, **same config schema, same backbone**. That's the test of whether the framework is clean.

---

## 7. Two walked-through scenarios

### Scenario A — EFD&I replication on Wan 2.2

1. User: `python -m ttsd.runners.pipeline.run_pipeline --config configs/pipeline/efdi_style_dino.yaml --prompt "..."`
2. Orchestrator loads config → builds adapter (Wan 2.2), verifier (DINO combined, online wrap), policy (DynamicSlidingWindow), strategy (SequentialTrial).
3. Strategy launches Trial 0 = base generation. On each verifier-checkpoint step, the runtime captures `posterior_mean`, hands to the verifier (which decodes via VAE → DINOv2 → existing probe → scalar score), policy decides `Continue` until step 10.
4. At step 10, policy sees window-average ≥ τ → `Accept`. Strategy unwinds, returns the final video.
5. Logger emits ~50 events to `events.jsonl`. Budget = ~base time.

If at step 10 score < τ:
6. Policy emits `SingleFrameAnchorInject` action → Strategy launches Trial 1: spawn single-frame side-rollout, get its `x0_hat`, inject at steps 0–1 of a new video rollout. Verify again at step 10.
7. If Trial 1 still fails: `RefinePromptVLM` action → call VLM, get refined prompt, restart with refined prompt as Trial 2.
8. If Trial 2 still fails: `SingleFrameAnchorInject` once more on the refined prompt (EFD&I's "Trial 2→1"). Then terminate; return best of the trials.

### Scenario B — Naive BoN

1. User: `python -m ttsd.runners.pipeline.run_pipeline --config configs/pipeline/bon_dino.yaml --prompt "..."`
2. Strategy = `ParallelCandidateSearch(N=4)`. Launches 4 trajectories with seeds `[s0, s1, s2, s3]`. (Sequentially on 1 GPU; orchestrator-level concurrency is a Phase-6 nice-to-have, not v1.)
3. All 4 denoise to the verifier checkpoint (step 10). Verifier scores each.
4. Policy = `BestOfNPolicy` → keeps the highest, kills the other 3.
5. Winning trajectory denoises to the end.

Both scenarios reuse the same `ModelAdapter`, `Verifier`, `Logger`, `Budget`, and config schema. Only the strategy + policy + actions differ.

---

## 8. Implementation phases

Each phase is a self-contained PR-sized chunk. After each, we have a working e2e demo.


| Phase             | Deliverable                                                                                                                                                                                                                                                                                                                       | Wall-clock |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **P0** (this doc) | Plan + sign-off                                                                                                                                                                                                                                                                                                                   | done       |
| **P1**            | Skeleton: `pipeline/` package, `Registry`, `Config`, `Trajectory`, `Budget`, `Logger`, `Orchestrator` shell. Adapter `on_step_end` hook on `Wan22Adapter`. Smoke test: orchestrator drives Wan22Adapter end-to-end with NoOp verifier + `Continue`-only policy + SequentialTrial. Produces a video identical to current baseline. | ~1 day     |
| **P2**            | Wire up `Verifier.REQUIRES` extension. Build `DinoCombinedOnlineVerifier` (~80 LOC wrap of the existing offline verifier). Add `FixedThresholdPolicy`, `Continue` + `StopAndFail` actions. Smoke test: pipeline runs base generation, checks score at step 10, aborts if below τ. First real "decide at step 10" demo.            | ~1 day     |
| **P3**            | Add `SingleFrameAnchorInject` (Trial 1) + `RefinePromptVLM` (Trial 2) actions. Add `DynamicSlidingWindowPolicy`. Full EFD&I replication on our model. Numeric comparison: replication vs. paper's reported VBench / overhead.                                                                                                     | ~3 days    |
| **P4**            | Add `ParallelCandidateSearch` + `BestOfNPolicy` + `RolloutBoNCandidates` + `KeepBestCandidate`. Run a side-by-side experiment: BoN-N vs. EFD&I-tiered on the same prompt set. Pareto plot (VBench vs. wall-clock).                                                                                                                | ~1–2 days  |
| **P5**            | Hook into existing 4-GPU sweep launcher. Make `run_pipeline` work in the same shard pattern as `baseline.py`. Run a *pipeline sweep* across our 30 prompts × ≥5 seeds × 2 strategies (EFD&I, BoN) on GPUs 4–7.                                                                                                                    | ~2 days    |


**Total ~8 active days** for the full v1 contribution. After P3 we have a publishable replication; after P4 we have the BoN baseline; P5 makes it runnable at scale.

**Beam / MCTS / RL** — explicitly deferred. The backbone is shaped so adding them is "new files under `ttsd/pipeline/policy/` and `ttsd/search/`, no edits to `orchestrator.py`."

---

## 9. Design choices — LOCKED 2026-05-23

The original tradeoff analysis is preserved below; each item ends with the chosen path.

1. **Registry vs. import-by-string.** (a) decorator registry (HF/Diffusers style — `@register_verifier("dino")`), or (b) explicit `module:class` strings in YAML (Hydra/LightningCLI style). (a) is more discoverable; (b) avoids import cycles and lets users add components without editing the registry.
   **→ DECIDED:** (a) for built-ins, (b) as a fallback for external add-ons. `verifier.kind` in YAML accepts either a registered short name (`"dino_combined"`) or a `"module.path:ClassName"` string.

2. **Config format.** Dataclasses-only (typed, IDE-friendly) vs. Hydra (composable YAML, override syntax).
   **→ DECIDED:** dataclasses + YAML loader for v1. Hydra deferred — revisit only if config composition becomes painful.

3. **VLM choice for Trial 2 (`RefinePromptVLM`).** EFD&I uses GPT-5-mini. Options: Anthropic Claude (Haiku 4.5), OpenAI GPT, local VLM (LLaVA).
   **→ DECIDED:** leave open. Ship a `VLMClient` interface with `kind` + `api_key` slots in YAML; default left blank. Concrete client class chosen per-config; we'll wire one (or several) when Trial 2 is actually used. No hard dependency on any one provider baked into the framework.

4. **Synchronous vs. async orchestrator.** Async helps for multi-trajectory or overlapping VLM-API and GPU work. v1 single-trajectory is naturally sync; BoN with N candidates on 1 GPU is also sync.
   **→ DECIDED:** sync for v1. Async tracked as a future refactor — likely needed once we run BoN across multiple GPUs (P5 area) or want to overlap a slow VLM call with GPU denoising. Will be a self-contained orchestrator rewrite when that happens.

5. **How to wrap existing offline DINO verifiers for live use.** Offline impls currently consume cached features.
   **→ DECIDED:** add the online adapter alongside as `ttsd/verifiers/dino/online_adapter.py`. Do NOT refactor the offline code — it stays useful for training and post-hoc evaluation.

6. **Naming.** Top-level package name. Candidates: `pipeline`, `orchestrator`, `runtime`, `inference`, `tts`, `e2e`.
   **→ DECIDED:** `pipeline`. Reads naturally as `from ttsd.pipeline import Orchestrator`, matches HF/Diffusers conventions.

---

## 10. Open research questions left to the strategies (NOT solved by this doc)

This doc is about the *backbone*. The following are research questions the **strategies** answer, and we deliberately don't bake answers into the framework. **Owner / answers: TBD — revisit after v1 ships and we have runnable experiments to point at.**

- What's the optimal τ for our DINO verifier, and is it prompt-dependent?
- Does posterior-mean decoding give a better signal than the noisy latent at the same step? (We have evidence yes — see analysis runs on `dev`.)
- Does SDE sampling yield more diverse failures, making search more useful?
- Does Trial 1 (single-frame anchor injection) actually fire often on Wan 2.2, or is the failure mode different from CogVideoX?
- At what N does BoN saturate? Is there a prompt-dependent sweet spot?

All of these become trivially answerable A/B tests once the backbone exists.

---

## 11. What we get when this lands

- **One command** runs either strategy on any prompt: `python -m ttsd.runners.pipeline.run_pipeline --config <yaml> --prompt "..."`.
- **One JSON-lines file per run** records every verifier call, every decision, every action, every timing.
- **One headline plot** for the project: **EFD&I-replication vs. BoN-N on the Pareto front of (VBench score, wall-clock)** across our 30-prompt set. This is the v1 deliverable; it's the figure the eventual writeup hangs on.
- **A clean extension story** for whoever picks up the project next:
  - Wan 2.2 → Wan 2.5 → CogVideoX-7B → HunyuanVideo: add a `ModelAdapter`.
  - New verifier (MLLM, ViCLIP, learned probe): register it; usable in every strategy.
  - New search algorithm (beam with window rollback, MCTS, RL): add a `SearchStrategy` + maybe one or two new actions; gets all existing verifiers + budget + logging for free.

---

## 12. Sign-off — RESOLVED 2026-05-23

Top-level architecture (3 layers + cross-cutting) **approved**.
The `Action` / `DecisionPolicy` / `SearchStrategy` split **approved**.
Implementation order **delegated to Claude**: P1 → P2 → P3 → P4 → P5 as listed in §8. Deployment / operational decisions delegated to Claude.

Original sign-off questions, for the record:

Please react to:

1. **Top-level architecture** (3 layers + cross-cutting). Right shape?
2. **The `Action`/`DecisionPolicy`/`SearchStrategy` split** — does it make sense for v1, given we have only 2 strategies (EFD&I, BoN)? Or do you want to fold `Action` into `SearchStrategy` and accept the cost of refactoring later when beam/RL arrives?
3. **The 6 design choices in §9** — which way do you lean on each?
4. **Phase order in §8** — start with P1 (skeleton) as proposed, or jump elsewhere?

Once we converge, I'll implement P1 + P2 in one PR.