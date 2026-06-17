"""SequentialTrialSearch — EFD&I-style outer loop.

Run a list of Trials in order. Each Trial generates one trajectory. After
each Trial the strategy checks acceptance via the most recent verifier
score (compared to the policy's threshold). First accepted trial wins.
Otherwise escalate to the next trial. After all trials, return best.

For P1 the trial list is just `[Continue]` — produces output identical to
the bare baseline generation. P3 will add Trial 1 / Trial 2 escalation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ttsd.pipeline.actions import ApplyContext, build_action
from ttsd.pipeline.core import (
    ActionSpec,
    CostEstimate,
    RunResult,
    StepDirective,
    StepState,
    TrajectoryState,
)
from ttsd.pipeline.policy import DecisionContext
from ttsd.pipeline.registry import register_strategy
from ttsd.search.base import RunContext, SearchStrategy


def _save_video(frames, path: Path, fps: int = 16) -> None:
    """Mirror of ttsd.runners.generate.baseline._save_video so the orchestrator
    doesn't pull the runner module."""
    import imageio.v3 as iio
    import numpy as np

    arr = frames if hasattr(frames, "__array__") else frames.cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255).astype("uint8")
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    iio.imwrite(path, arr, fps=fps, codec="libx264")


@dataclass
class _Trial:
    """One trial in the sequence. v1 only has 'continue' / 'noop' trials;
    P3 adds 'inject', 'refine_prompt'."""
    name: str
    setup_actions: list[ActionSpec]    # actions applied before generation starts


@register_strategy("sequential_trial")
class SequentialTrialSearch(SearchStrategy):
    """Run trials in order until one is accepted or budget is exhausted.

    `trials` is a list of dicts (parsed from YAML), each with:
        name: str                  # human-readable label
        setup_actions: list[ActionSpec-as-dict]   # what to do before this trial's generation

    Default v1 trials list = [{name: "base", setup_actions: []}] — bare
    baseline generation, exactly equivalent to running the model directly.
    """

    def __init__(self, trials: list[dict[str, Any]] | None = None):
        if not trials:
            trials = [{"name": "base", "setup_actions": []}]
        self.trials: list[_Trial] = [
            _Trial(
                name=t.get("name", f"trial_{i}"),
                setup_actions=[ActionSpec(**a) for a in t.get("setup_actions", [])],
            )
            for i, t in enumerate(trials)
        ]

    def run(self, request, ctx: RunContext) -> RunResult:
        best_frames = None
        best_score: float | None = None
        best_trial_idx: int = -1
        terminating: int = -1
        cumulative_cost = CostEstimate()

        for trial_idx, trial in enumerate(self.trials):
            ctx.logger.log("trial_start", trial=trial_idx, name=trial.name)

            traj = TrajectoryState(
                prompt=request.prompt,
                seed=request.seed + trial_idx,    # naive deterministic re-seed per trial
                trial_index=trial_idx,
            )

            # Run setup actions (e.g. anchor injection, prompt refinement — P3).
            for spec in trial.setup_actions:
                action = build_action(spec)
                result = action.apply(
                    traj,
                    ApplyContext(
                        adapter=ctx.adapter,
                        budget=ctx.budget,
                        logger=ctx.logger,
                        prompt=request.prompt,
                        seed=traj.seed,
                    ),
                )
                cumulative_cost = cumulative_cost + result.cost_spent
                ctx.logger.log(
                    "setup_action", trial=trial_idx, kind=spec.kind,
                    status=result.status, reason=result.reason,
                )
                if result.status == "abort":
                    ctx.logger.log("trial_aborted_in_setup", trial=trial_idx)
                    break

            # Per-step handler: ask verifier (if policy wants), ask policy, dispatch action.
            should_run_verifier_at: set[int] | None = ctx.policy.decide_at_steps
            actions_taken: list[str] = []
            verifier_scores: list[float | None] = []

            def _on_step(step_state: StepState) -> StepDirective | None:
                run_verifier = (
                    should_run_verifier_at is None
                    or step_state.step in should_run_verifier_at
                )
                verifier_out = None
                if run_verifier and step_state.latent is not None:
                    # If the verifier needs posterior_mean, push it through a
                    # known attribute so the verifier's score() can read it.
                    # Online verifiers expose `set_posterior_mean`; others ignore.
                    setter = getattr(ctx.verifier, "set_posterior_mean", None)
                    if setter is not None:
                        setter(step_state.posterior_mean)
                    verifier_out = ctx.verifier.score(
                        latent=step_state.latent,
                        prompt=request.prompt,
                        step=step_state.step,
                        total_steps=step_state.total_steps,
                    )
                    if setter is not None:
                        setter(None)    # release reference; next step refreshes
                    traj.score_history.append(verifier_out)
                    verifier_scores.append(verifier_out.final_score_estimate)
                    ctx.logger.log(
                        "verifier_call", trial=trial_idx, step=step_state.step,
                        score=verifier_out.final_score_estimate,
                        score_breakdown=verifier_out.score,
                    )
                ctx.logger.log(
                    "step", trial=trial_idx, step=step_state.step,
                    total=step_state.total_steps,
                )
                spec = ctx.policy.decide(
                    verifier_out,
                    DecisionContext(
                        state=traj, step=step_state.step,
                        total_steps=step_state.total_steps,
                        budget=ctx.budget, prompt=request.prompt,
                    ),
                )
                if spec is None:
                    return None
                actions_taken.append(spec.kind)
                ctx.logger.log(
                    "policy_decision", trial=trial_idx, step=step_state.step,
                    action_kind=spec.kind, params=spec.params,
                )
                action = build_action(spec)
                _result = action.apply(
                    traj,
                    ApplyContext(
                        adapter=ctx.adapter, budget=ctx.budget, logger=ctx.logger,
                        prompt=request.prompt, seed=traj.seed,
                    ),
                )
                if _result.status == "abort":
                    return StepDirective(abort=True)
                if _result.new_latent is not None:
                    return StepDirective(replace_latent=_result.new_latent)
                return None

            # Drive the model. Tell the adapter what to capture (e.g.
            # posterior_mean) based on what the verifier asked for.
            capture = getattr(ctx.verifier, "REQUIRES", set()) or set()
            t0 = time.monotonic()
            gen_out = ctx.adapter.generate(request, on_step=_on_step, capture=capture)
            gen_wall_s = time.monotonic() - t0
            cumulative_cost = cumulative_cost + CostEstimate(
                wall_clock_s=gen_wall_s, gpu_seconds=gen_wall_s,
            )
            ctx.budget.record(CostEstimate(gpu_seconds=gen_wall_s))
            ctx.logger.log(
                "trial_end", trial=trial_idx, name=trial.name,
                aborted=gen_out.aborted, abort_step=gen_out.abort_at_step,
                wall_s=round(gen_wall_s, 3),
                n_actions=len(actions_taken), actions=actions_taken,
            )

            if gen_out.aborted:
                # Strategy: try next trial. If no more trials, will fall
                # through to the final RunResult below with success=False.
                continue

            # Save the video to the run directory under a per-trial subdir.
            trial_video = ctx.output_dir / f"trial_{trial_idx:02d}_video.mp4"
            _save_video(gen_out.frames, trial_video)

            # Headline score: last verifier score we collected (None if policy
            # said decide_at_steps=set(), i.e. NoOp).
            this_score = next(
                (s for s in reversed(verifier_scores) if s is not None), None
            )
            if best_frames is None or (
                this_score is not None and (best_score is None or this_score > best_score)
            ):
                best_frames = gen_out.frames
                best_score = this_score
                best_trial_idx = trial_idx
                terminating = trial_idx

            # Acceptance check: for v1 we have no threshold (NoOp policy), so
            # the FIRST trial is always accepted. P3's DynamicSlidingWindowPolicy
            # will add proper acceptance criteria via policy.decide() during
            # the trial, or a post-trial check we can wire in then.
            if ctx.policy.decide_at_steps == set():
                # NoOp shortcut: accept and return.
                terminating = trial_idx
                break

        if best_frames is None:
            # No trial produced output.
            return RunResult(success=False, n_trials=len(self.trials), cost=cumulative_cost)

        final_path = ctx.output_dir / "video.mp4"
        _save_video(best_frames, final_path)
        return RunResult(
            success=True,
            video_path=final_path,
            final_score=best_score,
            events_path=ctx.output_dir / "events.jsonl",
            cost=cumulative_cost,
            n_trials=len(self.trials),
            terminating_trial=terminating,
            metadata={"best_trial": best_trial_idx},
        )
