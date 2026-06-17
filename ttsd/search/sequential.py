"""SequentialTrialSearch — EFD&I-style outer loop.

Runs a list of trials in order. For each trial:
  1. Run its `setup_actions` (e.g. anchor injection, prompt refinement).
     These actions can populate per-trial *overrides* (prompt / seed /
     initial_latent) via ActionResult.metadata["overrides"], and can also
     signal "skip" to bail before generation.
  2. Build a GenerationRequest from the original request + the overrides.
  3. Drive the model adapter; per step, ask the verifier (if the policy
     decides at this step) and dispatch the policy's chosen action.
  4. If the trial completed without abort → ACCEPT the result, save the
     video, return immediately (the trial sequence is escalation-on-failure,
     not best-of-all-trials).
  5. If the trial aborted → escalate to the next trial.
After all trials, return success=False with a list of failed trials.

This is faithful to EFD&I's tiered intervention: only the failed samples
incur the cost of escalations.
"""

from __future__ import annotations

import dataclasses
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
    name: str
    setup_actions: list[ActionSpec]


def _resolve_initial_latent(
    overrides: dict[str, Any],
    target_shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor | None:
    """Build the initial latent for a trial from setup-action overrides.

    Supports two patterns:
      - overrides['initial_latent']: a fully-formed tensor of `target_shape`.
        Used directly.
      - overrides['_anchor_broadcast_latent'] (+ '_anchor_blend_alpha'):
        a (1, C, 1, H, W) broadcast anchor latent. Tiled along the temporal
        axis to match target_shape and blended with fresh seeded noise:
            init = (1 - alpha) * fresh_noise + alpha * tiled_anchor
        This is the EFD&I-style Trial 1 mechanism.
    Returns None when no override touches the initial latent.
    """
    direct = overrides.get("initial_latent")
    if direct is not None:
        return direct.to(device, dtype=dtype)

    anchor = overrides.get("_anchor_broadcast_latent")
    if anchor is None:
        return None

    alpha = float(overrides.get("_anchor_blend_alpha", 0.5))
    # Tile anchor (1, C, 1, H, W) → target_shape's temporal axis.
    if anchor.shape[2] != 1:
        anchor = anchor.mean(dim=2, keepdim=True)
    tiled = anchor.expand(target_shape).contiguous()
    fresh_noise = torch.randn(
        target_shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    )
    init = (1.0 - alpha) * fresh_noise + alpha * tiled.to(torch.float32).cpu()
    return init.to(device, dtype=dtype)


@register_strategy("sequential_trial")
class SequentialTrialSearch(SearchStrategy):
    """Run trials in order; first trial that completes without abort wins.

    `trials` is a list of dicts (parsed from YAML), each with:
        name: str
        setup_actions: list[{kind: str, params: {...}}]

    Default trials list = [{name: "base", setup_actions: []}] — bare
    baseline generation, equivalent to running the model directly.
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
        cumulative_cost = CostEstimate()
        failed_trials: list[dict[str, Any]] = []

        for trial_idx, trial in enumerate(self.trials):
            ctx.logger.log("trial_start", trial=trial_idx, name=trial.name)

            traj = TrajectoryState(
                prompt=request.prompt,
                seed=request.seed + trial_idx,    # deterministic per-trial seed
                trial_index=trial_idx,
                metadata={"height": request.height, "width": request.width},
            )

            # Run setup actions; collect per-trial overrides as we go.
            overrides: dict[str, Any] = {}
            skip_trial = False
            for spec in trial.setup_actions:
                action = build_action(spec)
                apply_ctx = ApplyContext(
                    adapter=ctx.adapter,
                    budget=ctx.budget,
                    logger=ctx.logger,
                    prompt=overrides.get("prompt", request.prompt),
                    seed=overrides.get("seed", traj.seed),
                )
                result = action.apply(traj, apply_ctx)
                cumulative_cost = cumulative_cost + result.cost_spent
                ctx.logger.log(
                    "setup_action", trial=trial_idx, kind=spec.kind,
                    status=result.status, reason=result.reason,
                )
                if result.status == "skip":
                    skip_trial = True
                    break
                if result.status == "abort":
                    ctx.logger.log("trial_aborted_in_setup", trial=trial_idx)
                    skip_trial = True
                    break
                overrides.update(result.metadata.get("overrides", {}))

            if skip_trial:
                failed_trials.append({"trial": trial_idx, "name": trial.name, "reason": "setup_skipped"})
                continue

            # Build the per-trial generation request from overrides.
            trial_prompt = overrides.get("prompt", request.prompt)
            seed_offset = overrides.get("seed_offset", 0)
            trial_seed = overrides.get("seed", traj.seed + seed_offset)

            # Resolve initial_latent (handles anchor-broadcast or direct latent).
            # Latent shape (1, C, F_lat, H_lat, W_lat) computed from generation request.
            # Wan VAE: H_lat = H/16, W_lat = W/16, F_lat = (num_frames - 1) // 4 + 1, C = 48.
            target_h_lat = request.height // 16
            target_w_lat = request.width // 16
            target_f_lat = (request.num_frames - 1) // 4 + 1
            target_shape = (1, 48, target_f_lat, target_h_lat, target_w_lat)
            initial_latent = _resolve_initial_latent(
                overrides,
                target_shape,
                device=getattr(ctx.adapter, "_impl", ctx.adapter).device,
                dtype=getattr(getattr(ctx.adapter, "_impl", ctx.adapter), "dtype", torch.bfloat16),
                seed=trial_seed,
            )

            trial_request = dataclasses.replace(
                request,
                prompt=trial_prompt,
                seed=trial_seed,
                initial_latent=initial_latent,
            )

            # Per-step handler.
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
                    setter = getattr(ctx.verifier, "set_posterior_mean", None)
                    if setter is not None:
                        setter(step_state.posterior_mean)
                    verifier_out = ctx.verifier.score(
                        latent=step_state.latent,
                        prompt=trial_prompt,
                        step=step_state.step,
                        total_steps=step_state.total_steps,
                    )
                    if setter is not None:
                        setter(None)
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
                        budget=ctx.budget, prompt=trial_prompt,
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
                        prompt=trial_prompt, seed=trial_seed,
                    ),
                )
                if _result.status == "abort":
                    return StepDirective(abort=True)
                if _result.new_latent is not None:
                    return StepDirective(replace_latent=_result.new_latent)
                return None

            capture = getattr(ctx.verifier, "REQUIRES", set()) or set()
            t0 = time.monotonic()
            gen_out = ctx.adapter.generate(trial_request, on_step=_on_step, capture=capture)
            gen_wall_s = time.monotonic() - t0
            cumulative_cost = cumulative_cost + CostEstimate(
                wall_clock_s=gen_wall_s, gpu_seconds=gen_wall_s,
            )
            ctx.budget.record(CostEstimate(gpu_seconds=gen_wall_s))
            ctx.logger.log(
                "trial_end", trial=trial_idx, name=trial.name,
                prompt=trial_prompt,
                seed=trial_seed,
                used_initial_latent=initial_latent is not None,
                aborted=gen_out.aborted, abort_step=gen_out.abort_at_step,
                wall_s=round(gen_wall_s, 3),
                n_actions=len(actions_taken), actions=actions_taken,
            )

            if gen_out.aborted:
                failed_trials.append({
                    "trial": trial_idx, "name": trial.name,
                    "reason": "policy_abort", "abort_step": gen_out.abort_at_step,
                })
                continue

            # Trial completed without abort → ACCEPT.
            final_path = ctx.output_dir / "video.mp4"
            _save_video(gen_out.frames, final_path)
            # Also save per-trial videos for diagnostics in escalations.
            if trial_idx > 0:
                _save_video(gen_out.frames, ctx.output_dir / f"trial_{trial_idx:02d}_video.mp4")
            final_score = next(
                (s for s in reversed(verifier_scores) if s is not None), None
            )
            return RunResult(
                success=True,
                video_path=final_path,
                final_score=final_score,
                events_path=ctx.output_dir / "events.jsonl",
                cost=cumulative_cost,
                n_trials=trial_idx + 1,
                terminating_trial=trial_idx,
                metadata={"accepted_trial": trial.name, "failed_trials": failed_trials},
            )

        # All trials failed.
        return RunResult(
            success=False,
            n_trials=len(self.trials),
            cost=cumulative_cost,
            metadata={"failed_trials": failed_trials},
        )
