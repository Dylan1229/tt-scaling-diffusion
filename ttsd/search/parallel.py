"""ParallelCandidateSearch — naive best-of-N strategy.

Generates `n_candidates` independent trajectories with different seeds,
scores each at one verifier checkpoint, returns the highest-scoring one.
The simplest meaningful test-time scaling baseline; what every TTS-for-video
paper publishes as the comparison point.

On a single GPU the candidates run sequentially (orchestrator-level
concurrency is a Phase-6 nice-to-have, not v1 — see docs/e2e_framework_plan.md
§9 decision 4). Multi-GPU candidate parallelism is the obvious extension
once we have a per-shard launcher (P5).

This strategy bypasses the DecisionPolicy for mid-flight decisions — it
makes its own verifier calls at the configured checkpoint. The policy slot
in the config is purely advisory; `BestOfNPolicy` is the canonical choice
because it just declares the checkpoint step.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from ttsd.pipeline.core import CostEstimate, RunResult, StepDirective, StepState
from ttsd.pipeline.registry import register_strategy
from ttsd.search.base import RunContext, SearchStrategy
from ttsd.search.sequential import _save_video


@register_strategy("parallel_candidates")
class ParallelCandidateSearch(SearchStrategy):
    """Generate N candidates, score each, return the best.

    Params:
        n_candidates: number of independent trajectories to roll out.
        score_at_step: 0-indexed denoising step at which to call the verifier.
                       Default = last step (num_inference_steps - 1) — the
                       x0_hat there is essentially the final video.
        seed_step: seed stride between candidates. seed_i = request.seed + i*seed_step.
                   Use 1 for consecutive seeds, larger values to space them out.
    """

    def __init__(
        self,
        n_candidates: int = 4,
        score_at_step: int | None = None,
        seed_step: int = 1,
    ):
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
        self.n_candidates = int(n_candidates)
        self.score_at_step = score_at_step
        self.seed_step = int(seed_step)

    def run(self, request, ctx: RunContext) -> RunResult:
        target_step = (
            self.score_at_step
            if self.score_at_step is not None
            else request.num_inference_steps - 1
        )
        score_at: set[int] = {target_step}
        capture = getattr(ctx.verifier, "REQUIRES", set()) or set()

        candidates: list[dict[str, Any]] = []
        cumulative_cost = CostEstimate()

        for i in range(self.n_candidates):
            seed = request.seed + i * self.seed_step
            cand_request = dataclasses.replace(request, seed=seed)
            captured_score: list[float | None] = [None]
            captured_breakdown: list[dict[str, float] | None] = [None]

            def _on_step(state: StepState, _cap=captured_score, _bd=captured_breakdown,
                         _ci=i, _seed=seed) -> StepDirective | None:
                if state.step in score_at and state.latent is not None:
                    setter = getattr(ctx.verifier, "set_posterior_mean", None)
                    if setter is not None:
                        setter(state.posterior_mean)
                    vout = ctx.verifier.score(
                        latent=state.latent,
                        prompt=request.prompt,
                        step=state.step,
                        total_steps=state.total_steps,
                    )
                    if setter is not None:
                        setter(None)
                    _cap[0] = vout.final_score_estimate
                    _bd[0] = vout.score
                    ctx.logger.log(
                        "verifier_call", candidate=_ci, seed=_seed, step=state.step,
                        score=vout.final_score_estimate, score_breakdown=vout.score,
                    )
                ctx.logger.log("step", candidate=_ci, step=state.step, total=state.total_steps)
                return None

            ctx.logger.log("candidate_start", candidate=i, seed=seed)
            t0 = time.monotonic()
            gen_out = ctx.adapter.generate(cand_request, on_step=_on_step, capture=capture)
            cand_wall = time.monotonic() - t0
            cumulative_cost = cumulative_cost + CostEstimate(
                wall_clock_s=cand_wall, gpu_seconds=cand_wall,
            )
            ctx.budget.record(CostEstimate(gpu_seconds=cand_wall))

            score = captured_score[0]
            ctx.logger.log(
                "candidate_end", candidate=i, seed=seed,
                score=score, score_breakdown=captured_breakdown[0],
                wall_s=round(cand_wall, 3),
                aborted=gen_out.aborted,
            )
            candidates.append({
                "candidate_idx": i,
                "seed": seed,
                "score": score,
                "frames": gen_out.frames if not gen_out.aborted else None,
                "wall_s": cand_wall,
            })

        # Rank candidates. Drop None scores and NaN; if no valid candidates
        # remain, return success=False.
        def _valid(c) -> bool:
            s = c["score"]
            return s is not None and s == s and c["frames"] is not None    # NaN-safe

        valid = [c for c in candidates if _valid(c)]
        if not valid:
            ctx.logger.log("no_valid_candidate", n_total=len(candidates))
            return RunResult(
                success=False,
                n_trials=self.n_candidates,
                cost=cumulative_cost,
                metadata={"all_candidates": [
                    {"candidate_idx": c["candidate_idx"], "seed": c["seed"], "score": c["score"]}
                    for c in candidates
                ]},
            )

        best = max(valid, key=lambda c: c["score"])
        final_path = ctx.output_dir / "video.mp4"
        _save_video(best["frames"], final_path)
        # Also save per-candidate videos for diagnostics.
        for c in candidates:
            if c["frames"] is None:
                continue
            cand_path = ctx.output_dir / f"candidate_{c['candidate_idx']:02d}_seed{c['seed']:04d}.mp4"
            _save_video(c["frames"], cand_path)

        ctx.logger.log(
            "best_selected", candidate=best["candidate_idx"], seed=best["seed"],
            score=best["score"], n_candidates=len(candidates),
        )

        return RunResult(
            success=True,
            video_path=final_path,
            final_score=best["score"],
            events_path=ctx.output_dir / "events.jsonl",
            cost=cumulative_cost,
            n_trials=self.n_candidates,
            terminating_trial=best["candidate_idx"],
            metadata={
                "best_seed": best["seed"],
                "best_candidate_idx": best["candidate_idx"],
                "all_candidates": [
                    {"candidate_idx": c["candidate_idx"], "seed": c["seed"], "score": c["score"]}
                    for c in candidates
                ],
            },
        )
