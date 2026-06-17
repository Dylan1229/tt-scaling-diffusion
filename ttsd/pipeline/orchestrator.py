"""Orchestrator — the thin conductor that owns one user request's lifecycle.

Loads plugins via the registry, hands them to the SearchStrategy, runs the
strategy, returns a RunResult. Owns no domain logic itself.

This is what users call. The CLI runner is a tiny wrapper around this.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from ttsd.pipeline.budget import Budget
from ttsd.pipeline.config import PipelineConfig
from ttsd.pipeline.core import RunResult
from ttsd.pipeline.logger import JsonlLogger
from ttsd.pipeline.model_adapter import GenerationRequest
from ttsd.pipeline.policy import build_policy
from ttsd.pipeline.registry import MODELS, STRATEGIES, VERIFIERS
from ttsd.search.base import RunContext


class Orchestrator:
    """Owns config → instantiate plugins → drive strategy."""

    def __init__(self, config: PipelineConfig):
        # Importing the registration-side modules registers their plugins.
        # Done here at construction so config kinds can be looked up below.
        import ttsd.pipeline.actions    # noqa: F401  registers continue/noop/stop_and_fail/stop_and_accept
        import ttsd.pipeline.policy     # noqa: F401  registers noop, fixed_threshold
        import ttsd.search.sequential   # noqa: F401  registers sequential_trial
        import ttsd.verifiers.noop      # noqa: F401  registers noop verifier
        import ttsd.verifiers.dino.online_adapter   # noqa: F401  registers dino_frame_cos_mean_online

        self.config = config
        self.adapter = MODELS.get(config.model.kind)(**config.model.params)
        self.verifier = VERIFIERS.get(config.verifier.kind)(**config.verifier.params)
        self.policy = build_policy(config.policy)
        self.strategy = STRATEGIES.get(config.strategy.kind)(**config.strategy.params)
        self.budget = Budget.from_config(config.budget)

    def run(self, prompt: str, seed: int, run_id: str | None = None) -> RunResult:
        """Drive one user request to completion."""
        run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.config.output_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the resolved config for reproducibility.
        cfg_dump = asdict(self.config)
        (out_dir / "config.snapshot.json").write_text(json.dumps(cfg_dump, indent=2))

        log_path = Path(self.config.log.path.format(run_id=run_id))
        if not log_path.is_absolute():
            log_path = out_dir / log_path.name

        with JsonlLogger(log_path) as logger:
            logger.log(
                "run_start", run_id=run_id, prompt=prompt, seed=seed,
                model_kind=self.config.model.kind,
                verifier_kind=self.config.verifier.kind,
                policy_kind=self.config.policy.kind,
                strategy_kind=self.config.strategy.kind,
            )
            self.budget.start_wall_clock()

            request = GenerationRequest(
                prompt=prompt,
                seed=seed,
                num_frames=self.config.generation.num_frames,
                height=self.config.generation.height,
                width=self.config.generation.width,
                num_inference_steps=self.config.generation.num_inference_steps,
                guidance_scale=self.config.generation.guidance_scale,
            )
            ctx = RunContext(
                adapter=self.adapter,
                verifier=self.verifier,
                policy=self.policy,
                budget=self.budget,
                logger=logger,
                output_dir=out_dir,
                run_id=run_id,
            )

            try:
                result = self.strategy.run(request, ctx)
            except Exception as exc:    # pragma: no cover - safety log
                logger.log("run_error", error=repr(exc))
                raise

            logger.log(
                "run_end", run_id=run_id, success=result.success,
                final_score=result.final_score, n_trials=result.n_trials,
                terminating_trial=result.terminating_trial,
                budget=self.budget.summary(),
            )

        # Persist a tiny manifest so a downstream tool doesn't have to parse JSONL.
        (out_dir / "result.json").write_text(json.dumps({
            "success": result.success,
            "video_path": str(result.video_path) if result.video_path else None,
            "events_path": str(result.events_path) if result.events_path else None,
            "final_score": result.final_score,
            "n_trials": result.n_trials,
            "terminating_trial": result.terminating_trial,
            "budget": self.budget.summary(),
            "metadata": result.metadata,
        }, indent=2))

        return result
