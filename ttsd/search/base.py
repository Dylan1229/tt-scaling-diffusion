"""SearchStrategy ABC — the outer-loop algorithm.

A SearchStrategy owns:
  - the set of active trajectories (1 for sequential, N for BoN),
  - the trial sequence (e.g. EFD&I's [T0, T1, T2, T2→1]),
  - the dispatch of Action results (e.g. "Trial 1 accepted → return").

The strategy delegates per-step decisions to the DecisionPolicy and per-step
interventions to Actions. It does not itself read the verifier — that's the
orchestrator's job. The strategy receives `RunContext` which gives access to
the adapter, verifier, policy, budget, logger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsd.pipeline.core import RunResult
from ttsd.pipeline.registry import STRATEGIES

if TYPE_CHECKING:
    from ttsd.pipeline.budget import Budget
    from ttsd.pipeline.logger import JsonlLogger
    from ttsd.pipeline.model_adapter import GenerationRequest, ModelAdapter
    from ttsd.pipeline.policy import DecisionPolicy
    from ttsd.verifiers.base import Verifier


@dataclass
class RunContext:
    """Everything a strategy needs to drive one user request."""
    adapter: "ModelAdapter"
    verifier: "Verifier"
    policy: "DecisionPolicy"
    budget: "Budget"
    logger: "JsonlLogger"
    output_dir: object        # pathlib.Path; typed as object to avoid eager import
    run_id: str


class SearchStrategy(ABC):
    """Outer loop. Drives one user request through whatever trial / candidate
    scheme this strategy implements; returns a final RunResult."""

    @abstractmethod
    def run(self, request: "GenerationRequest", ctx: RunContext) -> RunResult: ...


def build_strategy(cfg) -> SearchStrategy:
    cls = STRATEGIES.get(cfg.kind)
    return cls(**cfg.params)
