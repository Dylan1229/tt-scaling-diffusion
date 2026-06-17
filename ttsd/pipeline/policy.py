"""DecisionPolicy ABC + the v1 policies.

A DecisionPolicy is a pure function of `(VerifierOutput, DecisionContext) →
ActionSpec | None`. None means "no action — just continue". Returning an
ActionSpec means "fire this intervention now".

Policies are stateless: any state they need (sliding window history, etc.)
is read from `ctx.state.score_history` etc. — never stored on the policy
object. This makes them trivial to unit-test and lets the orchestrator
replay decisions deterministically from the log.

P1 ships only NoOpPolicy. Real policies land in P2-P4:
  - FixedThresholdPolicy            (P2)
  - DynamicSlidingWindowPolicy      (P3)
  - BestOfNPolicy                   (P4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ttsd.pipeline.core import ActionSpec, TrajectoryState
from ttsd.pipeline.registry import POLICIES, register_policy

if TYPE_CHECKING:
    from ttsd.pipeline.budget import Budget
    from ttsd.verifiers.base import VerifierOutput


@dataclass
class DecisionContext:
    state: TrajectoryState
    step: int
    total_steps: int
    budget: "Budget"
    prompt: str


class DecisionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        verifier_out: "VerifierOutput | None",
        ctx: DecisionContext,
    ) -> ActionSpec | None: ...

    @property
    def decide_at_steps(self) -> set[int] | None:
        """Steps at which the orchestrator should bother running the verifier.
        None = run at every step. Empty set = never run (degenerate). Letting
        the policy declare this avoids wasted verifier work in long
        trajectories where we only care about a handful of checkpoints."""
        return None


def build_policy(cfg) -> DecisionPolicy:
    cls = POLICIES.get(cfg.kind)
    return cls(**cfg.params)


@register_policy("noop")
class NoOpPolicy(DecisionPolicy):
    """Never returns an ActionSpec. The pipeline behaves like the bare
    baseline generation. Used in P1 smoke tests to prove the orchestrator
    preserves baseline behavior."""

    def decide(self, verifier_out, ctx):
        return None

    @property
    def decide_at_steps(self) -> set[int]:
        return set()    # never bother the verifier
