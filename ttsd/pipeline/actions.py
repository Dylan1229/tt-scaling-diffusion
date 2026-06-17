"""Action ABC + the v1 action repertoire.

Each Action is one unit of intervention. Built lazily from an ActionSpec
(declarative handle) by the orchestrator. New actions added under
ttsd/pipeline/actions/ in later phases register themselves with
@register_action("name").

P1 ships only `Continue` and `Noop` (synonyms). Real actions land in P2-P4:
  - StopAndFail / StopAndAccept              (P2)
  - SingleFrameAnchorInject (EFD&I Trial 1)  (P3)
  - RefinePromptVLM         (EFD&I Trial 2)  (P3)
  - RolloutBoNCandidates / KeepBestCandidate (P4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ttsd.pipeline.core import ActionResult, CostEstimate, TrajectoryState
from ttsd.pipeline.registry import ACTIONS, register_action


@dataclass
class ApplyContext:
    """Read-only context handed to Action.apply()."""
    adapter: object                                # ModelAdapter; typed as object to avoid circular
    budget: object                                 # Budget
    logger: object                                 # JsonlLogger
    prompt: str
    seed: int


class Action(ABC):
    """Pluggable intervention primitive."""

    @abstractmethod
    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult: ...

    @property
    def estimated_cost(self) -> CostEstimate:
        return CostEstimate()


def build_action(spec) -> Action:
    """Resolve an ActionSpec (kind + params) to a concrete Action instance."""
    cls = ACTIONS.get(spec.kind)
    return cls(**spec.params)


@register_action("continue")
@register_action("noop")
class Continue(Action):
    """No-op: let the trajectory keep going. The default 'do nothing' action.

    A DecisionPolicy can return ActionSpec(kind='continue') explicitly, or
    return None — both mean the same thing to the orchestrator. We support
    both because some policies want to *log* their no-op decisions and others
    want to stay quiet.
    """

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        return ActionResult(status="continue", reason="noop")
