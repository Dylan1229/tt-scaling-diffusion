"""Wall-clock + GPU-seconds + VLM-token budget tracking.

Lets Actions and Policies ask "can I afford this?" before doing expensive
things. None for a limit = unlimited. Designed to implement EFD&I's
"worst-case ≤ 56% of base regeneration cost" guarantee in three lines:

    if not budget.can_afford(action.estimated_cost):
        return StopAndFail
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ttsd.pipeline.core import CostEstimate


@dataclass
class BudgetLimits:
    wall_clock_s: float | None = None
    gpu_seconds: float | None = None
    vlm_tokens: int | None = None


class Budget:
    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self.spent = CostEstimate()
        self._wall_start: float | None = None

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(
            limits=BudgetLimits(
                wall_clock_s=cfg.wall_clock_s,
                gpu_seconds=cfg.gpu_seconds,
                vlm_tokens=cfg.vlm_tokens,
            )
        )

    def start_wall_clock(self) -> None:
        """Begin measuring wall-clock from now."""
        self._wall_start = time.monotonic()

    def _wall_elapsed(self) -> float:
        if self._wall_start is None:
            return 0.0
        return time.monotonic() - self._wall_start

    def record(self, cost: CostEstimate) -> None:
        """Add cost to the spent total. Wall-clock is read live from the
        monotonic clock, but explicit `cost.wall_clock_s` is also added so
        callers can attribute cost to specific actions for the log."""
        self.spent = self.spent + cost

    def remaining_wall_s(self) -> float:
        if self.limits.wall_clock_s is None:
            return float("inf")
        return max(0.0, self.limits.wall_clock_s - self._wall_elapsed())

    def remaining_gpu_s(self) -> float:
        if self.limits.gpu_seconds is None:
            return float("inf")
        return max(0.0, self.limits.gpu_seconds - self.spent.gpu_seconds)

    def remaining_vlm_tokens(self) -> int:
        if self.limits.vlm_tokens is None:
            return 10**12
        return max(0, self.limits.vlm_tokens - self.spent.vlm_tokens)

    def can_afford(self, cost: CostEstimate) -> bool:
        return (
            cost.wall_clock_s <= self.remaining_wall_s()
            and cost.gpu_seconds <= self.remaining_gpu_s()
            and cost.vlm_tokens <= self.remaining_vlm_tokens()
        )

    def summary(self) -> dict:
        return {
            "spent": {
                "wall_clock_s": round(self._wall_elapsed(), 3),
                "gpu_seconds": self.spent.gpu_seconds,
                "vlm_tokens": self.spent.vlm_tokens,
            },
            "limits": {
                "wall_clock_s": self.limits.wall_clock_s,
                "gpu_seconds": self.limits.gpu_seconds,
                "vlm_tokens": self.limits.vlm_tokens,
            },
        }
