"""Shared dataclasses passed between pipeline components.

These are the value types the orchestrator, policies, actions, and strategies
all agree on. Defined here once so there's no circular import between the
component packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from ttsd.verifiers.base import VerifierOutput


@dataclass(frozen=True)
class CostEstimate:
    """Predicted or measured cost of an Action / generation step."""
    wall_clock_s: float = 0.0
    gpu_seconds: float = 0.0
    vlm_tokens: int = 0

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            vlm_tokens=self.vlm_tokens + other.vlm_tokens,
        )


@dataclass
class StepState:
    """Snapshot of one denoising step. Built by the ModelAdapter callback,
    passed to the orchestrator's per-step handler."""
    step: int                                  # 0-indexed step in the schedule
    total_steps: int
    timestep: Any                              # scheduler timestep (tensor or float)
    latent: "torch.Tensor | None" = None       # current noisy x_t
    posterior_mean: "torch.Tensor | None" = None  # x0_hat (lazy)
    decoded_preview: "torch.Tensor | None" = None  # VAE-decoded preview (lazier)


@dataclass
class StepDirective:
    """What the orchestrator wants the ModelAdapter to do after a step.
    Returned by the on_step callback. None / default = continue unchanged."""
    replace_latent: "torch.Tensor | None" = None   # swap callback_kwargs["latents"]
    abort: bool = False                            # raise AbortTrajectory


class AbortTrajectory(Exception):
    """Raised to signal that the current generation should stop NOW.

    Caught by the ModelAdapter's generate() call site. The strategy then
    decides whether to start a new trajectory (Trial 1 / Trial 2 / new BoN
    candidate) or return failure to the caller."""


@dataclass
class TrajectoryState:
    """Per-rollout state that persists across steps and across trials.

    One TrajectoryState per (prompt, seed, trial) combination. Strategies
    may fork it (e.g. BoN spawns N children sharing a parent)."""
    prompt: str
    seed: int
    trial_index: int = 0
    score_history: list["VerifierOutput"] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    accumulated_cost: CostEstimate = field(default_factory=CostEstimate)
    parent: "TrajectoryState | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec:
    """Lightweight, declarative handle to an Action.

    Built by a DecisionPolicy from config; instantiated to a concrete Action
    only at apply time. Keeping it as data (not as a class instance) makes
    policy decisions cheap to log and replay."""
    kind: str                                       # registry key
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """What an Action returns after applying."""
    status: str                                     # "continue" / "abort" / "restart" / "accept"
    new_state: TrajectoryState | None = None
    new_latent: "torch.Tensor | None" = None
    cost_spent: CostEstimate = field(default_factory=CostEstimate)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """What the orchestrator returns to the CLI / library caller."""
    success: bool
    video_path: Path | None = None
    final_score: float | None = None
    events_path: Path | None = None
    cost: CostEstimate = field(default_factory=CostEstimate)
    n_trials: int = 0
    terminating_trial: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)
