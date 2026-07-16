from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from ttsd.verifiers.base import VerifierOutput


class InterventionAction(Enum):
    CONTINUE = "continue"
    BRANCH_SELECT = "branch_select"
    RENOISE = "renoise"


@dataclass
class InterventionContext:
    """State available when a verifier-triggered intervention runs."""

    latent: torch.Tensor
    prompt: str
    step: int
    total_steps: int
    verifier_output: VerifierOutput | None = None
    history: list[VerifierOutput] = field(default_factory=list)
    timestep: int | float | torch.Tensor | None = None
    sigma: float | torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionCandidate:
    latent: torch.Tensor
    preview_latent: torch.Tensor
    verifier_output: VerifierOutput
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionResult:
    latent: torch.Tensor
    action: InterventionAction
    selected_index: int | None = None
    candidates: list[InterventionCandidate] = field(default_factory=list)
    extra_nfe: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def scalar_score(output: VerifierOutput, key: str | None = None) -> float:
    """Extract one optimization scalar from a verifier output."""

    if key is not None:
        if key not in output.score:
            raise KeyError(f"score key {key!r} not found in verifier output")
        return float(output.score[key])

    for fallback in ("reward", "score", "overall", "overall_consistency"):
        if fallback in output.score:
            return float(output.score[fallback])

    if len(output.score) == 1:
        return float(next(iter(output.score.values())))

    raise ValueError("VerifierOutput has multiple score keys; pass score_key explicitly")
