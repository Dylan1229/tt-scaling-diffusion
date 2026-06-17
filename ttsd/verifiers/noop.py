"""NoOpVerifier — always returns a constant score.

Used in P1 smoke tests to prove the orchestrator works end-to-end without
depending on any real verifier model. Also useful as a baseline reference
for cost/time measurements (any real verifier MUST be more expensive).
"""

from __future__ import annotations

from typing import ClassVar

import torch

from ttsd.pipeline.registry import register_verifier
from ttsd.verifiers.base import Verifier, VerifierOutput


@register_verifier("noop")
class NoOpVerifier(Verifier):
    REQUIRES: ClassVar[set[str]] = set()

    def __init__(self, constant_score: float = 1.0):
        self.constant_score = float(constant_score)

    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> VerifierOutput:
        return VerifierOutput(
            score={"noop": self.constant_score},
            final_score_estimate=self.constant_score,
        )


def build_verifier(cfg):
    """Resolve a VerifierConfig to a concrete Verifier."""
    from ttsd.pipeline.registry import VERIFIERS
    cls = VERIFIERS.get(cfg.kind)
    return cls(**cfg.params)
