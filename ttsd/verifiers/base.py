from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass
class VerifierOutput:
    score: dict[str, float]
    uncertainty: dict[str, float] | None = None
    # Headline scalar the orchestrator's DecisionPolicy reads. Optional so
    # existing offline verifiers stay backward-compatible.
    final_score_estimate: float | None = None


class Verifier(ABC):
    """Predicts final-video quality from intermediate denoising state.

    Subclasses operate at one of three loci:
      - pixel preview (decode latent → score with ViCLIP / aesthetic)
      - latent MLLM head (operate on hidden states, no decode)
      - lightweight probe (PCA + MLP on intermediate features)
    """

    # What state the verifier wants on each call. The pipeline orchestrator
    # uses this to skip expensive captures (VAE decode, DINOv2 forward) when
    # no registered verifier needs them. Default = empty = "just the noisy
    # latent". Concrete verifiers override.
    REQUIRES: ClassVar[set[str]] = set()

    @abstractmethod
    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> VerifierOutput: ...
