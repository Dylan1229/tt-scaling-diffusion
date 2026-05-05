from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class VerifierOutput:
    score: dict[str, float]
    uncertainty: dict[str, float] | None = None


class Verifier(ABC):
    """Predicts final-video quality from intermediate denoising state.

    Subclasses operate at one of three loci (see ROADMAP §4):
      - pixel preview (decode latent → score with ViCLIP / aesthetic)
      - latent MLLM head (operate on hidden states, no decode)
      - lightweight probe (PCA + MLP on intermediate features)
    """

    @abstractmethod
    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> VerifierOutput: ...
