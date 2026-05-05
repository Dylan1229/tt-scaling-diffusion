from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from ttsd.verifiers.base import Verifier, VerifierOutput


class Decision(Enum):
    CONTINUE = "continue"
    EARLY_STOP_FAIL = "early_stop_fail"
    EARLY_STOP_ACCEPT = "early_stop_accept"
    BRANCH = "branch"
    REPLACE_NOISE = "replace_noise"


@dataclass
class StepContext:
    latent: torch.Tensor
    step: int
    total_steps: int
    prompt: str
    history: list[VerifierOutput]


class SearchPolicy(ABC):
    """Decides what to do at each verifier checkpoint during sampling."""

    def __init__(self, verifier: Verifier):
        self.verifier = verifier

    @abstractmethod
    def step(self, ctx: StepContext) -> Decision: ...
