"""Shared base helpers for DINO-based concrete verifiers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ttsd.verifiers.base import Verifier

DEFAULT_STEP014_INDEX = 3
DEFAULT_UPTO014_INDICES = (0, 1, 2, 3)
DEFAULT_QUANTILES = (0.05, 0.50, 0.95)

ClsProvider = Callable[[torch.Tensor, str, int, int], np.ndarray]


class ProbeBackedDinoVerifier(Verifier):
    def __init__(
        self,
        *,
        cls_provider: ClsProvider,
        score_name: str,
    ) -> None:
        self.cls_provider = cls_provider
        self.score_name = score_name

    def _load_cls_features(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> np.ndarray:
        cls_features = self.cls_provider(latent, prompt, step, total_steps)
        return np.asarray(cls_features)
