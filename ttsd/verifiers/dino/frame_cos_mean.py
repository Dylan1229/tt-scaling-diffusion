"""Standalone DINO CLS adjacent-frame cosine-mean verifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ttsd.features.dino_cls import frame_cosine_mean, select_step
from ttsd.verifiers.base import VerifierOutput
from ttsd.verifiers.dino.base import DEFAULT_STEP014_INDEX, ClsProvider, ProbeBackedDinoVerifier


@dataclass(frozen=True)
class DinoFrameCosMeanProbe:
    step_index: int | None = DEFAULT_STEP014_INDEX
    gap: int = 1

    def score_cls_features(self, cls_features: np.ndarray) -> float:
        step_features = select_step(cls_features, self.step_index)
        return frame_cosine_mean(step_features, gap=self.gap)


class DinoFrameCosMeanVerifier(ProbeBackedDinoVerifier):
    def __init__(
        self,
        *,
        cls_provider: ClsProvider,
        step_index: int | None = DEFAULT_STEP014_INDEX,
        gap: int = 1,
        score_name: str = "dino_cls_frame_cos_mean",
    ) -> None:
        super().__init__(cls_provider=cls_provider, score_name=score_name)
        self.probe = DinoFrameCosMeanProbe(step_index=step_index, gap=gap)

    def score(self, latent: torch.Tensor, prompt: str, step: int, total_steps: int) -> VerifierOutput:
        cls_features = self._load_cls_features(latent, prompt, step, total_steps)
        return VerifierOutput(score={self.score_name: self.probe.score_cls_features(cls_features)})
