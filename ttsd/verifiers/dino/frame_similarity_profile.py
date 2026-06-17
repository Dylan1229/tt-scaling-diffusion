"""DINO CLS cross-frame similarity-profile verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ttsd.features.dino_cls import upto_gap_cosine_profiles
from ttsd.verifiers.base import VerifierOutput
from ttsd.verifiers.dino.artifacts import RbfSvrArtifact
from ttsd.verifiers.dino.base import (
    DEFAULT_UPTO014_INDICES,
    ClsProvider,
    ProbeBackedDinoVerifier,
)


@dataclass(frozen=True)
class DinoFrameSimilarityProfileProbe:
    artifact: RbfSvrArtifact
    step_indices: tuple[int, ...] = DEFAULT_UPTO014_INDICES
    gap: int = 1

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        step_indices: tuple[int, ...] = DEFAULT_UPTO014_INDICES,
        gap: int = 1,
    ) -> "DinoFrameSimilarityProfileProbe":
        return cls(RbfSvrArtifact.from_npz(path), step_indices=step_indices, gap=gap)

    def feature_vector(self, cls_features: np.ndarray) -> np.ndarray:
        return upto_gap_cosine_profiles(cls_features, self.step_indices, gap=self.gap)

    def score_cls_features(self, cls_features: np.ndarray) -> float:
        return float(self.artifact.predict(self.feature_vector(cls_features))[0])


class DinoFrameSimilarityProfileVerifier(ProbeBackedDinoVerifier):
    def __init__(
        self,
        probe: DinoFrameSimilarityProfileProbe,
        *,
        cls_provider: ClsProvider,
        score_name: str = "dino_frame_similarity_profile",
    ) -> None:
        super().__init__(cls_provider=cls_provider, score_name=score_name)
        self.probe = probe

    @classmethod
    def from_npz(
        cls,
        artifact_path: str | Path,
        *,
        cls_provider: ClsProvider,
        step_indices: tuple[int, ...] = DEFAULT_UPTO014_INDICES,
        gap: int = 1,
    ) -> "DinoFrameSimilarityProfileVerifier":
        probe = DinoFrameSimilarityProfileProbe.from_npz(artifact_path, step_indices=step_indices, gap=gap)
        return cls(probe, cls_provider=cls_provider)

    def score(self, latent: torch.Tensor, prompt: str, step: int, total_steps: int) -> VerifierOutput:
        cls_features = self._load_cls_features(latent, prompt, step, total_steps)
        return VerifierOutput(score={self.score_name: self.probe.score_cls_features(cls_features)})
