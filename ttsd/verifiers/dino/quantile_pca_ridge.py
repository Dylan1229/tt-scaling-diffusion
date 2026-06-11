"""DINO temporal-quantile PCA/ridge verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ttsd.features.dino_cls import select_step, temporal_quantile_features
from ttsd.verifiers.base import VerifierOutput
from ttsd.verifiers.dino.artifacts import PCARidgeArtifact
from ttsd.verifiers.dino.base import (
    DEFAULT_QUANTILES,
    DEFAULT_STEP014_INDEX,
    ClsProvider,
    ProbeBackedDinoVerifier,
)


@dataclass(frozen=True)
class DinoTemporalQuantilePCARidgeProbe:
    artifact: PCARidgeArtifact
    step_index: int | None = DEFAULT_STEP014_INDEX
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        step_index: int | None = DEFAULT_STEP014_INDEX,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> "DinoTemporalQuantilePCARidgeProbe":
        return cls(PCARidgeArtifact.from_npz(path), step_index=step_index, quantiles=quantiles)

    def feature_vector(self, cls_features: np.ndarray) -> np.ndarray:
        step_features = select_step(cls_features, self.step_index)
        return temporal_quantile_features(step_features, self.quantiles)

    def score_cls_features(self, cls_features: np.ndarray) -> float:
        return float(self.artifact.predict(self.feature_vector(cls_features))[0])


class DinoTemporalQuantilePCARidgeVerifier(ProbeBackedDinoVerifier):
    def __init__(
        self,
        probe: DinoTemporalQuantilePCARidgeProbe,
        *,
        cls_provider: ClsProvider,
        score_name: str = "dino_temporal_quantile_pca_ridge",
    ) -> None:
        super().__init__(cls_provider=cls_provider, score_name=score_name)
        self.probe = probe

    @classmethod
    def from_npz(
        cls,
        artifact_path: str | Path,
        *,
        cls_provider: ClsProvider,
        step_index: int | None = DEFAULT_STEP014_INDEX,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> "DinoTemporalQuantilePCARidgeVerifier":
        probe = DinoTemporalQuantilePCARidgeProbe.from_npz(
            artifact_path,
            step_index=step_index,
            quantiles=quantiles,
        )
        return cls(probe, cls_provider=cls_provider)

    def score(self, latent: torch.Tensor, prompt: str, step: int, total_steps: int) -> VerifierOutput:
        cls_features = self._load_cls_features(latent, prompt, step, total_steps)
        return VerifierOutput(score={self.score_name: self.probe.score_cls_features(cls_features)})
