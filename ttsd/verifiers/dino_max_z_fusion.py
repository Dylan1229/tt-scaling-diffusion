"""Combined DINO verifier using max-z late fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ttsd.verifiers.base import VerifierOutput
from ttsd.verifiers.dino_frame_similarity_profile import DinoFrameSimilarityProfileProbe
from ttsd.verifiers.dino_quantile_pca_ridge import DinoTemporalQuantilePCARidgeProbe
from ttsd.verifiers.dino_score_artifacts import ScoreZFusionArtifact
from ttsd.verifiers.dino_verifier_base import ClsProvider, ProbeBackedDinoVerifier


@dataclass(frozen=True)
class CombinedDinoProbe:
    quantile_probe: DinoTemporalQuantilePCARidgeProbe
    profile_probe: DinoFrameSimilarityProfileProbe
    fusion_artifact: ScoreZFusionArtifact

    def base_scores(self, cls_features: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self.quantile_probe.score_cls_features(cls_features),
                self.profile_probe.score_cls_features(cls_features),
            ],
            dtype=np.float64,
        )

    def score_cls_features(self, cls_features: np.ndarray) -> float:
        return float(self.fusion_artifact.predict(self.base_scores(cls_features))[0])

    def score_details(self, cls_features: np.ndarray) -> dict[str, float]:
        base = self.base_scores(cls_features)
        fused = float(self.fusion_artifact.predict(base)[0])
        return {
            "combined_dino_max_z": fused,
            "dino_temporal_quantile_pca_ridge": float(base[0]),
            "dino_frame_similarity_profile": float(base[1]),
        }


class CombinedDinoVerifier(ProbeBackedDinoVerifier):
    def __init__(
        self,
        probe: CombinedDinoProbe,
        *,
        cls_provider: ClsProvider,
        score_name: str = "combined_dino_max_z",
        include_base_scores: bool = True,
    ) -> None:
        super().__init__(cls_provider=cls_provider, score_name=score_name)
        self.probe = probe
        self.include_base_scores = include_base_scores

    def score(self, latent: torch.Tensor, prompt: str, step: int, total_steps: int) -> VerifierOutput:
        cls_features = self._load_cls_features(latent, prompt, step, total_steps)
        if self.include_base_scores:
            details = self.probe.score_details(cls_features)
            if self.score_name != "combined_dino_max_z":
                details[self.score_name] = details["combined_dino_max_z"]
            return VerifierOutput(score=details)
        return VerifierOutput(score={self.score_name: self.probe.score_cls_features(cls_features)})
