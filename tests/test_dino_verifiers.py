from __future__ import annotations

import math

import numpy as np
import torch

from ttsd.features.dino_cls import (
    frame_neighbor_similarity,
    temporal_quantile_features,
    upto_gap_cosine_profiles,
)
from ttsd.verifiers.dino.artifacts import (
    PCARidgeArtifact,
    RbfSvrArtifact,
    ScoreZFusionArtifact,
)
from ttsd.verifiers.dino.frame_cos_mean import DinoFrameCosMeanProbe, DinoFrameCosMeanVerifier
from ttsd.verifiers.dino.frame_similarity_profile import DinoFrameSimilarityProfileProbe
from ttsd.verifiers.dino.max_z_fusion import CombinedDinoProbe
from ttsd.verifiers.dino.quantile_pca_ridge import DinoTemporalQuantilePCARidgeProbe


def test_frame_neighbor_similarity_and_mean_probe() -> None:
    cls_step = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    grid = np.stack([cls_step, cls_step], axis=0)

    profile = frame_neighbor_similarity(cls_step)

    np.testing.assert_allclose(profile, np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert DinoFrameCosMeanProbe(step_index=1).score_cls_features(grid) == 0.5


def test_temporal_quantile_features_concatenates_by_quantile() -> None:
    cls_step = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
        ],
        dtype=np.float32,
    )

    features = temporal_quantile_features(cls_step, quantiles=(0.0, 0.5, 1.0))

    np.testing.assert_allclose(
        features,
        np.asarray([1.0, 10.0, 3.0, 20.0, 5.0, 30.0], dtype=np.float32),
        atol=1e-6,
    )


def test_pca_ridge_probe_scores_quantile_features() -> None:
    cls_step = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
        ],
        dtype=np.float32,
    )
    grid = np.stack([cls_step, cls_step], axis=0)
    artifact = PCARidgeArtifact(
        scaler_mean=np.zeros(6),
        scaler_scale=np.ones(6),
        pca_mean=np.zeros(6),
        pca_components=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        ridge_coef=np.asarray([2.0], dtype=np.float64),
        ridge_intercept=1.0,
    )
    probe = DinoTemporalQuantilePCARidgeProbe(
        artifact=artifact,
        step_index=0,
        quantiles=(0.0, 0.5, 1.0),
    )

    assert probe.score_cls_features(grid) == 3.0


def test_upto_gap_cosine_profiles_concatenates_selected_steps() -> None:
    step_a = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    step_b = np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    step_c = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    grid = np.stack([step_a, step_b, step_c], axis=0)

    features = upto_gap_cosine_profiles(grid, (0, 2), gap=1)

    np.testing.assert_allclose(features, np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_rbf_svr_artifact_predicts_kernel_expansion() -> None:
    artifact = RbfSvrArtifact(
        scaler_mean=np.zeros(2),
        scaler_scale=np.ones(2),
        support_vectors=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        dual_coef=np.asarray([1.0, -1.0], dtype=np.float64),
        intercept=0.5,
        gamma=1.0,
    )

    pred = artifact.predict(np.asarray([0.0, 0.0], dtype=np.float64))[0]

    assert math.isclose(pred, 1.0 - math.exp(-1.0) + 0.5, rel_tol=1e-9)


def test_profile_probe_uses_rbf_svr_artifact() -> None:
    step = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    grid = np.stack([step, step], axis=0)
    artifact = RbfSvrArtifact(
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        support_vectors=np.zeros((1, 4), dtype=np.float64),
        dual_coef=np.asarray([2.0], dtype=np.float64),
        intercept=-1.0,
        gamma=0.0,
    )
    probe = DinoFrameSimilarityProfileProbe(artifact=artifact, step_indices=(0, 1), gap=1)

    assert probe.score_cls_features(grid) == 1.0


def test_combined_probe_max_z_fusion() -> None:
    cls_step = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    grid = np.stack([cls_step, cls_step], axis=0)
    quantile_artifact = PCARidgeArtifact(
        scaler_mean=np.zeros(2),
        scaler_scale=np.ones(2),
        pca_mean=np.zeros(2),
        pca_components=np.asarray([[1.0, 0.0]], dtype=np.float64),
        ridge_coef=np.asarray([1.0], dtype=np.float64),
        ridge_intercept=0.0,
    )
    profile_artifact = RbfSvrArtifact(
        scaler_mean=np.zeros(2),
        scaler_scale=np.ones(2),
        support_vectors=np.zeros((1, 2), dtype=np.float64),
        dual_coef=np.asarray([10.0], dtype=np.float64),
        intercept=0.0,
        gamma=0.0,
    )
    combined = CombinedDinoProbe(
        quantile_probe=DinoTemporalQuantilePCARidgeProbe(
            quantile_artifact,
            step_index=0,
            quantiles=(0.0,),
        ),
        profile_probe=DinoFrameSimilarityProfileProbe(
            profile_artifact,
            step_indices=(0, 1),
            gap=1,
        ),
        fusion_artifact=ScoreZFusionArtifact(
            base_score_mean=np.asarray([0.0, 2.0], dtype=np.float64),
            base_score_scale=np.asarray([1.0, 4.0], dtype=np.float64),
            mode="max_z",
        ),
    )

    details = combined.score_details(grid)

    assert details["dino_temporal_quantile_pca_ridge"] == 1.0
    assert details["dino_frame_similarity_profile"] == 10.0
    assert details["combined_dino_max_z"] == 2.0


def test_concrete_verifier_uses_cls_provider() -> None:
    cls_step = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    grid = np.stack([cls_step, cls_step], axis=0)

    def provider(latent: torch.Tensor, prompt: str, step: int, total_steps: int) -> np.ndarray:
        assert prompt == "prompt"
        assert step == 14
        assert total_steps == 50
        assert latent.shape == (1,)
        return grid

    verifier = DinoFrameCosMeanVerifier(cls_provider=provider, step_index=1)
    output = verifier.score(torch.zeros(1), "prompt", 14, 50)

    assert output.score == {"dino_cls_frame_cos_mean": 0.5}
