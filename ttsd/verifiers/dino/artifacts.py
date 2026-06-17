"""Serializable score-model artifacts for DINO-based verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PCARidgeArtifact:
    """Frozen ``StandardScaler -> PCA -> Ridge`` parameters."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: float

    @classmethod
    def from_sklearn_pipeline(cls, estimator: object) -> "PCARidgeArtifact":
        """Build an artifact from a fitted sklearn pipeline."""
        scaler = _find_estimator_step(estimator, "StandardScaler")
        pca = _find_estimator_step(estimator, "PCA")
        ridge = _find_estimator_step(estimator, "Ridge")
        return cls(
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            pca_mean=np.asarray(pca.mean_, dtype=np.float64),
            pca_components=np.asarray(pca.components_, dtype=np.float64),
            ridge_coef=np.asarray(ridge.coef_, dtype=np.float64),
            ridge_intercept=float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "PCARidgeArtifact":
        artifact_path = Path(path)
        with np.load(artifact_path) as data:
            _require_keys(
                data,
                [
                    "scaler_mean",
                    "scaler_scale",
                    "pca_mean",
                    "pca_components",
                    "ridge_coef",
                    "ridge_intercept",
                ],
                artifact_path,
            )
            return cls(
                scaler_mean=np.asarray(data["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(data["scaler_scale"], dtype=np.float64),
                pca_mean=np.asarray(data["pca_mean"], dtype=np.float64),
                pca_components=np.asarray(data["pca_components"], dtype=np.float64),
                ridge_coef=np.asarray(data["ridge_coef"], dtype=np.float64),
                ridge_intercept=float(np.asarray(data["ridge_intercept"]).reshape(-1)[0]),
            )

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            ridge_coef=self.ridge_coef,
            ridge_intercept=np.asarray([self.ridge_intercept], dtype=np.float64),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        _check_width(x, self.scaler_mean, "features", "scaler_mean")
        x_scaled = (x - self.scaler_mean) / _safe_scale(self.scaler_scale)
        _check_width(x_scaled, self.pca_mean, "scaled features", "pca_mean")

        components = np.asarray(self.pca_components, dtype=np.float64)
        if components.ndim != 2:
            raise ValueError(f"pca_components must be 2D, got shape {components.shape}")
        if components.shape[1] != x_scaled.shape[1]:
            raise ValueError(
                "pca_components width does not match feature width: "
                f"{components.shape[1]} vs {x_scaled.shape[1]}"
            )

        x_pca = (x_scaled - self.pca_mean) @ components.T
        coef = np.asarray(self.ridge_coef, dtype=np.float64).reshape(-1)
        if coef.shape[0] != x_pca.shape[1]:
            raise ValueError(f"ridge_coef length {coef.shape[0]} does not match PCA width {x_pca.shape[1]}")
        return x_pca @ coef + self.ridge_intercept


@dataclass(frozen=True)
class RbfSvrArtifact:
    """Frozen ``StandardScaler -> SVR(kernel='rbf')`` predictor parameters."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    support_vectors: np.ndarray
    dual_coef: np.ndarray
    intercept: float
    gamma: float

    @classmethod
    def from_sklearn_pipeline(cls, estimator: object) -> "RbfSvrArtifact":
        """Build an artifact from a fitted ``StandardScaler -> SVR`` pipeline."""
        scaler = _find_estimator_step(estimator, "StandardScaler")
        svr = _find_estimator_step(estimator, "SVR")
        gamma = getattr(svr, "_gamma", svr.gamma)
        if isinstance(gamma, str):
            raise ValueError("SVR gamma must be fitted to a numeric value before export")
        return cls(
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            support_vectors=np.asarray(svr.support_vectors_, dtype=np.float64),
            dual_coef=np.asarray(svr.dual_coef_, dtype=np.float64),
            intercept=float(np.asarray(svr.intercept_).reshape(-1)[0]),
            gamma=float(gamma),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "RbfSvrArtifact":
        artifact_path = Path(path)
        with np.load(artifact_path) as data:
            _require_keys(
                data,
                [
                    "scaler_mean",
                    "scaler_scale",
                    "support_vectors",
                    "dual_coef",
                    "intercept",
                    "gamma",
                ],
                artifact_path,
            )
            return cls(
                scaler_mean=np.asarray(data["scaler_mean"], dtype=np.float64),
                scaler_scale=np.asarray(data["scaler_scale"], dtype=np.float64),
                support_vectors=np.asarray(data["support_vectors"], dtype=np.float64),
                dual_coef=np.asarray(data["dual_coef"], dtype=np.float64),
                intercept=float(np.asarray(data["intercept"]).reshape(-1)[0]),
                gamma=float(np.asarray(data["gamma"]).reshape(-1)[0]),
            )

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            support_vectors=self.support_vectors,
            dual_coef=self.dual_coef,
            intercept=np.asarray([self.intercept], dtype=np.float64),
            gamma=np.asarray([self.gamma], dtype=np.float64),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        _check_width(x, self.scaler_mean, "features", "scaler_mean")
        x_scaled = (x - self.scaler_mean) / _safe_scale(self.scaler_scale)

        support = np.asarray(self.support_vectors, dtype=np.float64)
        if support.ndim != 2:
            raise ValueError(f"support_vectors must be 2D, got shape {support.shape}")
        if support.shape[1] != x_scaled.shape[1]:
            raise ValueError(
                "support vector width does not match feature width: "
                f"{support.shape[1]} vs {x_scaled.shape[1]}"
            )

        dual = np.asarray(self.dual_coef, dtype=np.float64).reshape(-1)
        if dual.shape[0] != support.shape[0]:
            raise ValueError(f"dual_coef length {dual.shape[0]} does not match support count {support.shape[0]}")
        squared_dist = np.sum((x_scaled[:, None, :] - support[None, :, :]) ** 2, axis=-1)
        kernel = np.exp(-float(self.gamma) * squared_dist)
        return kernel @ dual + self.intercept


@dataclass(frozen=True)
class ScoreZFusionArtifact:
    """Score-level formula fusion calibration."""

    base_score_mean: np.ndarray
    base_score_scale: np.ndarray
    mode: str = "max_z"

    @classmethod
    def from_base_scores(cls, base_scores: np.ndarray, *, mode: str = "max_z") -> "ScoreZFusionArtifact":
        scores = _as_2d(base_scores)
        return cls(
            base_score_mean=scores.mean(axis=0),
            base_score_scale=_safe_scale(scores.std(axis=0)),
            mode=mode,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "ScoreZFusionArtifact":
        artifact_path = Path(path)
        with np.load(artifact_path) as data:
            mean_key = "base_score_mean" if "base_score_mean" in data else "mean"
            scale_key = "base_score_scale" if "base_score_scale" in data else "scale"
            _require_keys(data, [mean_key, scale_key], artifact_path)
            mode = _read_optional_string(data, "mode", default="max_z")
            return cls(
                base_score_mean=np.asarray(data[mean_key], dtype=np.float64),
                base_score_scale=np.asarray(data[scale_key], dtype=np.float64),
                mode=mode,
            )

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            base_score_mean=self.base_score_mean,
            base_score_scale=self.base_score_scale,
            mode=np.asarray(self.mode),
        )

    def predict(self, base_scores: np.ndarray) -> np.ndarray:
        scores = _as_2d(base_scores)
        _check_width(scores, self.base_score_mean, "base_scores", "base_score_mean")
        z_scores = (scores - self.base_score_mean) / _safe_scale(self.base_score_scale)
        if self.mode == "max_z":
            return z_scores.max(axis=1)
        if self.mode == "min_z":
            return z_scores.min(axis=1)
        if self.mode == "equal_zmean":
            return z_scores.mean(axis=1)
        raise ValueError(f"Unsupported fusion mode: {self.mode}")


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D features, got shape {arr.shape}")


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.asarray(values, dtype=np.float64)
    return np.where(np.abs(scale) < 1e-12, 1.0, scale)


def _check_width(features: np.ndarray, reference: np.ndarray, feature_name: str, reference_name: str) -> None:
    ref = np.asarray(reference)
    if ref.ndim != 1:
        raise ValueError(f"{reference_name} must be 1D, got shape {ref.shape}")
    if features.shape[1] != ref.shape[0]:
        raise ValueError(f"{feature_name} width {features.shape[1]} does not match {reference_name} {ref.shape[0]}")


def _require_keys(data: np.lib.npyio.NpzFile, keys: list[str], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {', '.join(missing)}")


def _read_optional_string(data: np.lib.npyio.NpzFile, key: str, *, default: str) -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def _find_estimator_step(estimator: object, class_name: str) -> object:
    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.values())
    elif hasattr(estimator, "steps"):
        steps = [step for _, step in estimator.steps]
    else:
        steps = [estimator]

    for step in steps:
        if step.__class__.__name__ == class_name:
            return step
    raise ValueError(f"Could not find fitted {class_name} in estimator {estimator!r}")
