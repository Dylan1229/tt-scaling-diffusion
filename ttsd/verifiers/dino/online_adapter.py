"""Online DINO verifier — runs live during the diffusion loop.

P2 ships only `DinoFrameCosMeanOnlineVerifier`. The fancier
quantile/PCA/ridge variants need trained probe artifacts (.npz files) which
aren't on this branch yet; this single-step probe is artifact-free.

Per call:
  1. Receive the posterior-mean latent (x0_hat) captured by the model adapter
     at the current step.
  2. VAE-decode it to RGB frames using Wan's own AutoencoderKLWan.
  3. Run DINOv2-base forward to get one CLS token per frame.
  4. Compute mean adjacent-frame cosine similarity → scalar score.

Both the VAE and DINOv2 are lazy-loaded on the first score() call and reused
for the rest of the run. Memory budget: ~32 GB Wan model + ~2 GB VAE + ~0.5
GB DINOv2 ≈ 35 GB — fits on one Blackwell GPU with margin.
"""

from __future__ import annotations

import os
from typing import ClassVar

import numpy as np
import torch

from ttsd.features.dino_cls import DEFAULT_DINO_MODEL_NAME, frame_cosine_mean
from ttsd.pipeline.registry import register_verifier
from ttsd.verifiers.base import Verifier, VerifierOutput


def _resolve_snapshot(model_path: str) -> str:
    """Resolve HF hub cache snapshot directory."""
    snap_dir = os.path.join(model_path, "snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            return os.path.join(snap_dir, snaps[-1])
    return model_path


class _LazyWanVae:
    """Loads Wan's VAE on first decode call; reused after that."""

    def __init__(self, model_path: str, dtype: torch.dtype, device: str):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self._vae = None
        self._processor = None

    def _load(self) -> None:
        if self._vae is not None:
            return
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor

        resolved = _resolve_snapshot(self.model_path)
        self._vae = AutoencoderKLWan.from_pretrained(
            resolved, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device).eval()
        # Tiling protects against OOM on large spatial resolutions.
        self._vae.enable_tiling(
            tile_sample_min_height=128, tile_sample_min_width=128,
            tile_sample_stride_height=96, tile_sample_stride_width=96,
        )
        self._processor = VideoProcessor(vae_scale_factor=self._vae.config.scale_factor_spatial)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Decode one (B, C, F_lat, H_lat, W_lat) latent → (F, H, W, 3) uint8 frames.
        Mirrors ttsd.runners.generate.decode_latents._decode_latents."""
        self._load()
        z = latent.to(self.device, dtype=self._vae.dtype)
        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(z.device, z.dtype)
        )
        latents_std_inv = (
            1.0 / torch.tensor(self._vae.config.latents_std)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(z.device, z.dtype)
        )
        z = z / latents_std_inv + latents_mean
        video = self._vae.decode(z, return_dict=False)[0]
        video_np = self._processor.postprocess_video(video, output_type="np")[0]
        return (np.clip(video_np, 0.0, 1.0) * 255.0).astype(np.uint8)


class _LazyDinoCls:
    """Loads DINOv2 + image processor on first call; reused after that."""

    def __init__(self, model_name: str, device: str, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def cls_features(self, frames_uint8: np.ndarray) -> np.ndarray:
        """frames_uint8: (F, H, W, 3) uint8 → (F, dim) L2-normalized float32."""
        self._load()
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames_uint8]
        outs: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            outs.append(outputs.last_hidden_state[:, 0].cpu())
        feats = torch.cat(outs, dim=0)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.numpy().astype(np.float32)


@register_verifier("dino_frame_cos_mean_online")
class DinoFrameCosMeanOnlineVerifier(Verifier):
    """Live DINO frame-cosine-mean verifier.

    Reads posterior_mean from the StepState (so REQUIRES it). When the
    orchestrator passes a non-None posterior_mean, decodes → DINOv2 →
    frame_cosine_mean → scalar. When posterior_mean is missing (e.g.
    capture wasn't requested), returns score=NaN so the policy can decide
    how to handle it (typically: skip the threshold check for that step).
    """

    REQUIRES: ClassVar[set[str]] = {"posterior_mean"}

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "bf16",
        device: str = "cuda",
        dino_model_name: str = DEFAULT_DINO_MODEL_NAME,
        dino_batch_size: int = 32,
        gap: int = 1,
        score_name: str = "dino_frame_cos_mean_online",
    ):
        if model_path is None:
            model_path = os.environ.get(
                "WAN22_MODEL_PATH",
                "/data/datasets/fanjiang/.cache/huggingface/hub/"
                "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",
            )
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._vae = _LazyWanVae(model_path, dtype_map[dtype], device)
        self._dino = _LazyDinoCls(dino_model_name, device, batch_size=dino_batch_size)
        self.gap = int(gap)
        self.score_name = score_name
        # latent-side state for plumbing posterior_mean from StepState.
        # The orchestrator stuffs the just-captured posterior mean here
        # before each score() call; cleared after.
        self._posterior_mean: torch.Tensor | None = None

    def set_posterior_mean(self, x: torch.Tensor | None) -> None:
        self._posterior_mean = x

    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> VerifierOutput:
        if self._posterior_mean is None:
            return VerifierOutput(
                score={self.score_name: float("nan")},
                final_score_estimate=None,
            )
        frames = self._vae.decode(self._posterior_mean)
        cls = self._dino.cls_features(frames)
        scalar = frame_cosine_mean(cls, gap=self.gap)
        return VerifierOutput(
            score={self.score_name: scalar},
            final_score_estimate=scalar,
        )
