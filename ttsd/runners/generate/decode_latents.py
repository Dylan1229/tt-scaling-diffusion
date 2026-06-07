"""Decode saved Wan latent snapshots into per-step mp4 clips.

Usage:
    python -m ttsd.runners.generate.decode_latents \
        --latents-dir /path/to/latents \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor

from ttsd.runners.generate.baseline import _save_video

DEFAULT_MODEL_PATH = (
    "/data/datasets/fanjiang/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _sorted_latent_files(latents_dir: Path) -> list[Path]:
    files = sorted(latents_dir.glob("step_*.pt"))
    if not files:
        raise FileNotFoundError(f"No step_*.pt files found under {latents_dir}")
    return files


def _resolved_model_path(model_path: str | None) -> str:
    path = model_path or os.environ.get("WAN22_MODEL_PATH", DEFAULT_MODEL_PATH)
    snap_dir = os.path.join(path, "snapshots")
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            path = os.path.join(snap_dir, snaps[-1])
    return path


def _candidate_devices(device_arg: str, gpu_indices: str | None) -> list[str]:
    if device_arg != "cuda":
        return [device_arg]

    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA decode but torch.cuda.is_available() is false")

    if gpu_indices:
        indices = [part.strip() for part in gpu_indices.split(",") if part.strip()]
    else:
        indices = [str(i) for i in range(torch.cuda.device_count())]

    if not indices:
        raise ValueError("No CUDA devices were selected")

    return [f"cuda:{idx}" for idx in indices]


def _load_decoder(
    model_path: str | None,
    dtype: torch.dtype,
    device: str,
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_sample_stride_height: int,
    tile_sample_stride_width: int,
):
    resolved_path = _resolved_model_path(model_path)
    vae = AutoencoderKLWan.from_pretrained(resolved_path, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.enable_tiling(
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_sample_stride_height=tile_sample_stride_height,
        tile_sample_stride_width=tile_sample_stride_width,
    )
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    return vae, video_processor


def _decode_latents(
    vae: AutoencoderKLWan,
    video_processor: VideoProcessor,
    latents: torch.Tensor,
    device: str,
    output_type: str = "np",
):
    with torch.no_grad():
        latents = latents.to(device, dtype=vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(vae.config.latents_std)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        video = vae.decode(latents, return_dict=False)[0]
        return video_processor.postprocess_video(video, output_type=output_type)


def _decode_with_failover(
    model_path: str | None,
    dtype: torch.dtype,
    candidate_devices: list[str],
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_sample_stride_height: int,
    tile_sample_stride_width: int,
    latents: torch.Tensor,
):
    last_error: Exception | None = None

    for device in candidate_devices:
        vae = None
        try:
            print(f"[decode_latents] trying {device}")
            vae, video_processor = _load_decoder(
                model_path,
                dtype,
                device,
                tile_sample_min_height,
                tile_sample_min_width,
                tile_sample_stride_height,
                tile_sample_stride_width,
            )
            return _decode_latents(vae, video_processor, latents, device, output_type="np"), device
        except torch.OutOfMemoryError as exc:
            last_error = exc
            print(f"[decode_latents] OOM on {device}; trying next device")
        finally:
            if vae is not None:
                del vae
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"Decode failed on all candidate devices: {candidate_devices}") from last_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-indices", type=str, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--tile-sample-min-height", type=int, default=128)
    parser.add_argument("--tile-sample-min-width", type=int, default=128)
    parser.add_argument("--tile-sample-stride-height", type=int, default=96)
    parser.add_argument("--tile-sample-stride-width", type=int, default=96)
    args = parser.parse_args(argv)

    latents_dir = args.latents_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_from_name(args.dtype)
    candidate_devices = _candidate_devices(args.device, args.gpu_indices)

    latent_files = _sorted_latent_files(latents_dir)
    print(f"[decode_latents] decoding {len(latent_files)} files from {latents_dir}")
    print(f"[decode_latents] candidate devices: {candidate_devices}")

    for latent_path in latent_files:
        latents = torch.load(latent_path, map_location="cpu")
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected a tensor in {latent_path}, got {type(latents).__name__}")

        decoded, used_device = _decode_with_failover(
            args.model_path,
            dtype,
            candidate_devices,
            args.tile_sample_min_height,
            args.tile_sample_min_width,
            args.tile_sample_stride_height,
            args.tile_sample_stride_width,
            latents,
        )
        frames = decoded[0]
        out_path = output_dir / f"{latent_path.stem}.mp4"
        _save_video(frames, out_path, fps=args.fps)
        print(f"[decode_latents] wrote {out_path} via {used_device}")


if __name__ == "__main__":
    main()