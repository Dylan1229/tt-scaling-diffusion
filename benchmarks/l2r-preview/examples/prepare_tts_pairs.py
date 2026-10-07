"""Build a small teacher-fitting dataset from cached clean TTS latents.

Loads only Wan's VAE. No diffusion generation, downloads, or scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from l2r_preview.io import load_latent
from l2r_preview.latents import WanLatentSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Cached Diffusers model directory")
    parser.add_argument("--train-latent", action="append", default=[], type=Path)
    parser.add_argument("--holdout-latent", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sources = [("train", p) for p in args.train_latent] + [("holdout", p) for p in args.holdout_latent]
    if not sources:
        parser.error("Provide at least one clean latent")
    if len({p.resolve() for _, p in sources}) != len(sources):
        parser.error("A source latent may appear only once")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty to avoid mixing datasets")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    spec = WanLatentSpec.from_vae_config(args.model_path / "vae" / "config.json")
    is_cuda = str(args.device).startswith("cuda")
    dtype = torch.bfloat16 if is_cuda else torch.float32
    vae = (
        AutoencoderKLWan.from_pretrained(
            str(args.model_path),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=dtype,
        )
        .to(args.device)
        .eval()
    )
    vae.requires_grad_(False)
    counts = {
        "vae_total": sum(p.numel() for p in vae.parameters()),
        "native_decoder": sum(p.numel() for p in vae.decoder.parameters()),
        "post_quant_conv": sum(p.numel() for p in vae.post_quant_conv.parameters()),
    }
    print(json.dumps({"pid": os.getpid(), "device": args.device, "parameters": counts}), flush=True)
    (args.output / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    records = []
    indices = {"train": 0, "holdout": 0}
    for split, source in sources:
        diffusion_z = load_latent(source)
        if diffusion_z.shape[0] != 1:
            raise ValueError("The smoke example expects one video per latent")
        z = spec.denormalize(diffusion_z)
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            decoded = vae.decode(z.to(device=args.device, dtype=dtype), return_dict=False)[0]
        if is_cuda:
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        expected_frames = (z.shape[2] - 1) * spec.temporal_scale + 1
        if decoded.shape != (
            1,
            3,
            expected_frames,
            z.shape[3] * spec.spatial_scale,
            z.shape[4] * spec.spatial_scale,
        ):
            raise ValueError(f"Unexpected teacher shape: {tuple(decoded.shape)}")
        frame_indices = spec.rgb_frame_indices(z.shape[2])
        rgb = decoded.index_select(2, frame_indices.to(decoded.device)).cpu().half()
        if not torch.isfinite(rgb).all():
            raise RuntimeError("Teacher produced non-finite RGB values")
        record = {
            "split": split,
            "source": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "input_shape": list(diffusion_z.shape),
            "full_rgb_shape": list(decoded.shape),
            "preview_rgb_shape": list(rgb.shape),
            "rgb_frame_indices": frame_indices.tolist(),
            "native_decode_seconds": seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if is_cuda else 0,
        }
        meta = source.parent.parent / "meta.json"
        if meta.is_file():
            record["generation_metadata"] = json.loads(meta.read_text())
        path = args.output / f"{split}_{indices[split]:02d}.pt"
        torch.save({"latent": z.half(), "rgb": rgb, "provenance": record}, path)
        record["pair_file"] = path.name
        records.append(record)
        indices[split] += 1
        print(json.dumps(record), flush=True)
        del decoded, rgb
    manifest = {
        "kind": "synthetic_native_vae_teacher_pairs",
        "model_path": str(args.model_path.resolve()),
        "parameters": counts,
        "temporal_alignment": "causal endpoints 0, stride, 2*stride, ...",
        "paper_training_reproduction": False,
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
