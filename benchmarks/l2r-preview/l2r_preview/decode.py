"""CLI: saved predicted-clean TTS latent -> RGB preview; no scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .io import load_checkpoint, load_latent, save_video
from .latents import flow_prediction_to_clean
from .runtime import make_preview_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--latent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latent-space", choices=("diffusion", "vae"), default="diffusion")
    parser.add_argument("--velocity", type=Path, help="Optional SAME pre-step flow prediction")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-frames", type=int, default=4, help="0 decodes all latent frames together")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--compile-mode", choices=("none", "default", "max-autotune-no-cudagraphs"), default="none", help="CUDA whole-preview inference; first call includes compilation")
    parser.add_argument("--source-fps", type=float, default=16)
    args = parser.parse_args()
    if (args.velocity is None) != (args.sigma is None):
        parser.error("--velocity and --sigma must be supplied together")
    if args.velocity is not None and args.latent_space != "diffusion":
        parser.error("Flow prediction conversion expects diffusion-space tensors")
    if args.compile_mode != "none" and (args.chunk_frames != 0 or not args.device.startswith("cuda")):
        parser.error("Compiled preview requires --device cuda and --chunk-frames 0")
    model, spec, training = load_checkpoint(args.checkpoint, args.device)
    latent = load_latent(args.latent)
    if args.velocity is not None:
        latent = flow_prediction_to_clean(latent, load_latent(args.velocity), args.sigma)
    if args.latent_space == "diffusion":
        latent = spec.denormalize(latent)
    if args.chunk_frames < 0:
        parser.error("--chunk-frames must be nonnegative")
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    model = model.to(dtype=dtype)
    latent = latent.to(device=args.device, dtype=dtype)
    with torch.inference_mode():
        if args.compile_mode != "none":
            decode = make_preview_decoder(model, args.compile_mode)
            rgb = decode(latent).cpu().float().mul(2).sub(1)
        elif args.chunk_frames == 0:
            rgb = model(latent).cpu()
        else:
            rgb = torch.cat([rgb.cpu() for _, rgb in model.iter_chunks(latent, args.chunk_frames)], dim=2)
    fps = args.source_fps / spec.temporal_scale
    save_video(args.output, rgb, fps)
    result = {
        "latent": str(args.latent.resolve()),
        "latent_space": args.latent_space,
        "precision": args.precision, "chunk_frames": args.chunk_frames, "compile_mode": args.compile_mode,
        "architecture": model.architecture,
        "input_shape": list(latent.shape),
        "output_shape": list(rgb.shape),
        "output": str(args.output.resolve()),
        "preview_fps": fps,
        "temporal_upsampling": False,
        "training": training,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
