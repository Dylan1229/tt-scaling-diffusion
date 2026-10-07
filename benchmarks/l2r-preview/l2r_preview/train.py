"""Bounded patch training on explicit latent/RGB pairs, without video scoring."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import save_checkpoint
from .latents import WanLatentSpec
from .model import L2RDecoder


def load_pairs(folder: Path, spec: WanLatentSpec):
    result = []
    for path in sorted(folder.glob("train_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        z, rgb = data["latent"].float(), data["rgb"].float()
        if z.ndim != 5 or z.shape[1] != spec.latent_channels:
            raise ValueError(f"Invalid latent shape in {path}")
        expected = (
            z.shape[0],
            3,
            z.shape[2],
            z.shape[3] * spec.spatial_scale,
            z.shape[4] * spec.spatial_scale,
        )
        if z.ndim != 5 or z.shape[1] != spec.latent_channels or tuple(rgb.shape) != expected:
            raise ValueError(f"Pair shape mismatch in {path}")
        if z.shape[0] != 1 or not torch.isfinite(z).all() or not torch.isfinite(rgb).all():
            raise ValueError(f"Expected one finite video per pair: {path}")
        result.append((z, rgb))
    if not result:
        raise ValueError("No train_*.pt pairs found")
    return result


def sample_patch(pair, scale, rng, patch_size=16, target_frames=2, context=7):
    z, rgb = pair
    _, _, frames, height, width = z.shape
    size_h, size_w = min(height, patch_size), min(width, patch_size)
    y, x = rng.randrange(height - size_h + 1), rng.randrange(width - size_w + 1)
    count = min(frames, target_frames)
    start = rng.randrange(frames - count + 1)
    context_start = max(0, start - context)
    stop = start + count
    z = z[:, :, context_start:stop, y : y + size_h, x : x + size_w].contiguous()
    rgb = rgb[
        :, :, start:stop, y * scale : (y + size_h) * scale, x * scale : (x + size_w) * scale
    ].contiguous()
    return z, rgb, start - context_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.patch_size, args.max_seconds, args.learning_rate) <= 0:
        parser.error("Training bounds and learning rate must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    spec = WanLatentSpec.from_dict(json.loads((args.data / "spec.json").read_text()))
    pairs = load_pairs(args.data, spec)
    model = L2RDecoder(spec.latent_channels, spec.spatial_scale).to(args.device).train()
    print(json.dumps({"pid": os.getpid(), "device": args.device, "seed": args.seed}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    is_cuda = str(args.device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses = []
    # Ignore patch boundary context when there is enough interior area.
    border = 4 * spec.spatial_scale + 8
    for step in range(args.steps):
        z, target, offset = sample_patch(
            pairs[rng.randrange(len(pairs))], spec.spatial_scale, rng, args.patch_size
        )
        z, target = z.to(args.device), target.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.bfloat16, enabled=is_cuda):
            prediction = model(z)[:, :, offset:]
            if min(prediction.shape[-2:]) > 2 * border + 8:
                prediction = prediction[..., border:-border, border:-border]
                target = target[..., border:-border, border:-border]
            loss = F.mse_loss(prediction.float(), target.float())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        if step == 0 or (step + 1) % 50 == 0:
            print(
                json.dumps(
                    {"step": step + 1, "loss": value, "seconds": round(time.monotonic() - started, 2)}
                ),
                flush=True,
            )
        if time.monotonic() - started >= args.max_seconds:
            break
    training = {
        "status": "small_teacher_fit_not_general_pretrained_decoder",
        "steps": len(losses),
        "seconds": time.monotonic() - started,
        "seed": args.seed,
        "train_pairs": len(pairs),
        "loss": "RGB MSE",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if is_cuda else 0,
        "data": str(args.data.resolve()),
    }
    save_checkpoint(args.output, model, spec, training)
    args.output.with_suffix(".training.json").write_text(
        json.dumps({**training, "loss_history": losses}, indent=2)
    )
    print(
        json.dumps(
            {"checkpoint": str(args.output), "parameters": model.num_parameters, **training}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
