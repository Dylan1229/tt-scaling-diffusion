"""Bounded fixed-pair fitting check before the general OpenVid training run."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from l2r_preview.model import L2RDecoder
from l2r_preview.openvid import OpenVidPatches, validate_cache
from l2r_preview.train_openvid import atomic_json, predict_patch, resource_guard

CACHE = Path("/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16")
OUT = Path("outputs/openvid5k_preflight_20261006")
GPU = "GPU-d9ab7e74-2814-efab-eefd-445eeda9faa2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    resource_guard(GPU, OUT)
    validate_cache(CACHE)
    data = OpenVidPatches(CACHE, "train")
    samples = [data[(0, index)] for index in (0, 1500)]
    model = L2RDecoder(48, 16).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    with torch.inference_mode():
        initial_outputs = [predict_patch(model, item, "cuda")[0].cpu() for item in samples]
    initial = sum(F.mse_loss(pred, sample["rgb"].unsqueeze(0)).item()
                  for pred, sample in zip(initial_outputs, samples))/len(samples)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    steps = 0
    for step in range(400):
        sample = samples[step % len(samples)]
        optimizer.zero_grad(set_to_none=True)
        pred, target = predict_patch(model, sample, "cuda")
        loss = F.mse_loss(pred, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        if step % 100 == 0:
            resource_guard(GPU, OUT)
            print(json.dumps({"step": step+1, "loss": float(loss), "seconds": time.monotonic()-started}), flush=True)
        if time.monotonic()-started > 120:
            break
    torch.cuda.synchronize()
    seconds = time.monotonic()-started
    model.eval()
    with torch.inference_mode():
        final_outputs = [predict_patch(model, item, "cuda")[0].cpu() for item in samples]
    final = sum(F.mse_loss(pred, sample["rgb"].unsqueeze(0)).item()
                for pred, sample in zip(final_outputs, samples))/len(samples)
    rows = []
    for initial_rgb, final_rgb, sample in zip(initial_outputs, final_outputs, samples):
        panels = []
        for label, rgb in (("Source RGB", sample["rgb"].unsqueeze(0)),
                           ("Before fitting", initial_rgb), ("After fitting", final_rgb)):
            array = rgb[0, :, 0].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().numpy()
            panel = Image.new("RGB", (384, 288), "white")
            panel.paste(Image.fromarray(array).resize((384, 256)), (0, 32))
            ImageDraw.Draw(panel).text((10, 10), label, fill="black")
            panels.append(np.array(panel))
        rows.append(np.concatenate(panels, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT/"fixed_pair_fit.jpg")
    report = {
        "steps": steps, "seconds": seconds, "seconds_per_step": seconds/steps,
        "initial_mse": initial, "final_mse": final, "loss_ratio": final/initial,
        "parameters": model.num_parameters, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "ids": [s["id"] for s in samples], "passed": steps >= 100 and final < initial*0.6,
        "scope": "overfit two fixed training crops, not held-out quality",
    }
    atomic_json(OUT/"report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Small-pair fitting gate failed")


if __name__ == "__main__":
    main()
