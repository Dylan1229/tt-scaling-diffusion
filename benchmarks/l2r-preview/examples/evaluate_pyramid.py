"""Compare trained pyramid checkpoints and TAEHV under fixed preview contracts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from l2r_preview.io import load_checkpoint
from l2r_preview.openvid import OpenVidPatches, read_rgb, sha256, validate_cache
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.train_openvid import atomic_json, resource_guard

BASE = Path("/data/datasets/peihao/l2r-preview")
CACHE = Path("/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16")
UPSTREAM = BASE/"outputs/taehv_20261002/upstream"


def selected_records(records):
    ordered = sorted(records, key=lambda r: hashlib.sha256(("pyramid-validation-22:"+r["id"]).encode()).hexdigest())
    chosen, topics = [], set()
    for band in ("low", "medium", "high"):
        for _ in range(2):
            candidates = [r for r in ordered if r["motion_band"] == band and r not in chosen]
            record = next((r for r in candidates if r["topic"] not in topics), candidates[0])
            chosen.append(record)
            topics.add(record["topic"])
    return chosen


def build_model(args, spec):
    if args.model.startswith("taehv"):
        revision = json.loads((UPSTREAM/"revision.json").read_text())
        for f in revision["downloads"]:
            if f["file"] in ("taehv.py", "taew2_2.safetensors"):
                assert sha256(UPSTREAM/f["file"]) == f["sha256"]
        source = importlib.util.spec_from_file_location("pyramid_taehv", UPSTREAM/"taehv.py")
        module = importlib.util.module_from_spec(source)
        source.loader.exec_module(module)
        kwargs = {"decoder_time_upscale": (False, False, False)} if args.model == "taehv21" else {}
        model = module.TAEHV(str(UPSTREAM/"taew2_2.safetensors"), **kwargs)
        model.encoder = torch.nn.Identity()
        model = model.eval().cuda().bfloat16()
        def convert(z):
            return spec.normalize(z).permute(0, 2, 1, 3, 4).contiguous().cuda().bfloat16()
        def forward(z):
            rgb = model.decode_video(z, parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4)
            return rgb.clamp(0, 1)
        computed_frames = 21 if args.model == "taehv21" else 81
        checkpoint_hash = sha256(UPSTREAM/"taew2_2.safetensors")
        training = None
    else:
        model, loaded_spec, training = load_checkpoint(args.checkpoint)
        assert loaded_spec == spec
        model = model.cuda().bfloat16()
        def convert(z):
            return z.cuda().bfloat16()
        def forward(z):
            return model(z).add(1).mul(0.5).clamp(0, 1)
        computed_frames = 21
        checkpoint_hash = sha256(args.checkpoint)
    model.requires_grad_(False)
    return model, convert, forward, computed_frames, checkpoint_hash, training


@torch.inference_mode()
def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Explicit GPU pinning required")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/(args.model+".json")
    if destination.exists():
        raise ValueError("Refusing to overwrite completed evaluation")
    validate_cache(CACHE)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.4)
    resource_guard(args.gpu_uuid, args.output)
    ds = OpenVidPatches(CACHE, args.split)
    chosen = selected_records(ds.records)
    chosen_ids = {r["id"] for r in chosen}
    model, convert, forward, computed_frames, checkpoint_hash, training = build_model(args, ds.spec)
    tensors = args.output/"tensors"
    tensors.mkdir(exist_ok=True)
    report = {"model": args.model, "split": args.split, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": checkpoint_hash, "training": training,
              "parameters": sum(p.numel() for p in model.parameters()), "gpu_uuid": args.gpu_uuid,
              "gpu_name": torch.cuda.get_device_name(), "precision": "bfloat16", "batch": 1,
              "computed_frames": computed_frames, "compared_frames": 21, "resolution": [480, 832],
              "timing_scope": "5 warmups, 10 repeats, synchronized forward including output clamp; no I/O/transfer/normalization",
              "selection": "hash seed22; two per motion band; diverse heuristic topics; independent of results",
              "visual_records": chosen, "records": [], "timing": [], "completed": False,
              "source_sha256": sha256(__file__)}
    began = time.monotonic()
    for index, record in enumerate(ds.records):
        resource_guard(args.gpu_uuid, args.output) if index % 25 == 0 else None
        sample = ds[index]
        z = load_file(CACHE/record["latent_path"])["latent"].unsqueeze(0).float()
        z = convert(z)
        if index < 3:
            for _ in range(5):
                warm = forward(z)
                del warm
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                rgb = forward(z)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                del rgb
            report["timing"].append({"id": record["id"], "milliseconds": times,
                                      "median_ms": statistics.median(times),
                                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        rgb = forward(z)
        assert rgb.shape == (1, 3, computed_frames, 480, 832) and torch.isfinite(rgb).all()
        selected = rgb if computed_frames == 21 else rgb[:, :, ::4]
        start = sample["plan"]["latent_start"]
        count = sample["roi"][1]
        y, x, h, w = sample["plan"]["rgb_crop"]
        target = (sample["rgb"].cuda().unsqueeze(0)+1)*0.5
        patch = selected[:, :, start:start+count, y:y+h, x:x+w].float()
        mse = float((patch-target).square().mean())
        row = {"id": record["id"], "mse_01": mse, "psnr_db": -10*math.log10(max(mse, 1e-12))}
        if args.full_metrics or record["id"] in chosen_ids:
            full = read_rgb(ds.root/record["clip_path"], list(range(0, 81, 4)), (480, 832)).unsqueeze(0).add(1).mul(0.5)
            row.update(full_video_metrics(full.cuda(), selected))
        if record["id"] in chosen_ids:
            torch.save(selected.cpu().half(), tensors/(record["id"]+"_"+args.model+".pt"))
            if not (tensors/(record["id"]+"_source.pt")).exists():
                torch.save(full.half(), tensors/(record["id"]+"_source.pt"))
        report["records"].append(row)
        if (index+1) % 50 == 0:
            print(json.dumps({"model": args.model, "completed": index+1, "total": len(ds),
                              "seconds": time.monotonic()-began}), flush=True)
        del z, rgb, selected, patch, target
    mean_mse = statistics.mean(r["mse_01"] for r in report["records"])
    report.update(completed=True, clips=len(ds), mse_01=mean_mse, psnr_db=-10*math.log10(mean_mse),
                  mean_clip_psnr_db=statistics.mean(r["psnr_db"] for r in report["records"]),
                  median_ms=statistics.median(r["median_ms"] for r in report["timing"]),
                  peak_allocated_gib=max(r["peak_allocated_bytes"] for r in report["timing"])/1024**3,
                  elapsed_seconds=time.monotonic()-began,
                  metric_scope="one deterministic 128x192 spatial crop and two frames per clip; six complete videos separately")
    if args.full_metrics:
        full_mse = statistics.mean(r["full_mse_01"] for r in report["records"])
        report.update(full_mse_01=full_mse, full_psnr_db=-10*math.log10(full_mse),
                      full_mean_clip_psnr_db=statistics.mean(r["full_psnr_db"] for r in report["records"]),
                      ssim=statistics.mean(r["ssim"] for r in report["records"]),
                      temporal_delta_mae=statistics.mean(r["temporal_delta_mae"] for r in report["records"]),
                      full_metric_scope="250 full spatial videos, all 21 endpoints for PSNR/delta; SSIM frames 0/10/20, Gaussian11 sigma1.5, population covariance")
    atomic_json(destination, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("visual_records","records","timing","training")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--full-metrics", action="store_true")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    args = parser.parse_args()
    if not args.model.startswith("taehv") and not args.checkpoint:
        parser.error("Checkpoint required")
    run(args)


if __name__ == "__main__":
    main()
