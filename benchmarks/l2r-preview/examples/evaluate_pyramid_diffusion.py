"""Saved diffusion latents are a qualitative domain-shift check, not training data."""
import argparse
import json
import os
from pathlib import Path

import torch

from examples.evaluate_pyramid import BASE, CACHE, build_model
from l2r_preview.io import load_latent
from l2r_preview.latents import WanLatentSpec
from l2r_preview.openvid import sha256
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.train_openvid import atomic_json, resource_guard


@torch.inference_mode()
def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("GPU pinning required")
    args.output.mkdir(parents=True, exist_ok=True)
    resource_guard(args.gpu_uuid, args.output)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.4)
    config = json.loads((BASE/"outputs/taehv_20261002/config.json").read_text())
    spec = WanLatentSpec.from_vae_config(CACHE/"vae_config.json")
    _model, convert, forward, computed, ckpt_hash, training = build_model(args, spec)
    tensors = args.output/"tensors"
    tensors.mkdir(exist_ok=True)
    report = {"model": args.model, "checkpoint_sha256": ckpt_hash, "training": training,
              "computed_frames": computed, "cases": [], "completed": False,
              "scope": "three final videos plus three clean predictions from one swimmer trajectory; qualitative check only"}
    for case in config["cases"]:
        resource_guard(args.gpu_uuid, args.output)
        assert sha256(case["source"]) == case["sha256"]
        z = spec.denormalize(load_latent(case["source"]))
        output = forward(convert(z))
        selected = output if computed == 21 else output[:, :, ::4]
        ref_path = BASE/"outputs/openvid5k_review_20261006/tensors"/(case["id"]+"_native.pt")
        reference = torch.load(ref_path, weights_only=True).cuda()
        row = {**case, **full_video_metrics(reference, selected),
               "reference": str(ref_path), "reference_sha256": sha256(ref_path)}
        torch.save(selected.cpu(), tensors/(case["id"]+"_"+args.model+".pt"))
        if not (tensors/(case["id"]+"_source.pt")).exists():
            torch.save(reference.cpu(), tensors/(case["id"]+"_source.pt"))
        report["cases"].append(row)
        del z, output, selected, reference
    report["completed"] = True
    atomic_json(args.output/(args.model+".json"), report)
    print(json.dumps({"model": args.model, "completed": len(report["cases"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
