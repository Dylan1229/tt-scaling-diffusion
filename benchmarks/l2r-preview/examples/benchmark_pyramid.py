"""Isolated, eager-BF16 speed/memory checks; new pyramid weights are UNTRAINED."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import statistics
from pathlib import Path

import torch

from examples.compare_taehv import require_idle_gpu, timed_decode
from examples.evaluate_openvid import load_input
from l2r_preview.io import load_checkpoint
from l2r_preview.latents import WanLatentSpec
from l2r_preview.openvid import sha256
from l2r_preview.pyramid import PyramidPreviewDecoder
from l2r_preview.train_openvid import atomic_json, resource_guard

BASE = Path("/data/datasets/peihao/l2r-preview")
REVIEW = BASE/"outputs/openvid5k_review_20261006"
UPSTREAM = BASE/"outputs/taehv_20261002/upstream"
OUT = BASE/"outputs/pyramid_design_20261006"
GPU = "GPU-d9ab7e74-2814-efab-eefd-445eeda9faa2"


def benchmark(name):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/f"{name}.json").exists():
        raise ValueError("Preserve existing benchmark result; choose a new output directory")
    preflight = require_idle_gpu()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise ValueError("Wrong GPU allocation")
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.25)
    config = json.loads((REVIEW/"config.json").read_text())
    spec = WanLatentSpec.from_dict(config["spec"])
    wanted = (config["cases"][0]["id"], "coffee_final", "swimmer_step10")
    cases = [next(r for r in config["cases"] if r["id"] == key) for key in wanted]
    if name in ("compact", "balanced"):
        model = PyramidPreviewDecoder.from_variant(name)
        architecture = model.architecture
        weights = "RANDOM INITIALIZATION; NOT TRAINED; quality is unknown"
        expected = 21
    elif name == "l2r":
        assert sha256(config["checkpoint"]) == config["checkpoint_sha256"]
        model, restored_spec, _ = load_checkpoint(config["checkpoint"])
        assert restored_spec == spec
        architecture = model.architecture
        weights = {"checkpoint": config["checkpoint"], "sha256": config["checkpoint_sha256"]}
        expected = 21
    else:
        revision = json.loads((UPSTREAM/"revision.json").read_text())
        for r in revision["downloads"]:
            if r["file"] in ("taehv.py", "taew2_2.safetensors"):
                assert sha256(UPSTREAM/r["file"]) == r["sha256"]
        module_spec = importlib.util.spec_from_file_location("bench_pinned_taehv", UPSTREAM/"taehv.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        kwargs = {"decoder_time_upscale": (False, False, False)} if name == "taehv_preview" else {}
        model = module.TAEHV(str(UPSTREAM/"taew2_2.safetensors"), **kwargs)
        model.encoder = torch.nn.Identity()
        architecture = {"kind": name, "temporal_upscale": model.t_upscale, "parallel": True,
                        "note": "21-frame mode uses upstream support for disabled temporal upsampling; quality not checked here"}
        weights = {"upstream_commit": revision["commit"], "sha256": sha256(UPSTREAM/"taew2_2.safetensors")}
        expected = 21 if name == "taehv_preview" else 81
    model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)

    def convert(case):
        z = load_input(case)
        if name.startswith("taehv"):
            if case["latent_space"] == "native_vae_unnormalized":
                z = spec.normalize(z)
            z = z.permute(0, 2, 1, 3, 4).contiguous()
        elif case["latent_space"] != "native_vae_unnormalized":
            z = spec.denormalize(z)
        return z.to(device="cuda", dtype=torch.bfloat16)

    def forward(z):
        if name.startswith("taehv"):
            return model.decode_video(z, parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4).clamp_(0, 1)
        return model(z).add_(1).mul_(0.5).clamp_(0, 1)

    resource_guard(GPU, OUT)
    convs = []
    handles = []
    def hook(layer_name):
        def record(module, inputs, output):
            macs = output.numel() * (module.in_channels//module.groups) * math.prod(module.kernel_size)
            convs.append({"layer": layer_name, "output_shape": list(output.shape), "macs": macs})
        return record
    for layer_name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
            handles.append(layer.register_forward_hook(hook(layer_name)))
    with torch.inference_mode():
        z = convert(cases[0])
        out = forward(z)
        assert tuple(out.shape) == (1, 3, expected, 480, 832)
        assert torch.isfinite(out).all()
        for handle in handles:
            handle.remove()
        del out
        for _ in range(5):
            out = forward(z)
            del out
        torch.cuda.synchronize()
        del z
        gc.collect()
        torch.cuda.empty_cache()
        results = []
        for case in cases:
            resource_guard(GPU, OUT)
            z = convert(case)
            rgb, timing = timed_decode(forward, z, 10)
            assert tuple(rgb.shape) == (1, 3, expected, 480, 832)
            assert torch.isfinite(rgb).all()
            resource_guard(GPU, OUT)
            results.append({"id": case["id"], "source": case["source"], "source_sha256": case["sha256"],
                            "output_shape": list(rgb.shape), **timing})
            print(json.dumps({"model": name, **results[-1]}), flush=True)
            del rgb, z
            gc.collect()
            torch.cuda.empty_cache()
    report = {
        "model": name, "architecture": architecture, "weights": weights,
        "parameters": sum(p.numel() for p in model.parameters()),
        "median_decode_ms": statistics.median(r["median_seconds"] for r in results)*1000,
        "peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in results)/2**30,
        "output_frames": expected, "output_resolution": [480, 832], "batch": 1,
        "conv_macs": sum(r["macs"] for r in convs), "conv_layers": convs,
        "mac_note": "Counts executed Conv2d/Conv3d multiply-accumulates; excludes other operators and memory traffic",
        "records": results, "pid": os.getpid(), "gpu_uuid": GPU, "gpu_preflight": preflight,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "precision": "bfloat16", "execution": "eager, no torch.compile, no CUDA graphs",
        "warmups": 5, "repeats_per_case": 10,
        "timing": "CUDA-synchronized model forward plus RGB range conversion/clamp; excludes transfers/file IO",
        "no_competing_gpu_process_at_case_boundaries": True,
        "quality_evaluated": False,
        "pyramid_source_sha256": sha256(BASE/"l2r_preview/pyramid.py"),
        "benchmark_source_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUT/f"{name}.json", report)
    (OUT/"source").mkdir(exist_ok=True)
    shutil.copy2(BASE/"l2r_preview/pyramid.py", OUT/"source/pyramid.py")
    shutil.copy2(__file__, OUT/"source/benchmark_pyramid.py")
    print(json.dumps({k:report[k] for k in ("model","parameters","median_decode_ms","peak_allocated_GiB","conv_macs","output_frames","quality_evaluated")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("compact","balanced","l2r","taehv_full","taehv_preview"))
    benchmark(parser.parse_args().model)
