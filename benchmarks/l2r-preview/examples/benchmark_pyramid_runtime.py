"""Optional inference-only CUDA graph/layout experiment on frozen checkpoints."""
import argparse
import copy
import json
import os
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from l2r_preview.io import load_checkpoint
from l2r_preview.openvid import load_records, sha256
from l2r_preview.train_openvid import atomic_json, resource_guard

CACHE = Path("/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16")


def measure(fn, z):
    for _ in range(10):
        output = fn(z)
        del output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(40):
        start = time.perf_counter()
        output = fn(z)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        del output
    return {"median_ms": statistics.median(times), "p10_ms": sorted(times)[4],
            "p90_ms": sorted(times)[35], "samples_ms": times,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3}


@torch.inference_mode()
def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("GPU pinning required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resource_guard(args.gpu_uuid, args.output.parent)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.4)
    model, _, _ = load_checkpoint(args.checkpoint)
    model = model.cuda().bfloat16().requires_grad_(False)
    record = load_records(CACHE, "validation")[0]
    z = load_file(CACHE/record["latent_path"])["latent"].unsqueeze(0).cuda().bfloat16()
    def eager(x):
        return model(x).add(1).mul(0.5).clamp(0, 1)
    reference = eager(z).clone()
    report = {"checkpoint": str(args.checkpoint), "sha256": sha256(args.checkpoint),
              "gpu_uuid": args.gpu_uuid, "input_id": record["id"], "shape": list(z.shape),
              "scope": "40 synchronized repeats after 10 warmups; inference only",
              "eager": measure(eager, z)}
    alternate = copy.deepcopy(model)
    for module in alternate.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.contiguous(memory_format=torch.channels_last)
    hook = alternate.stem.register_forward_pre_hook(
        lambda module, inputs: (inputs[0].contiguous(memory_format=torch.channels_last),))
    def channels_last(x):
        return alternate(x).add(1).mul(0.5).clamp(0, 1)
    output = channels_last(z)
    report["channels_last_max_error"] = float((output-reference).abs().max())
    report["channels_last_mean_error"] = float((output-reference).abs().mean())
    report["channels_last"] = measure(channels_last, z)
    buffer = z.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(6):
            graph_output = channels_last(buffer)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = channels_last(buffer)
    def replay(x):
        buffer.copy_(x)
        graph.replay()
        # Stable ownership: caller may retain its output across subsequent previews.
        return graph_output.clone()
    output = replay(z)
    report["graph_max_error"] = float((output-channels_last(z)).abs().max())
    changed = z*0.75
    changed_reference = channels_last(changed)
    changed_output = replay(changed)
    report["graph_new_input_max_error"] = float((changed_output-changed_reference).abs().max())
    retained = output.clone()
    replay(changed)
    assert torch.equal(output, retained)
    report["graph_retained_output_unchanged"] = True
    report["channels_last_graph"] = measure(replay, z)
    resource_guard(args.gpu_uuid, args.output.parent)
    atomic_json(args.output, report)
    hook.remove()
    print(json.dumps({k:v if not isinstance(v,dict) else {a:b for a,b in v.items() if a!="samples_ms"}
                      for k,v in report.items()}), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    run(parser.parse_args())
