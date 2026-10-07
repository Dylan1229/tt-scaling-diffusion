"""Compare the pinned TAEW2.2 decoder with the native Wan VAE on saved clean latents.

Run each stage in a fresh process for isolated GPU memory measurements.
No diffusion generation, training, or search is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from l2r_preview.io import load_latent
from l2r_preview.latents import WanLatentSpec


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def prepare(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    by_prompt = {
        r["generation_metadata"]["prompt_id"]: r for r in manifest["records"] if r["split"] == "train"
    }
    bear = next(r for r in manifest["records"] if r["generation_metadata"]["prompt_id"] == "p04")
    early = next(r for r in manifest["records"] if "/posterior_means/" in r["source"])
    chosen = [
        ("coffee_final", by_prompt["p02"]["source"]),
        ("bear_zebra_final", bear["source"]),
        ("bird_cat_final", by_prompt["p03"]["source"]),
    ]
    for step in (4, 9, 29):
        chosen.append(
            (f"swimmer_step{step + 1:02d}", str(Path(early["source"]).with_name(f"step_{step:03d}.pt")))
        )
    cases = []
    for name, filename in chosen:
        source = Path(filename)
        if not source.is_file():
            raise FileNotFoundError(source)
        meta = json.loads((source.parent.parent / "meta.json").read_text())
        z = load_latent(source)
        if tuple(z.shape) != (1, 48, 21, 30, 52):
            raise ValueError(f"Unexpected latent shape: {source}: {z.shape}")
        cases.append(
            {
                "id": name,
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "prompt": meta["prompt_text"],
                "input_shape": list(z.shape),
                "stage": "final latent"
                if source.parent.name == "latents"
                else f"predicted clean latent after step {int(source.stem.split('_')[-1]) + 1}/50",
            }
        )
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": manifest["model_path"],
        "cases": cases,
        "source_fps": 16,
        "dtype": "bfloat16",
        "repeats": 3,
        "warmups": 1,
        "timing": "CUDA-synchronized decode only; excludes load, transfers, metrics, and video encoding.",
        "memory": "Fresh process per model; decoder-only weights resident; peak PyTorch allocated memory includes output.",
        "scope": "Six cached latents: three final clips and three steps from one swimmer trajectory.",
    }
    write_json(root / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)


def timed_decode(fn, z, repeats):
    samples = []
    output = None
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for _ in range(repeats):
        if output is not None:
            del output
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = fn(z)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    memory = {
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    output = output.detach().to(device="cpu", dtype=torch.float16)
    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite decoder output")
    return output, {"seconds": samples, "median_seconds": statistics.median(samples), **memory}


def require_idle_gpu():
    """Reject occupied GPUs before loading model weights or creating CUDA tensors."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one available GPU.")
    status = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    uuid, memory_mb, utilization = [part.strip() for part in status.split(",")]
    if float(memory_mb) > 128 or float(utilization) > 5:
        raise RuntimeError(
            f"Selected GPU is occupied ({memory_mb} MiB, {utilization}% use); benchmark not launched."
        )
    return {
        "uuid": uuid,
        "memory_used_MiB_before": float(memory_mb),
        "utilization_percent_before": float(utilization),
    }


def run_decoder(root, which):
    gpu_preflight = require_idle_gpu()
    import diffusers
    from diffusers import AutoencoderKLWan

    config = json.loads((root / "config.json").read_text())
    spec = WanLatentSpec.from_vae_config(Path(config["model_path"]) / "vae" / "config.json")
    device, dtype = "cuda:0", torch.bfloat16
    expected = (1, 3, 81, 480, 832)
    if which == "native":
        model = AutoencoderKLWan.from_pretrained(
            config["model_path"], subfolder="vae", torch_dtype=dtype, local_files_only=True
        ).eval()
        total_parameters = sum(p.numel() for p in model.parameters())
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters())
        post_quant_parameters = sum(p.numel() for p in model.post_quant_conv.parameters())
        model.encoder = torch.nn.Identity()
        model = model.to(device)

        def prepare_input(z):
            return spec.denormalize(z).to(device=device, dtype=dtype)

        def decode(z):
            return model.decode(z, return_dict=False)[0].add_(1).mul_(0.5).clamp_(0, 1)

        architecture = "Native Wan2.2 VAE decoder; unused encoder removed before GPU transfer."
    else:
        module_spec = importlib.util.spec_from_file_location("pinned_taehv", root / "upstream" / "taehv.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.TAEHV(str(root / "upstream" / "taew2_2.safetensors")).eval()
        total_parameters = sum(p.numel() for p in model.parameters())
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters())
        post_quant_parameters = 0
        model.encoder = torch.nn.Identity()
        model = model.to(device=device, dtype=dtype)

        def prepare_input(z):
            # TAEHV expects normalized diffusion-space NTCHW, unlike native VAE.
            return z.permute(0, 2, 1, 3, 4).contiguous().to(device=device, dtype=dtype)

        def decode(z):
            return model.decode_video(z, parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4)

        architecture = "Upstream taew2_2, parallel=True; unused encoder removed before GPU transfer."
    model.requires_grad_(False)
    report = {
        "model": which,
        "gpu_preflight": gpu_preflight,
        "architecture": architecture,
        "total_autoencoder_parameters": total_parameters,
        "decoder_parameters": decoder_parameters,
        "post_quant_parameters": post_quant_parameters,
        "resident_parameters": sum(p.numel() for p in model.parameters()),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "diffusers": diffusers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "dtype": config["dtype"],
        "records": [],
    }
    print(json.dumps({"pid": os.getpid(), **report}), flush=True)
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        warm_input = prepare_input(load_latent(config["cases"][0]["source"]))
        for _ in range(config["warmups"]):
            warm = decode(warm_input)
            if tuple(warm.shape) != expected:
                raise ValueError(f"Unexpected warmup shape: {warm.shape}")
            del warm
        del warm_input
        gc.collect()
        torch.cuda.empty_cache()
        for case in config["cases"]:
            z = prepare_input(load_latent(case["source"]))
            rgb, measurement = timed_decode(decode, z, config["repeats"])
            if tuple(rgb.shape) != expected:
                raise ValueError(f"Unexpected shape: {rgb.shape}")
            torch.save(rgb, tensor_dir / f"{case['id']}_{which}.pt")
            record = {"id": case["id"], "output_shape": list(rgb.shape), **measurement}
            report["records"].append(record)
            print(json.dumps(record), flush=True)
            del z, rgb
        if which == "taehv":
            z = prepare_input(load_latent(config["cases"][0]["source"]))

            def sequential(z):
                return model.decode_video(z, parallel=False, show_progress_bar=False).permute(0, 2, 1, 3, 4)

            warm = sequential(z)
            del warm
            torch.cuda.empty_cache()
            seq, measurement = timed_decode(sequential, z, config["repeats"])
            parallel = torch.load(tensor_dir / f"{config['cases'][0]['id']}_taehv.pt", weights_only=True)
            difference = (seq.float() - parallel.float()).abs()
            report["sequential_mode_single_clip"] = {
                "id": config["cases"][0]["id"],
                **measurement,
                "parallel_mean_absolute_difference": difference.mean().item(),
                "parallel_max_absolute_difference": difference.max().item(),
            }
            print(json.dumps(report["sequential_mode_single_clip"]), flush=True)
    write_json(root / f"{which}.json", report)


def frames_u8(rgb):
    return rgb[0].permute(1, 2, 3, 0).float().clamp(0, 1).mul(255).round().byte().numpy()


def save_mp4(path, array, fps):
    imageio.mimwrite(
        path,
        array,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-threads", "4", "-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )


def compare_rgb(reference, candidate):
    mse_sum, element_count = 0.0, 0
    selected = sorted(set(np.linspace(0, reference.shape[2] - 1, 9, dtype=int).tolist()))
    ssim = []
    for frame in range(reference.shape[2]):
        a, b = reference[0, :, frame].float(), candidate[0, :, frame].float()
        mse_sum += (a - b).square().sum(dtype=torch.float64).item()
        element_count += a.numel()
        if frame in selected:
            ssim.append(
                float(
                    structural_similarity(
                        a.permute(1, 2, 0).numpy(),
                        b.permute(1, 2, 0).numpy(),
                        data_range=1.0,
                        channel_axis=-1,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                )
            )
    mse = mse_sum / element_count
    return {
        "psnr_vs_native_db": -10 * math.log10(max(mse, 1e-15)),
        "ssim_vs_native": statistics.mean(ssim),
        "ssim_frame_indices": selected,
        "psnr_frames": reference.shape[2],
    }


def get_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render(root):
    config = json.loads((root / "config.json").read_text())
    native = json.loads((root / "native.json").read_text())
    tiny = json.loads((root / "taehv.json").read_text())
    if any(r.get("timing_valid_as_uncontended_benchmark") is False for r in (native, tiny)):
        raise RuntimeError("Contended timing: use render_taehv_cached for quality-only reporting.")
    revision = json.loads((root / "upstream" / "revision.json").read_text())
    native_rows = {r["id"]: r for r in native["records"]}
    tiny_rows = {r["id"]: r for r in tiny["records"]}
    results, sections, panels = [], [], []
    for case in config["cases"]:
        name = case["id"]
        ref = torch.load(root / "tensors" / f"{name}_native.pt", weights_only=True)
        candidate = torch.load(root / "tensors" / f"{name}_taehv.pt", weights_only=True)
        if ref.shape != candidate.shape:
            raise ValueError("Reference and candidate frame alignment mismatch")
        record = {
            **case,
            **compare_rgb(ref, candidate),
            "native_seconds": native_rows[name]["median_seconds"],
            "taehv_seconds": tiny_rows[name]["median_seconds"],
        }
        record["speedup"] = record["native_seconds"] / record["taehv_seconds"]
        results.append(record)
        a, b = frames_u8(ref), frames_u8(candidate)
        save_mp4(root / f"{name}_native.mp4", a, config["source_fps"])
        save_mp4(root / f"{name}_taehv.mp4", b, config["source_fps"])
        paired = np.concatenate((a, b), axis=2)
        save_mp4(root / f"{name}_comparison.mp4", paired, config["source_fps"])
        Image.fromarray(paired[len(paired) // 2]).save(root / f"{name}_poster.jpg", quality=94)
        title = case["prompt"] + " | " + case["stage"]
        panel = Image.new("RGB", (832, 286), "#f1f4f7")
        draw = ImageDraw.Draw(panel)
        draw.text((9, 4), title, font=get_font(14), fill="#17212e")
        draw.text((9, 25), "Native VAE", font=get_font(13), fill="#536171")
        draw.text((425, 25), "TAEHV / Wan 2.2 5B", font=get_font(13), fill="#536171")
        panel.paste(
            Image.fromarray(paired[len(paired) // 2]).resize((832, 240), Image.Resampling.LANCZOS), (0, 46)
        )
        panels.append(panel)
        sections.append(
            f"<section><h2>{html.escape(case['prompt'])}</h2><p>{html.escape(case['stage'])} · "
            f"PSNR vs native {record['psnr_vs_native_db']:.2f} dB · SSIM {record['ssim_vs_native']:.3f}</p>"
            '<div class="labels"><span>Original Wan VAE</span><span>Pretrained TAEHV</span></div>'
            f'<video controls muted loop playsinline preload="auto" poster="{name}_poster.jpg" '
            f'src="{name}_comparison.mp4"></video>'
            f'<p><a href="{name}_native.mp4">Original</a> · <a href="{name}_taehv.mp4">TAEHV</a></p></section>'
        )
        print(json.dumps(record), flush=True)
        del ref, candidate, a, b, paired
    contact = Image.new("RGB", (832, sum(p.height for p in panels)), "white")
    cursor = 0
    for panel in panels:
        contact.paste(panel, (0, cursor))
        cursor += panel.height
    contact.save(root / "contact_sheet.jpg", quality=94)
    summary = {
        "native_median_seconds": statistics.median(r["native_seconds"] for r in results),
        "taehv_median_seconds": statistics.median(r["taehv_seconds"] for r in results),
        "median_paired_speedup": statistics.median(r["speedup"] for r in results),
        "native_peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in native["records"]) / 2**30,
        "taehv_peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in tiny["records"]) / 2**30,
        "taehv_mean_psnr_vs_native_db": statistics.mean(r["psnr_vs_native_db"] for r in results),
        "taehv_mean_ssim_vs_native": statistics.mean(r["ssim_vs_native"] for r in results),
    }
    report = {
        "config": config,
        "upstream_revision": revision["commit"],
        "summary": summary,
        "native": native,
        "taehv": tiny,
        "quality_records": results,
        "interpretation": "PSNR/SSIM measure agreement with native decoding, not ground-truth video quality.",
        "limitations": "Six cached latents, only four prompts, one GPU; not a general paper benchmark.",
    }
    write_json(root / "report.json", report)
    rows = []
    for key, label, model_report in [
        ("native", "Original Wan VAE", native),
        ("taehv", "TAEHV parallel", tiny),
    ]:
        rows.append(
            f"<tr><td>{label}</td><td>{model_report['decoder_parameters'] / 1e6:.2f}M</td>"
            f"<td>{summary[key + '_median_seconds']:.3f} s</td>"
            f"<td>{summary[key + '_peak_allocated_GiB']:.2f} GiB</td><td>81</td></tr>"
        )
    seq = tiny["sequential_mode_single_clip"]
    rows.append(
        f"<tr><td>TAEHV sequential*</td><td>{tiny['decoder_parameters'] / 1e6:.2f}M</td>"
        f"<td>{seq['median_seconds']:.3f} s</td><td>{seq['peak_allocated_bytes'] / 2**30:.2f} GiB</td><td>81</td></tr>"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TAEHV vs Wan VAE</title>
<style>
body{margin:0;background:#edf1f5;color:#17212e;font:16px/1.55 system-ui}
main{max-width:1140px;margin:auto;padding:30px 22px}h1{font-size:34px;margin:0}
h2{font-size:20px;margin:0}p{color:#536171}section{background:#fff;border-radius:12px;padding:20px;margin:24px 0}
.labels{display:grid;grid-template-columns:1fr 1fr;text-align:center;font-size:14px;margin-bottom:6px}
video{display:block;width:100%;border-radius:6px}table{width:100%;border-collapse:collapse;background:white}
td,th{padding:10px 14px;border-bottom:1px solid #e0e6ef;text-align:left}
a{color:#2453a4}.note{font-size:14px}code{background:#e4eaf1;padding:2px 5px}
</style><main><h1>TAEHV vs original Wan VAE</h1>
<p>Same saved latents. Same 81 frames at 480×832, played at 16 fps. No extra training.</p>
"""
    page += (
        f"<p><b>{summary['median_paired_speedup']:.1f}× faster</b> median paired decode speed on these six examples. "
        "“5B” names the compatible Wan generator, not the tiny decoder’s parameter count.</p>"
        "<table><tr><th>Decoder</th><th>Decoder parameters</th><th>Median decode</th>"
        "<th>Peak GPU allocation</th><th>Output frames</th></tr>" + "".join(rows) + "</table>"
        '<p class="note">BF16, RTX PRO 6000 Blackwell, eager PyTorch, 1 full warm-up + 3 timed decodes per clip. '
        "Fresh process per model; unused encoder removed. Timing excludes disk/CPU transfers and MP4 encoding. "
        "*Sequential mode measured only on the coffee clip.</p>"
        '<p class="note">PSNR/SSIM compare with the original decoder, not real-video ground truth. '
        "PSNR uses all 81 frames; SSIM uses 9 evenly spaced frames. Six examples are a smoke test, "
        "not evidence of general quality or reliable search rankings.</p>"
    )
    page += "".join(sections)
    page += (
        f'<p>Upstream: <a href="https://github.com/madebyollin/taehv/tree/{revision["commit"]}">'
        f'TAEHV {revision["commit"][:12]}</a> · <a href="upstream/LICENSE">MIT license</a> · '
        '<a href="report.json">Full results</a> · <a href="contact_sheet.jpg">Contact sheet</a></p></main></html>'
    )
    (root / "index.html").write_text(page)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "native", "taehv", "render"], required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/taehv_20261002"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/smoke/pairs/manifest.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.output, args.manifest)
    elif args.stage == "render":
        render(args.output)
    else:
        run_decoder(args.output, args.stage)


if __name__ == "__main__":
    main()
