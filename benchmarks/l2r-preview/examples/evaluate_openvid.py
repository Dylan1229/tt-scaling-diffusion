"""Full-spatial/temporal-span visual check for the trained OpenVid L2R decoder."""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.util
import json
import os
import shutil
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from examples.compare_taehv import compare_rgb, frames_u8, get_font, require_idle_gpu, save_mp4, timed_decode
from l2r_preview.io import load_checkpoint, load_latent
from l2r_preview.latents import WanLatentSpec
from l2r_preview.openvid import load_records, read_rgb, sha256, validate_cache
from l2r_preview.train_openvid import atomic_json, resource_guard

BASE = Path("/data/datasets/peihao/l2r-preview")
CACHE = Path("/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16")
TRAIN = BASE/"outputs/openvid5k_20261006"
PREVIOUS = BASE/"outputs/taehv_20261002"
ROOT = BASE/"outputs/openvid5k_review_20261006"
GPU = "GPU-d9ab7e74-2814-efab-eefd-445eeda9faa2"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/"tensors").mkdir(exist_ok=True)
    (ROOT/"web").mkdir(exist_ok=True)
    validate_cache(CACHE)
    test_report = json.loads((TRAIN/"test.json").read_text())
    assert sha256(TRAIN/"best.pt") == test_report["checkpoint_sha256"]
    model, spec, training = load_checkpoint(TRAIN/"best.pt")
    assert training["step"] == 90000
    assert spec == WanLatentSpec.from_vae_config(CACHE/"vae_config.json")
    assert all(torch.isfinite(p).all() for p in model.parameters())
    records = sorted(load_records(CACHE, "test"),
                     key=lambda r: hashlib.sha256(("preview-17:"+r["id"]).encode()).hexdigest())
    chosen, used_topics = [], set()
    for band in ("low", "medium", "high"):
        for _ in range(2):
            candidates = [r for r in records if r["motion_band"] == band and r not in chosen]
            r = next((r for r in candidates if r["topic"] not in used_topics), candidates[0])
            chosen.append(r)
            used_topics.add(r["topic"])
    dataset_root = Path(json.loads((CACHE/"config.json").read_text())["dataset_root"])
    cases = []
    for r in chosen:
        path = CACHE/r["latent_path"]
        source_clip = dataset_root/r["clip_path"]
        assert sha256(path) == r["latent_sha256"] and sha256(source_clip) == r["clip_sha256"]
        case = {
            "id": "openvid_"+r["id"], "group": "openvid_test", "source": str(path),
            "sha256": r["latent_sha256"], "latent_space": "native_vae_unnormalized",
            "prompt": f"{r['topic']} | {r['motion_band']} motion", "stage": "held-out real-video reconstruction",
            "source_clip": str(source_clip), "clip_sha256": r["clip_sha256"],
            "source_id": r["source_id"], "split": "test", "topic": r["topic"], "motion_band": r["motion_band"],
        }
        cases.append(case)
        target = read_rgb(source_clip, list(range(0, 81, 4)), (480, 832)).unsqueeze(0).add(1).mul(0.5)
        torch.save(target.half(), ROOT/"tensors"/f"{case['id']}_source.pt")
    previous = json.loads((PREVIOUS/"config.json").read_text())
    for r in previous["cases"]:
        assert sha256(r["source"]) == r["sha256"]
        cases.append({**r, "group": "diffusion", "latent_space": "wan_diffusion_normalized"})
    config = {
        "cases": cases, "checkpoint": str(TRAIN/"best.pt"),
        "checkpoint_sha256": test_report["checkpoint_sha256"], "model_path": previous["model_path"],
        "spec": spec.to_dict(), "parameters_l2r": model.num_parameters, "training_step": training["step"],
        "selection": "seed-17 hash order; two test clips per metadata motion band, prefer distinct topics; no output-based selection",
        "source_frames": 81, "preview_indices": list(range(0, 81, 4)), "preview_fps": 4,
        "precision": "BF16", "warmups_per_shape": 1, "repeats": 3,
        "gpu_uuid": GPU, "scope": "12 visual smoke cases, not a complete quality benchmark",
        "timing": "CUDA-synchronized decode only, batch 1; no disk/transfer/video-encoding time",
        "timing_caveat": "L2R computes 21 RGB frames; native VAE and TAEHV compute 81 then select 21; output workloads differ",
        "script_sha256": sha256(Path(__file__)),
    }
    atomic_json(ROOT/"config.json", config)
    shutil.copy2(__file__, ROOT/"evaluate_openvid.py")
    shutil.copy2(PREVIOUS/"upstream/LICENSE", ROOT/"web/TAEHV_LICENSE.txt")
    shutil.copy2(TRAIN/"test.json", ROOT/"web/training_test_patches.json")
    print(json.dumps({"prepared_cases": len(cases), "checkpoint_verified": True, "cases": cases}, indent=2), flush=True)


def load_input(case):
    if sha256(case["source"]) != case["sha256"]:
        raise ValueError("Latent changed")
    if case["latent_space"] == "native_vae_unnormalized":
        z = load_file(case["source"])["latent"].unsqueeze(0).float()
    else:
        z = load_latent(case["source"])
    if tuple(z.shape) != (1, 48, 21, 30, 52):
        raise ValueError(f"Bad latent shape: {z.shape}")
    return z


def decode(which):
    preflight = require_idle_gpu()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise ValueError("Wrong GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.45)
    config = json.loads((ROOT/"config.json").read_text())
    spec = WanLatentSpec.from_dict(config["spec"])
    resource_guard(GPU, ROOT)
    dtype = torch.bfloat16
    if which == "l2r":
        if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        model, loaded_spec, _ = load_checkpoint(config["checkpoint"])
        assert loaded_spec == spec
        model = model.cuda().to(dtype).eval()
        parameters = model.num_parameters

        def fn(z):
            return model(z).add_(1).mul_(0.5).clamp_(0, 1)
        expected_frames = 21
    elif which == "taehv":
        revision = json.loads((PREVIOUS/"upstream/revision.json").read_text())
        for file in revision["downloads"]:
            if file["file"] in ("taehv.py", "taew2_2.safetensors"):
                assert sha256(PREVIOUS/"upstream"/file["file"]) == file["sha256"]
        module_spec = importlib.util.spec_from_file_location("review_taehv", PREVIOUS/"upstream/taehv.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.TAEHV(str(PREVIOUS/"upstream/taew2_2.safetensors")).eval()
        model.encoder = torch.nn.Identity()
        model = model.cuda().to(dtype)
        parameters = sum(p.numel() for p in model.parameters())

        def fn(z):
            return model.decode_video(z, parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4).clamp_(0, 1)
        expected_frames = 81
    else:
        from diffusers import AutoencoderKLWan
        model = AutoencoderKLWan.from_pretrained(config["model_path"], subfolder="vae",
                                                torch_dtype=dtype, local_files_only=True).eval()
        model.encoder = torch.nn.Identity()
        model = model.cuda()
        parameters = sum(p.numel() for p in model.parameters())

        def fn(z):
            return model.decode(z, return_dict=False)[0].add_(1).mul_(0.5).clamp_(0, 1)
        expected_frames = 81
    model.requires_grad_(False)

    def convert(case):
        z = load_input(case)
        if which == "taehv":
            if case["latent_space"] == "native_vae_unnormalized":
                z = spec.normalize(z)
            z = z.permute(0, 2, 1, 3, 4).contiguous()
        elif case["latent_space"] != "native_vae_unnormalized":
            z = spec.denormalize(z)
        return z.to(device="cuda", dtype=dtype)

    report = {
        "model": which, "pid": os.getpid(), "gpu_uuid": GPU, "gpu_preflight": preflight,
        "resident_parameters": parameters, "dtype": "bfloat16", "output_frames": expected_frames,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(), "records": [],
    }
    with torch.inference_mode():
        z = convert(config["cases"][0])
        warm = fn(z)
        assert tuple(warm.shape) == (1, 3, expected_frames, 480, 832)
        del warm, z
        gc.collect()
        torch.cuda.empty_cache()
        for case in config["cases"]:
            resource_guard(GPU, ROOT)
            z = convert(case)
            rgb, measurement = timed_decode(fn, z, config["repeats"])
            assert tuple(rgb.shape) == (1, 3, expected_frames, 480, 832)
            assert torch.isfinite(rgb).all()
            selected = rgb if expected_frames == 21 else rgb[:, :, ::4].contiguous()
            assert tuple(selected.shape) == (1, 3, 21, 480, 832)
            torch.save(selected, ROOT/"tensors"/f"{case['id']}_{which}.pt")
            resource_guard(GPU, ROOT)
            record = {"id": case["id"], "output_frames": expected_frames, **measurement}
            report["records"].append(record)
            atomic_json(ROOT/f"{which}.json", report)
            print(json.dumps({"model": which, **record}), flush=True)
            del rgb, selected, z
            gc.collect()
            torch.cuda.empty_cache()
    report["completed"] = True
    report["no_competing_gpu_process_at_case_boundaries"] = True
    atomic_json(ROOT/f"{which}.json", report)


def metrics(reference, candidate):
    values = compare_rgb(reference, candidate)
    return {"psnr_db": values["psnr_vs_native_db"], "ssim": values["ssim_vs_native"],
            "psnr_frames": values["psnr_frames"], "ssim_frame_indices": values["ssim_frame_indices"]}


def render():
    torch.set_num_threads(4)
    config = json.loads((ROOT/"config.json").read_text())
    reports = {m: json.loads((ROOT/f"{m}.json").read_text()) for m in ("l2r", "taehv", "native")}
    assert all(r.get("completed") and len(r["records"]) == len(config["cases"]) for r in reports.values())
    web = ROOT/"web"
    cards, records, contact_panels = [], [], []
    for case in config["cases"]:
        name = case["id"]
        decoded = {m: torch.load(ROOT/"tensors"/f"{name}_{m}.pt", map_location="cpu", weights_only=True)
                   for m in reports}
        record = {**case, "l2r_vs_native": metrics(decoded["native"], decoded["l2r"]),
                  "taehv_vs_native": metrics(decoded["native"], decoded["taehv"])}
        order = [("native", "Wan VAE"), ("taehv", "TAEHV"), ("l2r", "Our L2R")]
        if case["group"] == "openvid_test":
            decoded["source"] = torch.load(ROOT/"tensors"/f"{name}_source.pt", weights_only=True)
            order.insert(0, ("source", "Original video"))
            record["vs_source"] = {m: metrics(decoded["source"], decoded[m]) for m in reports}
        arrays = {m: frames_u8(decoded[m]) for m, _ in order}
        # One combined file guarantees synchronized playback in all columns.
        for m, _ in order:
            save_mp4(web/f"{name}_{m}.mp4", arrays[m], 4)
        width = 416
        scaled = {
            m: np.stack([np.array(Image.fromarray(frame).resize((width, 240), Image.Resampling.LANCZOS))
                         for frame in array]) for m, array in arrays.items()
        }
        combined = np.concatenate([scaled[m] for m, _ in order], axis=2)
        save_mp4(web/f"{name}_comparison.mp4", combined, 4)
        Image.fromarray(combined[10]).save(web/f"{name}_poster.jpg", quality=94)
        # Contact sheet has all three time points and all decoder columns.
        panel = Image.new("RGB", (width*len(order), 44+3*262), "#ffffff")
        draw = ImageDraw.Draw(panel)
        title = f"{case['prompt']} | {case['stage']}"
        draw.text((10, 8), title, fill="#17212e", font=get_font(15))
        for row, frame in enumerate((0, 10, 20)):
            for col, (m, label) in enumerate(order):
                x, y = col*width, 35+row*262
                draw.text((x+8, y), f"{label} - t={frame/4:g}s", fill="#536171", font=get_font(13))
                panel.paste(Image.fromarray(scaled[m][frame]), (x, y+20))
        panel.save(web/f"{name}_frames.jpg", quality=94)
        # Small overview uses just the middle frame, with explicit labels.
        overview = Image.new("RGB", (width*len(order), 300), "white")
        od = ImageDraw.Draw(overview)
        od.text((10, 7), title, fill="#17212e", font=get_font(15))
        for col, (m, label) in enumerate(order):
            od.text((col*width+8, 33), label, fill="#536171", font=get_font(13))
        overview.paste(Image.fromarray(combined[10]), (0, 58))
        contact_panels.append(overview)
        if case["group"] == "openvid_test":
            scores = record["vs_source"]
            quality = "PSNR vs original: " + " · ".join(
                f"{label} {scores[m]['psnr_db']:.2f} dB" for m, label in order if m != "source")
        else:
            quality = (f"PSNR vs Wan VAE: TAEHV {record['taehv_vs_native']['psnr_db']:.2f} dB"
                       f" · L2R {record['l2r_vs_native']['psnr_db']:.2f} dB")
        labels = "".join(f"<span>{html.escape(label)}</span>" for _, label in order)
        links = " · ".join(f'<a href="{name}_{m}.mp4">{html.escape(label)}, full resolution</a>' for m, label in order)
        cards.append(
            f'<section data-group="{case["group"]}"><h2>{html.escape(case["prompt"])}</h2>'
            f'<p>{html.escape(case["stage"])} · {html.escape(quality)}</p>'
            f'<div class="labels" style="grid-template-columns:repeat({len(order)},1fr)">{labels}</div>'
            f'<video controls loop muted playsinline preload="metadata" poster="{name}_poster.jpg"'
            f' src="{name}_comparison.mp4"></video><p class="links">{links}</p>'
            f'<details><summary>First / middle / last frames</summary>'
            f'<img src="{name}_frames.jpg" loading="lazy" alt="Decoded frames comparison"></details></section>')
        records.append(record)
        print(json.dumps({"rendered": name, "quality": quality}), flush=True)
    summary = {}
    for model, report in reports.items():
        summary[model] = {
            "parameters": report["resident_parameters"],
            "median_decode_ms": statistics.median(r["median_seconds"] for r in report["records"])*1000,
            "peak_allocated_GiB": max(r["peak_allocated_bytes"] for r in report["records"])/2**30,
            "output_frames": report["output_frames"],
        }
    real = [r for r in records if r["group"] == "openvid_test"]
    generated = [r for r in records if r["group"] == "diffusion"]
    summary["real_video_mean_psnr"] = {
        m: statistics.mean(r["vs_source"][m]["psnr_db"] for r in real) for m in reports
    }
    summary["generated_mean_psnr_vs_native"] = {
        m: statistics.mean(r[f"{m}_vs_native"]["psnr_db"] for r in generated) for m in ("l2r", "taehv")
    }
    result = {"config": config, "summary": summary, "records": records, "decoders": reports,
              "metric_notes": "PSNR on all 21 aligned full-resolution RGB frames; SSIM on 9; pre-video-compression RGB in [0,1]. Mean PSNR is arithmetic mean of per-clip values.",
              "limits": "Six test clips and six generated cases; not a large benchmark. Motion/topic tags are metadata heuristics. L2R renders 21 frames, baselines render 81.",
              "scope": "reconstruction/preview, no failure detector or generation-quality selection evaluation"}
    atomic_json(web/"report.json", result)
    sheet = Image.new("RGB", (1664, sum(p.height for p in contact_panels)), "#edf1f5")
    y = 0
    for panel in contact_panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(web/"contact_sheet.jpg", quality=93)
    table = "".join(
        f"<tr><td>{label}</td><td>{summary[m]['parameters']/1e6:.3f}M</td>"
        f"<td>{summary[m]['output_frames']}</td><td>{summary[m]['median_decode_ms']:.1f} ms</td>"
        f"<td>{summary[m]['peak_allocated_GiB']:.2f} GiB</td></tr>"
        for m, label in (("native", "Wan VAE"), ("taehv", "TAEHV"), ("l2r", "Our L2R")))
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenVid-trained L2R preview</title>
<style>
body{margin:0;background:#edf1f5;color:#17212e;font:16px/1.5 system-ui}
main{max-width:1664px;margin:auto;padding:28px 20px}h1{margin:0;font-size:30px}
h2{font-size:20px;margin:0}p{color:#536171}.note{background:#fff3d9;padding:12px 16px;border-radius:8px}
section{background:white;border-radius:10px;padding:18px;margin:24px 0}
.labels{display:grid;text-align:center;font-weight:600;margin:12px 0 8px}
video,img{display:block;width:100%;height:auto;border-radius:5px}
table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:9px 14px;border-bottom:1px solid #dce4ee}
a{color:#2453a4}.links{font-size:13px}details{margin-top:12px}summary{cursor:pointer}
button{padding:9px 16px;border:1px solid #adb9c9;border-radius:8px;background:white;cursor:pointer;margin-right:8px}
button.active{background:#17212e;color:white}
</style><main><h1>OpenVid-trained L2R preview</h1>
<p>4,500 training clips · 90,000 updates · 0.525M parameters. Six held-out real videos and six saved diffusion latents.</p>
<p><b>All comparison videos are local and synchronized.</b> Each shows 21 corresponding frames at 4 fps.
Full-resolution individual videos are linked under each comparison.</p>
"""
    page += "<table><tr><th>Decoder</th><th>Parameters*</th><th>Frames computed</th><th>Decode median</th><th>Peak GPU</th></tr>"+table+"</table>"
    page += ('<p class="note">Timing: BF16, batch 1, RTX PRO 6000 Blackwell; one warm-up and three repeats per case. '
             'Model loading, CPU transfer and MP4 encoding excluded. <b>Output workloads differ:</b> L2R computes 21 frames; '
             'Wan VAE and TAEHV compute 81, then we select the same 21 for viewing. *Resident parameters include '
             'both native VAE quantization projections.</p>'
             '<p>PSNR uses raw decoded RGB before video compression. Six clips are a visual smoke check, not the whole test set.</p>'
             '<nav><button class="active" data-filter="all">All 12</button>'
             '<button data-filter="openvid_test">Real video reconstruction</button>'
             '<button data-filter="diffusion">Diffusion previews</button></nav>')
    page += "".join(cards)
    page += ('<p><a href="contact_sheet.jpg">Contact sheet</a> · <a href="report.json">Full results</a> · '
             '<a href="training_test_patches.json">250-clip training test report (patches)</a> · '
             '<a href="TAEHV_LICENSE.txt">TAEHV license</a></p></main>'
             '<script>document.querySelectorAll("button[data-filter]").forEach(b=>b.onclick=()=>{'
             'document.querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));'
             'document.querySelectorAll("section[data-group]").forEach(s=>{s.hidden=b.dataset.filter!=="all"&&s.dataset.group!==b.dataset.filter;'
             'if(s.hidden)s.querySelector("video").pause()})});'
             'document.querySelectorAll("video").forEach(v=>v.addEventListener("play",()=>'
             'document.querySelectorAll("video").forEach(o=>{if(o!==v)o.pause()})));</script></html>')
    (web/"index.html").write_text(page)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "l2r", "taehv", "native", "render"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "render":
        render()
    else:
        decode(args.stage)


if __name__ == "__main__":
    main()
