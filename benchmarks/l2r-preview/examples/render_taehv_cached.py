"""Compare pretrained TAEHV against previously cached native-VAE references.

CPU-only report generation. Exact RGB references are available for four cases;
final-video MP4s are used for full-frame-rate visual playback only.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .compare_taehv import compare_rgb, frames_u8, get_font, save_mp4, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/taehv_20261002"))
    parser.add_argument("--pairs", type=Path, default=Path("outputs/smoke/pairs"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output
    config = json.loads((root / "config.json").read_text())
    tiny = json.loads((root / "taehv.json").read_text())
    upstream = json.loads((root / "upstream" / "revision.json").read_text())
    manifest = json.loads((args.pairs / "manifest.json").read_text())
    mapping = {
        "coffee_final": "train_01.pt",
        "bear_zebra_final": "holdout_00.pt",
        "bird_cat_final": "train_02.pt",
        "swimmer_step10": "holdout_01.pt",
    }
    results, panels, sections, standalone = [], [], [], []
    for case in config["cases"]:
        name = case["id"]
        candidate = torch.load(root / "tensors" / f"{name}_taehv.pt", weights_only=True)
        b_full = frames_u8(candidate)
        save_mp4(root / f"{name}_taehv.mp4", b_full, config["source_fps"])
        if name not in mapping:
            standalone.append(f'<a href="{name}_taehv.mp4">{html.escape(case["stage"])}</a>')
            continue
        pair_path = args.pairs / mapping[name]
        pair = torch.load(pair_path, weights_only=True)
        if pair["provenance"]["source"] != case["source"]:
            raise ValueError("Cached reference provenance mismatch")
        reference = pair["rgb"].float().add(1).mul(0.5)
        sampled = candidate[:, :, ::4]
        record = {
            **case,
            **compare_rgb(reference, sampled),
            "reference": str(pair_path),
            "metric_reference_frames": 21,
            "reference_kind": "uncompressed native-VAE RGB",
        }
        if name.endswith("final"):
            source_video = Path(case["source"]).parent.parent / "video.mp4"
            reader = imageio.get_reader(source_video, input_params=["-threads", "4"])
            meta = reader.get_meta_data()
            record["cached_mp4_fps"] = float(meta["fps"])
            if abs(float(meta["fps"]) - config["source_fps"]) > 0.01:
                raise ValueError("Cached MP4 frame rate mismatch")
            a = np.stack([frame for frame in reader])
            reader.close()
            if a.shape != b_full.shape:
                raise ValueError(f"Cached MP4 shape mismatch: {a.shape} / {b_full.shape}")
            source_rgb = torch.from_numpy(a).permute(3, 0, 1, 2).unsqueeze(0).float().div(255)
            fidelity = compare_rgb(reference, source_rgb[:, :, ::4])
            record["cached_mp4_psnr_vs_raw_native_db"] = fidelity["psnr_vs_native_db"]
            if fidelity["psnr_vs_native_db"] < 28:
                raise ValueError("Cached MP4 does not sufficiently agree with the exact native reference")
            shutil.copy2(source_video, root / f"{name}_native.mp4")
            paired = np.concatenate([a, b_full], axis=2)
            fps = config["source_fps"]
            playback_note = "81 frames · 16 fps. Left uses the original cached MP4; metrics use raw RGB."
        else:
            a, b = frames_u8(reference), frames_u8(sampled)
            save_mp4(root / f"{name}_native.mp4", a, config["source_fps"] / 4)
            paired = np.concatenate([a, b], axis=2)
            fps = config["source_fps"] / 4
            playback_note = "21 matched frames · 4 fps. The full 81-frame TAEHV output is linked below."
        save_mp4(root / f"{name}_comparison.mp4", paired, fps)
        Image.fromarray(paired[len(paired) // 2]).save(root / f"{name}_poster.jpg", quality=94)
        title = case["prompt"] + " | " + case["stage"]
        panel = Image.new("RGB", (832, 286), "#f1f4f7")
        draw = ImageDraw.Draw(panel)
        draw.text((9, 4), title, font=get_font(14), fill="#17212e")
        draw.text((9, 25), "Native VAE", font=get_font(13), fill="#536171")
        draw.text((425, 25), "Pretrained TAEHV", font=get_font(13), fill="#536171")
        exact_pair = np.concatenate([frames_u8(reference)[10], frames_u8(sampled)[10]], axis=1)
        panel.paste(Image.fromarray(exact_pair).resize((832, 240), Image.Resampling.LANCZOS), (0, 46))
        panels.append(panel)
        results.append(record)
        sections.append(
            f"<section><h2>{html.escape(case['prompt'])}</h2><p>{html.escape(case['stage'])} · "
            f"PSNR {record['psnr_vs_native_db']:.2f} dB · SSIM {record['ssim_vs_native']:.3f}</p>"
            '<div class="labels"><span>Original Wan VAE</span><span>Pretrained TAEHV</span></div>'
            f'<video controls muted loop playsinline preload="auto" poster="{name}_poster.jpg" '
            f'src="{name}_comparison.mp4"></video><p class="note">{playback_note}</p>'
            f'<p><a href="{name}_native.mp4">Original</a> · <a href="{name}_taehv.mp4">TAEHV, full 16 fps</a></p></section>'
        )
        print(json.dumps(record), flush=True)
    contact = Image.new("RGB", (832, sum(p.height for p in panels)), "white")
    cursor = 0
    for panel in panels:
        contact.paste(panel, (0, cursor))
        cursor += panel.height
    contact.save(root / "contact_sheet.jpg", quality=94)
    summary = {
        "native_decoder_parameters": manifest["parameters"]["native_decoder"],
        "taehv_decoder_parameters": tiny["decoder_parameters"],
        "decoder_parameter_ratio": manifest["parameters"]["native_decoder"] / tiny["decoder_parameters"],
        "quality_cases": len(results),
        "mean_psnr_vs_native_db": statistics.mean(r["psnr_vs_native_db"] for r in results),
        "mean_ssim_vs_native": statistics.mean(r["ssim_vs_native"] for r in results),
        "observed_taehv_parallel_seconds": statistics.median(r["median_seconds"] for r in tiny["records"]),
        "taehv_parallel_peak_GiB": max(r["peak_allocated_bytes"] for r in tiny["records"]) / 2**30,
        "taehv_sequential_seconds_one_case": tiny["sequential_mode_single_clip"]["median_seconds"],
        "taehv_sequential_peak_GiB": tiny["sequential_mode_single_clip"]["peak_allocated_bytes"] / 2**30,
        "timing_valid_as_uncontended_benchmark": False,
        "timing_caveat": "GPU 0 was occupied by another job at benchmark launch and during the run. No new native-VAE GPU benchmark was launched.",
    }
    report = {
        "config": config,
        "summary": summary,
        "quality_records": results,
        "taehv": tiny,
        "upstream_revision": upstream["commit"],
        "upstream_files": upstream["downloads"],
        "comparison_protocol": "Identical saved clean latent. TAEHV takes normalized NTCHW; native references use denormalized NCTHW.",
        "quality_protocol": "PSNR on all 21 raw reference frames (indices 0,4,...,80); SSIM on 9 evenly spaced frames. Range [0,1].",
        "interpretation": "Metrics measure agreement with native decoding, not real-video ground truth or downstream decision quality.",
        "scope": "Three final clips plus one intermediate-step clip for fidelity; two additional intermediate clips have TAEHV-only playback.",
    }
    write_json(root / "report.json", report)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TAEHV vs Wan VAE</title>
<style>
body{margin:0;background:#edf1f5;color:#17212e;font:16px/1.55 system-ui}
main{max-width:1140px;margin:auto;padding:30px 22px}h1{font-size:34px;margin:0}
h2{font-size:20px;margin:0}p{color:#536171}section{background:#fff;border-radius:12px;padding:20px;margin:24px 0}
.labels{display:grid;grid-template-columns:1fr 1fr;text-align:center;font-size:14px;margin-bottom:6px}
video{display:block;width:100%;border-radius:6px}table{width:100%;border-collapse:collapse;background:white}
td,th{padding:10px 14px;border-bottom:1px solid #e0e6ef;text-align:left}
a{color:#2453a4}.note{font-size:14px}.caution{background:#fff1d4;padding:12px 16px;border-radius:8px}
</style><main><h1>TAEHV vs original Wan VAE</h1>
<p>The same saved latents decoded with pretrained TAEHV. No new training.</p>
"""
    page += (
        f"<p><b>{summary['native_decoder_parameters'] / 1e6:.2f}M → {summary['taehv_decoder_parameters'] / 1e6:.2f}M decoder parameters</b>"
        f" ({summary['decoder_parameter_ratio']:.1f}× fewer). “5B” names the compatible Wan generator.</p>"
        "<table><tr><th>TAEHV mode</th><th>Observed decode</th><th>Peak GPU allocation</th><th>Output</th></tr>"
        f"<tr><td>Parallel</td><td>{summary['observed_taehv_parallel_seconds'] * 1000:.0f} ms</td>"
        f"<td>{summary['taehv_parallel_peak_GiB']:.2f} GiB</td><td>81 × 480 × 832</td></tr>"
        f"<tr><td>Sequential*</td><td>{summary['taehv_sequential_seconds_one_case'] * 1000:.0f} ms</td>"
        f"<td>{summary['taehv_sequential_peak_GiB']:.2f} GiB</td><td>81 × 480 × 832</td></tr></table>"
        '<p class="caution"><b>Timing is provisional:</b> another job shared the GPU during measurement. '
        "No matching new native-VAE timing was taken, so this page makes no runtime speedup claim.</p>"
        '<p class="note">BF16, RTX PRO 6000 Blackwell, decoder-only weights, 1 warm-up + 3 repeats. '
        "Timing excludes disk transfers and MP4 encoding. *Sequential mode measured on the coffee clip only.</p>"
        '<p class="note">PSNR/SSIM measure similarity to cached, uncompressed native-VAE RGB, not ground-truth quality. '
        "Metrics use 21 corresponding frames; SSIM samples 9 of those. Four clips are a small smoke test.</p>"
    )
    page += "".join(sections)
    page += "<section><h2>Additional TAEHV previews</h2><p>81 frames at 16 fps; no cached native reference for these steps.</p><p>"
    page += " · ".join(standalone) + "</p></section>"
    page += (
        f'<p><a href="https://github.com/madebyollin/taehv/tree/{upstream["commit"]}">Pinned upstream source</a> · '
        '<a href="upstream/LICENSE">MIT license</a> · <a href="report.json">Full results</a> · '
        '<a href="contact_sheet.jpg">Contact sheet</a></p></main></html>'
    )
    (root / "index.html").write_text(page)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
