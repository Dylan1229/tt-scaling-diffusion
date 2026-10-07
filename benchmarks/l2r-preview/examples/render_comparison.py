"""Render native-VAE references beside L2R previews from the smoke pairs."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from l2r_preview.io import load_checkpoint, rgb_to_uint8, save_video


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pair", action="append", help="Pair filename, repeat to choose examples")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-fps", type=float, default=16)
    parser.add_argument("--chunk-frames", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    model, spec, training = load_checkpoint(args.checkpoint, args.device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    pairs = [args.data / name for name in args.pair] if args.pair else sorted(args.data.glob("*.pt"))
    if not pairs:
        raise ValueError("No pair files selected")
    fps = args.source_fps / spec.temporal_scale
    records, cards, panels = [], [], []
    for path in pairs:
        pair = torch.load(path, map_location="cpu", weights_only=True)
        z, reference = pair["latent"].float().to(args.device), pair["rgb"].float()
        info = pair["provenance"]
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            preview = torch.cat([rgb.cpu() for _, rgb in model.iter_chunks(z, args.chunk_frames)], dim=2)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.monotonic() - start
        if preview.shape != reference.shape or not torch.isfinite(preview).all():
            raise RuntimeError("Invalid preview")
        stem = path.stem
        save_video(args.output / f"{stem}_preview.mp4", preview, fps)
        save_video(args.output / f"{stem}_reference.mp4", reference, fps)
        save_video(args.output / f"{stem}_comparison.mp4", torch.cat([reference, preview], dim=-1), fps)
        prompt = info.get("generation_metadata", {}).get("prompt_text", stem)
        seen = "Seen training clip" if info["split"] == "train" else "Held-out latent"
        source = Path(info["source"])
        step = int(source.stem.split("_")[-1])
        stage = (
            f"clean prediction at step index {step}"
            if source.parent.name == "posterior_means"
            else "final latent"
        )
        title = f"{seen} | {prompt}"
        arrays = [rgb_to_uint8(reference), rgb_to_uint8(preview)]
        frame_indices = [0, z.shape[2] // 2, z.shape[2] - 1]
        thumb_w, thumb_h = 320, round(320 * preview.shape[-2] / preview.shape[-1])
        panel = Image.new("RGB", (960, 88 + 2 * (thumb_h + 26)), "#f4f5f6")
        draw = ImageDraw.Draw(panel)
        draw.text((12, 8), title, font=font(17), fill="#101827")
        draw.text(
            (12, 33), stage + " | columns: first, middle, last latent frame", font=font(14), fill="#536171"
        )
        for row, (array, label) in enumerate(zip(arrays, ["Native VAE reference", "L2R preview"])):
            y = 62 + row * (thumb_h + 26)
            draw.text((12, y), label, font=font(15), fill="#101827")
            for col, frame in enumerate(frame_indices):
                thumbnail = Image.fromarray(array[frame]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                panel.paste(thumbnail, (col * thumb_w, y + 23))
        panel.save(args.output / f"{stem}.jpg", quality=92)
        panels.append(panel)
        record = {
            "pair": path.name,
            "source": str(source),
            "prompt": prompt,
            "split": info["split"],
            "stage": stage,
            "input_shape": list(z.shape),
            "output_shape": list(preview.shape),
            "decode_seconds": seconds,
            "precision": "float32",
            "chunk_frames": args.chunk_frames,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if str(args.device).startswith("cuda")
            else 0,
        }
        records.append(record)
        cards.append(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(stage)}</p>"
            '<div class="labels"><span>Native VAE reference</span><span>L2R preview</span></div>'
            f'<video controls loop muted playsinline preload="metadata" poster="{stem}_poster.jpg" '
            f'src="{stem}_comparison.mp4"></video>'
            f'<details><summary>First / middle / last frames</summary><img src="{stem}.jpg" '
            f'alt="Native VAE and preview comparison"></details></section>'
        )
        poster = Image.fromarray(rgb_to_uint8(torch.cat([reference, preview], dim=-1))[z.shape[2] // 2])
        poster.save(args.output / f"{stem}_poster.jpg", quality=90)
        print(json.dumps(record), flush=True)
    sheet = Image.new("RGB", (960, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(args.output / "contact_sheet.jpg", quality=92)
    report = {
        "architecture": model.architecture,
        "parameters": model.num_parameters,
        "training": training,
        "native_parameters": manifest["parameters"],
        "preview_fps": fps,
        "records": records,
        "limitation": "Tiny teacher fit; not paper weights or general quality validation.",
        "timing_note": "Chunked preview timing includes CPU transfers. Native decoder restores more frames; no speedup claim.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>L2R preview demo</title>
<style>
body{margin:0;background:#eef1f4;color:#17212e;font:16px/1.55 system-ui}
main{max-width:1100px;margin:auto;padding:36px 22px}h1{margin:0;font-size:34px}
p{color:#536171}section{background:white;margin:25px 0;padding:20px;border-radius:12px}
h2{font-size:20px;margin:0}video,img{display:block;width:100%;border-radius:6px}
.labels{display:grid;grid-template-columns:1fr 1fr;text-align:center;font-size:14px;margin:12px 0 6px}
small{display:block}details{margin-top:16px}summary{cursor:pointer;margin-bottom:8px}
code{background:#e6ebf1;padding:2px 5px;border-radius:3px}a{color:#2453a4}
</style><main><h1>L2R preview demo</h1>
<p>Predicted clean Wan latent → two narrow causal residual blocks, learned spatial upsampling, RGB head.</p>
"""
    page += (
        f"<p><b>{model.num_parameters:,} parameters · {spec.latent_channels} latent channels · "
        f"{spec.spatial_scale}× spatial upsampling · no temporal upsampling.</b><br>"
        f"The source videos contain 81 frames; these previews contain 21 frames at {fps:g} fps. "
        "References are sampled at the same timestamps.</p>"
        f"<p><b>Small fitting demo:</b> {training['steps']} steps on {training['train_pairs']} cached clips. "
        "The held-out latents were never used for fitting. These are independently trained smoke-test weights, "
        "not the authors’ weights or a broadly trained decoder.</p>"
        "<p>The intermediate example uses <code>posterior_means/step_009.pt</code>: "
        "the clean estimate from the tenth denoising iteration, not the noisy latent at that step.</p>"
    )
    page += "".join(cards)
    page += '<p><a href="report.json">Run metadata</a> · <a href="contact_sheet.jpg">Contact sheet</a></p></main></html>'
    (args.output / "index.html").write_text(page)


if __name__ == "__main__":
    main()
