"""Run a fixed RENOISE visual pilot and build its comparison page."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import importlib
import json
from pathlib import Path

EXPECTED_PILOTS = {
    2: ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], None),
    35: ([0.0, 0.4, 0.6, 0.8, 1.0], None),
}
EXPECTED_PROMPT_IDS = ["p01", "p03", "p05"]


def validate_config(cfg: dict) -> None:
    generation = cfg["generation"]
    renoise = cfg["renoise"]
    prompts = cfg["prompts"]
    if generation["num_inference_steps"] != 50:
        raise ValueError("num_inference_steps must be 50")
    branch_step = renoise["branch_step"]
    if branch_step not in EXPECTED_PILOTS:
        raise ValueError(f"branch_step must be one of {sorted(EXPECTED_PILOTS)}")
    expected_amplitudes, expected_independent_seed = EXPECTED_PILOTS[branch_step]
    if [float(value) for value in renoise["amplitudes"]] != expected_amplitudes:
        raise ValueError(f"amplitudes must be {expected_amplitudes}")
    if renoise["root_seed"] != 0:
        raise ValueError("root_seed must be 0")
    if renoise["independent_seed"] != expected_independent_seed:
        raise ValueError(f"independent_seed must be {expected_independent_seed}")
    if list(prompts["ids"]) != EXPECTED_PROMPT_IDS:
        raise ValueError(f"prompt ids must be {EXPECTED_PROMPT_IDS}")


def _amplitude_slug(amplitude: float) -> str:
    return f"alpha_{amplitude:.1f}".replace(".", "p")


def build_comparison_html(manifest: dict) -> str:
    branch_step = manifest["branch_step"]
    rows = manifest["rows"]
    column_labels = [video["label"] for video in rows[0]["videos"]] if rows else []
    column_count = len(column_labels)
    minimum_width = 180 + column_count * 270
    cells = ['<div class="corner">Prompt</div>']
    cells.extend(f'<div class="header">{html.escape(label)}</div>' for label in column_labels)
    for row in rows:
        cells.append(
            '<div class="prompt"><strong>'
            + html.escape(row["prompt_id"])
            + "</strong><span>"
            + html.escape(row["prompt_text"])
            + "</span></div>"
        )
        for video in row["videos"]:
            cells.append(
                '<div class="clip"><video autoplay muted loop controls playsinline preload="auto" '
                f'src="{html.escape(video["path"], quote=True)}"></video></div>'
            )
    grid = "\n".join(cells)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step-{branch_step} RENOISE Visual Pilot</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; padding: 24px; background: #111; color: #eee; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
p {{ margin: 0 0 20px; color: #aaa; }}
button {{ margin-bottom: 16px; padding: 8px 12px; cursor: pointer; }}
.grid {{ display: grid; grid-template-columns: 180px repeat({column_count}, minmax(260px, 1fr)); gap: 10px; min-width: {minimum_width}px; align-items: center; }}
.header, .corner {{ position: sticky; top: 0; z-index: 2; padding: 10px; background: #202020; text-align: center; font-weight: 700; }}
.prompt {{ align-self: stretch; display: flex; flex-direction: column; justify-content: center; gap: 8px; padding: 12px; background: #1b1b1b; }}
.prompt span {{ color: #bbb; line-height: 1.35; }}
.clip {{ background: #000; }}
video {{ display: block; width: 100%; aspect-ratio: 832 / 480; object-fit: contain; }}
</style>
</head>
<body>
<h1>Step-{branch_step} RENOISE Visual Pilot</h1>
<p>Root seed 0 · UniPC 50 steps · same fresh-noise direction across amplitudes</p>
<button type="button" onclick="syncVideos()">Restart all videos together</button>
<div class="grid">
{grid}
</div>
<script>
function syncVideos() {{
  document.querySelectorAll('video').forEach((video) => {{
    video.currentTime = 0;
    video.play().catch(() => {{}});
  }});
}}
window.addEventListener('load', syncVideos);
</script>
</body>
</html>
"""


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    return getattr(importlib.import_module(module_path), attr)


def _complete(directory: Path) -> bool:
    return (directory / "video.mp4").is_file() and (directory / "meta.json").is_file()


def _write_meta(directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _row_from_artifacts(
    run_root: Path,
    prompt: dict,
    amplitudes: list[float],
    independent_seed: int | None,
) -> dict:
    videos = [
        {
            "label": f"alpha={amplitude:.1f}",
            "path": str(
                (Path(prompt["id"]) / _amplitude_slug(amplitude) / "video.mp4").as_posix()
            ),
        }
        for amplitude in amplitudes
    ]
    if independent_seed is not None:
        videos.append(
            {
                "label": f"independent seed={independent_seed}",
                "path": str(
                    (
                        Path(prompt["id"])
                        / f"independent_seed_{independent_seed}"
                        / "video.mp4"
                    ).as_posix()
                ),
            }
        )
    for video in videos:
        if not (run_root / video["path"]).is_file():
            raise RuntimeError(f"missing comparison video: {run_root / video['path']}")
    return {
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "axis": prompt.get("axis", ""),
        "videos": videos,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-ids", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch
    import yaml

    from ttsd.models.wan22_adapter import Wan22Adapter
    from ttsd.runners.generate.baseline import _save_video

    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    renoise_cfg = cfg["renoise"]
    out_cfg = cfg["output"]
    if model_cfg.get("scheduler", "unipc") != "unipc":
        raise ValueError("scheduler must be unipc")

    requested = [part.strip() for part in args.prompt_ids.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(EXPECTED_PROMPT_IDS))
    if unknown:
        raise ValueError(f"unknown prompt ids: {unknown}")
    selected_ids = requested or EXPECTED_PROMPT_IDS
    all_prompts = _load_prompts(cfg["prompts"]["source"])
    by_id = {prompt["id"]: prompt for prompt in all_prompts}
    prompts = [by_id[prompt_id] for prompt_id in selected_ids]

    device = args.device or model_cfg["device"]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[model_cfg["dtype"]]
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=device,
        scheduler_kind="unipc",
    )

    run_id = (
        args.run_id
        or out_cfg.get("run_id")
        or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = run_root / "config.snapshot.yaml"
    if not snapshot.exists():
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))

    height, width = gen_cfg["resolution"]
    branch_step = renoise_cfg["branch_step"]
    experiment = f"step{branch_step}_renoise_visual_pilot"
    log_prefix = f"[step{branch_step}_renoise]"
    independent_seed = renoise_cfg["independent_seed"]
    amplitudes = [float(value) for value in renoise_cfg["amplitudes"]]
    common_generation = {
        "num_frames": gen_cfg["num_frames"],
        "height": height,
        "width": width,
        "num_inference_steps": gen_cfg["num_inference_steps"],
        "guidance_scale": gen_cfg["guidance_scale"],
    }
    print(f"{log_prefix} run_root={run_root}")

    for prompt in prompts:
        branch_dirs = [run_root / prompt["id"] / _amplitude_slug(a) for a in amplitudes]
        if all(_complete(directory) for directory in branch_dirs):
            print(f"{log_prefix} SKIP {prompt['id']} amplitudes (complete)")
        else:
            print(f"{log_prefix} GENERATE {prompt['id']} amplitudes")
            result = adapter.generate_with_renoise_branches(
                prompt=prompt["text"],
                seed=renoise_cfg["root_seed"],
                amplitudes=amplitudes,
                branch_step=renoise_cfg["branch_step"],
                noise_seed=renoise_cfg["noise_seed"],
                **common_generation,
            )
            for amplitude, frames, directory in zip(
                result.amplitudes, result.frames_by_amplitude, branch_dirs, strict=True
            ):
                directory.mkdir(parents=True, exist_ok=True)
                _save_video(frames, directory / "video.mp4", fps=out_cfg["fps"])
                _write_meta(
                    directory,
                    {
                        "experiment": experiment,
                        "kind": "renoise",
                        "prompt_id": prompt["id"],
                        "prompt_text": prompt["text"],
                        "axis": prompt.get("axis", ""),
                        "root_seed": renoise_cfg["root_seed"],
                        "branch_step": result.branch_step,
                        "branch_sigma": result.branch_sigma,
                        "amplitude": amplitude,
                        "noise_seed": result.noise_seed,
                        "model": model_cfg["name"],
                        "scheduler": "unipc",
                        **gen_cfg,
                    },
                )

        if independent_seed is not None:
            independent_dir = run_root / prompt["id"] / f"independent_seed_{independent_seed}"
            if _complete(independent_dir):
                print(f"{log_prefix} SKIP {prompt['id']} independent seed (complete)")
            else:
                print(
                    f"{log_prefix} GENERATE {prompt['id']} "
                    f"independent seed={independent_seed}"
                )
                independent = adapter.generate(
                    prompt=prompt["text"],
                    seed=independent_seed,
                    **common_generation,
                )
                independent_dir.mkdir(parents=True, exist_ok=True)
                _save_video(
                    independent.frames,
                    independent_dir / "video.mp4",
                    fps=out_cfg["fps"],
                )
                _write_meta(
                    independent_dir,
                    {
                        "experiment": experiment,
                        "kind": "independent_seed",
                        "prompt_id": prompt["id"],
                        "prompt_text": prompt["text"],
                        "axis": prompt.get("axis", ""),
                        "seed": independent_seed,
                        "model": model_cfg["name"],
                        "scheduler": "unipc",
                        **gen_cfg,
                    },
                )

    completed_prompts = [
        prompt
        for prompt_id in EXPECTED_PROMPT_IDS
        if (prompt := by_id[prompt_id])
        and all(
            _complete(run_root / prompt_id / _amplitude_slug(amplitude))
            for amplitude in amplitudes
        )
        and (
            independent_seed is None
            or _complete(run_root / prompt_id / f"independent_seed_{independent_seed}")
        )
    ]
    manifest = {
        "experiment": experiment,
        "branch_step": branch_step,
        "run_id": run_id,
        "rows": [
            _row_from_artifacts(run_root, prompt, amplitudes, independent_seed)
            for prompt in completed_prompts
        ],
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_root / "comparison.html").write_text(build_comparison_html(manifest))
    print(f"{log_prefix} complete_rows={len(manifest['rows'])}")
    print(f"{log_prefix} comparison={run_root / 'comparison.html'}")


if __name__ == "__main__":
    main()
