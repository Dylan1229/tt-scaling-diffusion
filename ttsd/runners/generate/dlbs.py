"""Run Wan 2.2 generation with DLBS branch/lookahead/preview scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ttsd.models.wan22_dlbs import WanDLBSConfig
from ttsd.search.dlbs_reward import VBenchRewardConfig, VBenchWeightedReward


@dataclass
class DLBSRunMeta:
    prompt_id: str
    prompt_text: str
    seed: int
    model: str
    scheduler: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    num_beams: int
    num_candidates: int
    num_lookahead_steps: int
    reward_weights: list[float]
    timestamp: str


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _save_video(frames, path: Path, fps: int = 16) -> None:
    import imageio.v3 as iio
    import numpy as np

    arr = frames if hasattr(frames, "__array__") else frames.cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255).astype("uint8")
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    iio.imwrite(path, arr, fps=fps, codec="libx264")


def _snapshot_step_indices(num_steps: int, every_n: int, also_keep: list[int]) -> list[int]:
    steps = set(range(every_n - 1, num_steps, every_n))
    steps.update(i for i in also_keep if 0 <= i < num_steps)
    return sorted(steps)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    from ttsd.models.wan22_adapter import Wan22Adapter

    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    dlbs_cfg = cfg["dlbs"]
    reward_cfg = cfg["reward"]
    snap_cfg = cfg.get("snapshots", {})
    out_cfg = cfg["output"]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "euler_sde")
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )
    reward_model = VBenchWeightedReward(
        VBenchRewardConfig.from_weight_sequence(
            reward_cfg["weights"],
            t2v_search_root=Path(reward_cfg["t2v_search_root"]) if reward_cfg.get("t2v_search_root") else None,
            vbench_root=Path(reward_cfg["vbench_root"]) if reward_cfg.get("vbench_root") else None,
            pretrained_root=Path(reward_cfg["pretrained_root"]) if reward_cfg.get("pretrained_root") else None,
            device=reward_cfg.get("device", model_cfg["device"]),
        )
    )

    prompts = _load_prompts(cfg["prompts"]["source"])
    seeds = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]
    if args.smoke:
        prompts = prompts[:1]
        seeds = seeds[:1]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]

    snapshot_steps = _snapshot_step_indices(
        gen_cfg["num_inference_steps"],
        snap_cfg.get("every_n_steps", gen_cfg["num_inference_steps"]),
        snap_cfg.get("also_keep", []),
    )

    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot_config = run_root / "config.snapshot.yaml"
    if not snapshot_config.exists():
        snapshot_config.write_text(yaml.safe_dump(cfg))

    print(f"[dlbs] run_root={run_root}")
    print(f"[dlbs] scheduler={scheduler_kind}")
    print(f"[dlbs] snapshot_steps={snapshot_steps}")
    print(
        "[dlbs] beams={num_beams} candidates={num_candidates} lookahead={num_lookahead_steps}".format(
            **dlbs_cfg
        )
    )

    height, width = gen_cfg["resolution"]
    for prompt in prompts:
        for seed in prompt.get("seeds") or seeds:
            out_dir = run_root / prompt["id"] / f"seed{int(seed):04d}"
            if (out_dir / "DONE").exists():
                print(f"[dlbs] SKIP {prompt['id']} seed={seed} (already done)")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "latents").mkdir(exist_ok=True)

            print(f"[dlbs] {prompt['id']} seed={seed} :: {prompt['text'][:60]}")
            dlbs_config = WanDLBSConfig(
                num_beams=int(dlbs_cfg["num_beams"]),
                num_candidates=int(dlbs_cfg["num_candidates"]),
                num_lookahead_steps=int(dlbs_cfg["num_lookahead_steps"]),
                branch_noise_scale=float(dlbs_cfg.get("branch_noise_scale", 0.0)),
                branch_noise_std=dlbs_cfg.get("branch_noise_std"),
                include_deterministic_candidate=bool(dlbs_cfg.get("include_deterministic_candidate", False)),
                reward_frame_stride=int(dlbs_cfg.get("reward_frame_stride", 2)),
                reward_max_frames=dlbs_cfg.get("reward_max_frames", 16),
                output_type=out_cfg.get("output_type", "np"),
                trace_path=out_dir / "search_trace.jsonl",
            )
            result = adapter.generate_with_dlbs(
                prompt=prompt["text"],
                seed=int(seed),
                reward_model=reward_model,
                num_frames=gen_cfg["num_frames"],
                height=height,
                width=width,
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                snapshot_steps=snapshot_steps,
                dlbs_config=dlbs_config,
            )

            if out_cfg.get("save_video", True):
                _save_video(result.frames, out_dir / "video.mp4")
            if out_cfg.get("save_latents", True):
                for step_idx, latent in result.latents_by_step.items():
                    torch.save(latent, out_dir / "latents" / f"step_{step_idx:03d}.pt")
            (out_dir / "search_trace.json").write_text(json.dumps(result.search_trace, indent=2))

            meta = DLBSRunMeta(
                prompt_id=prompt["id"],
                prompt_text=prompt["text"],
                seed=int(seed),
                model=model_cfg["name"],
                scheduler=scheduler_kind,
                height=height,
                width=width,
                num_frames=gen_cfg["num_frames"],
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                num_beams=dlbs_config.num_beams,
                num_candidates=dlbs_config.num_candidates,
                num_lookahead_steps=dlbs_config.num_lookahead_steps,
                reward_weights=[float(v) for v in reward_cfg["weights"]],
                timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            )
            (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
            (out_dir / "DONE").touch()

    print(f"[dlbs] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
