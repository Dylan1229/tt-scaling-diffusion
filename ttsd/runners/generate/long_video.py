"""Generate stitched long-video baselines from repeated Wan T2V chunks.

This runner intentionally mirrors `ttsd.runners.generate.baseline` while
keeping the chunking policy simple: each chunk is generated independently from
the same prompt, then stitched into one MP4 for VBench-Long.

Usage:
    python -m ttsd.runners.generate.long_video --config configs/long_wan22_480p.yaml
    python -m ttsd.runners.generate.long_video --config configs/long_wan22_480p.yaml --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ttsd.runners.generate.baseline import _load_prompts, _save_video


@dataclass
class LongRunMeta:
    prompt_id: str
    prompt_text: str
    axis: str
    seed: int
    model: str
    height: int
    width: int
    long_video: bool
    chunk_num_frames: int
    num_chunks: int
    overlap_frames: int
    stitched_num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    chunk_seed_stride: int
    chunk_seeds: list[int]
    chunk_frame_counts: list[int]
    scheduler: str
    timestamp: str


def _as_uint8_frame_array(frames):
    import numpy as np

    arr = frames if hasattr(frames, "__array__") else frames.cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0.0, 1.0) * 255).astype("uint8")
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected video frames with shape (T,H,W,3), got {arr.shape}")
    return arr


def _stitch_chunks(chunk_arrays: list, overlap_frames: int):
    import numpy as np

    if not chunk_arrays:
        raise ValueError("Cannot stitch an empty chunk list")
    if overlap_frames < 0:
        raise ValueError(f"overlap_frames must be non-negative, got {overlap_frames}")

    pieces = [chunk_arrays[0]]
    for i, chunk in enumerate(chunk_arrays[1:], start=1):
        if overlap_frames >= chunk.shape[0]:
            raise ValueError(
                f"overlap_frames={overlap_frames} removes all frames from chunk {i} "
                f"with {chunk.shape[0]} frames"
            )
        pieces.append(chunk[overlap_frames:])
    return np.concatenate(pieces, axis=0)


def _validate_generation_config(gen_cfg: dict) -> None:
    chunk_num_frames = int(gen_cfg["chunk_num_frames"])
    num_chunks = int(gen_cfg["num_chunks"])
    overlap_frames = int(gen_cfg.get("overlap_frames", 0))

    if chunk_num_frames < 1:
        raise ValueError(f"chunk_num_frames must be positive, got {chunk_num_frames}")
    if chunk_num_frames % 4 != 1:
        raise ValueError(
            f"Wan frame counts must be 4n+1. Use 81, 121, etc.; got {chunk_num_frames}"
        )
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if not (0 <= overlap_frames < chunk_num_frames):
        raise ValueError(
            f"overlap_frames must be in [0, chunk_num_frames), got {overlap_frames}"
        )


def _build_work(cfg: dict, smoke: bool, limit_prompts: int | None, limit_seeds: int | None) -> list[tuple[dict, int]]:
    prompts = _load_prompts(cfg["prompts"]["source"])
    default_seeds = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]

    if smoke:
        prompts = prompts[:1]
        default_seeds = default_seeds[:1]
    if limit_prompts:
        prompts = prompts[:limit_prompts]

    work: list[tuple[dict, int]] = []
    for prompt in prompts:
        seeds_for_prompt = prompt.get("seeds") or default_seeds
        if smoke:
            seeds_for_prompt = seeds_for_prompt[:1]
        if limit_seeds:
            seeds_for_prompt = seeds_for_prompt[:limit_seeds]
        for seed in seeds_for_prompt:
            work.append((prompt, int(seed)))
    return work


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="One prompt, one seed sanity check.")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    out_cfg = cfg["output"]
    _validate_generation_config(gen_cfg)

    from ttsd.models.wan22_adapter import Wan22Adapter

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "unipc")
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )

    work = _build_work(cfg, args.smoke, args.limit_prompts, args.limit_seeds)
    total_clips = len(work)
    if args.num_shards > 1:
        work = [pair for i, pair in enumerate(work) if i % args.num_shards == args.shard_index]

    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snap_path = run_root / "config.snapshot.yaml"
    if not snap_path.exists():
        snap_path.write_text(yaml.safe_dump(cfg))

    height, width = gen_cfg["resolution"]
    chunk_num_frames = int(gen_cfg["chunk_num_frames"])
    num_chunks = int(gen_cfg["num_chunks"])
    overlap_frames = int(gen_cfg.get("overlap_frames", 0))
    fps = int(gen_cfg.get("fps", 16))
    chunk_seed_stride = int(gen_cfg.get("chunk_seed_stride", 1_000_000))
    save_chunks = bool(out_cfg.get("save_chunks", False))

    shard_tag = f" (shard {args.shard_index}/{args.num_shards})" if args.num_shards > 1 else ""
    print(f"[long-video] run_root={run_root}")
    print(f"[long-video] total clips across all shards: {total_clips}; this shard: {len(work)}{shard_tag}")
    print(
        "[long-video] chunks="
        f"{num_chunks} x {chunk_num_frames} frames, overlap={overlap_frames}, fps={fps}"
    )
    print(f"[long-video] scheduler: {scheduler_kind}")

    for prompt, seed in work:
        out_dir = run_root / prompt["id"] / f"seed{seed:04d}"
        if (out_dir / "DONE").exists():
            print(f"[long-video] SKIP {prompt['id']} seed={seed} (already done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_chunks:
            (out_dir / "chunks").mkdir(exist_ok=True)

        print(f"[long-video] > {prompt['id']} seed={seed} :: {prompt['text'][:60]}...")
        chunk_arrays = []
        chunk_seeds = []
        chunk_frame_counts = []
        for chunk_idx in range(num_chunks):
            chunk_seed = seed + chunk_idx * chunk_seed_stride
            chunk_seeds.append(chunk_seed)
            print(f"[long-video]   chunk {chunk_idx + 1}/{num_chunks} seed={chunk_seed}")
            result = adapter.generate(
                prompt=prompt["text"],
                seed=chunk_seed,
                num_frames=chunk_num_frames,
                height=height,
                width=width,
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                snapshot_steps=(),
            )
            chunk_arr = _as_uint8_frame_array(result.frames)
            chunk_arrays.append(chunk_arr)
            chunk_frame_counts.append(int(chunk_arr.shape[0]))
            if save_chunks:
                _save_video(chunk_arr, out_dir / "chunks" / f"chunk_{chunk_idx:03d}.mp4", fps=fps)

        stitched = _stitch_chunks(chunk_arrays, overlap_frames=overlap_frames)
        if out_cfg.get("save_video", True):
            _save_video(stitched, out_dir / "video.mp4", fps=fps)

        meta = LongRunMeta(
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            axis=prompt.get("axis", ""),
            seed=seed,
            model=model_cfg["name"],
            height=height,
            width=width,
            long_video=True,
            chunk_num_frames=chunk_num_frames,
            num_chunks=num_chunks,
            overlap_frames=overlap_frames,
            stitched_num_frames=int(stitched.shape[0]),
            fps=fps,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            chunk_seed_stride=chunk_seed_stride,
            chunk_seeds=chunk_seeds,
            chunk_frame_counts=chunk_frame_counts,
            scheduler=scheduler_kind,
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        )
        (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        (out_dir / "DONE").touch()

    print(f"[long-video] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
