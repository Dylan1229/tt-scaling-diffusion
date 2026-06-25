"""Generate a binary last-frame I2V branch tree for chunk-level TTS.

This runner is intentionally separate from ``long_video.py``. It fixes chunk 0
for each (prompt, root_seed), branches chunks 1..5 with two I2V seeds per node,
and writes every leaf as a normal VBench-Long-compatible video directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ttsd.runners.generate.baseline import _load_prompts, _save_video
from ttsd.runners.generate.long_video import (
    _as_uint8_frame_array,
    _frame_to_pil,
    _stitch_chunks,
    _target_num_frames,
    _validate_generation_config,
)


@dataclass
class BranchState:
    path_bits: str
    chunk_arrays: list
    chunk_seeds: list[int]
    branch_steps: list[dict]
    last_frame: object


MANIFEST_FIELDS = [
    "prompt_id",
    "prompt_text",
    "root_seed",
    "path_bits",
    "path_id",
    "seed_idx",
    "video_path",
]


def _selected_prompts(cfg: dict) -> list[dict]:
    prompts = _load_prompts(cfg["prompts"]["source"])
    prompt_ids = [str(item) for item in cfg["branch"]["prompt_ids"]]
    by_id = {str(prompt["id"]): prompt for prompt in prompts}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if missing:
        raise ValueError(f"Requested prompt ids not found: {missing}; available={sorted(by_id)}")
    return [by_id[prompt_id] for prompt_id in prompt_ids]


def _root_work(cfg: dict) -> list[tuple[dict, int]]:
    prompts = _selected_prompts(cfg)
    root_seeds = [int(seed) for seed in cfg["branch"]["root_seeds"]]
    return [(prompt, seed) for prompt in prompts for seed in root_seeds]


def _path_bits(path_id: int, depth: int) -> str:
    return format(path_id, f"0{depth}b")


def _seed_idx(root_seed: int, path_id: int, seed_index_stride: int) -> int:
    return root_seed * seed_index_stride + path_id


def _leaf_dir(run_root: Path, prompt_id: str, root_seed: int, path_id: int, depth: int, seed_index_stride: int) -> Path:
    seed_idx = _seed_idx(root_seed, path_id, seed_index_stride)
    return run_root / prompt_id / f"seed{seed_idx:06d}_path{_path_bits(path_id, depth)}"


def _root_complete(run_root: Path, prompt_id: str, root_seed: int, depth: int, seed_index_stride: int) -> bool:
    return all(
        (_leaf_dir(run_root, prompt_id, root_seed, path_id, depth, seed_index_stride) / "DONE").exists()
        for path_id in range(2**depth)
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_rows_from_meta(run_root: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(run_root.glob("p*/seed*_path*/meta.json")):
        meta = json.loads(meta_path.read_text())
        rows.append(
            {
                "prompt_id": str(meta["prompt_id"]),
                "prompt_text": str(meta["prompt_text"]),
                "root_seed": int(meta["root_seed"]),
                "path_bits": str(meta["path_bits"]),
                "path_id": int(meta["path_id"]),
                "seed_idx": int(meta["seed"]),
                "video_path": str((meta_path.parent / "video.mp4").relative_to(run_root)),
            }
        )
    return sorted(rows, key=lambda r: (r["prompt_id"], int(r["root_seed"]), int(r["path_id"])))


def merge_manifest(run_root: Path) -> Path:
    rows = _manifest_rows_from_meta(run_root)
    out = run_root / "branch_manifest.csv"
    _write_rows(out, rows)
    print(f"[chunk-branch] wrote {out} ({len(rows)} rows)")
    return out


def _branch_seed(root_seed: int, chunk_idx: int, path_bits: str, chunk_seed_stride: int) -> int:
    branch_id = int(path_bits, 2)
    return root_seed + chunk_idx * chunk_seed_stride + branch_id


def _validate_branch_config(cfg: dict) -> None:
    gen_cfg = cfg["generation"]
    branch_cfg = cfg["branch"]
    if gen_cfg.get("mode") != "last_frame_i2v_chunks":
        raise ValueError("chunk branch runner expects generation.mode=last_frame_i2v_chunks")
    _validate_generation_config(gen_cfg)
    if int(branch_cfg.get("branch_factor", 2)) != 2:
        raise ValueError("This pilot supports branch_factor=2 only")
    num_chunks = int(gen_cfg["num_chunks"])
    branch_chunks = [int(item) for item in branch_cfg.get("branch_chunks", list(range(1, num_chunks)))]
    expected = list(range(1, num_chunks))
    if branch_chunks != expected:
        raise ValueError(f"branch_chunks must be {expected}, got {branch_chunks}")


def _save_leaf(
    *,
    run_root: Path,
    prompt: dict,
    root_seed: int,
    state: BranchState,
    cfg: dict,
    fps: int,
    overlap_frames: int,
    seed_index_stride: int,
    scheduler_kind: str,
) -> dict:
    gen_cfg = cfg["generation"]
    model_cfg = cfg["model"]
    branch_cfg = cfg["branch"]
    depth = int(gen_cfg["num_chunks"]) - 1
    path_id = int(state.path_bits, 2)
    seed_idx = _seed_idx(root_seed, path_id, seed_index_stride)
    out_dir = _leaf_dir(run_root, prompt["id"], root_seed, path_id, depth, seed_index_stride)
    if (out_dir / "DONE").exists():
        return {
            "prompt_id": prompt["id"],
            "prompt_text": prompt["text"],
            "root_seed": root_seed,
            "path_bits": state.path_bits,
            "path_id": path_id,
            "seed_idx": seed_idx,
            "video_path": str((out_dir / "video.mp4").relative_to(run_root)),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    stitched = _stitch_chunks(state.chunk_arrays, overlap_frames=overlap_frames)
    _save_video(stitched, out_dir / "video.mp4", fps=fps)

    height, width = gen_cfg["resolution"]
    meta = {
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "axis": prompt.get("axis", ""),
        "seed": seed_idx,
        "root_seed": root_seed,
        "path_bits": state.path_bits,
        "path_id": path_id,
        "branch_factor": int(branch_cfg.get("branch_factor", 2)),
        "fixed_first_chunk": True,
        "model": model_cfg["name"],
        "height": int(height),
        "width": int(width),
        "long_video": True,
        "generation_mode": "last_frame_i2v_chunk_branch_tree",
        "conditioning": "previous_last_frame",
        "target_num_frames": _target_num_frames(gen_cfg),
        "chunk_num_frames": int(gen_cfg["chunk_num_frames"]),
        "num_chunks": int(gen_cfg["num_chunks"]),
        "overlap_frames": int(gen_cfg.get("overlap_frames", 0)),
        "stitched_num_frames": int(stitched.shape[0]),
        "fps": fps,
        "num_inference_steps": int(gen_cfg["num_inference_steps"]),
        "guidance_scale": float(gen_cfg["guidance_scale"]),
        "chunk_seed_stride": int(gen_cfg.get("chunk_seed_stride", 1_000_000)),
        "chunk_seeds": state.chunk_seeds,
        "chunk_frame_counts": [int(chunk.shape[0]) for chunk in state.chunk_arrays],
        "branch_steps": state.branch_steps,
        "scheduler": scheduler_kind,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "DONE").touch()
    return {
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "root_seed": root_seed,
        "path_bits": state.path_bits,
        "path_id": path_id,
        "seed_idx": seed_idx,
        "video_path": str((out_dir / "video.mp4").relative_to(run_root)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-manifest", action="store_true")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_cfg = cfg["output"]
    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id

    if args.merge_manifest:
        if not run_root.exists():
            raise SystemExit(f"Run root not found: {run_root}")
        merge_manifest(run_root)
        return

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )
    _validate_branch_config(cfg)

    from ttsd.models.wan22_adapter import Wan22Adapter
    from ttsd.models.wan22_i2v_adapter import Wan22I2VAdapter

    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    branch_cfg = cfg["branch"]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "unipc")
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )
    i2v_adapter = Wan22I2VAdapter.from_t2v_adapter(adapter)

    work = _root_work(cfg)
    total_roots = len(work)
    if args.num_shards > 1:
        work = [pair for i, pair in enumerate(work) if i % args.num_shards == args.shard_index]

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
    seed_index_stride = int(branch_cfg.get("seed_index_stride", 1000))
    depth = num_chunks - 1

    shard_tag = f" (shard {args.shard_index}/{args.num_shards})" if args.num_shards > 1 else ""
    print(f"[chunk-branch] run_root={run_root}")
    print(f"[chunk-branch] total roots: {total_roots}; this shard: {len(work)}{shard_tag}")
    print(f"[chunk-branch] prompts={branch_cfg['prompt_ids']} root_seeds={branch_cfg['root_seeds']}")
    print(f"[chunk-branch] leaves per root={2**depth}, continuation nodes per root={sum(2**i for i in range(1, depth + 1))}")
    print(f"[chunk-branch] chunks={num_chunks} x {chunk_num_frames}, overlap={overlap_frames}, fps={fps}")
    print(f"[chunk-branch] scheduler={scheduler_kind}")

    for prompt, root_seed in work:
        prompt_id = str(prompt["id"])
        part_manifest = run_root / f"branch_manifest_{prompt_id}_root{root_seed:04d}.csv"
        if _root_complete(run_root, prompt_id, root_seed, depth, seed_index_stride):
            print(f"[chunk-branch] SKIP {prompt_id} root_seed={root_seed} (all leaves done)")
            rows = [
                row
                for row in _manifest_rows_from_meta(run_root)
                if row["prompt_id"] == prompt_id and int(row["root_seed"]) == root_seed
            ]
            _write_rows(part_manifest, rows)
            continue

        print(f"[chunk-branch] > {prompt_id} root_seed={root_seed} :: {prompt['text'][:60]}...")
        print(f"[chunk-branch]   chunk 0 fixed T2V seed={root_seed}")
        result = adapter.generate(
            prompt=prompt["text"],
            seed=root_seed,
            num_frames=chunk_num_frames,
            height=height,
            width=width,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            snapshot_steps=(),
        )
        chunk0 = _as_uint8_frame_array(result.frames)
        states = [
            BranchState(
                path_bits="",
                chunk_arrays=[chunk0],
                chunk_seeds=[root_seed],
                branch_steps=[],
                last_frame=chunk0[-1],
            )
        ]

        for chunk_idx in range(1, num_chunks):
            next_states = []
            print(f"[chunk-branch]   expand chunk {chunk_idx}/{num_chunks - 1}: parents={len(states)}")
            for state in states:
                for choice in range(2):
                    child_bits = state.path_bits + str(choice)
                    chunk_seed = _branch_seed(root_seed, chunk_idx, child_bits, chunk_seed_stride)
                    print(
                        f"[chunk-branch]     prefix={state.path_bits or '-'} choice={choice} "
                        f"path={child_bits} seed={chunk_seed}"
                    )
                    result = i2v_adapter.generate(
                        prompt=prompt["text"],
                        image=_frame_to_pil(state.last_frame),
                        seed=chunk_seed,
                        num_frames=chunk_num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=gen_cfg["num_inference_steps"],
                        guidance_scale=gen_cfg["guidance_scale"],
                    )
                    chunk_arr = _as_uint8_frame_array(result.frames)
                    next_states.append(
                        BranchState(
                            path_bits=child_bits,
                            chunk_arrays=state.chunk_arrays + [chunk_arr],
                            chunk_seeds=state.chunk_seeds + [chunk_seed],
                            branch_steps=state.branch_steps
                            + [
                                {
                                    "chunk_idx": chunk_idx,
                                    "parent_path_bits": state.path_bits,
                                    "choice": choice,
                                    "path_bits": child_bits,
                                    "chunk_seed": chunk_seed,
                                }
                            ],
                            last_frame=chunk_arr[-1],
                        )
                    )
            states = next_states

        rows = []
        for state in states:
            rows.append(
                _save_leaf(
                    run_root=run_root,
                    prompt=prompt,
                    root_seed=root_seed,
                    state=state,
                    cfg=cfg,
                    fps=fps,
                    overlap_frames=overlap_frames,
                    seed_index_stride=seed_index_stride,
                    scheduler_kind=scheduler_kind,
                )
            )
        _write_rows(part_manifest, sorted(rows, key=lambda r: int(r["path_id"])))
        print(f"[chunk-branch]   wrote {len(rows)} leaves and {part_manifest}")

    print(f"[chunk-branch] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
