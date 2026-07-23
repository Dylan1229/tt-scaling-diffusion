"""Generate late-stage Best-of-M branches from a shared Wan denoising prefix.

Usage:
    python -m ttsd.runners.generate.late_branching \
        --config configs/late_branching_s35_wan22_480p.yaml --smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ttsd.runners.generate.baseline import _save_video
from ttsd.search.late_branching import LateBranchConfig


@dataclass
class LateBranchRunMeta:
    experiment: str
    prompt_id: str
    prompt_text: str
    axis: str
    seed: int
    root_seed: int
    candidate_index: int
    branch_kind: str
    perturbation_seed: int | None
    model: str
    scheduler: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    branch_step: int
    branch_step_index: int
    branch_sigma: float
    perturbation_scale: float
    perturbation_std: float
    noise_seed_stride: int
    posterior_mean_steps: list[int]
    total_candidates: int
    denoising_step_equivalents: int
    compute_ratio_vs_baseline: float
    elapsed_seconds_for_group: float
    timestamp: str


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    return getattr(importlib.import_module(module_path), attr)


def _select_prompts(
    cfg: dict,
    *,
    smoke: bool,
    limit_prompts: int | None,
    prompt_ids: str,
) -> list[dict]:
    prompts = _load_prompts(cfg["prompts"]["source"])
    wanted = {part.strip() for part in prompt_ids.split(",") if part.strip()}
    if wanted:
        prompts = [prompt for prompt in prompts if prompt["id"] in wanted]
        missing = wanted - {prompt["id"] for prompt in prompts}
        if missing:
            raise SystemExit(f"Unknown prompt ids: {', '.join(sorted(missing))}")
    if smoke:
        prompts = prompts[:1]
    if limit_prompts is not None:
        prompts = prompts[:limit_prompts]
    return prompts


def _flatten_work(
    prompts: list[dict],
    cfg: dict,
    *,
    smoke: bool,
    limit_seeds: int | None,
    seed_idxs: str,
) -> list[tuple[dict, int]]:
    explicit = [int(part.strip()) for part in seed_idxs.split(",") if part.strip()]
    default = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]
    work: list[tuple[dict, int]] = []
    for prompt in prompts:
        seeds = explicit or prompt.get("seeds") or default
        if smoke:
            seeds = seeds[:1]
        if limit_seeds is not None:
            seeds = seeds[:limit_seeds]
        work.extend((prompt, int(seed)) for seed in seeds)
    return work


def _load_pairs(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"prompt_id", "root_seed"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        pairs = [
            (str(row["prompt_id"]).strip(), int(row["root_seed"])) for row in reader
        ]
    if not pairs:
        raise ValueError(f"{path} contains no prompt/root pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} contains duplicate prompt/root pairs")
    return pairs


def _filter_work_by_pairs(
    work: list[tuple[dict, int]], pairs: list[tuple[str, int]]
) -> list[tuple[dict, int]]:
    lookup = {(prompt["id"], seed): (prompt, seed) for prompt, seed in work}
    missing = [pair for pair in pairs if pair not in lookup]
    if missing:
        formatted = ", ".join(f"{prompt_id}/{seed}" for prompt_id, seed in missing[:8])
        raise ValueError(f"Unknown prompt/root pairs: {formatted}")
    return [lookup[pair] for pair in pairs]


def _candidate_seed(root_seed: int, candidate_index: int, stride: int) -> int:
    if root_seed < 0:
        raise ValueError("root seeds must be non-negative")
    if not 0 <= candidate_index < stride:
        raise ValueError(
            f"candidate index {candidate_index} must be smaller than stride {stride}"
        )
    return root_seed * stride + candidate_index


def _write_group_manifest(run_root: Path, prompt_id: str, root_seed: int, paths: list[Path]) -> None:
    entries = []
    for path in paths:
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text())
        entries.append(
            {
                "candidate_index": meta["candidate_index"],
                "candidate_seed": meta["seed"],
                "branch_kind": meta["branch_kind"],
                "video": str((path / "video.mp4").relative_to(run_root)),
                "meta": str(meta_path.relative_to(run_root)),
            }
        )
    manifest_dir = run_root / "_branch_groups"
    manifest_dir.mkdir(exist_ok=True)
    manifest_path = manifest_dir / f"{prompt_id}_rootseed{root_seed:04d}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "root_seed": root_seed,
                "candidates": sorted(entries, key=lambda row: row["candidate_index"]),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--prompt-ids", default="", help="Comma-separated prompt ids.")
    parser.add_argument("--seed-idxs", default="", help="Comma-separated root seeds.")
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="CSV with explicit prompt_id,root_seed pairs.",
    )
    parser.add_argument("--device", default=None, help="Override model.device, e.g. cuda:1.")
    parser.add_argument("--branch-step", type=int, default=None)
    parser.add_argument("--num-noise-branches", type=int, default=None)
    parser.add_argument("--perturbation-scale", type=float, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    branch_cfg = cfg["late_branching"]
    out_cfg = cfg["output"]
    if args.device is not None:
        model_cfg["device"] = args.device
    if args.branch_step is not None:
        branch_cfg["branch_step"] = args.branch_step
    if args.num_noise_branches is not None:
        branch_cfg["num_noise_branches"] = args.num_noise_branches
    if args.perturbation_scale is not None:
        branch_cfg["perturbation_scale"] = args.perturbation_scale

    late_config = LateBranchConfig(
        branch_step=int(branch_cfg["branch_step"]),
        num_noise_branches=int(branch_cfg["num_noise_branches"]),
        perturbation_scale=float(branch_cfg["perturbation_scale"]),
        include_batched_control=bool(branch_cfg.get("include_batched_control", True)),
        noise_seed_offset=int(branch_cfg.get("noise_seed_offset", 10_000_000)),
        noise_seed_stride=(
            int(branch_cfg["noise_seed_stride"])
            if branch_cfg.get("noise_seed_stride") is not None
            else None
        ),
    )
    late_config.validate(int(gen_cfg["num_inference_steps"]))
    if not late_config.include_batched_control:
        raise SystemExit(
            "This feasibility experiment requires include_batched_control: true"
        )

    candidate_seed_stride = int(out_cfg.get("candidate_seed_stride", 100))
    if late_config.total_branches > candidate_seed_stride:
        raise SystemExit(
            "output.candidate_seed_stride must be at least the total number of candidates"
        )

    prompts = _select_prompts(
        cfg,
        smoke=args.smoke,
        limit_prompts=args.limit_prompts,
        prompt_ids=args.prompt_ids,
    )
    work = _flatten_work(
        prompts,
        cfg,
        smoke=args.smoke,
        limit_seeds=args.limit_seeds,
        seed_idxs=args.seed_idxs,
    )
    if args.pairs_file is not None:
        work = _filter_work_by_pairs(work, _load_pairs(args.pairs_file))
    if not work:
        raise SystemExit("No prompt/seed groups selected")
    total_groups = len(work)
    if args.num_shards > 1:
        work = [item for i, item in enumerate(work) if i % args.num_shards == args.shard_index]

    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_root / "config.snapshot.yaml"
    if not snapshot_path.exists():
        snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    from ttsd.models.wan22_adapter import Wan22Adapter

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "unipc")
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )

    print(f"[late_branching] run_root={run_root}")
    print(
        f"[late_branching] fork=after-step-{late_config.branch_step}; "
        f"batched_control=1; noisy={late_config.num_noise_branches}; "
        f"scale={late_config.perturbation_scale}"
    )
    print(
        f"[late_branching] total groups={total_groups}; this shard={len(work)} "
        f"({args.shard_index}/{args.num_shards})"
    )

    height, width = gen_cfg["resolution"]
    posterior_mean_offsets = [
        int(offset) for offset in branch_cfg.get("posterior_mean_offsets", [])
    ]
    for prompt, root_seed in work:
        candidate_paths = [
            run_root
            / prompt["id"]
            / f"seed{_candidate_seed(root_seed, index, candidate_seed_stride):06d}"
            for index in range(late_config.total_branches)
        ]
        if all((path / "DONE").exists() for path in candidate_paths):
            print(f"[late_branching] SKIP {prompt['id']} root_seed={root_seed}")
            _write_group_manifest(run_root, prompt["id"], root_seed, candidate_paths)
            continue

        print(
            f"[late_branching] RUN {prompt['id']} root_seed={root_seed} :: "
            f"{prompt['text'][:70]}"
        )
        started = time.perf_counter()
        result = adapter.generate_with_late_branches(
            prompt=prompt["text"],
            seed=root_seed,
            branch_config=late_config,
            num_frames=gen_cfg["num_frames"],
            height=height,
            width=width,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            posterior_mean_offsets=posterior_mean_offsets,
            decode_batch_size=int(branch_cfg.get("decode_batch_size", 8)),
        )
        elapsed_seconds = time.perf_counter() - started
        timestamp = dt.datetime.now().isoformat(timespec="seconds")

        for frames, spec, out_dir in zip(
            result.frames_by_branch, result.branch_specs, candidate_paths, strict=True
        ):
            if (out_dir / "DONE").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_cfg.get("save_video", True):
                _save_video(frames, out_dir / "video.mp4", fps=int(out_cfg.get("fps", 16)))
            for step, batched_posterior in result.posterior_means_by_step.items():
                posterior_dir = out_dir / "posterior_means"
                posterior_dir.mkdir(exist_ok=True)
                torch.save(
                    batched_posterior[spec.index : spec.index + 1].contiguous(),
                    posterior_dir / f"step_{step:03d}.pt",
                )

            candidate_seed = _candidate_seed(root_seed, spec.index, candidate_seed_stride)
            meta = LateBranchRunMeta(
                experiment="late_branching_best_of_m",
                prompt_id=prompt["id"],
                prompt_text=prompt["text"],
                axis=prompt.get("axis", ""),
                seed=candidate_seed,
                root_seed=root_seed,
                candidate_index=spec.index,
                branch_kind=spec.kind,
                perturbation_seed=spec.perturbation_seed,
                model=model_cfg["name"],
                scheduler=scheduler_kind,
                height=height,
                width=width,
                num_frames=gen_cfg["num_frames"],
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                branch_step=result.branch_step,
                branch_step_index=result.branch_step - 1,
                branch_sigma=result.branch_sigma,
                perturbation_scale=late_config.perturbation_scale,
                perturbation_std=spec.perturbation_std,
                noise_seed_stride=late_config.resolved_noise_seed_stride,
                posterior_mean_steps=sorted(result.posterior_means_by_step),
                total_candidates=late_config.total_branches,
                denoising_step_equivalents=result.denoising_step_equivalents,
                compute_ratio_vs_baseline=(
                    result.denoising_step_equivalents / gen_cfg["num_inference_steps"]
                ),
                elapsed_seconds_for_group=elapsed_seconds,
                timestamp=timestamp,
            )
            (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
            (out_dir / "DONE").touch()

        _write_group_manifest(run_root, prompt["id"], root_seed, candidate_paths)
        print(
            f"[late_branching] DONE {prompt['id']} root_seed={root_seed} "
            f"in {elapsed_seconds:.1f}s"
        )

    print(f"[late_branching] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
