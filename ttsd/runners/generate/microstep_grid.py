"""Generate a grid of Wan 2.2 runs with local extra denoising microsteps.

Each variant is written as an independent VBench-compatible run:

    runs/microstep_grid/<run_id>/<variant>/<prompt_id>/seed<NNNN>/video.mp4

Pass a single variant directory to ``ttsd.eval.vbench --run``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ttsd.models.microsteps import MicrostepSchedule, build_microstep_schedule
from ttsd.models.wan22_adapter import Wan22Adapter
from ttsd.runners.generate.baseline import _save_video, _snapshot_step_indices


@dataclass(frozen=True)
class GridVariant:
    name: str
    extra_steps_by_step: dict[int, int]


@dataclass
class MicrostepRunMeta:
    prompt_id: str
    prompt_text: str
    axis: str
    seed: int
    variant: str
    extra_steps_by_step: dict[int, int]
    index_base: int
    model: str
    height: int
    width: int
    num_frames: int
    base_num_inference_steps: int
    effective_num_inference_steps: int
    guidance_scale: float
    scheduler: str
    timestamp: str
    microstep_schedule: dict


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _variant_name(extra_steps_by_step: dict[int, int]) -> str:
    if not extra_steps_by_step:
        return "baseline"
    parts = []
    for step in sorted(extra_steps_by_step, reverse=True):
        parts.append(f"s{step:02d}x{extra_steps_by_step[step]:02d}")
    return "_".join(parts)


def _coerce_step_map(value: dict | None) -> dict[int, int]:
    return {int(k): int(v) for k, v in (value or {}).items() if int(v) > 0}


def _build_variants(grid_cfg: dict) -> list[GridVariant]:
    variants: dict[str, GridVariant] = {}

    if grid_cfg.get("include_baseline", True):
        variants["baseline"] = GridVariant("baseline", {})

    anchors = [int(v) for v in grid_cfg.get("anchors", [])]
    extra_counts = [int(v) for v in grid_cfg.get("extra_steps", [])]
    if anchors and extra_counts:
        min_active = int(grid_cfg.get("min_active_anchors", 1))
        max_active = int(grid_cfg.get("max_active_anchors", 1))
        if min_active < 1:
            raise ValueError(f"min_active_anchors must be >= 1, got {min_active}")
        if max_active < min_active:
            raise ValueError(
                f"max_active_anchors must be >= min_active_anchors, got {max_active} < {min_active}"
            )
        max_active = min(max_active, len(anchors))
        for active in range(min_active, max_active + 1):
            for selected_anchors in itertools.combinations(anchors, active):
                for selected_counts in itertools.product(extra_counts, repeat=active):
                    plan = dict(zip(selected_anchors, selected_counts, strict=True))
                    name = _variant_name(plan)
                    variants.setdefault(name, GridVariant(name, plan))

    for explicit in grid_cfg.get("variants", []):
        plan = _coerce_step_map(explicit.get("extra_steps_by_step"))
        name = explicit.get("name") or _variant_name(plan)
        if name in variants and variants[name].extra_steps_by_step != plan:
            raise ValueError(f"Duplicate variant name with different plan: {name}")
        variants[name] = GridVariant(name, plan)

    return list(variants.values())


def _write_variant_manifest(path: Path, variants: list[GridVariant]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "extra_steps_by_step"])
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "variant": variant.name,
                    "extra_steps_by_step": json.dumps(variant.extra_steps_by_step, sort_keys=True),
                }
            )


def _select_prompts(cfg: dict, *, prompt_ids: str, limit_prompts: int | None, smoke: bool) -> list[dict]:
    prompts = _load_prompts(cfg["prompts"]["source"])
    wanted = {p.strip() for p in prompt_ids.split(",") if p.strip()}
    if wanted:
        prompts = [p for p in prompts if p["id"] in wanted]
    if smoke:
        prompts = prompts[:1]
    elif limit_prompts:
        prompts = prompts[:limit_prompts]
    return prompts


def _select_seeds(
    prompt: dict,
    cfg: dict,
    *,
    seed_idxs: str,
    limit_seeds: int | None,
    smoke: bool,
) -> list[int]:
    explicit = [int(s.strip()) for s in seed_idxs.split(",") if s.strip()]
    default = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]
    seeds = explicit or prompt.get("seeds") or default
    if smoke:
        seeds = seeds[:1]
    elif limit_seeds:
        seeds = seeds[:limit_seeds]
    return [int(s) for s in seeds]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true", help="One prompt, one seed, first selected variant.")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--prompt-ids", default="", help="Comma-separated prompt ids to include.")
    parser.add_argument("--seed-idxs", default="", help="Comma-separated seed values to include.")
    parser.add_argument("--variant-names", default="", help="Comma-separated variant names to include.")
    parser.add_argument("--list-variants", action="store_true", help="Print expanded variants and exit.")
    parser.add_argument("--device", default=None, help="Override config model.device, e.g. cuda:1.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = dict(cfg["model"])
    if args.device:
        model_cfg["device"] = args.device
    gen_cfg = cfg["generation"]
    grid_cfg = cfg["microstep_grid"]
    out_cfg = cfg["output"]
    snap_cfg = cfg.get("snapshots", {})

    variants = _build_variants(grid_cfg)
    if args.variant_names:
        wanted = {v.strip() for v in args.variant_names.split(",") if v.strip()}
        variants = [v for v in variants if v.name in wanted]
        missing = sorted(wanted - {v.name for v in variants})
        if missing:
            raise SystemExit(f"Unknown variant name(s): {missing}")
    if args.smoke:
        variants = variants[:1]
    if args.list_variants:
        for variant in variants:
            print(f"{variant.name}\t{json.dumps(variant.extra_steps_by_step, sort_keys=True)}")
        return
    if not variants:
        raise SystemExit("No microstep variants selected")

    prompts = _select_prompts(
        cfg,
        prompt_ids=args.prompt_ids,
        limit_prompts=args.limit_prompts,
        smoke=args.smoke,
    )
    if not prompts:
        raise SystemExit("No prompts selected")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "unipc")
    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )

    height, width = gen_cfg["resolution"]
    base_steps = int(gen_cfg["num_inference_steps"])
    index_base = int(grid_cfg.get("index_base", 1))
    num_train_timesteps = int(grid_cfg.get("num_train_timesteps", 1000))
    schedules: dict[str, MicrostepSchedule] = {
        variant.name: build_microstep_schedule(
            base_num_steps=base_steps,
            extra_steps_by_step=variant.extra_steps_by_step,
            index_base=index_base,
            num_train_timesteps=num_train_timesteps,
        )
        for variant in variants
    }

    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snap_path = run_root / "config.snapshot.yaml"
    if not snap_path.exists():
        snap_path.write_text(yaml.safe_dump(cfg))
    _write_variant_manifest(run_root / "variant_manifest.csv", variants)

    work: list[tuple[GridVariant, dict, int]] = []
    for prompt in prompts:
        seeds = _select_seeds(
            prompt,
            cfg,
            seed_idxs=args.seed_idxs,
            limit_seeds=args.limit_seeds,
            smoke=args.smoke,
        )
        for seed in seeds:
            for variant in variants:
                work.append((variant, prompt, seed))

    total_jobs = len(work)
    if args.num_shards > 1:
        work = [item for i, item in enumerate(work) if i % args.num_shards == args.shard_index]

    print(f"[microstep_grid] run_root={run_root}")
    print(f"[microstep_grid] variants={len(variants)} prompts={len(prompts)} jobs={total_jobs}")
    if args.num_shards > 1:
        print(f"[microstep_grid] shard {args.shard_index}/{args.num_shards}: {len(work)} jobs")
    print(f"[microstep_grid] base_steps={base_steps} index_base={index_base}")

    save_latents = bool(out_cfg.get("save_latents", False))
    for variant, prompt, seed in work:
        schedule = schedules[variant.name]
        out_dir = run_root / variant.name / prompt["id"] / f"seed{seed:04d}"
        if (out_dir / "DONE").exists():
            print(f"[microstep_grid] SKIP {variant.name} {prompt['id']} seed={seed}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_latents:
            (out_dir / "latents").mkdir(exist_ok=True)

        print(
            f"[microstep_grid] RUN variant={variant.name} prompt={prompt['id']} "
            f"seed={seed} steps={schedule.effective_num_steps}"
        )

        snapshot_steps: list[int] = []
        if save_latents:
            snapshot_steps = _snapshot_step_indices(
                schedule.effective_num_steps,
                int(snap_cfg.get("every_n_steps", 5)),
                list(snap_cfg.get("also_keep", [])),
            )

        result = adapter.generate(
            prompt=prompt["text"],
            seed=seed,
            num_frames=gen_cfg["num_frames"],
            height=height,
            width=width,
            num_inference_steps=base_steps,
            guidance_scale=gen_cfg["guidance_scale"],
            snapshot_steps=snapshot_steps,
            microstep_schedule=schedule if variant.extra_steps_by_step else None,
        )

        if out_cfg.get("save_video", True):
            _save_video(result.frames, out_dir / "video.mp4")
        if save_latents:
            for step_idx, latent in result.latents_by_step.items():
                torch.save(latent, out_dir / "latents" / f"step_{step_idx:03d}.pt")

        meta = MicrostepRunMeta(
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            axis=prompt.get("axis", ""),
            seed=seed,
            variant=variant.name,
            extra_steps_by_step=variant.extra_steps_by_step,
            index_base=index_base,
            model=model_cfg["name"],
            height=height,
            width=width,
            num_frames=gen_cfg["num_frames"],
            base_num_inference_steps=base_steps,
            effective_num_inference_steps=schedule.effective_num_steps,
            guidance_scale=gen_cfg["guidance_scale"],
            scheduler=scheduler_kind,
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            microstep_schedule=schedule.to_dict(),
        )
        (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        (out_dir / "DONE").touch()

    print(f"[microstep_grid] done. outputs under {run_root}")
    print(f"[microstep_grid] VBench example: python -m ttsd.eval.vbench --run {run_root / variants[0].name}")


if __name__ == "__main__":
    main()
