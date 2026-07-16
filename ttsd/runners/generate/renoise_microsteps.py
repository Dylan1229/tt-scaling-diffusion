"""Run Wan 2.2 with a local Renoise + replay-microsteps intervention."""

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

from ttsd.models.wan22_renoise_microsteps import WanRenoiseMicrostepsConfig
from ttsd.runners.generate.baseline import _save_video, _snapshot_step_indices


@dataclass
class RenoiseMicrostepsRunMeta:
    variant: str
    prompt_id: str
    prompt_text: str
    axis: str
    seed: int
    model: str
    scheduler: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    trigger_step: int
    rollback_to_step: int
    extra_microsteps: int
    noise_scale: float
    index_base: int
    snapshot_steps: list[int]
    elapsed_seconds: float
    timestamp: str


@dataclass(frozen=True)
class RenoiseVariant:
    name: str
    trigger_step: int
    rollback_to_step: int
    extra_microsteps: int
    noise_scale: float
    index_base: int


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _select_prompts(cfg: dict, *, smoke: bool, limit_prompts: int | None, prompt_ids: str) -> list[dict]:
    prompts = _load_prompts(cfg["prompts"]["source"])
    wanted = {p.strip() for p in prompt_ids.split(",") if p.strip()}
    if wanted:
        prompts = [prompt for prompt in prompts if prompt["id"] in wanted]
    if smoke:
        return prompts[:1]
    if limit_prompts:
        return prompts[:limit_prompts]
    return prompts


def _load_variants(cfg: dict) -> tuple[list[RenoiseVariant], bool]:
    if "renoise_grid" not in cfg:
        item = cfg["renoise_microsteps"]
        return [
            RenoiseVariant(
                name="",
                trigger_step=int(item["trigger_step"]),
                rollback_to_step=int(item["rollback_to_step"]),
                extra_microsteps=int(item["extra_microsteps"]),
                noise_scale=float(item.get("noise_scale", 1.0)),
                index_base=int(item.get("index_base", 1)),
            )
        ], False

    grid = cfg["renoise_grid"]
    index_base = int(grid.get("index_base", 1))
    default_distance = int(grid.get("rollback_distance", 2))
    default_extra = int(grid.get("extra_microsteps", 5))
    default_noise_scale = float(grid.get("noise_scale", 1.0))
    variants: list[RenoiseVariant] = []
    for item in grid["variants"]:
        trigger_step = int(item["trigger_step"])
        rollback_to_step = int(
            item.get("rollback_to_step", trigger_step - default_distance)
        )
        extra_microsteps = int(item.get("extra_microsteps", default_extra))
        noise_scale = float(item.get("noise_scale", default_noise_scale))
        name = item.get(
            "name",
            f"s{trigger_step:02d}to{rollback_to_step:02d}x{extra_microsteps:02d}",
        )
        variants.append(
            RenoiseVariant(
                name=str(name),
                trigger_step=trigger_step,
                rollback_to_step=rollback_to_step,
                extra_microsteps=extra_microsteps,
                noise_scale=noise_scale,
                index_base=int(item.get("index_base", index_base)),
            )
        )
    if not variants:
        raise ValueError("renoise_grid.variants must not be empty")
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("renoise_grid variant names must be unique")
    return variants, True


def _select_variants(variants: list[RenoiseVariant], wanted_csv: str) -> list[RenoiseVariant]:
    wanted = {part.strip() for part in wanted_csv.split(",") if part.strip()}
    if not wanted:
        return variants
    selected = [variant for variant in variants if variant.name in wanted]
    missing = sorted(wanted - {variant.name for variant in selected})
    if missing:
        raise SystemExit(f"Unknown Renoise variants: {', '.join(missing)}")
    return selected


def _write_variant_manifest(run_root: Path, variants: list[RenoiseVariant]) -> None:
    path = run_root / "variant_manifest.csv"
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "trigger_step",
                "rollback_to_step",
                "extra_microsteps",
                "noise_scale",
                "index_base",
            ],
        )
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "variant": variant.name,
                    "trigger_step": variant.trigger_step,
                    "rollback_to_step": variant.rollback_to_step,
                    "extra_microsteps": variant.extra_microsteps,
                    "noise_scale": variant.noise_scale,
                    "index_base": variant.index_base,
                }
            )


def _flatten_work(
    variants: list[RenoiseVariant],
    prompts: list[dict],
    cfg: dict,
    *,
    smoke: bool,
    limit_seeds: int | None,
    seed_idxs: str,
) -> list[tuple[RenoiseVariant, dict, int]]:
    explicit = [int(s.strip()) for s in seed_idxs.split(",") if s.strip()]
    default = [cfg["seeds"]["base"] + i for i in range(cfg["seeds"]["count"])]
    work: list[tuple[RenoiseVariant, dict, int]] = []
    for variant in variants:
        for prompt in prompts:
            seeds = explicit or prompt.get("seeds") or default
            if smoke:
                seeds = seeds[:1]
            if limit_seeds:
                seeds = seeds[:limit_seeds]
            for seed in seeds:
                work.append((variant, prompt, int(seed)))
    return work


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--prompt-ids", default="", help="Comma-separated prompt ids to include.")
    parser.add_argument("--seed-idxs", default="", help="Comma-separated seed values to include.")
    parser.add_argument("--variants", default="", help="Comma-separated grid variants to include.")
    parser.add_argument("--device", default=None, help="Override config model.device, e.g. cuda:1.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )

    cfg = yaml.safe_load(args.config.read_text())

    from ttsd.models.wan22_adapter import Wan22Adapter

    model_cfg = dict(cfg["model"])
    if args.device:
        model_cfg["device"] = args.device
    gen_cfg = cfg["generation"]
    snap_cfg = cfg.get("snapshots", {})
    out_cfg = cfg["output"]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_cfg["dtype"]]
    scheduler_kind = model_cfg.get("scheduler", "euler")
    if scheduler_kind == "unipc":
        raise SystemExit("Renoise+microsteps requires model.scheduler=euler or euler_sde, not unipc.")

    adapter = Wan22Adapter(
        model_path=model_cfg["path"],
        dtype=dtype,
        device=model_cfg["device"],
        scheduler_kind=scheduler_kind,
    )

    prompts = _select_prompts(
        cfg,
        smoke=args.smoke,
        limit_prompts=args.limit_prompts,
        prompt_ids=args.prompt_ids,
    )
    if not prompts:
        raise SystemExit("No prompts selected")

    variants, grid_mode = _load_variants(cfg)
    variants = _select_variants(variants, args.variants)
    work = _flatten_work(
        variants,
        prompts,
        cfg,
        smoke=args.smoke,
        limit_seeds=args.limit_seeds,
        seed_idxs=args.seed_idxs,
    )
    total_jobs = len(work)
    if args.num_shards > 1:
        work = [item for i, item in enumerate(work) if i % args.num_shards == args.shard_index]

    snapshot_steps = _snapshot_step_indices(
        gen_cfg["num_inference_steps"],
        snap_cfg.get("every_n_steps", gen_cfg["num_inference_steps"]),
        snap_cfg.get("also_keep", []),
    )
    if not out_cfg.get("save_latents", True):
        snapshot_steps = []

    run_id = args.run_id or out_cfg.get("run_id") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(out_cfg["root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snap_path = run_root / "config.snapshot.yaml"
    if not snap_path.exists():
        snap_path.write_text(yaml.safe_dump(cfg))
    if grid_mode:
        _write_variant_manifest(run_root, variants)

    print(f"[renoise_microsteps] run_root={run_root}")
    print(f"[renoise_microsteps] scheduler={scheduler_kind}")
    print(f"[renoise_microsteps] variants={[variant.name for variant in variants]}")
    print(f"[renoise_microsteps] total jobs={total_jobs}; this shard={len(work)}")

    height, width = gen_cfg["resolution"]
    for variant, prompt, seed in work:
        sample_root = run_root / variant.name if grid_mode else run_root
        out_dir = sample_root / prompt["id"] / f"seed{seed:04d}"
        if (out_dir / "DONE").exists():
            print(
                f"[renoise_microsteps] SKIP {variant.name or 'single'} "
                f"{prompt['id']} seed={seed} (already done)"
            )
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "latents").mkdir(exist_ok=True)

        print(
            f"[renoise_microsteps] {variant.name or 'single'} "
            f"{prompt['id']} seed={seed} :: {prompt['text'][:60]}"
        )
        run_config = WanRenoiseMicrostepsConfig(
            trigger_step=variant.trigger_step,
            rollback_to_step=variant.rollback_to_step,
            extra_microsteps=variant.extra_microsteps,
            noise_scale=variant.noise_scale,
            index_base=variant.index_base,
            output_type=out_cfg.get("output_type", "np"),
            trace_path=out_dir / "renoise_trace.jsonl",
        )
        started = time.perf_counter()
        result = adapter.generate_with_renoise_microsteps(
            prompt=prompt["text"],
            seed=seed,
            num_frames=gen_cfg["num_frames"],
            height=height,
            width=width,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            snapshot_steps=snapshot_steps,
            renoise_config=run_config,
        )
        elapsed_seconds = time.perf_counter() - started

        if out_cfg.get("save_video", True):
            _save_video(result.frames, out_dir / "video.mp4")
        if out_cfg.get("save_latents", True):
            for step_idx, latent in result.latents_by_step.items():
                torch.save(latent, out_dir / "latents" / f"step_{step_idx:03d}.pt")
        (out_dir / "renoise_trace.json").write_text(json.dumps(result.search_trace, indent=2))

        meta = RenoiseMicrostepsRunMeta(
            variant=variant.name,
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            axis=prompt.get("axis", ""),
            seed=seed,
            model=model_cfg["name"],
            scheduler=scheduler_kind,
            height=height,
            width=width,
            num_frames=gen_cfg["num_frames"],
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            trigger_step=run_config.trigger_step,
            rollback_to_step=run_config.rollback_to_step,
            extra_microsteps=run_config.extra_microsteps,
            noise_scale=run_config.noise_scale,
            index_base=run_config.index_base,
            snapshot_steps=snapshot_steps,
            elapsed_seconds=elapsed_seconds,
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        )
        (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        (out_dir / "DONE").touch()

    print(f"[renoise_microsteps] done. outputs under {run_root}")


if __name__ == "__main__":
    main()
