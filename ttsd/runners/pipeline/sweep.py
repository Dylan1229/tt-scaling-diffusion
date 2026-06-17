"""Pipeline sweep runner — drives Orchestrator over a (prompt × seed × strategy) grid.

Sharded for multi-GPU parallelism via --shard-index / --num-shards (combine with
CUDA_VISIBLE_DEVICES at launch time). Resumable: skips any work item whose
result.json already exists. Reuses one Orchestrator per (strategy × shard) so
the heavy Wan22 + verifier loads happen once per strategy, not once per item.

Usage:
    python -m ttsd.runners.pipeline.sweep \\
        --config configs/pipeline/sweeps/efdi_vs_bon_5x5.yaml \\
        --sweep-id efdi_vs_bon_5x5_20260617 \\
        --shard-index 0 --num-shards 4

Sweep config format (YAML):

    prompts:
      source: ttsd.prompts.sweep_v2:SWEEP_V2_PROMPTS
      limit: 5                # optional — slice the prompt list

    seeds:
      base: 0
      count: 5

    strategies:                # one entry per pipeline-config to run
      - id: efdi               # short name (drives output path)
        config: configs/pipeline/efdi_dino.yaml
      - id: bon4
        config: configs/pipeline/bon_dino.yaml

    output:
      sweep_root: runs/pipeline_sweeps    # parent dir for sweep_id subdir

Output layout:
    <sweep_root>/<sweep_id>/<strategy_id>/<prompt_id>__seed<NNNN>/
        ├── video.mp4
        ├── events.jsonl
        ├── result.json
        └── config.snapshot.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ttsd.pipeline import Orchestrator, load_config


def _load_prompts(spec: str) -> list[dict]:
    module_path, attr = spec.split(":")
    module = importlib.import_module(module_path)
    return list(getattr(module, attr))


def _build_work_groups(
    prompts: list[dict],
    seeds: list[int],
    strategies: list[dict],
) -> list[list[tuple[str, str, dict, int]]]:
    """Each *group* = all strategies for one (prompt, seed) tuple.

    Sharding assigns whole groups to shards (not individual items). This keeps
    the per-shard strategy mix balanced even when len(strategies) shares a
    common factor with num_shards (the naive item-level `i % num_shards` shard
    can clump all of one strategy onto a few shards in that case).
    """
    groups: list[list[tuple[str, str, dict, int]]] = []
    for prompt in prompts:
        for seed in seeds:
            group: list[tuple[str, str, dict, int]] = []
            for s in strategies:
                group.append((s["id"], s["config"], prompt, seed))
            groups.append(group)
    return groups


def _shard_groups(
    groups: list[list],
    shard_index: int,
    num_shards: int,
) -> list:
    if num_shards <= 1:
        return [item for group in groups for item in group]
    selected = [g for i, g in enumerate(groups) if i % num_shards == shard_index]
    return [item for g in selected for item in g]


def _free_orchestrator(orch: Orchestrator | None) -> None:
    """Best-effort GPU memory release between strategy switches."""
    if orch is None:
        return
    try:
        impl = getattr(orch.adapter, "_impl", None)
        if impl is not None and getattr(impl, "_pipe", None) is not None:
            del impl._pipe
            impl._pipe = None
    except Exception:    # pragma: no cover — defensive
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--sweep-id", type=str, default=None,
                   help="Shared identifier for the sweep root; all shards "
                        "must use the same one. Default = sweep_<timestamp>.")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit-prompts", type=int, default=None,
                   help="Test-mode: cap the prompt list (after the YAML's `limit`).")
    p.add_argument("--limit-seeds", type=int, default=None,
                   help="Test-mode: cap the seed count.")
    p.add_argument("--limit-strategies", type=int, default=None,
                   help="Test-mode: cap how many strategies to run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the work list and exit without invoking the orchestrator.")
    args = p.parse_args(argv)
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}"
        )

    sweep_cfg = yaml.safe_load(args.config.read_text())

    # Prompts.
    prompt_source = sweep_cfg["prompts"]["source"]
    prompts = _load_prompts(prompt_source)
    yaml_limit = sweep_cfg["prompts"].get("limit")
    if yaml_limit is not None:
        prompts = prompts[: int(yaml_limit)]
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    # Seeds.
    base = int(sweep_cfg["seeds"]["base"])
    count = int(sweep_cfg["seeds"]["count"])
    if args.limit_seeds is not None:
        count = min(count, args.limit_seeds)
    seeds = [base + i for i in range(count)]

    # Strategies.
    strategies = list(sweep_cfg["strategies"])
    if args.limit_strategies is not None:
        strategies = strategies[: args.limit_strategies]

    # Sweep ID and output root.
    sweep_id = args.sweep_id or f"sweep_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root = Path(sweep_cfg["output"]["sweep_root"]) / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    # Build + shard work. Sharding granularity = (prompt, seed) group, so
    # each shard runs every strategy for its assigned (prompt, seed) subset.
    groups = _build_work_groups(prompts, seeds, strategies)
    my_work = _shard_groups(groups, args.shard_index, args.num_shards)
    total_items = sum(len(g) for g in groups)

    shard_tag = f"shard {args.shard_index}/{args.num_shards}"
    print(f"[sweep] sweep_id   = {sweep_id}")
    print(f"[sweep] sweep_root = {sweep_root}")
    print(f"[sweep] prompts    = {len(prompts)} ({[p['id'] for p in prompts]})")
    print(f"[sweep] seeds      = {seeds}")
    print(f"[sweep] strategies = {[s['id'] for s in strategies]}")
    print(f"[sweep] total work = {total_items} ({shard_tag}: {len(my_work)})")

    if args.dry_run:
        print("[sweep] --dry-run; not invoking orchestrator")
        for sid, scfg, prompt, seed in my_work:
            print(f"  [{sid}] {prompt['id']}__seed{seed:04d} ({scfg})")
        return

    if not my_work:
        print("[sweep] this shard has no work; exiting cleanly")
        return

    # Group by strategy so we reload Wan22 only on strategy switches.
    by_strategy: dict[str, list[tuple[str, dict, int]]] = defaultdict(list)
    for sid, scfg, prompt, seed in my_work:
        by_strategy[sid].append((scfg, prompt, seed))

    # Process each strategy: instantiate Orchestrator once, iterate, free.
    n_done = 0
    n_skip = 0
    n_fail = 0
    for sid, items in by_strategy.items():
        scfg_path = items[0][0]
        try:
            base_pipeline_cfg = load_config(scfg_path)
        except Exception as exc:
            print(f"[sweep] ERROR loading config {scfg_path} for strategy '{sid}': {exc}",
                  file=sys.stderr)
            n_fail += len(items)
            continue
        strategy_root = sweep_root / sid
        strategy_root.mkdir(parents=True, exist_ok=True)
        pipeline_cfg = dataclasses.replace(base_pipeline_cfg, output_root=str(strategy_root))

        print(f"\n[sweep] === strategy '{sid}' :: {len(items)} item(s) ===")
        orch: Orchestrator | None = None
        try:
            for scfg, prompt, seed in items:
                run_id = f"{prompt['id']}__seed{seed:04d}"
                out_dir = strategy_root / run_id
                if (out_dir / "result.json").exists():
                    print(f"[sweep] SKIP {sid}/{run_id} (result.json already exists)")
                    n_skip += 1
                    continue
                if orch is None:
                    # Construct lazily so empty-shard runs don't load Wan22.
                    orch = Orchestrator(pipeline_cfg)
                try:
                    result = orch.run(prompt=prompt["text"], seed=seed, run_id=run_id)
                except Exception as exc:    # one bad run shouldn't kill the shard
                    print(f"[sweep] ERROR {sid}/{run_id}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    n_fail += 1
                    continue
                tag = "OK" if result.success else "FAIL"
                print(f"[sweep] {tag} {sid}/{run_id}: score={result.final_score} "
                      f"trials={result.n_trials} terminating={result.terminating_trial} "
                      f"wall_s={result.cost.wall_clock_s:.1f}")
                n_done += 1
        finally:
            _free_orchestrator(orch)
            orch = None

    print(f"\n[sweep] === shard summary === done={n_done} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
