"""Compare frozen reward-model selectors on saved chunk-branch videos.

This evaluator does not generate videos and does not run VBench.  It scores
the saved branch leaves with a frozen reward model, selects one path per root,
then maps that selection back to existing VBench-Long CSVs.

The first supported backend is VideoReward from VideoAlign:
https://github.com/KwaiVGI/VideoAlign
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BRANCH_RUN = Path("runs/baseline_long/chunk_branch_i2v_all_prompts_s0_1")
DEFAULT_BRANCH_COMPARE = Path("runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1")
DEFAULT_OUTPUT_SUBDIR = "frozen_reward_selectors"

VBENCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_mean",
)

VIDEOREWARD_DIMS = ("Overall", "VQ", "MQ", "TA")


@dataclass(frozen=True)
class RootKey:
    prompt_id: str
    root_seed: int


@dataclass(frozen=True)
class ScoreConfig:
    branch_run: Path
    branch_compare: Path
    output_dir: Path
    prompt_ids: tuple[str, ...] | None
    root_seeds: tuple[int, ...] | None
    limit_roots: int | None
    force: bool
    device: str
    dtype: str
    batch_size: int
    num_frames: int
    max_pixels: int
    use_norm: bool
    videoreward_repo: Path
    videoreward_checkpoint: Path
    videoreward_compat_path: Path | None
    disable_flash_attn2: bool


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return statistics.fmean(vals)


def _parse_csv_tuple(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    values = _parse_csv_tuple(raw)
    if values is None:
        return None
    return tuple(int(item) for item in values)


def _load_manifest(branch_run: Path) -> list[dict[str, str]]:
    path = branch_run / "branch_manifest.csv"
    if not path.exists():
        raise SystemExit(f"Missing branch manifest: {path}")
    return _read_csv(path)


def _root_keys(rows: list[dict[str, str]], cfg: ScoreConfig) -> list[RootKey]:
    keys = sorted(
        {RootKey(row["prompt_id"], int(row["root_seed"])) for row in rows},
        key=lambda item: (item.prompt_id, item.root_seed),
    )
    if cfg.prompt_ids is not None:
        allowed = set(cfg.prompt_ids)
        keys = [key for key in keys if key.prompt_id in allowed]
    if cfg.root_seeds is not None:
        allowed_seeds = set(cfg.root_seeds)
        keys = [key for key in keys if key.root_seed in allowed_seeds]
    if cfg.limit_roots is not None:
        keys = keys[: cfg.limit_roots]
    return keys


def _filter_manifest(rows: list[dict[str, str]], roots: list[RootKey]) -> list[dict[str, str]]:
    allowed = {(root.prompt_id, root.root_seed) for root in roots}
    return [
        row
        for row in rows
        if (row["prompt_id"], int(row["root_seed"])) in allowed
    ]


def _absolute_video_path(branch_run: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = branch_run / path
    return path.resolve()


def _score_key(row: dict[str, str]) -> tuple[str, int, str]:
    return (row["prompt_id"], int(row["root_seed"]), row["path_bits"])


def _load_existing_scores(path: Path, force: bool) -> list[dict[str, str]]:
    if force or not path.exists():
        return []
    return _read_csv(path)


def _load_vbench_path_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    path = compare_dir / "chunk_branch_by_path.csv"
    if not path.exists():
        raise SystemExit(f"Missing VBench path CSV: {path}")
    scores: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in _read_csv(path):
        key = (row["prompt_id"], int(row["root_seed"]), row["path_bits"])
        scores[key][row["dimension"]] = float(row["score"])
    return scores


def _load_vbench_root_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    path = compare_dir / "chunk_branch_by_root.csv"
    if not path.exists():
        raise SystemExit(f"Missing VBench root CSV: {path}")
    return {
        (row["prompt_id"], int(row["root_seed"]), row["dimension"]): row
        for row in _read_csv(path)
    }


def _load_videoreward_inferencer(cfg: ScoreConfig):
    if cfg.videoreward_compat_path is not None:
        sys.path.insert(0, str(cfg.videoreward_compat_path))
    sys.path.insert(0, str(cfg.videoreward_repo))

    import torch
    import inference

    if cfg.disable_flash_attn2:
        original_training_config = inference.TrainingConfig

        def training_config_no_flash(*args, **kwargs):
            kwargs["disable_flash_attn2"] = True
            return original_training_config(*args, **kwargs)

        inference.TrainingConfig = training_config_no_flash

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[cfg.dtype]
    inferencer = inference.VideoVLMRewardInference(
        str(cfg.videoreward_checkpoint),
        device=cfg.device,
        dtype=dtype,
    )
    return inferencer, torch


def _score_videoreward(cfg: ScoreConfig, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output_path = cfg.output_dir / "video_reward_scores.csv"
    if cfg.force and output_path.exists():
        output_path.unlink()

    fieldnames = [
        "prompt_id",
        "prompt_text",
        "root_seed",
        "seed_idx",
        "path_bits",
        "path_id",
        "video_path",
        "VQ",
        "MQ",
        "TA",
        "Overall",
    ]

    existing_rows = _load_existing_scores(output_path, force=cfg.force)
    existing_keys = {_score_key(row) for row in existing_rows}
    pending_rows = [row for row in rows if _score_key(row) not in existing_keys]

    if pending_rows:
        inferencer, torch = _load_videoreward_inferencer(cfg)
        for start in range(0, len(pending_rows), cfg.batch_size):
            batch_rows = pending_rows[start : start + cfg.batch_size]
            video_paths = [
                str(_absolute_video_path(cfg.branch_run, row["video_path"]))
                for row in batch_rows
            ]
            prompts = [row["prompt_text"] for row in batch_rows]
            with torch.no_grad():
                rewards = inferencer.reward(
                    video_paths,
                    prompts,
                    num_frames=cfg.num_frames,
                    max_pixels=cfg.max_pixels,
                    use_norm=cfg.use_norm,
                )

            scored_rows: list[dict[str, str]] = []
            for row, reward in zip(batch_rows, rewards):
                scored_rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "prompt_text": row["prompt_text"],
                        "root_seed": row["root_seed"],
                        "seed_idx": row["seed_idx"],
                        "path_bits": row["path_bits"],
                        "path_id": row["path_id"],
                        "video_path": str(_absolute_video_path(cfg.branch_run, row["video_path"])),
                        "VQ": _fmt(float(reward["VQ"])),
                        "MQ": _fmt(float(reward["MQ"])),
                        "TA": _fmt(float(reward["TA"])),
                        "Overall": _fmt(float(reward["Overall"])),
                    }
                )
            _append_csv(output_path, scored_rows, fieldnames)
            existing_rows.extend(scored_rows)
            print(
                f"scored {min(start + cfg.batch_size, len(pending_rows))}/{len(pending_rows)} pending videos",
                flush=True,
            )

    existing_rows.sort(key=lambda row: (row["prompt_id"], int(row["root_seed"]), int(row["path_id"])))
    _write_csv(output_path, existing_rows, fieldnames)
    return existing_rows


def _best_by_reward(
    score_rows: list[dict[str, str]],
    reward_dim: str,
) -> dict[RootKey, dict[str, str]]:
    grouped: dict[RootKey, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        grouped[RootKey(row["prompt_id"], int(row["root_seed"]))].append(row)

    selected = {}
    for root, candidates in grouped.items():
        selected[root] = max(
            candidates,
            key=lambda row: (float(row[reward_dim]), -int(row["path_id"])),
        )
    return selected


def _write_selection_tables(
    cfg: ScoreConfig,
    score_rows: list[dict[str, str]],
) -> None:
    path_scores = _load_vbench_path_scores(cfg.branch_compare)
    root_scores = _load_vbench_root_scores(cfg.branch_compare)

    by_root_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for reward_dim in VIDEOREWARD_DIMS:
        selector = f"video_reward_{reward_dim.lower()}"
        selected_by_root = _best_by_reward(score_rows, reward_dim)

        for root, selected in sorted(selected_by_root.items(), key=lambda item: (item[0].prompt_id, item[0].root_seed)):
            path_bits = selected["path_bits"]
            path_key = (root.prompt_id, root.root_seed, path_bits)
            prompt_text = selected["prompt_text"]
            for dim in VBENCH_DIMS:
                root_row = root_scores[(root.prompt_id, root.root_seed, dim)]
                selected_value = path_scores[path_key][dim]
                independent = float(root_row["independent_concat"])
                all_zero = float(root_row["all_zero"])
                branch_mean = float(root_row["branch_mean"])
                oracle = float(root_row["branch_max"])
                by_root_rows.append(
                    {
                        "selector": selector,
                        "reward_dim": reward_dim,
                        "prompt_id": root.prompt_id,
                        "prompt_text": prompt_text,
                        "root_seed": str(root.root_seed),
                        "selected_path_bits": path_bits,
                        "selected_path_id": selected["path_id"],
                        "selected_reward": selected[reward_dim],
                        "dimension": dim,
                        "selected": _fmt(selected_value),
                        "independent_concat": _fmt(independent),
                        "all_zero": _fmt(all_zero),
                        "branch_mean": _fmt(branch_mean),
                        "oracle_best_of_32": _fmt(oracle),
                        "selected_minus_independent": _fmt(selected_value - independent),
                        "selected_minus_all_zero": _fmt(selected_value - all_zero),
                        "selected_minus_branch_mean": _fmt(selected_value - branch_mean),
                        "gap_to_oracle": _fmt(oracle - selected_value),
                    }
                )

    by_root_fields = [
        "selector",
        "reward_dim",
        "prompt_id",
        "prompt_text",
        "root_seed",
        "selected_path_bits",
        "selected_path_id",
        "selected_reward",
        "dimension",
        "selected",
        "independent_concat",
        "all_zero",
        "branch_mean",
        "oracle_best_of_32",
        "selected_minus_independent",
        "selected_minus_all_zero",
        "selected_minus_branch_mean",
        "gap_to_oracle",
    ]
    _write_csv(cfg.output_dir / "video_reward_by_root.csv", by_root_rows, by_root_fields)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in by_root_rows:
        grouped[(row["selector"], row["dimension"])].append(row)

    for (selector, dim), rows in sorted(grouped.items()):
        selected_values = [float(row["selected"]) for row in rows]
        independent_values = [float(row["independent_concat"]) for row in rows]
        all_zero_values = [float(row["all_zero"]) for row in rows]
        branch_mean_values = [float(row["branch_mean"]) for row in rows]
        oracle_values = [float(row["oracle_best_of_32"]) for row in rows]
        summary_rows.append(
            {
                "selector": selector,
                "dimension": dim,
                "num_roots": str(len(rows)),
                "selected_mean": _fmt(_mean(selected_values)),
                "independent_mean": _fmt(_mean(independent_values)),
                "all_zero_mean": _fmt(_mean(all_zero_values)),
                "branch_mean": _fmt(_mean(branch_mean_values)),
                "oracle_best_of_32_mean": _fmt(_mean(oracle_values)),
                "selected_minus_independent": _fmt(_mean(a - b for a, b in zip(selected_values, independent_values))),
                "selected_minus_all_zero": _fmt(_mean(a - b for a, b in zip(selected_values, all_zero_values))),
                "selected_minus_branch_mean": _fmt(_mean(a - b for a, b in zip(selected_values, branch_mean_values))),
                "gap_to_oracle": _fmt(_mean(a - b for a, b in zip(oracle_values, selected_values))),
            }
        )

    summary_fields = [
        "selector",
        "dimension",
        "num_roots",
        "selected_mean",
        "independent_mean",
        "all_zero_mean",
        "branch_mean",
        "oracle_best_of_32_mean",
        "selected_minus_independent",
        "selected_minus_all_zero",
        "selected_minus_branch_mean",
        "gap_to_oracle",
    ]
    _write_csv(cfg.output_dir / "video_reward_summary.csv", summary_rows, summary_fields)

    correlation_rows = _compute_correlations(score_rows, path_scores)
    _write_csv(
        cfg.output_dir / "video_reward_correlations.csv",
        correlation_rows,
        ["reward_dim", "dimension", "pearson", "spearman", "num_videos"],
    )


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for order_index in range(index, end):
            ranks[order[order_index]] = rank
        index = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_den = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if x_den == 0 or y_den == 0:
        return float("nan")
    return num / (x_den * y_den)


def _compute_correlations(
    score_rows: list[dict[str, str]],
    path_scores: dict[tuple[str, int, str], dict[str, float]],
) -> list[dict[str, str]]:
    rows = []
    for reward_dim in VIDEOREWARD_DIMS:
        for dim in VBENCH_DIMS:
            reward_values = []
            vbench_values = []
            for row in score_rows:
                key = _score_key(row)
                if key not in path_scores or dim not in path_scores[key]:
                    continue
                reward_values.append(float(row[reward_dim]))
                vbench_values.append(path_scores[key][dim])
            rows.append(
                {
                    "reward_dim": reward_dim,
                    "dimension": dim,
                    "pearson": _fmt(_pearson(reward_values, vbench_values)),
                    "spearman": _fmt(_pearson(_rankdata(reward_values), _rankdata(vbench_values))),
                    "num_videos": str(len(reward_values)),
                }
            )
    return rows


def _parse_args() -> ScoreConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-run", type=Path, default=DEFAULT_BRANCH_RUN)
    parser.add_argument("--branch-compare", type=Path, default=DEFAULT_BRANCH_COMPARE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated prompt IDs to include.")
    parser.add_argument("--root-seeds", default=None, help="Comma-separated root seeds to include.")
    parser.add_argument("--limit-roots", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Discard existing reward scores.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=240 * 320)
    parser.add_argument("--no-norm", action="store_true", help="Use raw VideoReward outputs.")
    parser.add_argument("--videoreward-repo", type=Path, default=Path("/tmp/VideoAlign"))
    parser.add_argument("--videoreward-checkpoint", type=Path, default=Path("/tmp/VideoReward_ckpt"))
    parser.add_argument("--videoreward-compat-path", type=Path, default=Path("/tmp/videoreward_compat"))
    parser.add_argument("--allow-flash-attn2", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.branch_compare / DEFAULT_OUTPUT_SUBDIR

    compat_path = args.videoreward_compat_path
    if compat_path is not None and str(compat_path) == "":
        compat_path = None

    return ScoreConfig(
        branch_run=args.branch_run,
        branch_compare=args.branch_compare,
        output_dir=output_dir,
        prompt_ids=_parse_csv_tuple(args.prompt_ids),
        root_seeds=_parse_int_tuple(args.root_seeds),
        limit_roots=args.limit_roots,
        force=args.force,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        use_norm=not args.no_norm,
        videoreward_repo=args.videoreward_repo,
        videoreward_checkpoint=args.videoreward_checkpoint,
        videoreward_compat_path=compat_path,
        disable_flash_attn2=not args.allow_flash_attn2,
    )


def main() -> None:
    cfg = _parse_args()
    manifest_rows = _load_manifest(cfg.branch_run)
    roots = _root_keys(manifest_rows, cfg)
    selected_rows = _filter_manifest(manifest_rows, roots)
    if not selected_rows:
        raise SystemExit("No branch rows matched the requested filters.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"roots={len(roots)} videos={len(selected_rows)} output={cfg.output_dir}", flush=True)
    score_rows = _score_videoreward(cfg, selected_rows)
    _write_selection_tables(cfg, score_rows)
    print(f"wrote {cfg.output_dir / 'video_reward_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
