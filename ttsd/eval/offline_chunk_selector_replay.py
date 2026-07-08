"""Replay lightweight greedy chunk selectors on a saved chunk-branch run.

This module does not generate new videos.  It uses the existing binary branch
tree produced by ``ttsd.runners.generate.chunk_branch_i2v`` and simulates an
online greedy policy: at chunk k, compare the two child prefixes from the
current prefix, choose one, and continue.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DIMS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_mean",
)

DEFAULT_SELECTORS = (
    "boundary_continuity",
    "motion_amount",
    "motion_stability",
    "quality_proxy",
    "hybrid_light",
)

HYBRID_WEIGHTS = {
    "boundary_continuity": 1.00,
    "motion_amount": 0.75,
    "motion_stability": 0.35,
    "quality_proxy": 0.50,
}


@dataclass(frozen=True)
class ReplayConfig:
    branch_run: Path
    branch_compare: Path
    output_dir: Path
    chunk_num_frames: int
    overlap_frames: int
    tail_frames: int
    head_frames: int
    max_score_frames: int
    lowres_size: int
    selectors: tuple[str, ...]
    prompt_ids: tuple[str, ...] | None
    root_seeds: tuple[int, ...] | None
    limit_roots: int | None


@dataclass(frozen=True)
class RootKey:
    prompt_id: str
    root_seed: int


@dataclass
class CandidateEval:
    prefix: str
    path_bits: str
    path_id: int
    seed_idx: int
    video_path: str
    components: dict[str, float]
    selector_score: float = 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _load_manifest(branch_run: Path) -> dict[tuple[str, int, str], dict]:
    manifest_path = branch_run / "branch_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing branch manifest: {manifest_path}")
    rows = _read_csv(manifest_path)
    out = {}
    for row in rows:
        out[(row["prompt_id"], int(row["root_seed"]), row["path_bits"])] = row
    return out


def _load_path_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    path_csv = compare_dir / "chunk_branch_by_path.csv"
    if not path_csv.exists():
        raise SystemExit(f"Missing path score CSV: {path_csv}")
    scores: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in _read_csv(path_csv):
        key = (row["prompt_id"], int(row["root_seed"]), row["path_bits"])
        scores[key][row["dimension"]] = float(row["score"])
    return scores


def _load_root_scores(compare_dir: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    root_csv = compare_dir / "chunk_branch_by_root.csv"
    if not root_csv.exists():
        raise SystemExit(f"Missing root score CSV: {root_csv}")
    return {
        (row["prompt_id"], int(row["root_seed"]), row["dimension"]): row
        for row in _read_csv(root_csv)
    }


def _path_id(path_bits: str) -> int:
    return int(path_bits, 2) if path_bits else 0


def _root_keys(manifest: dict[tuple[str, int, str], dict], cfg: ReplayConfig) -> list[RootKey]:
    keys = sorted(
        {RootKey(prompt_id, root_seed) for prompt_id, root_seed, _ in manifest},
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


def _depth_for_root(manifest: dict[tuple[str, int, str], dict], root: RootKey) -> int:
    bits = [
        path_bits
        for prompt_id, root_seed, path_bits in manifest
        if prompt_id == root.prompt_id and root_seed == root.root_seed
    ]
    if not bits:
        raise KeyError(f"No manifest entries for {root}")
    return max(len(item) for item in bits)


def _representative_leaf(
    manifest: dict[tuple[str, int, str], dict],
    root: RootKey,
    prefix: str,
    depth: int,
) -> dict:
    padded = prefix + ("0" * (depth - len(prefix)))
    row = manifest.get((root.prompt_id, root.root_seed, padded))
    if row is not None:
        return row

    candidates = [
        row
        for (prompt_id, root_seed, path_bits), row in manifest.items()
        if prompt_id == root.prompt_id and root_seed == root.root_seed and path_bits.startswith(prefix)
    ]
    if not candidates:
        raise KeyError(f"No leaf found for {root} prefix={prefix!r}")
    return sorted(candidates, key=lambda item: item["path_bits"])[0]


def _video_path(branch_run: Path, manifest_row: dict) -> Path:
    raw = Path(manifest_row["video_path"])
    return raw if raw.is_absolute() else branch_run / raw


def _read_video(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
    except Exception:
        import imageio

        arr = imageio.mimread(path)
        arr = np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"Expected video array (frames,H,W,C), got {arr.shape} from {path}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _read_indexed_frames(path: Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    wanted = sorted({int(index) for index in indices if int(index) >= 0})
    if not wanted:
        return {}

    import imageio

    frames: dict[int, np.ndarray] = {}
    reader = imageio.get_reader(path)
    try:
        for index in wanted:
            try:
                frame = np.asarray(reader.get_data(index))
            except IndexError:
                continue
            if frame.ndim != 3:
                continue
            if frame.shape[-1] > 3:
                frame = frame[..., :3]
            frames[index] = frame.astype(np.uint8, copy=False)
    finally:
        reader.close()
    return frames


def _stack_existing(frames: dict[int, np.ndarray], indices: list[int]) -> np.ndarray:
    values = [frames[index] for index in indices if index in frames]
    if not values:
        return np.empty((0, 1, 1, 3), dtype=np.uint8)
    return np.stack(values, axis=0)


def _sample_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    arr = np.asarray(frames)
    if len(arr) <= max_frames:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_frames).round().astype(int)
    return arr[idx]


def _lowres(frames: np.ndarray, max_size: int) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.size == 0:
        return arr.astype(np.float32)
    h, w = arr.shape[1], arr.shape[2]
    stride_h = max(1, h // max_size)
    stride_w = max(1, w // max_size)
    small = arr[:, ::stride_h, ::stride_w, :]
    return small[:, :max_size, :max_size, :].astype(np.float32) / 255.0


def _luma(frames: np.ndarray) -> np.ndarray:
    arr = frames.astype(np.float32, copy=False)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    aa = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    bb = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return np.sum(aa * bb, axis=-1)


def _candidate_slices(video: np.ndarray, chunk_idx: int, cfg: ReplayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stride = cfg.chunk_num_frames - cfg.overlap_frames
    start = chunk_idx * stride
    end = min(start + cfg.chunk_num_frames, len(video))
    new_start = min(end, start + cfg.overlap_frames if chunk_idx > 0 else start)

    if cfg.overlap_frames > 0 and chunk_idx > 0:
        prev_tail_start = max(0, start - cfg.tail_frames + 1)
        prev_tail = video[prev_tail_start : start + 1]
    else:
        prev_tail_start = max(0, start - cfg.tail_frames)
        prev_tail = video[prev_tail_start:start]

    head = video[new_start : min(end, new_start + cfg.head_frames)]
    new_chunk = video[new_start:end]
    return prev_tail, head, new_chunk


def _candidate_frame_indices(chunk_idx: int, cfg: ReplayConfig) -> tuple[list[int], list[int], list[int]]:
    stride = cfg.chunk_num_frames - cfg.overlap_frames
    start = chunk_idx * stride
    end = start + cfg.chunk_num_frames
    new_start = start + cfg.overlap_frames if chunk_idx > 0 else start

    if cfg.overlap_frames > 0 and chunk_idx > 0:
        prev_tail_start = max(0, start - cfg.tail_frames + 1)
        prev_tail = list(range(prev_tail_start, start + 1))
    else:
        prev_tail_start = max(0, start - cfg.tail_frames)
        prev_tail = list(range(prev_tail_start, start))

    head = list(range(new_start, min(end, new_start + cfg.head_frames)))
    available = max(0, end - new_start)
    if available == 0:
        chunk_sample = []
    else:
        sample_count = min(cfg.max_score_frames, available)
        chunk_sample = (
            np.linspace(new_start, end - 1, sample_count).round().astype(int).tolist()
        )
    return prev_tail, head, chunk_sample


def _boundary_continuity(prev_tail: np.ndarray, head: np.ndarray, cfg: ReplayConfig) -> float:
    if len(prev_tail) == 0 or len(head) == 0:
        return 0.0
    n = min(len(prev_tail), len(head))
    prev = _lowres(prev_tail[-n:], cfg.lowres_size).reshape(n, -1)
    nxt = _lowres(head[:n], cfg.lowres_size).reshape(n, -1)
    return float(_cosine_rows(prev, nxt).mean())


def _motion_stats(chunk: np.ndarray, cfg: ReplayConfig) -> tuple[float, float, float]:
    frames = _sample_frames(chunk, cfg.max_score_frames)
    if len(frames) < 2:
        return 0.0, 0.0, 0.0
    small = _lowres(frames, cfg.lowres_size)
    luma = _luma(small)
    diffs = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    mean = float(diffs.mean())
    std = float(diffs.std())
    p95 = float(np.quantile(diffs, 0.95))
    return mean, std, p95


def _motion_stability(motion_mean: float, motion_std: float, motion_p95: float) -> float:
    if motion_mean <= 1e-8:
        return -10.0
    cv = motion_std / (motion_mean + 1e-8)
    spike = max(0.0, motion_p95 / (motion_mean + 1e-8) - 1.0)
    return float(-(0.70 * cv + 0.30 * spike))


def _quality_proxy(chunk: np.ndarray, cfg: ReplayConfig) -> float:
    frames = _sample_frames(chunk, min(cfg.max_score_frames, 8))
    if len(frames) == 0:
        return 0.0
    small = _lowres(frames, max(cfg.lowres_size, 96))
    luma = _luma(small)
    sharp_x = float(np.abs(np.diff(luma, axis=2)).mean()) if luma.shape[2] > 1 else 0.0
    sharp_y = float(np.abs(np.diff(luma, axis=1)).mean()) if luma.shape[1] > 1 else 0.0
    sharpness = 0.5 * (sharp_x + sharp_y)
    contrast = float(luma.std())
    exposure = float(max(0.0, 1.0 - 2.0 * abs(float(luma.mean()) - 0.5)))
    saturation = float(small.std(axis=-1).mean())
    return float(1.50 * sharpness + 0.50 * contrast + 0.25 * exposure + 0.25 * saturation)


def _component_scores(video: np.ndarray, chunk_idx: int, cfg: ReplayConfig) -> dict[str, float]:
    prev_tail, head, chunk = _candidate_slices(video, chunk_idx, cfg)
    motion_mean, motion_std, motion_p95 = _motion_stats(chunk, cfg)
    return {
        "boundary_continuity": _boundary_continuity(prev_tail, head, cfg),
        "motion_amount": motion_mean,
        "motion_stability": _motion_stability(motion_mean, motion_std, motion_p95),
        "quality_proxy": _quality_proxy(chunk, cfg),
        "motion_std": motion_std,
        "motion_p95": motion_p95,
    }


def _component_scores_from_path(video_path: Path, chunk_idx: int, cfg: ReplayConfig) -> dict[str, float]:
    prev_indices, head_indices, chunk_indices = _candidate_frame_indices(chunk_idx, cfg)
    frame_map = _read_indexed_frames(video_path, [*prev_indices, *head_indices, *chunk_indices])
    prev_tail = _stack_existing(frame_map, prev_indices)
    head = _stack_existing(frame_map, head_indices)
    chunk = _stack_existing(frame_map, chunk_indices)
    motion_mean, motion_std, motion_p95 = _motion_stats(chunk, cfg)
    return {
        "boundary_continuity": _boundary_continuity(prev_tail, head, cfg),
        "motion_amount": motion_mean,
        "motion_stability": _motion_stability(motion_mean, motion_std, motion_p95),
        "quality_proxy": _quality_proxy(chunk, cfg),
        "motion_std": motion_std,
        "motion_p95": motion_p95,
    }


def _score_candidates(selector: str, candidates: list[CandidateEval]) -> None:
    if selector != "hybrid_light":
        for candidate in candidates:
            candidate.selector_score = candidate.components[selector]
        return

    for candidate in candidates:
        candidate.selector_score = 0.0

    for name, weight in HYBRID_WEIGHTS.items():
        values = [candidate.components[name] for candidate in candidates]
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-12:
            normalized = [0.5 for _ in values]
        else:
            normalized = [(value - lo) / (hi - lo) for value in values]
        for candidate, norm_value in zip(candidates, normalized):
            candidate.selector_score += weight * norm_value
            candidate.components[f"{name}_pair_norm"] = float(norm_value)


def _choose_candidate(candidates: list[CandidateEval]) -> CandidateEval:
    return sorted(candidates, key=lambda item: (-item.selector_score, item.prefix))[0]


def _replay_selector_for_root(
    *,
    selector: str,
    root: RootKey,
    manifest: dict[tuple[str, int, str], dict],
    depth: int,
    component_cache: dict[tuple[str, int], dict[str, float]],
    cfg: ReplayConfig,
) -> tuple[str, list[dict]]:
    prefix = ""
    trace = []
    for chunk_idx in range(1, depth + 1):
        candidates: list[CandidateEval] = []
        for choice in ("0", "1"):
            child_prefix = prefix + choice
            row = _representative_leaf(manifest, root, child_prefix, depth)
            video_path = _video_path(cfg.branch_run, row)
            cache_key = (str(video_path), chunk_idx)
            if cache_key not in component_cache:
                component_cache[cache_key] = _component_scores_from_path(video_path, chunk_idx, cfg)
            components = dict(component_cache[cache_key])
            candidates.append(
                CandidateEval(
                    prefix=child_prefix,
                    path_bits=row["path_bits"],
                    path_id=int(row["path_id"]),
                    seed_idx=int(row["seed_idx"]),
                    video_path=str(video_path),
                    components=components,
                )
            )

        _score_candidates(selector, candidates)
        chosen = _choose_candidate(candidates)
        trace.append(
            {
                "chunk_idx": chunk_idx,
                "parent_prefix": prefix,
                "chosen_prefix": chosen.prefix,
                "chosen_score": chosen.selector_score,
                "candidates": [
                    {
                        "prefix": candidate.prefix,
                        "representative_path_bits": candidate.path_bits,
                        "path_id": candidate.path_id,
                        "seed_idx": candidate.seed_idx,
                        "video_path": candidate.video_path,
                        "selector_score": candidate.selector_score,
                        "components": candidate.components,
                    }
                    for candidate in sorted(candidates, key=lambda item: item.prefix)
                ],
            }
        )
        prefix = chosen.prefix
    return prefix, trace


def _score_lookup(
    path_scores: dict[tuple[str, int, str], dict[str, float]],
    root: RootKey,
    path_bits: str,
    dim: str,
) -> float | None:
    return path_scores.get((root.prompt_id, root.root_seed, path_bits), {}).get(dim)


def _prompt_text(manifest: dict[tuple[str, int, str], dict], root: RootKey, path_bits: str) -> str:
    row = manifest.get((root.prompt_id, root.root_seed, path_bits))
    if row is not None:
        return row["prompt_text"]
    for (prompt_id, root_seed, _), candidate in manifest.items():
        if prompt_id == root.prompt_id and root_seed == root.root_seed:
            return candidate["prompt_text"]
    return ""


def _summary_rows(metric_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["selector"], row["dimension"])].append(row)

    out = []
    for (selector, dim), rows in sorted(grouped.items()):
        selected = [float(row["selected_score"]) for row in rows if row["selected_score"] != ""]
        if not selected:
            continue

        def values(column: str) -> list[float]:
            return [float(row[column]) for row in rows if row[column] != ""]

        independent = values("independent_concat")
        all_zero = values("all_zero")
        branch_mean = values("branch_mean")
        branch_max = values("branch_max")
        selected_minus_ind = [
            float(row["selected_minus_independent"])
            for row in rows
            if row["selected_minus_independent"] != ""
        ]
        selected_minus_all_zero = [
            float(row["selected_minus_all_zero"])
            for row in rows
            if row["selected_minus_all_zero"] != ""
        ]
        gap_to_branch_max = [
            float(row["gap_to_branch_max"])
            for row in rows
            if row["gap_to_branch_max"] != ""
        ]
        out.append(
            {
                "selector": selector,
                "dimension": dim,
                "n_roots": len(selected),
                "selected_mean": _fmt(statistics.fmean(selected)),
                "independent_mean": _fmt(statistics.fmean(independent) if independent else None),
                "all_zero_mean": _fmt(statistics.fmean(all_zero) if all_zero else None),
                "branch_mean": _fmt(statistics.fmean(branch_mean) if branch_mean else None),
                "branch_max_mean": _fmt(statistics.fmean(branch_max) if branch_max else None),
                "avg_selected_minus_independent": _fmt(
                    statistics.fmean(selected_minus_ind) if selected_minus_ind else None
                ),
                "avg_selected_minus_all_zero": _fmt(
                    statistics.fmean(selected_minus_all_zero) if selected_minus_all_zero else None
                ),
                "avg_gap_to_branch_max": _fmt(
                    statistics.fmean(gap_to_branch_max) if gap_to_branch_max else None
                ),
                "beats_independent_frac": _fmt(
                    sum(value > 0 for value in selected_minus_ind) / len(selected_minus_ind)
                    if selected_minus_ind
                    else None
                ),
                "beats_all_zero_frac": _fmt(
                    sum(value > 0 for value in selected_minus_all_zero) / len(selected_minus_all_zero)
                    if selected_minus_all_zero
                    else None
                ),
            }
        )
    return out


def run(cfg: ReplayConfig) -> None:
    unknown = sorted(set(cfg.selectors) - set(DEFAULT_SELECTORS))
    if unknown:
        raise SystemExit(f"Unknown selectors: {unknown}; valid={list(DEFAULT_SELECTORS)}")

    manifest = _load_manifest(cfg.branch_run)
    path_scores = _load_path_scores(cfg.branch_compare)
    root_scores = _load_root_scores(cfg.branch_compare)
    roots = _root_keys(manifest, cfg)
    if not roots:
        raise SystemExit("No roots selected")

    trace_dir = cfg.output_dir / "selection_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    by_root_rows = []
    metric_rows = []
    component_cache: dict[tuple[str, int], dict[str, float]] = {}

    for root in roots:
        depth = _depth_for_root(manifest, root)
        for selector in cfg.selectors:
            selected_bits, trace = _replay_selector_for_root(
                selector=selector,
                root=root,
                manifest=manifest,
                depth=depth,
                component_cache=component_cache,
                cfg=cfg,
            )
            selected_row = _representative_leaf(manifest, root, selected_bits, depth)
            selected_path_id = _path_id(selected_bits)
            trace_path = trace_dir / f"{selector}_{root.prompt_id}_root{root.root_seed:04d}.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "selector": selector,
                        "prompt_id": root.prompt_id,
                        "prompt_text": _prompt_text(manifest, root, selected_bits),
                        "root_seed": root.root_seed,
                        "selected_path_bits": selected_bits,
                        "selected_path_id": selected_path_id,
                        "selected_seed_idx": int(selected_row["seed_idx"]),
                        "selected_video_path": str(_video_path(cfg.branch_run, selected_row)),
                        "trace": trace,
                    },
                    indent=2,
                )
            )

            overall_root = root_scores[(root.prompt_id, root.root_seed, "overall_mean")]
            selected_overall = _score_lookup(path_scores, root, selected_bits, "overall_mean")
            all_zero_overall = _as_float(overall_root.get("all_zero"))
            branch_mean_overall = _as_float(overall_root.get("branch_mean"))
            branch_max_overall = _as_float(overall_root.get("branch_max"))
            independent_overall = _as_float(overall_root.get("independent_concat"))

            by_root_rows.append(
                {
                    "selector": selector,
                    "prompt_id": root.prompt_id,
                    "prompt_text": _prompt_text(manifest, root, selected_bits),
                    "root_seed": root.root_seed,
                    "selected_path_bits": selected_bits,
                    "selected_path_id": selected_path_id,
                    "selected_seed_idx": int(selected_row["seed_idx"]),
                    "selected_video_path": str(_video_path(cfg.branch_run, selected_row)),
                    "selected_overall_mean": _fmt(selected_overall),
                    "independent_concat": _fmt(independent_overall),
                    "all_zero": _fmt(all_zero_overall),
                    "branch_mean": _fmt(branch_mean_overall),
                    "oracle_best_of_32": _fmt(branch_max_overall),
                    "selected_minus_independent": _fmt(
                        selected_overall - independent_overall
                        if selected_overall is not None and independent_overall is not None
                        else None
                    ),
                    "selected_minus_all_zero": _fmt(
                        selected_overall - all_zero_overall
                        if selected_overall is not None and all_zero_overall is not None
                        else None
                    ),
                    "selected_minus_branch_mean": _fmt(
                        selected_overall - branch_mean_overall
                        if selected_overall is not None and branch_mean_overall is not None
                        else None
                    ),
                    "gap_to_oracle": _fmt(
                        branch_max_overall - selected_overall
                        if selected_overall is not None and branch_max_overall is not None
                        else None
                    ),
                    "trace_path": str(trace_path),
                }
            )

            for dim in DEFAULT_DIMS:
                root_row = root_scores.get((root.prompt_id, root.root_seed, dim))
                if root_row is None:
                    continue
                selected_score = _score_lookup(path_scores, root, selected_bits, dim)
                independent = _as_float(root_row.get("independent_concat"))
                all_zero = _as_float(root_row.get("all_zero"))
                branch_mean = _as_float(root_row.get("branch_mean"))
                branch_max = _as_float(root_row.get("branch_max"))
                overall_best_path_value = _as_float(root_row.get("overall_best_path_value"))
                metric_rows.append(
                    {
                        "selector": selector,
                        "prompt_id": root.prompt_id,
                        "prompt_text": _prompt_text(manifest, root, selected_bits),
                        "root_seed": root.root_seed,
                        "dimension": dim,
                        "selected_path_bits": selected_bits,
                        "selected_score": _fmt(selected_score),
                        "independent_concat": _fmt(independent),
                        "all_zero": _fmt(all_zero),
                        "branch_mean": _fmt(branch_mean),
                        "branch_max": _fmt(branch_max),
                        "overall_best_path_value": _fmt(overall_best_path_value),
                        "selected_minus_independent": _fmt(
                            selected_score - independent
                            if selected_score is not None and independent is not None
                            else None
                        ),
                        "selected_minus_all_zero": _fmt(
                            selected_score - all_zero
                            if selected_score is not None and all_zero is not None
                            else None
                        ),
                        "selected_minus_branch_mean": _fmt(
                            selected_score - branch_mean
                            if selected_score is not None and branch_mean is not None
                            else None
                        ),
                        "gap_to_branch_max": _fmt(
                            branch_max - selected_score
                            if selected_score is not None and branch_max is not None
                            else None
                        ),
                    }
                )

    _write_csv(
        cfg.output_dir / "offline_selector_by_root.csv",
        by_root_rows,
        [
            "selector",
            "prompt_id",
            "prompt_text",
            "root_seed",
            "selected_path_bits",
            "selected_path_id",
            "selected_seed_idx",
            "selected_video_path",
            "selected_overall_mean",
            "independent_concat",
            "all_zero",
            "branch_mean",
            "oracle_best_of_32",
            "selected_minus_independent",
            "selected_minus_all_zero",
            "selected_minus_branch_mean",
            "gap_to_oracle",
            "trace_path",
        ],
    )
    _write_csv(
        cfg.output_dir / "offline_selector_by_metric.csv",
        metric_rows,
        [
            "selector",
            "prompt_id",
            "prompt_text",
            "root_seed",
            "dimension",
            "selected_path_bits",
            "selected_score",
            "independent_concat",
            "all_zero",
            "branch_mean",
            "branch_max",
            "overall_best_path_value",
            "selected_minus_independent",
            "selected_minus_all_zero",
            "selected_minus_branch_mean",
            "gap_to_branch_max",
        ],
    )
    _write_csv(
        cfg.output_dir / "offline_selector_summary.csv",
        _summary_rows(metric_rows),
        [
            "selector",
            "dimension",
            "n_roots",
            "selected_mean",
            "independent_mean",
            "all_zero_mean",
            "branch_mean",
            "branch_max_mean",
            "avg_selected_minus_independent",
            "avg_selected_minus_all_zero",
            "avg_gap_to_branch_max",
            "beats_independent_frac",
            "beats_all_zero_frac",
        ],
    )

    print(f"[offline-selector] roots={len(roots)} selectors={len(cfg.selectors)}")
    print(f"[offline-selector] wrote {cfg.output_dir}")


def _parse_csv_ints(values: Iterable[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    out = []
    for value in values:
        out.extend(int(part) for part in str(value).split(",") if part != "")
    return tuple(out)


def _parse_csv_strings(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    out = []
    for value in values:
        out.extend(part for part in str(value).split(",") if part)
    return tuple(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch-run",
        type=Path,
        default=Path("runs/baseline_long/chunk_branch_i2v_all_prompts_s0_1"),
    )
    parser.add_argument(
        "--branch-compare",
        type=Path,
        default=Path("runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/vbench_long_compare/chunk_branch_i2v_all_prompts_s0_1/offline_selectors"),
    )
    parser.add_argument("--chunk-num-frames", type=int, default=81)
    parser.add_argument("--overlap-frames", type=int, default=1)
    parser.add_argument("--tail-frames", type=int, default=4)
    parser.add_argument("--head-frames", type=int, default=4)
    parser.add_argument("--max-score-frames", type=int, default=16)
    parser.add_argument("--lowres-size", type=int, default=64)
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument("--root-seeds", nargs="*", default=None)
    parser.add_argument("--limit-roots", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = ReplayConfig(
        branch_run=args.branch_run,
        branch_compare=args.branch_compare,
        output_dir=args.output_dir,
        chunk_num_frames=args.chunk_num_frames,
        overlap_frames=args.overlap_frames,
        tail_frames=args.tail_frames,
        head_frames=args.head_frames,
        max_score_frames=args.max_score_frames,
        lowres_size=args.lowres_size,
        selectors=tuple(args.selectors),
        prompt_ids=_parse_csv_strings(args.prompt_ids),
        root_seeds=_parse_csv_ints(args.root_seeds),
        limit_roots=args.limit_roots,
    )
    run(cfg)


if __name__ == "__main__":
    main()
