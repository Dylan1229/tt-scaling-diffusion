from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

VALID_LABELS = {"success", "failure"}
DEFAULT_COARSE_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def _alpha_code(alpha: float) -> int:
    alpha = float(alpha)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    code = round(alpha * 10000)
    if not math.isclose(alpha, code / 10000, rel_tol=0, abs_tol=1e-12):
        raise ValueError("alpha must use at most four decimal places")
    return code


def format_sample_id(direction_index: int, alpha: float) -> str:
    if direction_index < 0:
        raise ValueError("direction_index must be non-negative")
    return f"d{direction_index:02d}_a{_alpha_code(alpha):05d}"


def expected_rms_radius(alpha: float) -> float:
    alpha = float(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1")
    return math.sqrt(2.0 - 2.0 * math.sqrt(1.0 - alpha**2))


def radial_specs(
    alphas: Sequence[float],
    direction_seeds: Sequence[int],
    *,
    start_index: int = 0,
) -> list[dict[str, int | float | str]]:
    values = tuple(float(alpha) for alpha in alphas)
    seeds = tuple(int(seed) for seed in direction_seeds)
    if not values or any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("alphas must be non-empty and strictly increasing")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("direction seeds must be non-empty and unique")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    specs = []
    for alpha in values:
        _alpha_code(alpha)
        for direction_index, perturb_seed in enumerate(seeds):
            specs.append(
                {
                    "index": start_index + len(specs),
                    "direction_index": direction_index,
                    "alpha": alpha,
                    "perturb_seed": perturb_seed,
                    "sample_id": format_sample_id(direction_index, alpha),
                }
            )
    return specs


def validate_scan_config(payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "version",
        "prompt",
        "input_path",
        "input_sha256",
        "parent_seed",
        "parent_label",
        "semantic_criterion",
        "direction_seeds",
        "coarse_alphas",
        "alpha_tolerance",
    }
    missing = sorted(required - payload.keys())
    extra = sorted(payload.keys() - required)
    if missing or extra:
        raise ValueError(f"scan config keys mismatch: missing={missing}, extra={extra}")

    if payload["version"] != 1:
        raise ValueError("scan config version must be 1")

    if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        raise ValueError("prompt must be non-empty")
    if not isinstance(payload["semantic_criterion"], str) or not payload["semantic_criterion"].strip():
        raise ValueError("semantic_criterion must be non-empty")

    if payload["parent_label"] not in VALID_LABELS:
        raise ValueError("parent_label must be success or failure")

    image_hash = str(payload["input_sha256"])
    if len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
        raise ValueError("input_sha256 must be 64 lowercase hex characters")

    direction_seeds = tuple(int(seed) for seed in payload["direction_seeds"])
    coarse_alphas = tuple(float(alpha) for alpha in payload["coarse_alphas"])
    radial_specs(coarse_alphas, direction_seeds)
    if coarse_alphas[-1] != 1.0:
        raise ValueError("coarse_alphas must end at 1.0")

    tolerance = float(payload["alpha_tolerance"])
    if not 0.0001 <= tolerance < 1:
        raise ValueError("alpha_tolerance must be at least 0.0001 and less than 1")

    return {
        "version": 1,
        "prompt": payload["prompt"].strip(),
        "input_path": str(payload["input_path"]),
        "input_sha256": image_hash,
        "parent_seed": int(payload["parent_seed"]),
        "parent_label": payload["parent_label"],
        "semantic_criterion": payload["semantic_criterion"].strip(),
        "direction_seeds": list(direction_seeds),
        "coarse_alphas": list(coarse_alphas),
        "alpha_tolerance": tolerance,
    }


def load_scan_config(path: Path) -> dict[str, object]:
    return validate_scan_config(json.loads(path.read_text()))


def scan_config_sha256(config: Mapping[str, object]) -> str:
    encoded = json.dumps(validate_scan_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_shell_plan(config: Mapping[str, object], alpha: float, *, start_index: int) -> dict[str, object]:
    validated = validate_scan_config(config)
    return {
        "version": 1,
        "kind": "coarse",
        "samples": radial_specs(
            (alpha,),
            tuple(validated["direction_seeds"]),
            start_index=start_index,
        ),
    }


def validate_sample_plan(payload: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    validated_config = validate_scan_config(config)
    required_root = {"version", "kind", "samples"}
    required_samples = {"index", "direction_index", "alpha", "perturb_seed", "sample_id"}
    missing = sorted(required_root - payload.keys())
    extra = sorted(payload.keys() - required_root)
    if missing or extra:
        raise ValueError(f"sample plan keys mismatch: missing={missing}, extra={extra}")

    if payload["version"] != 1:
        raise ValueError("sample plan version must be 1")

    if payload["kind"] not in {"coarse", "refinement"}:
        raise ValueError("sample plan kind must be coarse or refinement")

    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")

    seen_indices = set[int]()
    seen_sample_ids = set[str]()
    direction_seeds = tuple(validated_config["direction_seeds"])

    normalized_samples = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("each sample must be a mapping")

        sample_missing = sorted(required_samples - sample.keys())
        sample_extra = sorted(sample.keys() - required_samples)
        if sample_missing or sample_extra:
            raise ValueError(
                f"sample keys mismatch: missing={sample_missing}, extra={sample_extra}"
            )

        direction_index = int(sample["direction_index"])
        index = int(sample["index"])
        alpha = float(sample["alpha"])
        perturb_seed = int(sample["perturb_seed"])
        sample_id = str(sample["sample_id"])

        if direction_index < 0 or direction_index >= len(direction_seeds):
            raise ValueError(f"direction {direction_index} has no configured seed")
        if perturb_seed != direction_seeds[direction_index]:
            raise ValueError(
                f"direction {direction_index} must use seed {direction_seeds[direction_index]}"
            )
        if sample_id != format_sample_id(direction_index, alpha):
            raise ValueError("sample_id does not match direction_index and alpha")

        if index in seen_indices:
            raise ValueError("sample indices must be unique")
        if sample_id in seen_sample_ids:
            raise ValueError("sample IDs must be unique")
        seen_indices.add(index)
        seen_sample_ids.add(sample_id)

        normalized_samples.append(
            {
                "index": index,
                "direction_index": direction_index,
                "alpha": alpha,
                "perturb_seed": perturb_seed,
                "sample_id": sample_id,
            }
        )

    return {"version": 1, "kind": payload["kind"], "samples": normalized_samples}
