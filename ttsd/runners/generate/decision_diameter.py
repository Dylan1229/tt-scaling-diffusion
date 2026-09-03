from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

VALID_LABELS = {"success", "failure"}
DEFAULT_COARSE_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
MANUAL_LABELS = VALID_LABELS | {"ambiguous"}


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


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real number")
    return float(value)


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

    version = _require_int(payload["version"], name="version")
    if version != 1:
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

    direction_seeds_value = payload["direction_seeds"]
    if not isinstance(direction_seeds_value, Sequence) or isinstance(direction_seeds_value, (str, bytes, bytearray)):
        raise ValueError("direction_seeds must be a sequence")
    direction_seeds = tuple(
        _require_int(seed, name=f"direction_seeds[{index}]")
        for index, seed in enumerate(direction_seeds_value)
    )

    coarse_alphas_value = payload["coarse_alphas"]
    if not isinstance(coarse_alphas_value, Sequence) or isinstance(coarse_alphas_value, (str, bytes, bytearray)):
        raise ValueError("coarse_alphas must be a sequence")
    coarse_alphas = tuple(
        _require_real(alpha, name=f"coarse_alphas[{index}]")
        for index, alpha in enumerate(coarse_alphas_value)
    )
    radial_specs(coarse_alphas, direction_seeds)
    if coarse_alphas[-1] != 1.0:
        raise ValueError("coarse_alphas must end at 1.0")

    tolerance = _require_real(payload["alpha_tolerance"], name="alpha_tolerance")
    if not 0.0001 <= tolerance < 1:
        raise ValueError("alpha_tolerance must be at least 0.0001 and less than 1")

    return {
        "version": 1,
        "prompt": payload["prompt"].strip(),
        "input_path": str(payload["input_path"]),
        "input_sha256": image_hash,
        "parent_seed": _require_int(payload["parent_seed"], name="parent_seed"),
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
    alpha_value = _require_real(alpha, name="alpha")
    return {
        "version": 1,
        "kind": "coarse",
        "samples": radial_specs(
            (alpha_value,),
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

    if payload["kind"] not in {"coarse", "refinement"}:
        raise ValueError("sample plan kind must be coarse or refinement")

    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")

    if _require_int(payload["version"], name="version") != 1:
        raise ValueError("sample plan version must be 1")

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

        direction_index = _require_int(sample["direction_index"], name="direction_index")
        index = _require_int(sample["index"], name="sample index")
        alpha = _require_real(sample["alpha"], name="alpha")
        perturb_seed = _require_int(sample["perturb_seed"], name="perturb_seed")
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


def _opposite_label(parent_label: str) -> str:
    return "success" if parent_label == "failure" else "failure"


def _diameter_interval(alpha_interval: list[float] | None) -> list[float] | None:
    if alpha_interval is None:
        return None
    return [2 * expected_rms_radius(alpha) for alpha in alpha_interval]


def _manifest_entries(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    entries = manifest.get("neighbors")
    if entries is None:
        entries = manifest.get("samples")
    if not isinstance(entries, list):
        raise ValueError("manifest must provide a neighbors list")
    return entries


def _finite_metrics(sample: Mapping[str, object]) -> dict[str, float]:
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("sample metrics must be a mapping")
    normalized = {}
    for key in ("rms_distance", "cosine_similarity", "norm_ratio"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"sample metrics[{key}] must be finite")
        normalized[key] = float(value)
    return normalized


def _alpha_code_for_interval(alpha: float) -> int:
    if alpha == 0:
        return 0
    return _alpha_code(alpha)


def _lower_midpoint(lower: float, upper: float) -> float:
    return (_alpha_code_for_interval(lower) + _alpha_code_for_interval(upper)) // 2 / 10000


def _order_statistic(values: Sequence[float], rank: int) -> float:
    ordered = sorted(values)
    return ordered[rank - 1]


def analyze_scan(
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validated_config = validate_scan_config(config)
    parent_label = validated_config["parent_label"]
    opposite = _opposite_label(parent_label)
    entries = _manifest_entries(manifest)

    projected_samples = []
    sample_by_id: dict[str, Mapping[str, object]] = {}
    for sample in entries:
        if not isinstance(sample, Mapping):
            raise ValueError("manifest samples must be mappings")
        projected = {
            "index": sample["index"],
            "direction_index": sample["direction_index"],
            "alpha": sample["alpha"],
            "perturb_seed": sample["perturb_seed"],
            "sample_id": sample["sample_id"],
        }
        projected_samples.append(projected)
        sample_by_id[str(projected["sample_id"])] = sample
        _finite_metrics(sample)

    validate_sample_plan({"version": 1, "kind": "coarse", "samples": projected_samples}, validated_config)

    label_by_id: dict[str, str] = {}
    ambiguous_ids: list[str] = []
    for label_record in labels:
        if not isinstance(label_record, Mapping):
            raise ValueError("labels must be mappings")
        sample_id = str(label_record.get("sample_id"))
        label = label_record.get("label")
        if sample_id not in sample_by_id:
            raise ValueError(f"unknown label sample_id {sample_id}")
        if sample_id in label_by_id:
            raise ValueError(f"duplicate label for sample_id {sample_id}")
        if label not in MANUAL_LABELS:
            raise ValueError(f"invalid label {label}")
        label_by_id[sample_id] = str(label)
        if label == "ambiguous":
            ambiguous_ids.append(sample_id)

    if len(label_by_id) != len(sample_by_id):
        missing = sorted(set(sample_by_id) - set(label_by_id))
        raise ValueError(f"missing labels for sample_ids {missing}")

    label_counts = {label: 0 for label in ("success", "failure", "ambiguous")}
    for label in label_by_id.values():
        label_counts[label] += 1

    by_direction: dict[int, list[dict[str, object]]] = {index: [] for index in range(len(validated_config["direction_seeds"]))}
    for sample in entries:
        direction_index = int(sample["direction_index"])
        by_direction[direction_index].append(
            {
                "index": int(sample["index"]),
                "direction_index": direction_index,
                "alpha": float(sample["alpha"]),
                "sample_id": str(sample["sample_id"]),
                "label": label_by_id[str(sample["sample_id"])],
                "metrics": dict(sample["metrics"]),
            }
        )

    direction_records: list[dict[str, object]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    finite_upper_bounds: list[float] = []

    for direction_index in sorted(by_direction):
        records = sorted(by_direction[direction_index], key=lambda item: item["alpha"])
        definitive = [(0.0, parent_label)] + [
            (record["alpha"], record["label"]) for record in records if record["label"] != "ambiguous"
        ]
        has_ambiguous = any(record["label"] == "ambiguous" for record in records)
        first_opposite = next((alpha for alpha, label in definitive if alpha > 0.0 and label == opposite), None)
        upper_sample = next((record for record in records if record["alpha"] == first_opposite), None) if first_opposite is not None else None

        if first_opposite is not None:
            lower = max(alpha for alpha, label in definitive if alpha < first_opposite and label == parent_label)
            upper = first_opposite
            status = "crossed"
            finite_upper_bounds.append(upper)
        else:
            parent_at_one = any(alpha == 1.0 and label == parent_label for alpha, label in definitive)
            lower = max(alpha for alpha, label in definitive if label == parent_label)
            upper = math.inf
            status = "censored" if parent_at_one else "open"
        non_monotonic = bool(first_opposite is not None and any(alpha > first_opposite and label == parent_label for alpha, label in definitive))
        if has_ambiguous:
            status = "needs_adjudication"

        lower_bounds.append(lower)
        upper_bounds.append(upper)
        direction_record: dict[str, object] = {
            "direction_index": direction_index,
            "status": status,
            "lower_alpha": lower,
            "upper_alpha": upper,
            "non_monotonic": non_monotonic,
        }
        if upper_sample is not None:
            direction_record["upper_sample_id"] = upper_sample["sample_id"]
            direction_record["upper_sample_metrics"] = upper_sample["metrics"]
        direction_records.append(direction_record)

    nearest_alpha_interval = None
    finite_upper = [value for value in upper_bounds if math.isfinite(value)]
    if finite_upper:
        nearest_alpha_interval = [min(lower_bounds), min(finite_upper)]
    typical_alpha_interval = None
    rank = math.ceil(len(direction_records) / 2)
    typical_upper = _order_statistic(upper_bounds, rank)
    if math.isfinite(typical_upper):
        typical_alpha_interval = [
            _order_statistic(lower_bounds, rank),
            typical_upper,
        ]

    has_open = any(direction["status"] == "open" for direction in direction_records)
    needs_refinement = any(
        direction["status"] == "crossed"
        and float(direction["upper_alpha"]) - float(direction["lower_alpha"]) > validated_config["alpha_tolerance"]
        for direction in direction_records
    )
    all_censored = bool(direction_records) and all(direction["status"] == "censored" for direction in direction_records)

    profile = {
        "status": "needs_adjudication"
        if ambiguous_ids
        else "censored"
        if all_censored
        else "expand"
        if has_open
        else "refine"
        if needs_refinement
        else "complete",
        "label_counts": label_counts,
        "ambiguous_sample_ids": sorted(ambiguous_ids),
        "directions": direction_records,
        "nearest": {
            "alpha_interval": nearest_alpha_interval,
            "diameter_rms_interval": _diameter_interval(nearest_alpha_interval),
        },
        "typical": {
            "alpha_interval": typical_alpha_interval,
            "diameter_rms_interval": _diameter_interval(typical_alpha_interval),
        },
    }

    return profile


def _crossed_directions(profile: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [direction for direction in profile["directions"] if direction["status"] == "crossed"]


def next_sample_plan(
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    labels: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    profile = analyze_scan(config, manifest, labels)
    if profile["status"] == "needs_adjudication":
        return None

    validated_config = validate_scan_config(config)
    if profile["status"] == "censored":
        return None

    entries = _manifest_entries(manifest)
    if any(label.get("label") == "ambiguous" for label in labels if isinstance(label, Mapping)):
        return None

    crossed = _crossed_directions(profile)
    direction_count = len(profile["directions"])
    if len(crossed) < math.ceil(direction_count / 2):
        direction_samples: dict[int, set[float]] = {index: set() for index in range(direction_count)}
        for entry in entries:
            direction_samples[int(entry["direction_index"])] .add(float(entry["alpha"]))
        un_crossed = [direction for direction in profile["directions"] if direction["status"] != "crossed"]
        next_coarse = None
        for alpha in validated_config["coarse_alphas"]:
            if any(alpha not in direction_samples[direction["direction_index"]] for direction in un_crossed):
                next_coarse = float(alpha)
                break
        if next_coarse is not None:
            start_index = max((int(entry["index"]) for entry in entries), default=-1) + 1
            coarse_targets = [
                direction
                for direction in sorted(un_crossed, key=lambda item: item["direction_index"])
                if next_coarse not in direction_samples[direction["direction_index"]]
            ]
            samples = [
                {
                    "index": start_index + offset,
                    "direction_index": direction["direction_index"],
                    "alpha": next_coarse,
                    "perturb_seed": validated_config["direction_seeds"][direction["direction_index"]],
                    "sample_id": format_sample_id(direction["direction_index"], next_coarse),
                }
                for offset, direction in enumerate(coarse_targets)
            ]
            if not samples:
                return None
            plan = {"version": 1, "kind": "coarse", "samples": samples}
            return validate_sample_plan(plan, validated_config)

    refinement_samples = []
    start_index = max((int(entry["index"]) for entry in entries), default=-1) + 1
    offset = 0
    for direction in sorted(crossed, key=lambda item: item["direction_index"]):
        lower = float(direction["lower_alpha"])
        upper = float(direction["upper_alpha"])
        if upper - lower <= validated_config["alpha_tolerance"]:
            continue
        alpha = _lower_midpoint(lower, upper)
        refinement_samples.append(
            {
                "index": start_index + offset,
                "direction_index": direction["direction_index"],
                "alpha": alpha,
                "perturb_seed": validated_config["direction_seeds"][direction["direction_index"]],
                "sample_id": format_sample_id(direction["direction_index"], alpha),
            }
        )
        offset += 1

    if not refinement_samples:
        return None
    plan = {"version": 1, "kind": "refinement", "samples": refinement_samples}
    return validate_sample_plan(plan, validated_config)
