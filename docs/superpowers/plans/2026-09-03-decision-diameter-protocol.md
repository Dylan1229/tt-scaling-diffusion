# Decision-Diameter Calibration Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed, unrelated alpha shells with a reusable manual-label calibration workflow that estimates nearest and typical (`D50`) semantic decision diameters for every prompt and parent.

**Architecture:** Keep the existing Wan parent-capture and generation runner as the model-facing engine. Add a standard-library-only planner/analyzer for shared radial directions, coarse expansion, bounded diameter estimates, and midpoint refinement; extend the runner to consume append-only sample plans while preserving its legacy fixed-grid behavior.

**Tech Stack:** Python 3.10+, standard library, PyTorch, Pillow, NumPy, pytest, Ruff, Wan 2.2 TI2V-5B, deterministic UniPC.

**Spec:** `docs/superpowers/specs/2026-09-03-decision-diameter-protocol-design.md`

## Global Constraints

- Run every Python command, test, and model workload on SSH host `yukelab` in `/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo`.
- Prefix every remote Python, pytest, and model invocation with `PYTHONPATH=$PWD`; the shared remote environment otherwise resolves the base checkout before this linked worktree. This requirement overrides any command snippet below that omits the prefix.
- Use `/home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo` only for editing, Git operations, orchestration, and Windows viewing.
- Keep semantic labels manual: `success`, `failure`, or `ambiguous`; never add an automatic semantic scorer.
- Hold model, scheduler, prompt, input, dimensions, frame count, inference steps, guidance, and frame rate fixed within one calibration.
- Use `z(alpha, j) = sqrt(1 - alpha**2) * z_parent + alpha * epsilon_j` with `0 < alpha <= 1`.
- Reuse the same perturbation seed for one direction at every alpha.
- Default to eight direction seeds `10000` through `10007`, coarse alphas `(0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)`, and alpha bracket tolerance `0.02`.
- Preserve legacy fixed-grid behavior and never rewrite previous experiment artifacts.
- Keep the sample manifest append-only; only one process may append a plan, after which generation may shard across GPUs.
- Add no third-party dependency.
- Follow strict RED-GREEN-REFACTOR cycles and commit after each independently reviewable task.

## File Structure

- Create `ttsd/runners/generate/decision_diameter.py`: pure config validation, radial sample planning, label analysis, diameter calculation, refinement planning, and JSON CLI. It must not import Torch, Diffusers, Pillow, or the model runner.
- Modify `ttsd/runners/generate/noise_neighborhood_demo.py`: allow `alpha = 1`, apply scan configs, append explicit plans, validate manifest growth, and generate only selected plan samples.
- Create `tests/test_decision_diameter.py`: literal-fixture tests for shared directions, config validation, coarse progression, boundary brackets, censoring, ambiguity, non-monotonic labels, refinement, and CLI output.
- Modify `tests/test_noise_neighborhood_demo.py`: alpha-one construction, append/replay/conflict, custom-plan selection, and legacy compatibility tests.
- Create `docs/decision_diameter_calibration.md`: concise operator runbook for the manual iterative workflow.

---

### Task 1: Add deterministic radial sampling primitives

**Files:**
- Create: `ttsd/runners/generate/decision_diameter.py`
- Create: `tests/test_decision_diameter.py`
- Modify: `ttsd/runners/generate/noise_neighborhood_demo.py:58-64`
- Modify: `tests/test_noise_neighborhood_demo.py:16-30`

**Interfaces:**
- Produces: `format_sample_id(direction_index: int, alpha: float) -> str`
- Produces: `expected_rms_radius(alpha: float) -> float`
- Produces: `radial_specs(alphas: Sequence[float], direction_seeds: Sequence[int], *, start_index: int = 0) -> list[dict[str, int | float | str]]`
- Changes: `make_neighbor(parent: torch.Tensor, alpha: float, perturb_seed: int)` accepts `alpha = 1.0` and still rejects values outside `[0, 1]`.

- [ ] **Step 1: Write failing radial-spec and alpha-one tests**

Create `tests/test_decision_diameter.py` with hand-derived expected records:

```python
from __future__ import annotations

import math

import pytest

from ttsd.runners.generate.decision_diameter import expected_rms_radius, radial_specs


def test_radial_specs_reuse_each_direction_seed_at_every_alpha() -> None:
    assert radial_specs((0.2, 1.0), (10000, 10001), start_index=7) == [
        {
            "index": 7,
            "direction_index": 0,
            "alpha": 0.2,
            "perturb_seed": 10000,
            "sample_id": "d00_a02000",
        },
        {
            "index": 8,
            "direction_index": 1,
            "alpha": 0.2,
            "perturb_seed": 10001,
            "sample_id": "d01_a02000",
        },
        {
            "index": 9,
            "direction_index": 0,
            "alpha": 1.0,
            "perturb_seed": 10000,
            "sample_id": "d00_a10000",
        },
        {
            "index": 10,
            "direction_index": 1,
            "alpha": 1.0,
            "perturb_seed": 10001,
            "sample_id": "d01_a10000",
        },
    ]


def test_radial_specs_reject_invalid_or_unrepresentable_inputs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        radial_specs((0.2, 0.1), (10000,))
    with pytest.raises(ValueError, match="unique"):
        radial_specs((0.2,), (10000, 10000))
    with pytest.raises(ValueError, match="four decimal places"):
        radial_specs((0.20001,), (10000,))


def test_expected_rms_radius_has_independent_noise_endpoint() -> None:
    assert expected_rms_radius(0.0) == 0.0
    assert expected_rms_radius(1.0) == pytest.approx(math.sqrt(2.0))
```

Extend `tests/test_noise_neighborhood_demo.py`:

```python
def test_alpha_one_neighbor_is_exact_seeded_epsilon() -> None:
    runner = load_runner()
    parent = torch.ones((1, 2, 3), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(17)
    expected = torch.randn(parent.shape, generator=generator, dtype=torch.float32)

    torch.testing.assert_close(runner.make_neighbor(parent, 1.0, 17), expected, rtol=0, atol=0)
```

- [ ] **Step 2: Sync only the tests and verify RED remotely**

Run locally:

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -a tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/tests/
```

Run remotely:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py -q'
```

Expected: collection fails because `decision_diameter` does not exist; after adding only an empty module, the alpha-one test fails because `make_neighbor` rejects `1.0`.

- [ ] **Step 3: Implement the pure radial helpers and alpha-one endpoint**

Start `ttsd/runners/generate/decision_diameter.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Sequence


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
    if not values or any(left >= right for left, right in zip(values, values[1:])):
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
```

Change the guard in `make_neighbor` to:

```python
if not 0 <= alpha <= 1:
    raise ValueError("alpha must satisfy 0 <= alpha <= 1")
```

- [ ] **Step 4: Sync implementation and verify GREEN remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -aR ./ttsd/runners/generate/decision_diameter.py \
  ./ttsd/runners/generate/noise_neighborhood_demo.py \
  ./tests/test_decision_diameter.py ./tests/test_noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py -q'
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add ttsd/runners/generate/decision_diameter.py ttsd/runners/generate/noise_neighborhood_demo.py tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py
git commit -m "Add radial decision scan primitives"
```

---

### Task 2: Validate scan configs and sample plans

**Files:**
- Modify: `ttsd/runners/generate/decision_diameter.py`
- Modify: `tests/test_decision_diameter.py`

**Interfaces:**
- Produces: `validate_scan_config(payload: Mapping[str, object]) -> dict[str, object]`
- Produces: `load_scan_config(path: Path) -> dict[str, object]`
- Produces: `scan_config_sha256(config: Mapping[str, object]) -> str`
- Produces: `make_shell_plan(config: Mapping[str, object], alpha: float, *, start_index: int) -> dict[str, object]`
- Produces: `validate_sample_plan(payload: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]`
- Scan config keys: `version`, `prompt`, `input_path`, `input_sha256`, `parent_seed`, `parent_label`, `semantic_criterion`, `direction_seeds`, `coarse_alphas`, `alpha_tolerance`.
- Sample plan shape: `{"version": 1, "kind": "coarse" | "refinement", "samples": [...]}`.

- [ ] **Step 1: Write failing config and plan tests**

Append literal fixtures to `tests/test_decision_diameter.py`:

```python
from pathlib import Path

from ttsd.runners.generate.decision_diameter import (
    make_shell_plan,
    scan_config_sha256,
    validate_sample_plan,
    validate_scan_config,
)


def scan_config() -> dict[str, object]:
    return {
        "version": 1,
        "prompt": "A bud opens.",
        "input_path": "runs/flower/input.png",
        "input_sha256": "a" * 64,
        "parent_seed": 0,
        "parent_label": "failure",
        "semantic_criterion": "The petals visibly open.",
        "direction_seeds": [10000, 10001],
        "coarse_alphas": [0.2, 0.4, 1.0],
        "alpha_tolerance": 0.02,
    }


def test_scan_config_validation_and_hash_are_deterministic() -> None:
    first = validate_scan_config(scan_config())
    reordered = dict(reversed(list(scan_config().items())))
    second = validate_scan_config(reordered)
    assert first == second
    assert scan_config_sha256(first) == scan_config_sha256(second)
    assert len(scan_config_sha256(first)) == 64


def test_scan_config_requires_independent_endpoint_binary_parent_and_representable_tolerance() -> None:
    bad_endpoint = scan_config()
    bad_endpoint["coarse_alphas"] = [0.2, 0.8]
    with pytest.raises(ValueError, match="end at 1.0"):
        validate_scan_config(bad_endpoint)

    bad_label = scan_config()
    bad_label["parent_label"] = "ambiguous"
    with pytest.raises(ValueError, match="success or failure"):
        validate_scan_config(bad_label)

    bad_tolerance = scan_config()
    bad_tolerance["alpha_tolerance"] = 0.00001
    with pytest.raises(ValueError, match="at least 0.0001"):
        validate_scan_config(bad_tolerance)


def test_shell_plan_uses_configured_direction_seeds() -> None:
    config = validate_scan_config(scan_config())
    assert make_shell_plan(config, 0.2, start_index=4) == {
        "version": 1,
        "kind": "coarse",
        "samples": [
            {
                "index": 4,
                "direction_index": 0,
                "alpha": 0.2,
                "perturb_seed": 10000,
                "sample_id": "d00_a02000",
            },
            {
                "index": 5,
                "direction_index": 1,
                "alpha": 0.2,
                "perturb_seed": 10001,
                "sample_id": "d01_a02000",
            },
        ],
    }


def test_sample_plan_rejects_direction_seed_mismatch() -> None:
    config = validate_scan_config(scan_config())
    plan = make_shell_plan(config, 0.2, start_index=0)
    plan["samples"][0]["perturb_seed"] = 999
    with pytest.raises(ValueError, match="direction 0.*10000"):
        validate_sample_plan(plan, config)
```

- [ ] **Step 2: Sync tests and verify RED remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -a tests/test_decision_diameter.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/tests/test_decision_diameter.py
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py -q'
```

Expected: import fails for `validate_scan_config` and the other new interfaces.

- [ ] **Step 3: Implement strict standard-library validation**

Add imports and constants:

```python
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

VALID_LABELS = {"success", "failure"}
DEFAULT_COARSE_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
```

Implement validation with these exact observable rules:

```python
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

    directions = tuple(int(seed) for seed in payload["direction_seeds"])
    alphas = tuple(float(alpha) for alpha in payload["coarse_alphas"])
    radial_specs(alphas, directions)
    if alphas[-1] != 1.0:
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
        "direction_seeds": list(directions),
        "coarse_alphas": list(alphas),
        "alpha_tolerance": tolerance,
    }


def load_scan_config(path: Path) -> dict[str, object]:
    return validate_scan_config(json.loads(path.read_text()))


def scan_config_sha256(config: Mapping[str, object]) -> str:
    encoded = json.dumps(validate_scan_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

Implement `make_shell_plan` by calling `radial_specs((alpha,), direction_seeds, start_index=...)`, changing only `kind` to `"coarse"`. Implement `validate_sample_plan` so every sample has exactly `index`, `direction_index`, `alpha`, `perturb_seed`, and `sample_id`; IDs and indices are unique; each ID equals `format_sample_id(direction_index, alpha)`; and each direction uses its configured seed.

- [ ] **Step 4: Sync and verify GREEN remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -aR ./ttsd/runners/generate/decision_diameter.py ./tests/test_decision_diameter.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py -q'
```

Expected: all decision-diameter tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add ttsd/runners/generate/decision_diameter.py tests/test_decision_diameter.py
git commit -m "Validate decision scan configurations"
```

---

### Task 3: Analyze manual labels and plan adaptive refinement

**Files:**
- Modify: `ttsd/runners/generate/decision_diameter.py`
- Modify: `tests/test_decision_diameter.py`

**Interfaces:**
- Produces: `analyze_scan(config: Mapping[str, object], manifest: Mapping[str, object], labels: Sequence[Mapping[str, object]]) -> dict[str, object]`
- Produces: `next_sample_plan(config: Mapping[str, object], manifest: Mapping[str, object], labels: Sequence[Mapping[str, object]]) -> dict[str, object] | None`
- A profile has `status`, `label_counts`, `directions`, `nearest`, and `typical`.
- A direction has `direction_index`, `status`, `lower_alpha`, `upper_alpha`, and `non_monotonic`.
- Status values are `expand`, `refine`, `complete`, `censored`, or `needs_adjudication`.

- [ ] **Step 1: Write failing boundary-analysis tests**

Add a four-direction fixture whose expected order statistics are hand-derived:

```python
from ttsd.runners.generate.decision_diameter import analyze_scan, next_sample_plan


def four_direction_config() -> dict[str, object]:
    config = scan_config()
    config["direction_seeds"] = [10000, 10001, 10002, 10003]
    config["coarse_alphas"] = [0.2, 0.4, 0.6, 1.0]
    return validate_scan_config(config)


def manifest_and_labels(label_grid: dict[int, list[str]]):
    config = four_direction_config()
    specs = radial_specs((0.2, 0.4, 0.6, 1.0), config["direction_seeds"])
    for spec in specs:
        spec["metrics"] = {
            "rms_distance": spec["alpha"],
            "cosine_similarity": 1.0 - spec["alpha"],
            "norm_ratio": 1.0,
        }
    labels = [
        {
            "sample_id": spec["sample_id"],
            "label": label_grid[spec["direction_index"]][
                (0.2, 0.4, 0.6, 1.0).index(spec["alpha"])
            ],
        }
        for spec in specs
    ]
    return config, {"neighbors": specs}, labels


def test_analysis_reports_nearest_and_r50_brackets() -> None:
    config, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "success", "success"],
            1: ["failure", "failure", "success", "success"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["nearest"]["alpha_interval"] == [0.2, 0.4]
    assert profile["typical"]["alpha_interval"] == [0.4, 0.6]
    assert profile["directions"][0]["status"] == "crossed"
    assert profile["directions"][0]["upper_sample_metrics"] == {
        "rms_distance": 0.4,
        "cosine_similarity": 0.6,
        "norm_ratio": 1.0,
    }
    assert profile["directions"][2]["status"] == "censored"


def test_all_parent_labels_at_alpha_one_are_censored() -> None:
    config, manifest, labels = manifest_and_labels(
        {direction: ["failure"] * 4 for direction in range(4)}
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["status"] == "censored"
    assert profile["nearest"]["alpha_interval"] is None
    assert profile["typical"]["alpha_interval"] is None
    assert next_sample_plan(config, manifest, labels) is None


def test_ambiguous_label_blocks_new_sampling() -> None:
    config = four_direction_config()
    specs = radial_specs((0.2,), config["direction_seeds"])
    for spec in specs:
        spec["metrics"] = {"rms_distance": 0.2, "cosine_similarity": 0.98, "norm_ratio": 1.0}
    labels = [{"sample_id": spec["sample_id"], "label": "failure"} for spec in specs]
    labels[0]["label"] = "ambiguous"
    manifest = {"neighbors": specs}
    profile = analyze_scan(config, manifest, labels)
    assert profile["status"] == "needs_adjudication"
    assert profile["ambiguous_sample_ids"] == ["d00_a02000"]
    assert next_sample_plan(config, manifest, labels) is None


def test_next_plan_expands_then_refines_crossing_midpoints() -> None:
    config = four_direction_config()
    first_shell = radial_specs((0.2,), config["direction_seeds"])
    for spec in first_shell:
        spec["metrics"] = {"rms_distance": 0.2, "cosine_similarity": 0.98, "norm_ratio": 1.0}
    first_labels = [{"sample_id": spec["sample_id"], "label": "failure"} for spec in first_shell]
    expansion = next_sample_plan(config, {"neighbors": first_shell}, first_labels)
    assert [sample["alpha"] for sample in expansion["samples"]] == [0.4] * 4

    _, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "success", "success"],
            1: ["failure", "failure", "success", "success"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    refinement = next_sample_plan(config, manifest, labels)
    assert refinement["kind"] == "refinement"
    assert [(sample["direction_index"], sample["alpha"]) for sample in refinement["samples"]] == [
        (0, 0.3),
        (1, 0.5),
    ]


def test_analysis_flags_label_return_after_first_flip() -> None:
    config, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "failure", "success"],
            1: ["failure", "failure", "failure", "failure"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["directions"][0]["non_monotonic"] is True
```

- [ ] **Step 2: Sync tests and verify RED remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -a tests/test_decision_diameter.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/tests/test_decision_diameter.py
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py -q'
```

Expected: imports fail for `analyze_scan` and `next_sample_plan`.

- [ ] **Step 3: Implement label validation and directional brackets**

Implement these rules directly, without a class hierarchy:

```python
def _opposite_label(parent_label: str) -> str:
    return "success" if parent_label == "failure" else "failure"


def _diameter_interval(alpha_interval: list[float] | None) -> list[float] | None:
    if alpha_interval is None:
        return None
    return [2 * expected_rms_radius(alpha) for alpha in alpha_interval]
```

`analyze_scan` must:

1. Project each manifest entry to the five plan keys (`index`, `direction_index`, `alpha`, `perturb_seed`, `sample_id`) and validate those projected records through `validate_sample_plan`; require each manifest entry's real `metrics` object to contain finite `rms_distance`, `cosine_similarity`, and `norm_ratio` values.
2. Require exactly one label for every manifest sample and reject labels for unknown IDs.
3. Group records by `direction_index`, sort by `alpha`, and prepend the implicit parent point `(0.0, parent_label)`.
4. Find each direction's first opposite definitive label; use the greatest preceding parent-labeled alpha as the lower bracket.
5. Mark a no-flip direction `censored` only when it has a definitive parent label at `alpha = 1`; otherwise mark it `open`.
6. Mark `non_monotonic` when a parent label occurs above the first opposite label.
7. Compute nearest bounds as the minimum of directional lower bounds and the minimum of directional upper bounds.
8. Compute `r50` bounds as the `ceil(direction_count / 2)` order statistic of lower and upper bounds; an infinite upper bound becomes a censored typical result.
9. Include both `alpha_interval` and `diameter_rms_interval` in nearest and typical output, and preserve the actual manifest metrics plus sample ID for every directional upper-boundary sample.

- [ ] **Step 4: Implement deterministic next-plan selection**

`next_sample_plan` must use this exact priority:

1. Return `None` if any label is ambiguous.
2. If fewer than half the directions have crossed and any configured coarse alpha is untested for an un-crossed direction, emit the next coarse alpha only for un-crossed directions.
3. Otherwise, emit one four-decimal-grid midpoint for every crossed direction whose first bracket is wider than `alpha_tolerance`: average the integer alpha codes with floor division, then divide by `10000`. This is the nearest lower midpoint when an exact midpoint is not representable in the sample ID.
4. Assign indexes starting at `max(existing indexes) + 1`, sort samples by direction index, and use each direction's configured seed.
5. Return `None` when all required finite brackets meet tolerance and the remaining results are complete or censored.

Build refinement specs explicitly because different directions can receive different midpoint alphas; validate the resulting plan before returning it.

- [ ] **Step 5: Sync implementation and verify GREEN remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -aR ./ttsd/runners/generate/decision_diameter.py ./tests/test_decision_diameter.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py -q'
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add ttsd/runners/generate/decision_diameter.py tests/test_decision_diameter.py
git commit -m "Analyze decision diameter boundaries"
```

---

### Task 4: Make the generation manifest append-only and plan-driven

**Files:**
- Modify: `ttsd/runners/generate/noise_neighborhood_demo.py:282-465,517-544`
- Modify: `tests/test_noise_neighborhood_demo.py:66-253`

**Interfaces:**
- Produces: `_apply_scan_config(config: Mapping[str, object]) -> None`
- Produces: `_append_neighbors(output_root: Path, parent_noise: torch.Tensor, specs: Sequence[Mapping[str, object]]) -> dict[str, object]`
- Changes: `_prepare_neighbors(output_root, parent_noise, specs=None)` uses legacy `neighbor_specs()` when `specs is None`.
- Changes: `_validate_preparation(output_root, required_specs=None)` allows extra append-only entries and verifies every listed noise tensor.
- Changes: `_selected_specs(..., sample_ids: set[str] | None = None)` intersects sharding/indices with an optional plan selection.
- Adds CLI: `--scan-config PATH`, `--sample-plan PATH`, and `--append-plan`.

**CLI contract:**

- Legacy command without new flags behaves exactly as before.
- `--append-plan` requires `--sample-plan`, rejects sharded execution, prepares or appends once, and may be combined with `--prepare-only`.
- `--sample-plan` without `--append-plan` requires every plan sample already in the manifest and generates only those sample IDs.
- Parallel generation begins only after the single append command exits successfully.

- [ ] **Step 1: Write failing append, replay, conflict, and selection tests**

Add to `tests/test_noise_neighborhood_demo.py`:

```python
def radial_entry(index=100, alpha=0.4, seed=10000):
    return {
        "index": index,
        "direction_index": 0,
        "alpha": alpha,
        "perturb_seed": seed,
        "sample_id": f"d00_a{round(alpha * 10000):05d}",
    }


def test_append_neighbors_is_idempotent_and_preserves_legacy_entries(tmp_path) -> None:
    runner = load_runner()
    _write_complete_preparation_bundle(runner, tmp_path)
    parent = torch.zeros((1,), dtype=torch.float32)
    before = json.loads((tmp_path / "manifest.json").read_text())

    first = runner._append_neighbors(tmp_path, parent, [radial_entry()])
    second = runner._append_neighbors(tmp_path, parent, [radial_entry()])

    assert len(first["neighbors"]) == len(before["neighbors"]) + 1
    assert second == first
    assert (tmp_path / "noise/d00_a04000.pt").is_file()


def test_failed_append_does_not_publish_partial_manifest(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    _write_complete_preparation_bundle(runner, tmp_path)
    parent = torch.zeros((1,), dtype=torch.float32)
    before = (tmp_path / "manifest.json").read_bytes()
    original_save = runner._atomic_save_tensor

    def fail_second(path, tensor):
        if path.name == "d01_a04000.pt":
            raise OSError("injected save failure")
        original_save(path, tensor)

    monkeypatch.setattr(runner, "_atomic_save_tensor", fail_second)
    second = dict(radial_entry(index=101))
    second.update(direction_index=1, perturb_seed=10001, sample_id="d01_a04000")
    with pytest.raises(OSError, match="injected save failure"):
        runner._append_neighbors(tmp_path, parent, [radial_entry(), second])

    assert (tmp_path / "manifest.json").read_bytes() == before


def test_append_neighbors_rejects_conflicting_replay_or_direction_seed(tmp_path) -> None:
    runner = load_runner()
    _write_complete_preparation_bundle(runner, tmp_path)
    parent = torch.zeros((1,), dtype=torch.float32)
    runner._append_neighbors(tmp_path, parent, [radial_entry()])

    conflict = radial_entry(alpha=0.5)
    conflict["sample_id"] = "d00_a04000"
    with pytest.raises(RuntimeError, match="conflicting manifest sample"):
        runner._append_neighbors(tmp_path, parent, [conflict])

    with pytest.raises(RuntimeError, match="direction 0.*perturbation seed"):
        runner._append_neighbors(tmp_path, parent, [radial_entry(index=101, alpha=0.6, seed=10001)])


def test_plan_selection_generates_only_requested_manifest_samples(tmp_path) -> None:
    runner = load_runner()
    _write_complete_preparation_bundle(runner, tmp_path)
    parent = torch.zeros((1,), dtype=torch.float32)
    entry = radial_entry()
    manifest = runner._append_neighbors(tmp_path, parent, [entry])

    selected = runner._selected_specs(manifest, 0, 1, None, {entry["sample_id"]})
    assert [sample["sample_id"] for sample in selected] == ["d00_a04000"]


def test_parser_rejects_parallel_manifest_append() -> None:
    runner = load_runner()
    parser = runner.build_parser()
    args = parser.parse_args(
        ["--sample-plan", "plan.json", "--append-plan", "--shard-index", "0", "--num-shards", "2"]
    )
    with pytest.raises(ValueError, match="append-plan.*shard"):
        runner._validate_cli_args(args)
```

Add one stale-config test that uses `monkeypatch.setattr` to set `runner.SCAN_CONFIG_SHA256 = "b" * 64`, writes a complete bundle, changes the monkeypatched hash to `"c" * 64`, and asserts `_validate_preparation` reports `manifest.scan_config_sha256` without leaking global state to later tests.

- [ ] **Step 2: Sync tests and verify RED remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -a tests/test_noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/tests/test_noise_neighborhood_demo.py
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_noise_neighborhood_demo.py -q'
```

Expected: tests fail because `_append_neighbors`, `_validate_cli_args`, and plan-aware selection do not exist.

- [ ] **Step 3: Refactor manifest creation without changing legacy defaults**

Add `SCAN_CONFIG_SHA256: str | None = None`. Include it in manifest and parent metadata only when set:

```python
def _expected_prepare_fields() -> dict[str, object]:
    fields = {
        "prompt": PROMPT,
        "input_path": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "model_path": str(MODEL),
        "scheduler_class": SCHEDULER_CLASS,
        "parent_seed": PARENT_SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "fps": FPS,
    }
    if SCAN_CONFIG_SHA256 is not None:
        fields["scan_config_sha256"] = SCAN_CONFIG_SHA256
    return fields
```

Change `_prepare_neighbors` to accept an optional explicit spec list, create the fixed manifest header once, and delegate tensor creation to `_append_neighbors`. `_append_neighbors` must validate every conflict before writing, then atomically rewrite `manifest.json` only after every new tensor is safely written. A tensor-save failure may leave an unreferenced tensor that a retry safely overwrites, but the published manifest must remain byte-for-byte unchanged. Exact replays are no-ops.

Change `_validate_preparation` to validate every manifest entry and every referenced noise file, then require the requested specs as a subset. With `required_specs is None`, require the legacy `neighbor_specs()` subset so all old tests and invocations retain their contract.

- [ ] **Step 4: Apply immutable scan configuration and load plans**

At process start, load `--scan-config` through `load_scan_config`, then set only:

```python
INPUT = Path(config["input_path"])
INPUT_SHA256 = str(config["input_sha256"])
PROMPT = str(config["prompt"])
PARENT_SEED = int(config["parent_seed"])
SCAN_CONFIG_SHA256 = scan_config_sha256(config)
```

Load `--sample-plan` through `validate_sample_plan`. Never execute arbitrary Python from configuration.

- [ ] **Step 5: Implement CLI validation and plan-only selection**

Extend `build_parser()` with:

```python
parser.add_argument("--scan-config", type=Path)
parser.add_argument("--sample-plan", type=Path)
parser.add_argument("--append-plan", action="store_true")
```

Add `_validate_cli_args(args)` with these literal branches:

```python
if args.append_plan and args.sample_plan is None:
    raise ValueError("--append-plan requires --sample-plan")
if args.append_plan and (args.num_shards != 1 or args.shard_index not in (None, 0)):
    raise ValueError("--append-plan cannot run in a shard")
```

In `main`, an append command prepares a missing root with the plan specs or appends to an existing validated root. A later command with the same `--sample-plan` but no `--append-plan` validates that plan as a manifest subset and passes its sample IDs to `_selected_specs`. Keep the old auto-prepare and shard behavior when no plan is provided.

- [ ] **Step 6: Sync and verify targeted plus legacy GREEN remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -aR ./ttsd/runners/generate/noise_neighborhood_demo.py ./tests/test_noise_neighborhood_demo.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_noise_neighborhood_demo.py tests/test_decision_diameter.py -q'
```

Expected: all tests pass, including the original fixed 32-neighbor assertions.

- [ ] **Step 7: Commit Task 4**

```bash
git add ttsd/runners/generate/noise_neighborhood_demo.py tests/test_noise_neighborhood_demo.py
git commit -m "Support append-only noise sample plans"
```

---

### Task 5: Add the planner CLI, runbook, and end-to-end acceptance check

**Files:**
- Modify: `ttsd/runners/generate/decision_diameter.py`
- Modify: `tests/test_decision_diameter.py`
- Create: `docs/decision_diameter_calibration.md`
- Generate without committing: `runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1/`

**Interfaces:**
- Produces CLI: `python -m ttsd.runners.generate.decision_diameter --config CONFIG [--manifest MANIFEST] [--labels LABELS] --profile-out PROFILE --plan-out PLAN`
- `--manifest` and `--labels` may both be absent only for the first plan.
- Writes profile JSON every run; removes no prior artifact; writes a plan only when more samples are required.

- [ ] **Step 1: Write a failing real CLI test**

Append to `tests/test_decision_diameter.py`:

```python
import json
import sys

from ttsd.runners.generate import decision_diameter


def test_cli_writes_first_plan_without_importing_video_model(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    profile_path = tmp_path / "profile.json"
    plan_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(scan_config()))
    sys.modules.pop("diffusers", None)

    decision_diameter.main(
        [
            "--config",
            str(config_path),
            "--profile-out",
            str(profile_path),
            "--plan-out",
            str(plan_path),
        ]
    )

    assert json.loads(profile_path.read_text())["status"] == "expand"
    plan = json.loads(plan_path.read_text())
    assert [sample["alpha"] for sample in plan["samples"]] == [0.2, 0.2]
    assert "diffusers" not in sys.modules
```

- [ ] **Step 2: Sync and verify RED remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -a tests/test_decision_diameter.py \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/tests/test_decision_diameter.py
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py::test_cli_writes_first_plan_without_importing_video_model -q'
```

Expected: fails because `main` does not exist.

- [ ] **Step 3: Implement the dependency-free JSON CLI**

Use `argparse`, `json`, `tempfile`, and `os.replace`. With no manifest, analyze an empty scan and emit the first coarse shell. With a manifest, require labels and call `analyze_scan` plus `next_sample_plan`. Atomically write `profile-out`; atomically write `plan-out` only when the returned plan is not `None`, and delete a stale `plan-out` when calibration is complete.

Parser:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path, required=True)
    return parser
```

- [ ] **Step 4: Write the operator runbook**

Create `docs/decision_diameter_calibration.md` with these sections and exact defaults:

```markdown
# Decision-diameter calibration

1. Create one immutable scan config with the prompt, input hash, parent seed/label, semantic criterion, eight direction seeds, coarse alphas, and `0.02` tolerance.
2. Run the planner without a manifest to obtain the first plan.
3. Apply the plan once with `--append-plan --prepare-only`.
4. Generate that plan with one or more shards, without `--append-plan`.
5. Review every all-frame sheet and append manual labels.
6. Run the planner again; repeat until no next plan is emitted.

`D50` is the main comparison value. Always retain its interval, nearest-diameter interval, direction count, censored directions, ambiguous cases, and non-monotonic flags.
```

Include this command cycle verbatim, followed by a warning that two processes must never use `--append-plan` concurrently:

```bash
python -m ttsd.runners.generate.decision_diameter \
  --config scan_config.json \
  --profile-out decision_profile.json \
  --plan-out next_plan.json

python -m ttsd.runners.generate.noise_neighborhood_demo \
  --scan-config scan_config.json \
  --sample-plan next_plan.json \
  --append-plan --prepare-only \
  --output-root "$RUN_ROOT"

CUDA_VISIBLE_DEVICES=0 python -m ttsd.runners.generate.noise_neighborhood_demo \
  --scan-config scan_config.json \
  --sample-plan next_plan.json \
  --output-root "$RUN_ROOT" \
  --shard-index 0 --num-shards 2
CUDA_VISIBLE_DEVICES=1 python -m ttsd.runners.generate.noise_neighborhood_demo \
  --scan-config scan_config.json \
  --sample-plan next_plan.json \
  --output-root "$RUN_ROOT" \
  --shard-index 1 --num-shards 2

python -m ttsd.runners.generate.decision_diameter \
  --config scan_config.json \
  --manifest "$RUN_ROOT/manifest.json" \
  --labels review_labels.json \
  --profile-out decision_profile.json \
  --plan-out next_plan.json
```

- [ ] **Step 5: Run all tests and Ruff remotely**

```bash
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
rsync -aR ./ttsd/runners/generate/decision_diameter.py \
  ./ttsd/runners/generate/noise_neighborhood_demo.py \
  ./tests/test_decision_diameter.py ./tests/test_noise_neighborhood_demo.py \
  ./docs/decision_diameter_calibration.md \
  yukelab:/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py -q && ruff check ttsd/runners/generate/decision_diameter.py ttsd/runners/generate/noise_neighborhood_demo.py tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py'
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit Task 5**

```bash
git add ttsd/runners/generate/decision_diameter.py tests/test_decision_diameter.py docs/decision_diameter_calibration.md
git commit -m "Document decision diameter calibration"
```

- [ ] **Step 7: Request an independent code review**

Invoke `superpowers:requesting-code-review` against the committed diff from `7a66892` through current HEAD. Resolve only verified correctness, reproducibility, stale-state, or protocol-definition findings, rerun the full remote verification after every fix, and commit accepted fixes before model-backed acceptance.

- [ ] **Step 8: Create the ignored flower scan config and emit the first plan**

Use the proven prompt and parent:

```json
{
  "version": 1,
  "prompt": "A closed yellow flower bud opens its petals into a fully open flower.",
  "input_path": "runs/flower_bloom_i2v_v1/baseline_seed_sweep_v1/input.png",
  "input_sha256": "f5fe71e6e3f7e37fcd1a4e814abe2bea13a8b8de1e27af4e587422672c879c96",
  "parent_seed": 0,
  "parent_label": "failure",
  "semantic_criterion": "The initially closed bud clearly becomes a recognizable open flower at least once.",
  "direction_seeds": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007],
  "coarse_alphas": [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
  "alpha_tolerance": 0.02
}
```

Write that JSON to the ignored run root as `scan_config.json`, sync it to `yukelab`, then run and verify remotely:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && root=runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1 && python -m ttsd.runners.generate.decision_diameter --config "$root/scan_config.json" --profile-out "$root/decision_profile.json" --plan-out "$root/next_plan.json" && python - <<'"'"'PY'"'"'
import json
from pathlib import Path
root = Path("runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1")
plan = json.loads((root / "next_plan.json").read_text())
samples = plan["samples"]
assert len(samples) == 8
assert [row["sample_id"] for row in samples] == [f"d{i:02d}_a00200" for i in range(8)]
assert [row["perturb_seed"] for row in samples] == list(range(10000, 10008))
assert {row["alpha"] for row in samples} == {0.02}
print("first decision-diameter plan verified")
PY'
```

- [ ] **Step 9: Perform one complete model-backed iteration**

Run one unsharded append/prepare command:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && root=runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1 && CUDA_VISIBLE_DEVICES=4 python -m ttsd.runners.generate.noise_neighborhood_demo --scan-config "$root/scan_config.json" --sample-plan "$root/next_plan.json" --append-plan --prepare-only --output-root "$root"'
```

Sync and inspect `parent_control/all_frames.jpg`; stop if reinjection does not reproduce the failed seed 0 behavior. Then launch the eight plan samples on two GPUs:

```bash
ssh yukelab 'set -e
wt=/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
root=runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1
for gpu in 4 7; do tmux kill-session -t diameter_accept_gpu${gpu} 2>/dev/null || true; done
cmd4="cd $wt && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=4 python -u -m ttsd.runners.generate.noise_neighborhood_demo --scan-config $root/scan_config.json --sample-plan $root/next_plan.json --output-root $root --shard-index 0 --num-shards 2"
cmd7="cd $wt && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=7 python -u -m ttsd.runners.generate.noise_neighborhood_demo --scan-config $root/scan_config.json --sample-plan $root/next_plan.json --output-root $root --shard-index 1 --num-shards 2"
tmux new-session -d -s diameter_accept_gpu4 "$cmd4"
tmux new-session -d -s diameter_accept_gpu7 "$cmd7"'
```

After eight `DONE` markers exist, manually inspect every all-frame sheet and write one `success`, `failure`, or `ambiguous` label per sample to `review_labels.json`. Rerun the planner:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && root=runs/flower_bloom_i2v_v1/decision_diameter_seed0_v1 && python -m ttsd.runners.generate.decision_diameter --config "$root/scan_config.json" --manifest "$root/manifest.json" --labels "$root/review_labels.json" --profile-out "$root/decision_profile.json" --plan-out "$root/next_plan.json"'
```

Acceptance conditions:

- parent bundle is generated once and reused;
- manifest has exactly the eight planned direction/alpha records;
- every record has a tensor, video, sheet, metadata, and completion marker;
- rerunning the append command is a no-op;
- after eight complete labels, the planner's next action agrees with those labels: if all remain failures it emits all eight `alpha = 0.05` samples with the same seeds; a flip, ambiguity, or mixed shell instead follows the tested refinement/adjudication rules;
- no tracked file changes during model-backed execution.

- [ ] **Step 10: Run final verification before completion**

Invoke `superpowers:verification-before-completion`, then freshly run:

```bash
ssh yukelab 'cd /data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo && source .venv/bin/activate && pytest tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py -q && ruff check ttsd/runners/generate/decision_diameter.py ttsd/runners/generate/noise_neighborhood_demo.py tests/test_decision_diameter.py tests/test_noise_neighborhood_demo.py'
cd /home/atlas/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo
git status --short
git log -5 --oneline
```

Expected: tests and Ruff pass, Git shows no unintended tracked changes, and the recent commits correspond to the reviewed tasks above.
