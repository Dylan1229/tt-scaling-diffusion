"""Typed config dataclasses + YAML loader.

A whole experiment is described by one YAML file. Each component subconfig
has `kind: <registry key>` + `params: <free-form dict>`. The orchestrator
resolves `kind` via the registry and passes `params` to the constructor.

See docs/e2e_framework_plan.md §6 for example configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    wall_clock_s: float | None = None
    gpu_seconds: float | None = None
    vlm_tokens: int | None = None


@dataclass
class LogConfig:
    path: str = "runs/pipeline/{run_id}/events.jsonl"


@dataclass
class GenerationDefaults:
    """Default GenerationRequest params used when the caller doesn't override."""
    num_frames: int = 81
    height: int = 480
    width: int = 832
    num_inference_steps: int = 50
    guidance_scale: float = 5.0


@dataclass
class PipelineConfig:
    model: ModelConfig
    verifier: VerifierConfig
    policy: PolicyConfig
    strategy: StrategyConfig
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    log: LogConfig = field(default_factory=LogConfig)
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    output_root: str = "runs/pipeline"


def load_config(path: str | Path) -> PipelineConfig:
    """Parse a YAML file into a PipelineConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> PipelineConfig:
    """Coerce a plain dict (e.g. parsed YAML) into a PipelineConfig."""
    return PipelineConfig(
        model=ModelConfig(**raw["model"]),
        verifier=VerifierConfig(**raw["verifier"]),
        policy=PolicyConfig(**raw["policy"]),
        strategy=StrategyConfig(**raw["strategy"]),
        budget=BudgetConfig(**raw.get("budget", {})),
        log=LogConfig(**raw.get("log", {})),
        generation=GenerationDefaults(**raw.get("generation", {})),
        output_root=raw.get("output_root", "runs/pipeline"),
    )
