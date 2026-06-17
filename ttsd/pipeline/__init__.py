"""Pipeline package — the inference-time backbone for test-time scaling.

Public API:
    from ttsd.pipeline import Orchestrator, load_config

See docs/e2e_framework_plan.md for the full design.
"""

from ttsd.pipeline.config import PipelineConfig, load_config
from ttsd.pipeline.orchestrator import Orchestrator

__all__ = ["Orchestrator", "PipelineConfig", "load_config"]
