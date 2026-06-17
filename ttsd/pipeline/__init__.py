"""Pipeline package — the inference-time backbone for test-time scaling.

Public API:
    from ttsd.pipeline import Orchestrator, load_config

Importing this package triggers all built-in plugin registrations so the
registry tables are populated by the time a config is loaded.

See docs/e2e_framework_plan.md for the full design.
"""

# ── Register built-in plugins. Order doesn't matter; each module's
#    side-effect imports populate the relevant Registry singleton. ────────────
import ttsd.pipeline.actions       # noqa: F401  continue / stop_and_fail / anchor_inject / refine_prompt_vlm / stop_and_accept
import ttsd.pipeline.policy        # noqa: F401  noop / fixed_threshold / dynamic_sliding_window
import ttsd.pipeline.vlm           # noqa: F401  noop / quality_modifier_stub VLM clients
import ttsd.search.sequential      # noqa: F401  sequential_trial
import ttsd.verifiers.dino.online_adapter  # noqa: F401  dino_frame_cos_mean_online
import ttsd.verifiers.noop         # noqa: F401  noop verifier

from ttsd.pipeline.config import PipelineConfig, load_config
from ttsd.pipeline.orchestrator import Orchestrator

__all__ = ["Orchestrator", "PipelineConfig", "load_config"]
