"""Pluggable VLM client interface for Trial 2 prompt refinement.

We deliberately defer choosing a provider (per §9 decision 3 of the plan).
Concrete clients are registered by short name; the action picks one from its
config at instantiation time. v1 ships a NoOpVLMClient that appends quality
modifiers without making any external call — sufficient to demonstrate the
mechanism without depending on an API key.

To add a real provider:

    @register_vlm_client("claude_haiku_45")
    class ClaudeHaikuVLM:
        def __init__(self, api_key: str | None = None, ...):
            import anthropic  # lazy
            ...
        def refine_prompt(self, original_prompt: str, frames: np.ndarray | None) -> str:
            ...

The Action just looks the client up by `kind` and calls `refine_prompt(...)`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ttsd.pipeline.registry import Registry

VLM_CLIENTS = Registry("vlm_client")
register_vlm_client = VLM_CLIENTS.register


@runtime_checkable
class VLMClient(Protocol):
    """A VLM client must implement one method."""

    def refine_prompt(
        self,
        original_prompt: str,
        frames: np.ndarray | None,
    ) -> str:
        """Given the original prompt and optional reference frames of the
        failed video, return a refined prompt. Frames are (F, H, W, 3) uint8
        or None when no preview is available."""
        ...


def build_vlm_client(spec: dict) -> VLMClient:
    """Resolve a {'kind': '...', 'params': {...}} dict to a concrete client."""
    cls = VLM_CLIENTS.get(spec["kind"])
    return cls(**spec.get("params", {}))


@register_vlm_client("noop")
class NoOpVLMClient:
    """Returns the original prompt verbatim (or with an optional suffix).

    Doesn't make any external call; safe to use in CI, smoke tests, and
    anywhere an API key isn't available. Demonstrates the action mechanism."""

    def __init__(self, suffix: str = ""):
        self.suffix = suffix

    def refine_prompt(self, original_prompt: str, frames: np.ndarray | None) -> str:
        if self.suffix:
            return f"{original_prompt}, {self.suffix}".strip()
        return original_prompt


@register_vlm_client("quality_modifier_stub")
class QualityModifierStub:
    """Stub for real VLMs: appends common 'quality booster' phrases. Useful
    as a more visible test than NoOp (the prompt actually changes), still
    without any network call."""

    DEFAULT_SUFFIX = "high quality, detailed, smooth motion, photorealistic"

    def __init__(self, suffix: str | None = None):
        self.suffix = suffix or self.DEFAULT_SUFFIX

    def refine_prompt(self, original_prompt: str, frames: np.ndarray | None) -> str:
        return f"{original_prompt}, {self.suffix}".strip()
