"""Plugin registry — the one extensibility hook for the pipeline.

Components (verifiers, policies, actions, strategies, model adapters) are
registered by short name via decorator. The orchestrator never imports
concrete classes; it looks them up by the `kind` string in the config.

Two lookup paths:
  - Short name registered with @register_<kind>("name") — built-in plugins.
  - "module.path:ClassName" string — escape hatch for third-party adds without
    editing this file. Useful for experiment-specific verifiers or any
    out-of-tree extensions.

See docs/e2e_framework_plan.md §9 decision 1.
"""

from __future__ import annotations

import importlib


class Registry:
    """A name → class table for one component kind (verifier / policy / ...)."""

    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        """Decorator: @REG.register("foo") binds the class under that name."""
        def _decorate(cls: type) -> type:
            if name in self._items and self._items[name] is not cls:
                raise ValueError(
                    f"{self._kind} '{name}' already registered to {self._items[name]!r}; "
                    f"refusing to clobber with {cls!r}"
                )
            self._items[name] = cls
            return cls
        return _decorate

    def get(self, name_or_path: str) -> type:
        """Resolve a name or 'module.path:ClassName' string to a class."""
        if name_or_path in self._items:
            return self._items[name_or_path]
        if ":" in name_or_path:
            mod_path, cls_name = name_or_path.split(":", 1)
            try:
                module = importlib.import_module(mod_path)
            except ImportError as e:
                raise KeyError(
                    f"cannot import '{mod_path}' for {self._kind} '{name_or_path}': {e}"
                ) from e
            cls = getattr(module, cls_name, None)
            if cls is None:
                raise KeyError(f"'{mod_path}' has no attribute '{cls_name}'")
            return cls
        raise KeyError(
            f"unknown {self._kind} '{name_or_path}'. Registered: {sorted(self._items)}. "
            f"Or use 'module.path:ClassName' for an out-of-tree class."
        )

    def names(self) -> list[str]:
        return sorted(self._items)


# Singletons, one per plugin kind.
MODELS = Registry("model")
VERIFIERS = Registry("verifier")
POLICIES = Registry("policy")
ACTIONS = Registry("action")
STRATEGIES = Registry("strategy")

register_model = MODELS.register
register_verifier = VERIFIERS.register
register_policy = POLICIES.register
register_action = ACTIONS.register
register_strategy = STRATEGIES.register
