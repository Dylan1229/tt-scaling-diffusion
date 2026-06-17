"""JSON-lines event logger.

One line per event (step / verifier_call / policy_decision / action_dispatched /
budget_update). Cheap to parse, append-only, mirrors the existing `runs/_logs/`
pattern. Every event gets a wall-clock timestamp from the logger.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)  # line-buffered

    def log(self, event: str, **fields: Any) -> None:
        """Log one event. `event` is the type tag (free-form string).
        `fields` are arbitrary JSON-serializable values."""
        row: dict[str, Any] = {"ts": time.time(), "event": event}
        row.update(fields)
        self._f.write(json.dumps(row, default=_json_default) + "\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """Fallback for tensors, paths, etc."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)
