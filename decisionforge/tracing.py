from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    """Append-only event log. Every agent decision and tool call lands here."""

    def __init__(self, run_dir: str | Path | None = None):
        self.events: list[dict[str, Any]] = []
        self.run_dir = Path(run_dir) if run_dir else None
        self._t0 = time.time()
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **data: Any) -> None:
        ev = {"t": round(time.time() - self._t0, 3), "kind": kind, **data}
        self.events.append(ev)
        if self.run_dir:
            with open(self.run_dir / "trace.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
