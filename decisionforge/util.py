from __future__ import annotations

import math
from typing import Any


def get_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'groups.0.delta' inside nested dict/list data."""
    cur = obj
    for part in path.split("."):
        try:
            if isinstance(cur, list):
                cur = cur[int(part)]
            elif isinstance(cur, dict):
                cur = cur[part]
            else:
                return None
        except (KeyError, IndexError, ValueError):
            return None
    return cur


def num(x: Any, nd: int = 4) -> float | None:
    """JSON-safe rounded float (None for NaN/None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)
