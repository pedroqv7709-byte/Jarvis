"""Redaction applied to anything written to the audit log."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY = re.compile(
    r"(passw(or)?d|passphrase|secret|token|api[_-]?key|private[_-]?key|authorization|"
    r"credential|cookie|^pin$|cvv)",
    re.IGNORECASE,
)
MAX_STRING = 200
MAX_ITEMS = 50
REDACTED = "***"


def redact(value: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-safe copy with secrets masked and large values truncated."""
    if _depth > 6:
        return "<nested>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (key, item) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                out["<truncated>"] = f"{len(value) - MAX_ITEMS} more keys"
                break
            key_s = str(key)
            out[key_s] = (
                REDACTED if SENSITIVE_KEY.search(key_s) else redact(item, _depth=_depth + 1)
            )
        return out
    if isinstance(value, list | tuple):
        items = [redact(v, _depth=_depth + 1) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"<{len(value) - MAX_ITEMS} more items>")
        return items
    if isinstance(value, str):
        if len(value) > MAX_STRING:
            return f"{value[:MAX_STRING]}...<{len(value)} chars>"
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    return redact(str(value), _depth=_depth + 1)
