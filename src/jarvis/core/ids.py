"""Identifier and clock helpers shared by every module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def new_id(prefix: str) -> str:
    """Return a time-sortable unique identifier such as ``task_0199...``."""
    return f"{prefix}_{uuid.uuid7().hex}"


def utcnow() -> datetime:
    """Timezone-aware current time in UTC."""
    return datetime.now(UTC)
