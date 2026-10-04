"""Observations: snapshots of the world the agent loop looks at before deciding.

Observation data (window titles, tool outputs) is untrusted: it is shown to planners as data and
never interpreted as instructions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.ids import utcnow


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    status: Literal["ok", "unavailable", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    identifiers: tuple[str, ...] = ()
    """Stable ids the planner may refer to (e.g. window handles)."""
    at: datetime = Field(default_factory=utcnow)
