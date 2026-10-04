"""Inbound requests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.ids import new_id, utcnow


class Request(BaseModel):
    """Something the user asked Jarvis to do, as received from an interface."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("req"))
    text: str
    user_id: str
    interface: str = "cli"
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
