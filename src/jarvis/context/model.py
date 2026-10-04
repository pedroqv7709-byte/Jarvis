"""The explicit execution context handed to planners, policy and tools.

The context is rebuilt for every run (including resumes) from the persisted task plus the current
providers and grants; it is not itself persisted. That way a resumed task is evaluated against the
permissions in force *now*, not the ones in force when it started.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.ids import utcnow
from jarvis.core.task import TaskState
from jarvis.permissions.model import PermissionSet


class UserProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str | None = None


class ApplicationContext(BaseModel):
    """Where the request came from and what the user is working in."""

    model_config = ConfigDict(frozen=True)

    interface: str
    active_application: str | None = None
    """Foreground application, once a perception provider can report it."""
    workspace: str | None = None


class ContextItem(BaseModel):
    """A piece of relevant information contributed by a context provider."""

    model_config = ConfigDict(frozen=True)

    source: str
    key: str
    value: Any
    sensitive: bool = False
    """Sensitive items are available to the pipeline but never written to the audit log."""


class ExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    request_id: str
    goal: str
    user: UserProfile
    application: ApplicationContext
    permissions: PermissionSet
    task_state: TaskState
    items: tuple[ContextItem, ...] = ()
    degraded_sources: tuple[str, ...] = ()
    """Providers that failed while building this context."""
    ephemeral: dict[str, Any] = Field(default_factory=dict)
    """Temporary per-run data. Never persisted."""
    built_at: datetime = Field(default_factory=utcnow)
