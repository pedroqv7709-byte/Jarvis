"""Outbound responses returned to interfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jarvis.core.action import ActionStatus, ConfirmationRequest, VerificationStatus
from jarvis.core.decisions import Decision
from jarvis.core.errors import Failure
from jarvis.core.task import TaskState


class ActionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: str
    tool: str
    description: str
    status: ActionStatus
    decision: Decision | None = None
    verification: VerificationStatus | None = None
    attempts: int = 0
    output: dict[str, Any] | None = None
    failure: Failure | None = None


class Response(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    task_id: str
    state: TaskState
    message: str
    planner: str | None = None
    """Planner that produced the executed plan (``model``, ``rule_based``...)."""
    fallback_reason: str | None = None
    """Set when a fallback planner was used; explains why the preferred one was not."""
    actions: tuple[ActionSummary, ...] = ()
    pending_confirmations: tuple[ConfirmationRequest, ...] = ()
    failure: Failure | None = None
