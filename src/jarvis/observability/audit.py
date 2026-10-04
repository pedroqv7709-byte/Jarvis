"""Structured audit trail.

The audit log answers: what happened, for which task and action, with which tool, what the policy
decided and why, what the result was, whether it was verified, and which errors occurred.

It records operational facts only. It never stores model reasoning or chain-of-thought, and all
payloads pass through :func:`redact` before being written.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.core.action import ActionRecord
from jarvis.core.ids import new_id, utcnow
from jarvis.core.task import Task
from jarvis.observability.redaction import redact


class AuditEventType(StrEnum):
    REQUEST_RECEIVED = "request.received"
    REQUEST_REJECTED = "request.rejected"
    CONTEXT_BUILT = "context.built"
    AGENT_PLAN = "agent.plan"
    """A planner decision (next action / done / stop)."""
    PLAN_CREATED = "agent.plan"  # pre-v0.5 name, kept as an alias of AGENT_PLAN
    PLAN_ATTEMPT_FAILED = "plan.attempt_failed"
    PLAN_FAILED = "plan.failed"
    ACTION_SELECTED = "action.selected"
    ACTION_SELECTION_FAILED = "action.selection_failed"
    ACTION_RESOLVED = "action.resolved"
    ACTION_RESOLUTION_FAILED = "action.resolution_failed"
    POLICY_EVALUATED = "policy.evaluated"
    CONFIRMATION_REQUESTED = "confirmation.requested"
    CONFIRMATION_DECIDED = "confirmation.decided"
    ACTION_ATTEMPT_STARTED = "action.attempt_started"
    ACTION_EXECUTED = "action.executed"
    ACTION_ATTEMPT_FAILED = "action.attempt_failed"
    ACTION_RETRY_SCHEDULED = "action.retry_scheduled"
    ACTION_FAILED = "action.failed"
    ACTION_VERIFIED = "action.verified"
    AGENT_REQUIREMENTS = "agent.requirements"
    AGENT_OBSERVATION = "agent.observation"
    AGENT_ACTION_SELECTED = "agent.action_selected"
    AGENT_REPLAN = "agent.replan"
    AGENT_VERIFICATION = "agent.verification"
    AGENT_REQUIREMENT_MET = "agent.requirement_met"
    AGENT_LIMIT_REACHED = "agent.limit_reached"
    AGENT_CYCLE_DETECTED = "agent.cycle_detected"
    AGENT_COMPLETED = "agent.completed"
    AGENT_STOPPED = "agent.stopped"
    TASK_STATE_CHANGED = "task.state_changed"
    TASK_CANCEL_OBSERVED = "task.cancel_observed"
    TASK_RECOVERED = "task.recovered"
    RUNTIME_STARTED = "runtime.started"
    RUNTIME_STOPPED = "runtime.stopped"
    RESPONSE_SENT = "response.sent"


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("evt"))
    at: datetime = Field(default_factory=utcnow)
    type: AuditEventType
    task_id: str | None = None
    request_id: str | None = None
    action_id: str | None = None
    tool: str | None = None
    decision: str | None = None
    outcome: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data", mode="before")
    @classmethod
    def _redact(cls, value: Any) -> Any:
        return redact(value) if isinstance(value, dict) else value


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...

    def events(self, task_id: str | None = None) -> list[AuditEvent]: ...


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self, task_id: str | None = None) -> list[AuditEvent]:
        with self._lock:
            return [e for e in self._events if task_id is None or e.task_id == task_id]


class JsonlAuditLog:
    """Append-only JSON Lines file. One event per line; flushed on every write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: AuditEvent) -> None:
        line = event.model_dump_json()
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def events(self, task_id: str | None = None) -> list[AuditEvent]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as handle:
            parsed = _parse_lines(handle)
            return [e for e in parsed if task_id is None or e.task_id == task_id]


class Auditor:
    """Builds audit events from pipeline objects so call sites stay one line."""

    def __init__(self, sink: AuditSink) -> None:
        self.sink = sink

    def emit(
        self,
        type: AuditEventType,
        task: Task,
        *,
        record: ActionRecord | None = None,
        decision: str | None = None,
        outcome: str | None = None,
        message: str = "",
        **data: Any,
    ) -> None:
        self.sink.record(
            AuditEvent(
                type=type,
                task_id=task.id,
                request_id=task.request.id,
                action_id=record.id if record else None,
                tool=record.action.tool if record else None,
                decision=decision,
                outcome=outcome,
                message=message,
                data=data,
            )
        )


def _parse_lines(lines: Iterable[str]) -> list[AuditEvent]:
    events = []
    for line in lines:
        line = line.strip()
        if line:
            events.append(AuditEvent.model_validate(json.loads(line)))
    return events
