"""Public data shapes of the interface layer (HTTP API and CLI).

These are deliberately separate from the domain model: clients see a stable, minimal contract,
never internal objects such as ``Task`` or ``ActionRecord``. This module must stay free of
FastAPI imports so the CLI can use it without loading the web stack.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.errors import Failure
from jarvis.core.response import Response
from jarvis.core.task import Task, TaskState
from jarvis.observability.audit import AuditEvent
from jarvis.tools.base import ToolSpec

TaskStatus = Literal[
    "PENDING", "RUNNING", "WAITING_PERMISSION", "COMPLETED", "FAILED", "CANCELLED", "STOPPED"
]
ACTIVE_STATUSES: frozenset[str] = frozenset({"PENDING", "RUNNING"})


class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def status_of(state: TaskState) -> TaskStatus:
    return state.value.upper()  # type: ignore[return-value]


def state_of(status: str) -> TaskState:
    return TaskState(status.lower())


# -- requests -------------------------------------------------------------------------------


class CreateTaskRequest(_View):
    text: str = Field(min_length=1, max_length=20_000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


class DecisionRequest(_View):
    action_id: str = Field(min_length=1, max_length=64)


# -- responses ------------------------------------------------------------------------------


class ErrorView(_View):
    kind: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ActionView(_View):
    action_id: str
    tool: str
    description: str
    status: str
    decision: str | None = None
    verification: str | None = None
    attempts: int = 0
    output: dict[str, Any] | None = None
    error: ErrorView | None = None


class ConfirmationView(_View):
    action_id: str
    tool: str
    description: str
    arguments: dict[str, Any]
    reasons: list[str]
    decision: Literal["approved", "rejected"] | None = None
    """A decision already recorded but not yet applied: the task is about to resume."""
    target: dict[str, Any] | None = None
    """The concrete target Jarvis resolved (e.g. the exact window) that is being approved."""


class TaskAccepted(_View):
    task_id: str
    status: TaskStatus


class ProgressView(_View):
    """Operational progress of the agent loop (no reasoning, no secrets)."""

    phase: str
    iterations: int
    actions_used: int
    replans_used: int
    elapsed_seconds: float
    current_action: str | None = None
    stop_reason: str | None = None


class TaskView(_View):
    task_id: str
    status: TaskStatus
    goal: str
    message: str
    planner: str | None = None
    fallback_reason: str | None = None
    actions: list[ActionView] = Field(default_factory=list)
    pending_confirmations: list[ConfirmationView] = Field(default_factory=list)
    error: ErrorView | None = None
    progress: ProgressView | None = None
    created_at: datetime
    updated_at: datetime


class TaskSummary(_View):
    task_id: str
    status: TaskStatus
    goal: str
    created_at: datetime
    updated_at: datetime


class EventView(_View):
    id: str
    at: datetime
    type: str
    action_id: str | None = None
    tool: str | None = None
    decision: str | None = None
    outcome: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ToolView(_View):
    name: str
    description: str
    permissions: list[str]
    categories: list[str]


class HealthView(_View):
    status: Literal["ok", "error"]
    version: str
    runtime: str
    state_store: str
    worker: str
    queued: int
    planner: str


# -- mapping --------------------------------------------------------------------------------


def task_view(task: Task, response: Response) -> TaskView:
    return TaskView(
        task_id=task.id,
        status=status_of(response.state),
        goal=task.goal,
        message=response.message,
        planner=response.planner,
        fallback_reason=response.fallback_reason,
        actions=[
            ActionView(
                action_id=a.action_id,
                tool=a.tool,
                description=a.description,
                status=a.status.value,
                decision=a.decision.value if a.decision else None,
                verification=a.verification.value if a.verification else None,
                attempts=a.attempts,
                output=a.output,
                error=_error(a.failure),
            )
            for a in response.actions
        ],
        pending_confirmations=[
            ConfirmationView(
                action_id=c.action_id,
                tool=c.tool,
                description=c.description,
                arguments=c.arguments,
                reasons=list(c.reasons),
                decision=_recorded_decision(task, c.action_id),
                target=c.target,
            )
            for c in response.pending_confirmations
        ],
        error=_error(response.failure),
        progress=progress_view(task),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def is_settled(view: TaskView) -> bool:
    """True when the task will not move without new input (or is finished)."""
    if view.status in ACTIVE_STATUSES:
        return False
    return not any(c.decision is not None for c in view.pending_confirmations)


def _recorded_decision(task: Task, action_id: str) -> Literal["approved", "rejected"] | None:
    record = task.record(action_id)
    decision = record.confirmation_decision
    if decision is None or decision.consumed:
        return None
    return "approved" if decision.approved else "rejected"


def progress_view(task: Task) -> ProgressView:
    progress = task.progress
    pending = next(
        (r for r in task.actions if r.status.value in _IN_FLIGHT),
        None,
    )
    return ProgressView(
        phase=progress.phase.value,
        iterations=progress.iterations,
        actions_used=progress.actions_used,
        replans_used=progress.replans_used,
        elapsed_seconds=round(task.elapsed_seconds, 3),
        current_action=f"{pending.id}: {pending.action.tool}" if pending else None,
        stop_reason=progress.stop_reason,
    )


_IN_FLIGHT = frozenset({"planned", "selected", "authorized", "awaiting_confirmation"})


def task_summary(task: Task) -> TaskSummary:
    return TaskSummary(
        task_id=task.id,
        status=status_of(task.state),
        goal=task.goal,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def event_view(event: AuditEvent) -> EventView:
    return EventView(
        id=event.id,
        at=event.at,
        type=event.type.value,
        action_id=event.action_id,
        tool=event.tool,
        decision=event.decision,
        outcome=event.outcome,
        message=event.message,
        data=event.data,
    )


def tool_view(spec: ToolSpec) -> ToolView:
    return ToolView(
        name=spec.name,
        description=spec.description,
        permissions=sorted(spec.required_permissions),
        categories=sorted(spec.categories),
    )


def _error(failure: Failure | None) -> ErrorView | None:
    if failure is None:
        return None
    return ErrorView(kind=failure.kind.value, message=failure.message, details=failure.details)
