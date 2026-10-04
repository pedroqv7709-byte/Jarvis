"""Tasks and their explicit state machine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.action import ActionRecord
from jarvis.core.agent import AgentProgress, RequirementState
from jarvis.core.errors import Failure, InvalidTransitionError
from jarvis.core.ids import new_id, utcnow
from jarvis.core.request import Request


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"
    """Halted safely by Jarvis without completing: limit reached, cycle, or the planner stopped
    without a failed action to blame."""

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES


TERMINAL_STATES = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.STOPPED}
)

ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.WAITING_PERMISSION,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.STOPPED,
        }
    ),
    TaskState.WAITING_PERMISSION: frozenset(
        {TaskState.RUNNING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
    TaskState.STOPPED: frozenset(),
}


class TaskTransition(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_state: TaskState
    to_state: TaskState
    reason: str
    at: datetime = Field(default_factory=utcnow)


class PlanningAttempt(BaseModel):
    """A planner that failed to produce a usable plan for this task (kept for audit/response)."""

    model_config = ConfigDict(frozen=True)

    planner: str
    stage: Literal["plan", "validate"]
    """``plan``: the planner itself failed; ``validate``: its plan was rejected by validation."""
    failure: Failure


class Task(BaseModel):
    """A unit of work derived from a request. Fully serializable; this is what gets persisted."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: new_id("task"))
    request: Request
    state: TaskState = TaskState.PENDING
    planner: str | None = None
    reply: str | None = None
    """Text answer produced by the planner when the goal needs no actions."""
    planning_attempts: list[PlanningAttempt] = Field(default_factory=list)
    """Planners that failed before the one in ``planner`` succeeded (i.e. fallbacks)."""
    actions: list[ActionRecord] = Field(default_factory=list)
    failure: Failure | None = None
    transitions: list[TaskTransition] = Field(default_factory=list)
    total_attempts: int = 0
    execution_seconds: float = 0.0
    """Time spent running tools and backing off, across runs. Waiting for the user is excluded."""
    requirements: list[RequirementState] = Field(default_factory=list)
    """Verifiable conditions derived deterministically from the request (may be empty)."""
    progress: AgentProgress = Field(default_factory=AgentProgress)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    version: int = 0
    """Optimistic-concurrency version, managed by the state store."""

    @classmethod
    def from_request(cls, request: Request) -> Task:
        return cls(request=request)

    @property
    def goal(self) -> str:
        return self.request.text

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def elapsed_seconds(self) -> float:
        """Execution + planning time counted against the task's global time budget."""
        return self.execution_seconds + self.progress.planning_seconds

    def can_transition(self, to_state: TaskState) -> bool:
        return to_state in ALLOWED_TRANSITIONS[self.state]

    def transition(self, to_state: TaskState, reason: str) -> TaskTransition:
        if not self.can_transition(to_state):
            raise InvalidTransitionError(
                f"task {self.id}: transition {self.state} -> {to_state} is not allowed"
            )
        record = TaskTransition(from_state=self.state, to_state=to_state, reason=reason)
        self.transitions.append(record)
        self.state = to_state
        self.touch()
        return record

    def touch(self) -> None:
        self.updated_at = utcnow()

    def record(self, action_id: str) -> ActionRecord:
        for record in self.actions:
            if record.id == action_id:
                return record
        raise KeyError(action_id)
