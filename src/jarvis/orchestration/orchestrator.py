"""Central orchestrator: request -> context -> policy -> plan -> run -> response.

The orchestrator coordinates; it does not decide policy, select tools, execute or verify. Each of
those concerns lives in its own component and can be replaced independently (e.g. a model-based
planner, a PostgreSQL store, an OpenTelemetry tracer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jarvis.context.builder import ContextBuilder
from jarvis.context.model import ExecutionContext
from jarvis.core.action import ActionRecord, ActionStatus, ConfirmationDecision
from jarvis.core.agent import RequirementState
from jarvis.core.errors import (
    ErrorKind,
    Failure,
    InvalidTransitionError,
    JarvisError,
)
from jarvis.core.request import Request
from jarvis.core.response import Response
from jarvis.core.task import Task, TaskState
from jarvis.execution.runner import StopCheck, TaskRunner, skip_remaining
from jarvis.observability.audit import AuditEventType, Auditor
from jarvis.observability.tracing import NoopTracer, Tracer
from jarvis.orchestration.response import ResponseBuilder
from jarvis.safety.request_guard import RequestGuard
from jarvis.state.store import ConcurrencyError, StateStore


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Unfinished tasks found when a runtime starts, in the order they must be processed."""

    resume: tuple[str, ...]
    """Were RUNNING when the previous process stopped: resumed under crash-recovery rules."""
    requeue: tuple[str, ...]
    """Were PENDING: never started executing, safe to process."""
    waiting: tuple[str, ...]
    """WAITING_PERMISSION: left untouched until the user decides."""

    @property
    def to_process(self) -> tuple[str, ...]:
        return self.resume + self.requeue


RequirementExtractor = Callable[[str, ExecutionContext], list[RequirementState]]


class OrchestrationError(JarvisError):
    """A command could not be applied to a task (wrong state, unknown action...)."""


class Orchestrator:
    def __init__(
        self,
        *,
        contexts: ContextBuilder,
        guard: RequestGuard,
        runner: TaskRunner,
        store: StateStore,
        auditor: Auditor,
        responses: ResponseBuilder | None = None,
        tracer: Tracer | None = None,
        requirements: RequirementExtractor | None = None,
    ) -> None:
        self._contexts = contexts
        self._guard = guard
        self._runner = runner
        self._requirements = requirements
        self._store = store
        self._auditor = auditor
        self._responses = responses or ResponseBuilder()
        self._tracer = tracer or NoopTracer()

    # -- new requests -------------------------------------------------------------------------

    def handle(self, request: Request) -> Response:
        """Synchronous path: submit and process in the caller's thread."""
        task = self.submit(request)
        return self._respond(self.process(task.id))

    def submit(self, request: Request) -> Task:
        """Create and persist a PENDING task. Does not plan or execute anything."""
        task = Task.from_request(request)
        self._store.save(task)
        self._auditor.emit(
            AuditEventType.REQUEST_RECEIVED,
            task,
            text=request.text,
            interface=request.interface,
            user_id=request.user_id,
        )
        return task

    def process(self, task_id: str, should_stop: StopCheck | None = None) -> Task:
        """Advance a task as far as possible through the agent loop.

        Safe to call for any state: terminal tasks are returned unchanged, PENDING tasks go through
        the request guard first, RUNNING / WAITING_PERMISSION tasks are resumed by the runner,
        which applies the crash-recovery and confirmation rules. ``should_stop`` lets a
        shutting-down process stop at a safe point, leaving the task resumable.
        """
        task = self._store.get(task_id)
        if task.is_terminal:
            return task
        with self._tracer.span("jarvis.process", task_id=task.id):
            try:
                if task.state is TaskState.PENDING:
                    if should_stop is not None and should_stop():
                        return task
                    return self._handle(task, should_stop)
                return self._runner.run(task, self._contexts.build(task), should_stop)
            except Exception as exc:
                return self._fail_unexpected(task, exc)

    def _handle(self, task: Task, should_stop: StopCheck | None = None) -> Task:
        with self._tracer.span("jarvis.context"):
            context = self._contexts.build(task)
        self._auditor.emit(
            AuditEventType.CONTEXT_BUILT,
            task,
            keys=[item.key for item in context.items],
            degraded_sources=list(context.degraded_sources),
            permissions=dict(context.permissions.grants),
        )

        rejection = self._guard.check(task.request)
        if rejection is not None:
            self._auditor.emit(
                AuditEventType.REQUEST_REJECTED,
                task,
                outcome=rejection.kind,
                message=rejection.message,
            )
            return self._fail(task, rejection, "request rejected")

        if self._requirements is not None:
            # Deterministic, from the request text only: what Jarvis will later require as
            # verified evidence before accepting that the task is done.
            task.requirements = self._requirements(task.goal, context)
            self._auditor.emit(
                AuditEventType.AGENT_REQUIREMENTS,
                task,
                requirements=[
                    {"kind": s.requirement.kind.value, "application": s.requirement.application}
                    for s in task.requirements
                ],
            )
        task.touch()
        self._store.save(task)
        with self._tracer.span("jarvis.run", task_id=task.id):
            return self._runner.run(task, context, should_stop)

    # -- existing tasks -----------------------------------------------------------------------

    def resume(self, task_id: str) -> Response:
        return self._respond(self.process(task_id))

    def decide(self, task_id: str, action_id: str, *, approved: bool, decided_by: str) -> Response:
        """Record the user's answer to a pending confirmation, then resume the task."""
        self.record_decision(task_id, action_id, approved=approved, decided_by=decided_by)
        return self.resume(task_id)

    def record_decision(
        self, task_id: str, action_id: str, *, approved: bool, decided_by: str
    ) -> Task:
        """Persist the user's answer without resuming.

        The decision only *offers* an approval: on resume the executor re-evaluates the policy
        and accepts it only if it is unconsumed and bound to the exact same arguments.
        """

        def apply(task: Task) -> ActionRecord:
            if task.state is not TaskState.WAITING_PERMISSION:
                raise OrchestrationError(
                    f"task {task_id} is {task.state}, not waiting for permission"
                )
            try:
                record = task.record(action_id)
            except KeyError:
                raise OrchestrationError(f"task {task_id} has no action {action_id!r}") from None
            if (
                record.status is not ActionStatus.AWAITING_CONFIRMATION
                or record.confirmation is None
            ):
                raise OrchestrationError(f"action {action_id!r} is not awaiting confirmation")
            record.confirmation_decision = ConfirmationDecision(
                approved=approved,
                decided_by=decided_by,
                arguments_digest=record.confirmation.arguments_digest,
            )
            return record

        task, record = self._update(task_id, apply)
        self._auditor.emit(
            AuditEventType.CONFIRMATION_DECIDED,
            task,
            record=record,
            outcome="approved" if approved else "rejected",
            decided_by=decided_by,
        )
        return task

    def cancel(self, task_id: str, reason: str = "cancelled by user") -> Response:
        def apply(task: Task) -> TaskState:
            if task.is_terminal:
                raise OrchestrationError(f"task {task_id} is already {task.state}")
            previous = task.state
            details: dict[str, object] = {}
            if previous is TaskState.RUNNING:
                # Nothing is rolled back and a tool that is already running is not interrupted;
                # the runner stops at its next checkpoint and records what actually happened.
                details = {
                    "cancelled_while": "running",
                    "note": "an operation in progress is not interrupted or rolled back",
                }
            task.failure = Failure(kind=ErrorKind.CANCELLED, message=reason, details=details)
            skip_remaining(task)
            task.transition(TaskState.CANCELLED, reason)
            return previous

        task, previous = self._update(task_id, apply)
        self._auditor.emit(
            AuditEventType.TASK_STATE_CHANGED,
            task,
            outcome=TaskState.CANCELLED,
            message=reason,
            from_state=previous,
        )
        return self._respond(task)

    def recover(self) -> RecoveryReport:
        """Classify unfinished tasks after a restart. Executes nothing by itself.

        RUNNING tasks are resumed by the normal runner, which applies the existing rules: an
        attempt left STARTED is marked INTERRUPTED and retried only if the tool is idempotent;
        a non-idempotent action with unknown outcome fails instead of being repeated.
        """

        def oldest_first(state: TaskState) -> list[Task]:
            return sorted(self._store.list(state=state, limit=100_000), key=lambda t: t.created_at)

        running = oldest_first(TaskState.RUNNING)
        pending = oldest_first(TaskState.PENDING)
        waiting = oldest_first(TaskState.WAITING_PERMISSION)
        for task, outcome, message in (
            *((t, "resume", "was RUNNING when the previous process stopped") for t in running),
            *((t, "requeue", "was PENDING when the previous process stopped") for t in pending),
            *((t, "left_waiting", "waiting for the user's decision") for t in waiting),
        ):
            self._auditor.emit(
                AuditEventType.TASK_RECOVERED, task, outcome=outcome, message=message
            )
        return RecoveryReport(
            resume=tuple(t.id for t in running),
            requeue=tuple(t.id for t in pending),
            waiting=tuple(t.id for t in waiting),
        )

    def snapshot(self, task_id: str) -> tuple[Task, Response]:
        """One consistent read of a task and its response, without side effects.

        Built from a single load: reading twice could mix two versions of a task that the
        worker is changing in between.
        """
        task = self._store.get(task_id)
        return task, self._responses.build(task)

    def _update[R](
        self, task_id: str, apply: Callable[[Task], R], attempts: int = 5
    ) -> tuple[Task, R]:
        """Load-modify-save with optimistic concurrency, retrying on concurrent writes."""
        for _ in range(attempts):
            task = self._store.get(task_id)
            result = apply(task)
            task.touch()
            try:
                self._store.save(task)
            except ConcurrencyError:
                continue
            return task, result
        raise OrchestrationError(f"task {task_id} is changing too fast; try again")

    # -- helpers ------------------------------------------------------------------------------

    def _fail(self, task: Task, failure: Failure, reason: str) -> Task:
        task.failure = failure
        self._runner.transition(task, TaskState.FAILED, reason)
        return task

    def _fail_unexpected(self, task: Task, exc: Exception) -> Task:
        """Never leave a task stuck in a non-terminal state because of a bug."""
        failure = Failure(
            kind=ErrorKind.INTERNAL,
            message=f"unexpected error: {type(exc).__name__}: {exc}",
        )
        current = self._store.get(task.id)
        if current.is_terminal:
            return current
        current.failure = failure
        skip_remaining(current)
        try:
            self._runner.transition(current, TaskState.FAILED, "unexpected error")
        except InvalidTransitionError:
            return current
        return current

    def _respond(self, task: Task) -> Response:
        response = self._responses.build(task)
        self._auditor.emit(
            AuditEventType.RESPONSE_SENT,
            task,
            outcome=task.state,
            pending_confirmations=[c.action_id for c in response.pending_confirmations],
        )
        return response
