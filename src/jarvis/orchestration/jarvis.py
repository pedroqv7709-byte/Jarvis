"""The :class:`Jarvis` facade used by interfaces (CLI today; API, voice, etc. later)."""

from __future__ import annotations

from typing import Any

from jarvis.core.request import Request
from jarvis.core.response import Response
from jarvis.core.task import Task, TaskState
from jarvis.execution.runner import StopCheck
from jarvis.observability.audit import AuditEvent, AuditEventType, AuditSink
from jarvis.orchestration.orchestrator import Orchestrator, RecoveryReport
from jarvis.state.store import StateStore
from jarvis.tools.base import ToolSpec
from jarvis.tools.registry import ToolRegistry


class Jarvis:
    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        store: StateStore,
        audit: AuditSink,
        registry: ToolRegistry,
        user_id: str,
        planner_description: str = "",
    ) -> None:
        self._orchestrator = orchestrator
        self._store = store
        self._audit = audit
        self._registry = registry
        self.user_id = user_id
        self.planner_description = planner_description
        """Planner chain in use, e.g. ``model -> rule_based``."""

    def ask(
        self, text: str, *, interface: str = "cli", metadata: dict[str, Any] | None = None
    ) -> Response:
        request = Request(
            text=text, user_id=self.user_id, interface=interface, metadata=metadata or {}
        )
        return self._orchestrator.handle(request)

    def submit(
        self, text: str, *, interface: str = "cli", metadata: dict[str, Any] | None = None
    ) -> Task:
        """Create a PENDING task without processing it (asynchronous hosts process it later)."""
        request = Request(
            text=text, user_id=self.user_id, interface=interface, metadata=metadata or {}
        )
        return self._orchestrator.submit(request)

    def process(self, task_id: str, should_stop: StopCheck | None = None) -> Task:
        return self._orchestrator.process(task_id, should_stop)

    def record_decision(self, task_id: str, action_id: str, *, approved: bool) -> Task:
        """Store an approve/reject decision without resuming the task."""
        return self._orchestrator.record_decision(
            task_id, action_id, approved=approved, decided_by=f"user:{self.user_id}"
        )

    def recover(self) -> RecoveryReport:
        return self._orchestrator.recover()

    def record_runtime_event(self, type: AuditEventType, message: str, **data: Any) -> None:
        """Audit a process-level event that belongs to no task (start, stop)."""
        self._audit.record(AuditEvent(type=type, message=message, data=data))

    def snapshot(self, task_id: str) -> tuple[Task, Response]:
        """Consistent read-only view of a task (no audit event, no processing)."""
        return self._orchestrator.snapshot(task_id)

    def approve(self, task_id: str, action_id: str) -> Response:
        return self._orchestrator.decide(
            task_id, action_id, approved=True, decided_by=f"user:{self.user_id}"
        )

    def reject(self, task_id: str, action_id: str) -> Response:
        return self._orchestrator.decide(
            task_id, action_id, approved=False, decided_by=f"user:{self.user_id}"
        )

    def resume(self, task_id: str) -> Response:
        return self._orchestrator.resume(task_id)

    def cancel(self, task_id: str) -> Response:
        return self._orchestrator.cancel(task_id)

    def task(self, task_id: str) -> Task:
        return self._store.get(task_id)

    def tasks(self, state: TaskState | None = None, limit: int = 20) -> list[Task]:
        return self._store.list(state=state, limit=limit)

    def audit_trail(self, task_id: str) -> list[AuditEvent]:
        return self._audit.events(task_id)

    def tools(self) -> list[ToolSpec]:
        return self._registry.specs()
