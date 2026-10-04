"""The persistent Jarvis process.

Lifecycle::

    build (load config, wire components; LiteLLM imported here, once)
      -> start: acquire instance lock -> recover unfinished tasks -> start worker
      -> serve requests (create / approve / reject / cancel / read)
      -> stop: stop accepting work -> let the current task reach a safe point (bounded grace)
               -> release lock

Every state change is committed to SQLite and every audit event flushed when it happens, so
there is no buffered state to flush at shutdown. Work not finished in time stays in the database
and is recovered on the next start.

The runtime holds no task state of its own: all reads go to the state store, all writes through
the orchestrator. It only adds the worker queue and lifecycle on top of the existing facade.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jarvis import __version__
from jarvis.core.errors import JarvisError
from jarvis.core.response import Response
from jarvis.core.task import Task, TaskState
from jarvis.observability.audit import AuditEvent, AuditEventType
from jarvis.orchestration.jarvis import Jarvis
from jarvis.runtime.lock import InstanceLock
from jarvis.runtime.worker import TaskWorker
from jarvis.tools.base import ToolSpec


class RuntimeNotReadyError(JarvisError):
    pass


class RuntimeStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class Health:
    status: str
    runtime: RuntimeStatus
    state_store: str
    worker: str
    queued: int
    planner: str
    version: str = __version__


class JarvisRuntime:
    def __init__(self, jarvis: Jarvis, *, lock: InstanceLock, shutdown_grace_seconds: float):
        self._jarvis = jarvis
        self._lock = lock
        self._grace = shutdown_grace_seconds
        self._worker = TaskWorker(jarvis.process)
        self._status = RuntimeStatus.CREATED
        self._status_lock = threading.Lock()

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def worker(self) -> TaskWorker:
        return self._worker

    # -- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        with self._status_lock:
            if self._status is not RuntimeStatus.CREATED:
                raise RuntimeError(f"runtime cannot start from {self._status}")
            self._lock.acquire()
            try:
                report = self._jarvis.recover()
                self._worker.start()
            except BaseException:
                self._lock.release()
                raise
            for task_id in report.to_process:
                self._worker.submit(task_id)
            self._status = RuntimeStatus.READY
        self._jarvis.record_runtime_event(
            AuditEventType.RUNTIME_STARTED,
            "runtime started",
            resumed=list(report.resume),
            requeued=list(report.requeue),
            waiting=list(report.waiting),
        )

    def stop(self) -> bool:
        """Stop gracefully. Returns True if the worker reached a safe point within the grace."""
        with self._status_lock:
            if self._status is not RuntimeStatus.READY:
                return True
            self._status = RuntimeStatus.STOPPING
        in_progress = self._worker.current_task
        clean = self._worker.stop(self._grace)
        self._jarvis.record_runtime_event(
            AuditEventType.RUNTIME_STOPPED,
            "runtime stopped" if clean else "runtime stopped before the current task finished",
            clean=clean,
            task_in_progress=in_progress,
            left_queued=self._worker.queued,
        )
        self._lock.release()
        self._status = RuntimeStatus.STOPPED
        return clean

    # -- operations ---------------------------------------------------------------------------

    def create_task(self, text: str, metadata: dict[str, Any] | None = None) -> Task:
        self._require_ready()
        task = self._jarvis.submit(text, interface="api", metadata=metadata)
        self._worker.submit(task.id)
        return task

    def decide(self, task_id: str, action_id: str, *, approved: bool) -> None:
        """Hand the user's decision to the domain, then let the worker resume the task."""
        self._require_ready()
        self._jarvis.record_decision(task_id, action_id, approved=approved)
        self._worker.submit(task_id)

    def cancel(self, task_id: str) -> None:
        self._require_ready()
        self._jarvis.cancel(task_id)

    def snapshot(self, task_id: str) -> tuple[Task, Response]:
        return self._jarvis.snapshot(task_id)

    def tasks(self, state: TaskState | None, limit: int) -> list[Task]:
        return self._jarvis.tasks(state=state, limit=limit)

    def events(self, task_id: str) -> list[AuditEvent]:
        self._jarvis.snapshot(task_id)  # 404 for unknown tasks instead of an empty list
        return self._jarvis.audit_trail(task_id)

    def tools(self) -> list[ToolSpec]:
        return self._jarvis.tools()

    def health(self) -> Health:
        try:
            self._jarvis.tasks(limit=1)
            store = "ok"
        except Exception:
            store = "error"
        ready = self._status is RuntimeStatus.READY and self._worker.alive and store == "ok"
        return Health(
            status="ok" if ready else "error",
            runtime=self._status,
            state_store=store,
            worker="running" if self._worker.alive else "stopped",
            queued=self._worker.queued,
            planner=self._jarvis.planner_description,
        )

    def _require_ready(self) -> None:
        if self._status is not RuntimeStatus.READY:
            raise RuntimeNotReadyError(f"runtime is {self._status}")
