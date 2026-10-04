"""State store contract and an in-memory implementation.

Stores persist whole :class:`Task` documents with optimistic concurrency: ``save`` succeeds only
if the stored version equals ``task.version``, then increments it. This prevents two processes
(e.g. a running task and a ``cancel`` command) from silently overwriting each other, and maps
directly onto PostgreSQL (``UPDATE ... WHERE version = ?``) when that backend is added.
"""

from __future__ import annotations

import threading
from typing import Protocol

from jarvis.core.errors import JarvisError
from jarvis.core.task import Task, TaskState


class ConcurrencyError(JarvisError):
    """The task was modified by someone else since it was loaded."""


class TaskNotFoundError(JarvisError):
    pass


class StateStore(Protocol):
    def save(self, task: Task) -> None:
        """Insert (version 0) or update (matching version) a task; bumps ``task.version``."""
        ...

    def get(self, task_id: str) -> Task: ...

    def list(self, state: TaskState | None = None, limit: int = 50) -> list[Task]: ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._tasks: dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, task: Task) -> None:
        with self._lock:
            existing = self._tasks.get(task.id)
            if existing is None:
                if task.version != 0:
                    raise ConcurrencyError(
                        f"task {task.id} does not exist at version {task.version}"
                    )
            elif Task.model_validate_json(existing).version != task.version:
                raise ConcurrencyError(f"task {task.id} was modified concurrently")
            task.version += 1
            self._tasks[task.id] = task.model_dump_json()

    def get(self, task_id: str) -> Task:
        with self._lock:
            raw = self._tasks.get(task_id)
        if raw is None:
            raise TaskNotFoundError(task_id)
        return Task.model_validate_json(raw)

    def list(self, state: TaskState | None = None, limit: int = 50) -> list[Task]:
        with self._lock:
            tasks = [Task.model_validate_json(raw) for raw in self._tasks.values()]
        tasks = [t for t in tasks if state is None or t.state is state]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
