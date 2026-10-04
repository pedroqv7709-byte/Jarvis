"""The single background thread that executes tasks.

Invariant: inside a runtime, only this thread plans and executes tasks, one at a time. HTTP
handlers only create tasks, record decisions or cancel, all through the state store, and then
enqueue the task id here. This keeps execution sequential (as the runner expects) without any
extra locking around tools, and makes "is this task being executed right now?" unambiguous.

The queue is not a source of truth: it holds task ids only. Anything still queued at shutdown is
simply left in the database (PENDING / RUNNING) and picked up by recovery on the next start.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from jarvis.execution.runner import StopCheck

logger = logging.getLogger(__name__)

ProcessFn = Callable[[str, StopCheck], object]


class TaskWorker:
    def __init__(self, process: ProcessFn, *, name: str = "jarvis-worker") -> None:
        self._process = process
        self._name = name
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: str | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    @property
    def current_task(self) -> str | None:
        return self._current

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker already started")
        # Daemon so that a tool stuck past the shutdown grace period cannot keep the process
        # alive forever; its task stays RUNNING in the database and recovery handles it.
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def submit(self, task_id: str) -> None:
        if self._stopping.is_set():
            raise RuntimeError("worker is stopping")
        self._queue.put(task_id)

    def stop(self, grace_seconds: float) -> bool:
        """Stop after the current safe point. Returns True if the thread finished in time."""
        self._stopping.set()
        self._queue.put(None)
        if self._thread is None:
            return True
        self._thread.join(grace_seconds)
        return not self._thread.is_alive()

    def _loop(self) -> None:
        while True:
            task_id = self._queue.get()
            if task_id is None or self._stopping.is_set():
                return
            self._current = task_id
            try:
                self._process(task_id, self._stopping.is_set)
            except Exception:
                # The orchestrator already converts failures into FAILED tasks; this only guards
                # the loop itself so one bad task can never stop the worker.
                logger.exception("unexpected error while processing task %s", task_id)
            finally:
                self._current = None
