from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionRecord, ActionResult, ActionStatus, AttemptStatus
from jarvis.core.errors import ErrorKind
from jarvis.core.request import Request
from jarvis.core.task import Task, TaskState
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.state.sqlite import SqliteStateStore
from jarvis.state.store import (
    ConcurrencyError,
    InMemoryStateStore,
    StateStore,
    TaskNotFoundError,
)
from jarvis.tools.base import ToolContext

from support import CountingTool, EchoArgs, FlakyTool, NonIdempotentFlakyTool, StaticPlanner


def _task(text: str = "x") -> Task:
    return Task.from_request(Request(text=text, user_id="u"))


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> StateStore:
    if request.param == "memory":
        return InMemoryStateStore()
    return SqliteStateStore(tmp_path / "state.db")


def test_save_and_get_round_trip(store: StateStore) -> None:
    task = _task()
    task.actions.append(ActionRecord(action=Action(id="a1", tool="t", arguments={"k": [1, 2]})))
    store.save(task)
    assert task.version == 1
    loaded = store.get(task.id)
    assert loaded == task


def test_update_bumps_version(store: StateStore) -> None:
    task = _task()
    store.save(task)
    task.transition(TaskState.RUNNING, "go")
    store.save(task)
    assert task.version == 2
    assert store.get(task.id).state is TaskState.RUNNING


def test_stale_write_is_rejected(store: StateStore) -> None:
    task = _task()
    store.save(task)
    a = store.get(task.id)
    b = store.get(task.id)
    a.transition(TaskState.RUNNING, "a")
    store.save(a)
    b.transition(TaskState.CANCELLED, "b")
    with pytest.raises(ConcurrencyError):
        store.save(b)
    assert store.get(task.id).state is TaskState.RUNNING


def test_duplicate_insert_is_rejected(store: StateStore) -> None:
    task = _task()
    store.save(task)
    clone = task.model_copy(update={"version": 0})
    with pytest.raises(ConcurrencyError):
        store.save(clone)


def test_missing_task(store: StateStore) -> None:
    with pytest.raises(TaskNotFoundError):
        store.get("task_missing")


def test_list_filters_by_state_newest_first(store: StateStore) -> None:
    first, second = _task("1"), _task("2")
    store.save(first)
    store.save(second)
    second.transition(TaskState.RUNNING, "go")
    store.save(second)
    assert [t.id for t in store.list()] == [second.id, first.id]
    assert [t.id for t in store.list(state=TaskState.PENDING)] == [first.id]


def test_sqlite_survives_reopen_and_migrations_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = SqliteStateStore(path)
    assert store.applied_migrations() == [1]
    task = _task()
    store.save(task)

    reopened = SqliteStateStore(path)
    assert reopened.migrate() == []
    assert reopened.get(task.id) == task
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM tasks").fetchone()[0] == 1


# -- resumption -----------------------------------------------------------------------------


def test_waiting_task_resumes_in_a_new_process(settings: Settings) -> None:
    """A task waiting for permission is persisted and can be continued by a fresh instance."""
    first = build_jarvis(settings, audit=InMemoryAuditLog())
    (settings.workspace_dir / "keep.txt").parent.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "keep.txt").write_text("x", encoding="utf-8")
    waiting = first.ask("borra el archivo keep.txt")
    assert waiting.state is TaskState.WAITING_PERMISSION

    second = build_jarvis(settings, audit=InMemoryAuditLog())  # same SQLite file
    assert second.task(waiting.task_id).state is TaskState.WAITING_PERMISSION
    done = second.approve(waiting.task_id, "a1")
    assert done.state is TaskState.COMPLETED
    assert not (settings.workspace_dir / "keep.txt").exists()


def _crashed_task(store: StateStore, tool: str) -> Task:
    """Persist a task as a crashed process would leave it: RUNNING with a STARTED attempt."""
    task = _task()
    record = ActionRecord(action=Action(id="a1", tool=tool, arguments={"text": "hi"}))
    record.status = ActionStatus.AUTHORIZED
    record.attempts.append(ActionResult(action_id="a1", attempt=1, status=AttemptStatus.STARTED))
    task.actions.append(record)
    task.transition(TaskState.RUNNING, "starting")
    task.total_attempts = 1
    store.save(task)
    return task


def test_interrupted_idempotent_action_is_retried_on_resume(settings: Settings) -> None:
    store = InMemoryStateStore()
    tool = FlakyTool(failures=0)
    task = _crashed_task(store, "test.flaky")
    jarvis = build_jarvis(
        settings, planner=StaticPlanner(), tools=[tool], store=store, audit=InMemoryAuditLog()
    )
    response = jarvis.resume(task.id)
    assert response.state is TaskState.COMPLETED
    attempts = jarvis.task(task.id).actions[0].attempts
    assert attempts[0].failure is not None
    assert attempts[0].failure.kind is ErrorKind.INTERRUPTED
    assert attempts[1].status is AttemptStatus.SUCCEEDED
    assert tool.calls == 1


def test_interrupted_non_idempotent_action_is_not_repeated(settings: Settings) -> None:
    store = InMemoryStateStore()
    tool = NonIdempotentFlakyTool(failures=0)
    task = _crashed_task(store, "test.flaky_writer")
    jarvis = build_jarvis(
        settings, planner=StaticPlanner(), tools=[tool], store=store, audit=InMemoryAuditLog()
    )
    response = jarvis.resume(task.id)
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.INTERRUPTED
    assert tool.calls == 0


def test_cancel_waiting_task(settings: Settings) -> None:
    jarvis = build_jarvis(settings, audit=InMemoryAuditLog())
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "k.txt").write_text("x", encoding="utf-8")
    waiting = jarvis.ask("borra el archivo k.txt")
    cancelled = jarvis.cancel(waiting.task_id)
    assert cancelled.state is TaskState.CANCELLED
    assert cancelled.actions[0].status is ActionStatus.SKIPPED
    assert (settings.workspace_dir / "k.txt").exists()
    # A cancelled task cannot be resumed into execution.
    assert jarvis.resume(waiting.task_id).state is TaskState.CANCELLED


def test_cancel_from_another_process_stops_a_running_task(settings: Settings) -> None:
    """A cancel issued while a task runs wins: the runner detects it and stops."""
    store = InMemoryStateStore()
    audit = InMemoryAuditLog()
    other_process = build_jarvis(settings, store=store, audit=audit)

    class CancelsItsOwnTask(CountingTool):
        def run(self, args: EchoArgs, ctx: ToolContext) -> dict[str, Any]:
            other_process.cancel(ctx.task_id)
            return super().run(args, ctx)

    tool = CancelsItsOwnTask()
    later = CountingTool.spec.model_copy(update={"name": "test.later"})

    class Later(CountingTool):
        spec = later

    later_tool = Later()
    jarvis = build_jarvis(
        settings,
        planner=StaticPlanner(
            Action(id="a1", tool="test.echo", arguments={"text": "1"}),
            Action(id="a2", tool="test.later", arguments={"text": "2"}, depends_on=("a1",)),
        ),
        tools=[tool, later_tool],
        store=store,
        audit=audit,
    )
    response = jarvis.ask("x")
    assert response.state is TaskState.CANCELLED
    assert later_tool.calls == []
    assert store.get(response.task_id).state is TaskState.CANCELLED
