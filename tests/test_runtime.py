"""Persistent runtime lifecycle: recovery after a crash, clean shutdown, single instance."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionRecord, ActionResult, ActionStatus, AttemptStatus
from jarvis.core.errors import ErrorKind
from jarvis.core.request import Request
from jarvis.core.task import Task, TaskState
from jarvis.runtime.lock import InstanceLock, InstanceLockedError
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.state.sqlite import SqliteStateStore
from jarvis.tools.builtin.filesystem import filesystem_tools

from support import FlakyTool, GoalPlanner, HangingTool, NonIdempotentFlakyTool, wait_for

MakeApi = Callable[..., tuple[TestClient, JarvisRuntime]]


def _crashed(store: SqliteStateStore, tool: str) -> Task:
    """A task as a killed process leaves it: RUNNING, last attempt STARTED (outcome unknown)."""
    task = Task.from_request(Request(text=f"usa {tool}", user_id="tester"))
    task.planner = "static"
    record = ActionRecord(action=Action(id="a1", tool=tool, arguments={"text": "hi"}))
    record.status = ActionStatus.AUTHORIZED
    record.attempts.append(ActionResult(action_id="a1", attempt=1, status=AttemptStatus.STARTED))
    task.actions.append(record)
    task.transition(TaskState.RUNNING, "starting")
    task.total_attempts = 1
    store.save(task)
    return task


def test_recovery_after_restart_is_conservative(make_api: MakeApi, settings: Settings) -> None:
    store = SqliteStateStore(settings.state_db_path)
    idempotent = _crashed(store, "test.flaky")
    dangerous = _crashed(store, "test.flaky_writer")
    pending = Task.from_request(Request(text="crea un archivo de prueba", user_id="tester"))
    store.save(pending)

    # A task waiting for the user before the restart.
    first = build_jarvis(settings)
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "keep.txt").write_text("x", encoding="utf-8")
    waiting = first.ask("borra el archivo keep.txt")
    assert waiting.state is TaskState.WAITING_PERMISSION

    flaky, writer = FlakyTool(failures=0), NonIdempotentFlakyTool(failures=0)
    client, _ = make_api(tools=[flaky, writer, *filesystem_tools()])

    # Idempotent tool with unknown outcome: retried.
    done = wait_for(client, idempotent.id, until=("COMPLETED", "FAILED"))
    assert done["status"] == "COMPLETED"
    assert flaky.calls == 1
    # Non-idempotent tool with unknown outcome: never repeated.
    failed = wait_for(client, dangerous.id, until=("COMPLETED", "FAILED"))
    assert failed["status"] == "FAILED"
    assert failed["error"]["kind"] == ErrorKind.INTERRUPTED.value
    assert writer.calls == 0
    # Never-started task: processed normally.
    assert wait_for(client, pending.id)["status"] == "COMPLETED"
    assert (settings.workspace_dir / "prueba.txt").exists()
    # Waiting task: untouched until the user decides.
    assert client.get(f"/v1/tasks/{waiting.task_id}").json()["status"] == "WAITING_PERMISSION"
    assert (settings.workspace_dir / "keep.txt").exists()

    outcomes = {
        task_id: next(
            e["outcome"]
            for e in client.get(f"/v1/tasks/{task_id}/events").json()
            if e["type"] == "task.recovered"
        )
        for task_id in (idempotent.id, dangerous.id, pending.id, waiting.task_id)
    }
    assert outcomes == {
        idempotent.id: "resume",
        dangerous.id: "resume",
        pending.id: "requeue",
        waiting.task_id: "left_waiting",
    }


def test_clean_shutdown_leaves_no_worker_and_releases_the_lock(
    make_api: MakeApi, settings: Settings
) -> None:
    client, runtime = make_api()
    task_id = client.post("/v1/tasks", json={"text": "lista los archivos"}).json()["task_id"]
    assert wait_for(client, task_id)["status"] == "COMPLETED"
    assert runtime.stop() is True
    assert not runtime.worker.alive
    # The lock is free again: another runtime can take over the same data directory.
    with InstanceLock(settings.lock_path):
        pass


def test_shutdown_with_a_stuck_task_is_bounded_and_recoverable(
    make_api: MakeApi, settings: Settings, release: threading.Event
) -> None:
    tool = HangingTool(release)
    planner = GoalPlanner(
        {"bloquea": (Action(id="a1", tool="test.hang", arguments={"text": "x"}),)}
    )
    client, runtime = make_api(grace=0.3, planner=planner, tools=[tool, *filesystem_tools()])
    task_id = client.post("/v1/tasks", json={"text": "bloquea"}).json()["task_id"]
    assert tool.started.wait(5)

    started = time.monotonic()
    assert runtime.stop() is False  # did not reach a safe point within the grace period
    assert time.monotonic() - started < 3
    store = SqliteStateStore(settings.state_db_path)
    assert store.get(task_id).state is TaskState.RUNNING

    # Next start: the unknown-outcome, non-idempotent action is not repeated.
    client2, _ = make_api(planner=planner, tools=[tool, *filesystem_tools()])
    final = wait_for(client2, task_id, until=("COMPLETED", "FAILED"))
    assert final["status"] == "FAILED"
    assert final["error"]["kind"] == "interrupted"
    assert tool.calls == ["x"]

    # When the abandoned call finally returns, it cannot overwrite the recovered state.
    release.set()
    time.sleep(0.3)
    assert store.get(task_id).state is TaskState.FAILED


def test_pause_between_actions_on_shutdown(make_api: MakeApi, settings: Settings) -> None:
    """A run that sees the stop flag pauses before its next action and stays resumable."""
    jarvis = build_jarvis(settings)
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    task = jarvis.submit("crea el archivo a.txt y luego crea el archivo b.txt")
    calls = iter([False, True])  # planning check passes, then stop before the first action
    paused = jarvis.process(task.id, should_stop=lambda: next(calls, True))
    assert paused.state in (TaskState.PENDING, TaskState.RUNNING)
    assert not (settings.workspace_dir / "b.txt").exists()

    client, _ = make_api()
    assert wait_for(client, task.id)["status"] == "COMPLETED"
    assert (settings.workspace_dir / "a.txt").exists()
    assert (settings.workspace_dir / "b.txt").exists()


def test_only_one_runtime_per_data_directory(make_api: MakeApi, settings: Settings) -> None:
    make_api()
    second = JarvisRuntime(
        build_jarvis(settings), lock=InstanceLock(settings.lock_path), shutdown_grace_seconds=1
    )
    with pytest.raises(InstanceLockedError):
        second.start()


def test_shutdown_pauses_the_worker_at_the_next_safe_point(
    make_api: MakeApi, settings: Settings, release: threading.Event
) -> None:
    """Stop requested during action 1 of 2: action 1 completes, action 2 is left for later."""
    tool = HangingTool(release)
    planner = GoalPlanner(
        {
            "dos pasos": (
                Action(id="a1", tool="test.hang", arguments={"text": "x"}),
                Action(
                    id="a2",
                    tool="fs.write_file",
                    arguments={"path": "segundo.txt", "content": "2"},
                    depends_on=("a1",),
                ),
            )
        }
    )
    client, runtime = make_api(grace=5, planner=planner, tools=[tool, *filesystem_tools()])
    task_id = client.post("/v1/tasks", json={"text": "dos pasos"}).json()["task_id"]
    assert tool.started.wait(5)

    result: list[bool] = []
    stopper = threading.Thread(target=lambda: result.append(runtime.stop()))
    stopper.start()
    time.sleep(0.2)  # stop has been requested while action 1 is still running
    release.set()
    stopper.join(10)
    assert result == [True]

    store = SqliteStateStore(settings.state_db_path)
    paused = store.get(task_id)
    assert paused.state is TaskState.RUNNING
    assert paused.actions[1].status is ActionStatus.PLANNED
    assert not (settings.workspace_dir / "segundo.txt").exists()

    client2, _ = make_api(planner=planner, tools=[tool, *filesystem_tools()])
    assert wait_for(client2, task_id)["status"] == "COMPLETED"
    assert (settings.workspace_dir / "segundo.txt").exists()
    assert tool.calls == ["x"]  # action 1 was not repeated
