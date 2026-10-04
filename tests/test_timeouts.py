"""Hard timeouts for tools and the per-task execution budget."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, AttemptStatus
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.core.task import TaskState
from jarvis.core.timeouts import OperationTimeoutError, run_with_timeout
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool, ToolContext

from support import CountingTool, EchoArgs, StaticPlanner


@pytest.fixture
def release() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()


def _hanging_tool(release: threading.Event, *, idempotent: bool) -> Tool[Any]:
    name = "test.hang" if idempotent else "test.hang_writer"

    class Hanging(CountingTool):
        spec = CountingTool.spec.model_copy(
            update={"name": name, "idempotent": idempotent, "side_effects": not idempotent}
        )

        def run(self, args: EchoArgs, ctx: ToolContext) -> dict[str, Any]:
            self.calls.append(args.text)
            release.wait(10)
            return {"late": True}

    return Hanging()


def _jarvis(settings: Settings, tool: Tool[Any], sleeps: list[float] | None = None) -> Jarvis:
    action = Action(id="a1", tool=tool.spec.name, arguments={"text": "x"})
    return build_jarvis(
        settings,
        planner=StaticPlanner(action),
        tools=[tool],
        store=InMemoryStateStore(),
        audit=InMemoryAuditLog(),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


# -- primitive ------------------------------------------------------------------------------


def test_run_with_timeout_returns_raises_and_times_out() -> None:
    assert run_with_timeout(lambda: 42, 1, name="ok") == 42
    assert run_with_timeout(lambda: 7, None, name="inline") == 7
    with pytest.raises(ValueError):
        run_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 1, name="err")
    done = threading.Event()
    with pytest.raises(OperationTimeoutError) as excinfo:
        run_with_timeout(lambda: done.wait(5), 0.05, name="slow")
    assert excinfo.value.seconds == 0.05
    done.set()


# -- tool timeouts --------------------------------------------------------------------------


def test_hung_idempotent_tool_times_out_with_bounded_retries(
    settings: Settings, release: threading.Event
) -> None:
    settings = settings.model_copy(
        update={"tool_timeout_seconds": 0.1, "max_attempts_per_action": 2}
    )
    tool = _hanging_tool(release, idempotent=True)
    jarvis = _jarvis(settings, tool)
    started = time.monotonic()
    response = jarvis.ask("x")
    assert time.monotonic() - started < 5
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.TIMEOUT
    record = jarvis.task(response.task_id).actions[0]
    assert [a.status for a in record.attempts] == [AttemptStatus.FAILED, AttemptStatus.FAILED]
    assert all(a.failure and a.failure.kind is ErrorKind.TIMEOUT for a in record.attempts)
    types = [e.type for e in jarvis.audit_trail(response.task_id)]
    assert types.count(AuditEventType.ACTION_RETRY_SCHEDULED) == 1
    assert types.count(AuditEventType.ACTION_ATTEMPT_FAILED) == 2


def test_hung_non_idempotent_tool_is_never_retried(
    settings: Settings, release: threading.Event
) -> None:
    settings = settings.model_copy(
        update={"tool_timeout_seconds": 0.1, "max_attempts_per_action": 5}
    )
    tool = _hanging_tool(release, idempotent=False)
    jarvis = _jarvis(settings, tool)
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert tool.calls == ["x"]  # type: ignore[attr-defined]
    assert response.failure is not None
    assert response.failure.kind is ErrorKind.TIMEOUT
    assert response.failure.details["outcome_unknown"] is True


def test_tool_raising_its_own_timeout_follows_retry_rules(settings: Settings) -> None:
    class SlowBackend(CountingTool):
        spec = CountingTool.spec.model_copy(update={"name": "test.slow", "idempotent": False})

        def run(self, args: EchoArgs, ctx: ToolContext) -> dict[str, Any]:
            self.calls.append(args.text)
            if len(self.calls) == 1:
                # The tool states explicitly that nothing happened, so a retry is safe.
                raise ToolError(ErrorKind.TIMEOUT, "backend slow", retryable=True)
            return {"echo": args.text}

    tool = SlowBackend()
    response = _jarvis(settings, tool).ask("x")
    assert response.state is TaskState.COMPLETED
    assert len(tool.calls) == 2


# -- task budget ----------------------------------------------------------------------------


def test_task_budget_stops_retries(settings: Settings, release: threading.Event) -> None:
    settings = settings.model_copy(
        update={
            "tool_timeout_seconds": 0.2,
            "task_timeout_seconds": 0.5,
            "max_attempts_per_action": 20,
        }
    )
    tool = _hanging_tool(release, idempotent=True)
    jarvis = _jarvis(settings, tool)
    started = time.monotonic()
    response = jarvis.ask("x")
    assert time.monotonic() - started < 5
    assert response.state is TaskState.STOPPED  # a limit stops the task safely
    assert response.failure is not None
    assert response.failure.kind is ErrorKind.TIMEOUT
    assert response.failure.details == {"scope": "task", "limit": "time"}
    task = jarvis.task(response.task_id)
    assert len(task.actions[0].attempts) <= 3
    assert task.execution_seconds >= 0.5


def test_backoff_is_capped_by_the_task_budget(settings: Settings) -> None:
    class Flaky(CountingTool):
        spec = CountingTool.spec.model_copy(update={"name": "test.flaky2"})

        def run(self, args: EchoArgs, ctx: ToolContext) -> dict[str, Any]:
            raise ToolError(ErrorKind.TRANSIENT, "glitch", retryable=True)

    settings = settings.model_copy(
        update={
            "retry_backoff_seconds": 60,
            "retry_max_backoff_seconds": 60,
            "task_timeout_seconds": 1,
            "max_attempts_per_action": 5,
        }
    )
    sleeps: list[float] = []
    response = _jarvis(settings, Flaky(), sleeps).ask("x")
    assert response.state is TaskState.STOPPED
    assert sleeps and all(s <= 1 for s in sleeps)
    assert response.failure is not None and response.failure.kind is ErrorKind.TIMEOUT


def test_execution_time_is_persisted(settings: Settings) -> None:
    jarvis = _jarvis(settings, CountingTool())
    response = jarvis.ask("x")
    assert response.state is TaskState.COMPLETED
    assert jarvis.task(response.task_id).execution_seconds >= 0
