"""Cancellation at each phase of the agent loop.

The cancel comes from "another process" (a second Jarvis instance on the same store), at the
exact moment the loop is observing, planning, executing, verifying or replanning. The loop must
notice it at its next checkpoint, record what really happened and not continue.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionStatus
from jarvis.core.task import TaskState
from jarvis.desktop.catalog import build_catalog
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.orchestration.orchestrator import OrchestrationError
from jarvis.planning.base import Decision, DecisionKind, DecisionRequest
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.builtin.windows import windows_tools

from support import (
    WINDOWS_PERMISSIONS,
    FakeWindowsAutomation,
    ScriptedPlanner,
    VerifiedEcho,
    act,
)

CALC = ("Calculadora", "ApplicationFrameHost.exe")


class World:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings.model_copy(
            update={"permissions": {**WINDOWS_PERMISSIONS, "test.echo": "allow"}}
        )
        self.store = InMemoryStateStore()
        self.audit = InMemoryAuditLog()
        self.desktop = FakeWindowsAutomation()

    def jarvis(self, *extra: Tool[Any], **kw: Any) -> Jarvis:
        return build_jarvis(
            self.settings,
            windows=self.desktop,
            tools=[
                *filesystem_tools(),
                *windows_tools(self.desktop, build_catalog(), open_wait_seconds=0.05),
                *extra,
            ],
            store=self.store,
            audit=self.audit,
            **kw,
        )

    def cancel_current(self) -> None:
        """What `jarvis cancel <id>` from another terminal does."""
        (task,) = self.store.list()
        self.jarvis().cancel(task.id)

    def observed(self, task_id: str) -> bool:
        return AuditEventType.TASK_CANCEL_OBSERVED in [e.type for e in self.audit.events(task_id)]


@pytest.fixture
def world(settings: Settings) -> World:
    return World(settings)


def _two_steps() -> Decision:
    return Decision(
        kind=DecisionKind.NEXT_ACTION,
        planner="scripted",
        actions=(
            Action(id="s1", tool="test.verified", arguments={"text": "1"}),
            Action(id="s2", tool="test.verified", arguments={"text": "2"}),
        ),
        final=True,
    )


def test_cancel_while_observing(world: World) -> None:
    world.desktop.on_list = lambda n: world.cancel_current() if n == 1 else None
    planner = ScriptedPlanner(act("fs.list_dir", path="."))
    response = world.jarvis(planner=planner).ask("lista")
    assert response.state is TaskState.CANCELLED
    assert planner.requests == []  # the loop did not go on to plan
    assert world.store.get(response.task_id).actions == []


def test_cancel_while_waiting_for_the_planner(world: World) -> None:
    def cancel_then_propose(request: DecisionRequest) -> Decision:
        world.cancel_current()
        return act("fs.list_dir", path=".")

    planner = ScriptedPlanner(cancel_then_propose, act("fs.list_dir", path="./"))
    response = world.jarvis(planner=planner).ask("lista")
    assert response.state is TaskState.CANCELLED
    task = world.store.get(response.task_id)
    assert all(r.status is ActionStatus.SKIPPED and not r.attempts for r in task.actions)
    assert len(planner.requests) == 1
    assert world.observed(response.task_id)


def test_cancel_while_waiting_for_confirmation(world: World) -> None:
    world.desktop.add_window(*CALC)
    waiting = world.jarvis().ask("cierra la calculadora")
    assert waiting.state is TaskState.WAITING_PERMISSION
    world.jarvis().cancel(waiting.task_id)
    with pytest.raises(OrchestrationError):
        world.jarvis().approve(waiting.task_id, "a1")
    task = world.store.get(waiting.task_id)
    assert task.state is TaskState.CANCELLED
    assert task.actions[0].status is ActionStatus.SKIPPED
    assert world.desktop.close_calls == []


def test_cancel_while_executing(world: World) -> None:
    class CancelsDuringRun(VerifiedEcho):
        def run(self, args: Any, ctx: Any) -> dict[str, Any]:
            output = super().run(args, ctx)
            world.cancel_current()  # arrives while the action is running
            return output

    tool = CancelsDuringRun()
    response = world.jarvis(tool, planner=ScriptedPlanner(_two_steps())).ask("dos pasos")
    assert response.state is TaskState.CANCELLED
    task = world.store.get(response.task_id)
    # The running action finished (not rolled back) and is recorded as such; step 2 never ran.
    assert task.actions[0].status is ActionStatus.EXECUTED
    assert task.actions[1].status is ActionStatus.SKIPPED
    assert tool.calls == ["1"]
    assert task.failure is not None and task.failure.details["cancelled_while"] == "running"


def test_cancel_while_verifying(world: World) -> None:
    tool = VerifiedEcho()
    tool.on_verify = lambda n: world.cancel_current() if n == 1 else None
    response = world.jarvis(tool, planner=ScriptedPlanner(_two_steps())).ask("dos pasos")
    assert response.state is TaskState.CANCELLED
    task = world.store.get(response.task_id)
    assert task.actions[0].status is ActionStatus.VERIFIED  # what really happened
    assert task.actions[1].status is ActionStatus.SKIPPED
    assert tool.calls == ["1"]


def test_cancel_while_replanning(world: World) -> None:
    def cancel_during_replan(request: DecisionRequest) -> Decision:
        assert request.history[-1].failure_kind == "tool_error"
        world.cancel_current()
        return act("fs.list_dir", path=".")

    planner = ScriptedPlanner(
        act("fs.read_file", path="missing.txt"),
        cancel_during_replan,
        act("fs.list_dir", path="./"),
    )
    response = world.jarvis(planner=planner).ask("lee missing")
    assert response.state is TaskState.CANCELLED
    task = world.store.get(response.task_id)
    assert [r.status for r in task.actions] == [ActionStatus.FAILED, ActionStatus.SKIPPED]
    assert len(planner.requests) == 2
    assert task.progress.replans_used == 1
