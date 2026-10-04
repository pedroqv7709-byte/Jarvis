"""Recovery at each point of the agent loop.

``SimulatedCrash`` is a BaseException: it escapes every ``except Exception`` in Jarvis, exactly
like the process dying. Whatever was persisted at that moment is what a new Jarvis instance (same
store) resumes from.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import ActionStatus, AttemptStatus
from jarvis.core.agent import AgentPhase
from jarvis.core.errors import ErrorKind
from jarvis.core.task import TaskState
from jarvis.desktop.catalog import build_catalog
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.planning.base import DecisionRequest
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.builtin.windows import windows_tools

from support import (
    WINDOWS_PERMISSIONS,
    FakeWindowsAutomation,
    ScriptedPlanner,
    SimulatedCrash,
    VerifiedEcho,
    act,
    done,
)

CALC = ("Calculadora", "ApplicationFrameHost.exe")


class World:
    """One persistent store; each ``jarvis()`` call is a fresh process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.model_copy(
            update={
                "permissions": {**WINDOWS_PERMISSIONS, "test.echo": "allow"},
                "windows_open_wait_seconds": 0.05,
            }
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

    def crash(self, jarvis: Jarvis, goal: str) -> str:
        with pytest.raises(SimulatedCrash):
            jarvis.ask(goal)
        (task,) = self.store.list()
        assert task.state is TaskState.RUNNING  # what a killed process leaves behind
        return task.id


@pytest.fixture
def world(settings: Settings) -> World:
    return World(settings)


def _raise(*_: Any) -> Any:
    raise SimulatedCrash


def test_1_after_observation(world: World) -> None:
    task_id = world.crash(world.jarvis(planner=ScriptedPlanner(_raise)), "abre la calculadora")
    persisted = world.store.get(task_id)
    assert persisted.progress.phase is AgentPhase.PLANNING and persisted.actions == []
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC)
    response = world.jarvis().resume(task_id)
    assert response.state is TaskState.COMPLETED
    assert len(world.desktop.launches) == 1


def test_2_after_decision_before_execution(world: World) -> None:
    # Crash on the second desktop query: the resolution of the action just decided.
    world.desktop.on_list = lambda n: _raise() if n == 2 else None
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC)
    task_id = world.crash(world.jarvis(), "abre la calculadora")
    persisted = world.store.get(task_id)
    assert [r.status for r in persisted.actions] == [ActionStatus.PLANNED]
    assert world.desktop.launches == []
    world.desktop.on_list = None
    response = world.jarvis().resume(task_id)
    assert response.state is TaskState.COMPLETED
    assert len(world.desktop.launches) == 1
    assert world.store.get(task_id).progress.actions_used == 1  # not counted twice


def test_3_waiting_for_confirmation(world: World) -> None:
    calc = world.desktop.add_window(*CALC)
    waiting = world.jarvis().ask("cierra la calculadora")
    assert waiting.state is TaskState.WAITING_PERMISSION
    response = world.jarvis().approve(waiting.task_id, "a1")  # another process
    assert response.state is TaskState.COMPLETED
    assert world.desktop.close_calls == [calc.handle]


def test_4_after_execution_before_verification(world: World) -> None:
    tool = VerifiedEcho()
    tool.on_verify = lambda n: _raise() if n == 1 else None
    planner = ScriptedPlanner(act("test.verified", final=True, text="x"))
    task_id = world.crash(world.jarvis(tool, planner=planner), "haz eso")
    record = world.store.get(task_id).actions[0]
    assert record.status is ActionStatus.EXECUTED and record.verification is None
    response = world.jarvis(tool, planner=ScriptedPlanner()).resume(task_id)
    assert response.state is TaskState.COMPLETED
    assert tool.calls == ["x"]  # verified on resume, never executed again
    assert tool.verifies == 2
    assert response.actions[0].status is ActionStatus.VERIFIED
    resumed = [
        e
        for e in world.audit.events(task_id)
        if e.type is AuditEventType.AGENT_VERIFICATION and e.data.get("resumed")
    ]
    assert len(resumed) == 1


def test_5_after_verification_before_next_decision(world: World) -> None:
    world.desktop.add_window("Documento", "msedge.exe")
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC, focus=False)
    # The first foreground query after the launch is the observation that follows verification.
    world.desktop.on_foreground = lambda _: _raise() if world.desktop.launches else None
    task_id = world.crash(world.jarvis(), "abre la calculadora y enfócala")
    assert world.store.get(task_id).actions[0].status is ActionStatus.VERIFIED
    world.desktop.on_foreground = None
    response = world.jarvis().resume(task_id)
    assert response.state is TaskState.COMPLETED
    assert len(world.desktop.launches) == 1  # not reopened
    assert [a.tool for a in response.actions] == [
        "windows.open_application",
        "windows.focus_window",
    ]


def test_6_during_replanning(world: World) -> None:
    planner = ScriptedPlanner(act("fs.read_file", path="missing.txt"), _raise)
    task_id = world.crash(world.jarvis(planner=planner), "lee missing")
    persisted = world.store.get(task_id)
    assert persisted.progress.phase is AgentPhase.REPLANNING
    assert persisted.progress.replans_used == 1

    def check(request: DecisionRequest) -> Any:
        assert request.history[-1].failure_kind == "tool_error"
        return act("fs.list_dir", path=".")

    response = world.jarvis(planner=ScriptedPlanner(check, done())).resume(task_id)
    assert response.state is TaskState.COMPLETED
    task = world.store.get(task_id)
    assert task.progress.replans_used == 1  # the interrupted replan is not counted twice
    assert [r.action.tool for r in task.actions] == ["fs.read_file", "fs.list_dir"]


def _crash_in_launch(world: World) -> str:
    world.desktop.on_launch["calc.exe"] = _raise  # dies inside the launch: outcome unknown
    task_id = world.crash(world.jarvis(), "abre la calculadora")
    attempt = world.store.get(task_id).actions[0].attempts[-1]
    assert attempt.status is AttemptStatus.STARTED
    return task_id


def test_7_outcome_unknown_that_did_happen_is_observed_not_repeated(world: World) -> None:
    task_id = _crash_in_launch(world)
    world.desktop.add_window(*CALC)  # the launch had in fact worked
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC)
    response = world.jarvis().resume(task_id)
    assert response.state is TaskState.COMPLETED
    assert len(world.desktop.launches) == 1
    assert response.actions[0].failure is not None
    assert response.actions[0].failure.kind is ErrorKind.INTERRUPTED


def test_7_outcome_unknown_is_never_blindly_repeated(world: World) -> None:
    task_id = _crash_in_launch(world)
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC)
    response = world.jarvis().resume(task_id)
    # Nothing is open, but the launch of unknown outcome is not repeated: the task ends safely.
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.INTERRUPTED
    assert "not retried" in response.failure.message
    assert len(world.desktop.launches) == 1


def test_7_planner_insisting_on_repeating_an_unknown_outcome_is_refused(world: World) -> None:
    task_id = _crash_in_launch(world)
    world.desktop.on_launch["calc.exe"] = lambda: world.desktop.add_window(*CALC)
    insists = ScriptedPlanner(act("windows.open_application", application="calculadora"))
    response = world.jarvis(planner=insists).resume(task_id)
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.CYCLE_DETECTED
    assert "not idempotent" in response.failure.message
    assert len(world.desktop.launches) == 1


def test_8_after_a_limit_was_reached(world: World) -> None:
    world.settings = world.settings.model_copy(update={"agent_max_iterations": 1})
    planner = ScriptedPlanner(act("fs.list_dir", path="."), act("fs.list_dir", path="./"))
    stopped = world.jarvis(planner=planner).ask("explora")
    assert stopped.state is TaskState.STOPPED
    second = ScriptedPlanner(act("fs.list_dir", path="./."))
    response = world.jarvis(planner=second).resume(stopped.task_id)
    assert response.state is TaskState.STOPPED
    assert second.requests == []  # terminal: nothing runs again
    assert len(world.store.get(stopped.task_id).actions) == 1
