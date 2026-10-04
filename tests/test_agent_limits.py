"""Hard limits of the agent loop, cycle detection, bounded context and requirement extraction.

Limits are enforced by Jarvis; the scripted planners here never decide to stop on their own.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from jarvis.agent.context import bound_observation, build_request
from jarvis.agent.limits import AgentLimits, detect_cycle
from jarvis.agent.requirements import RequirementExtractor
from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionRecord, ActionStatus
from jarvis.core.errors import ErrorKind, Failure
from jarvis.core.observation import Observation
from jarvis.core.request import Request
from jarvis.core.task import Task, TaskState
from jarvis.desktop.catalog import build_catalog
from jarvis.models.router import ModelRouter
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.planning.base import Decision, DecisionKind, DecisionRequest
from jarvis.planning.model_planner import ModelPlanner
from jarvis.state.store import InMemoryStateStore

from support import ScriptedPlanner, act, make_context


def _jarvis(settings: Settings, planner: ScriptedPlanner, **limits: Any) -> Jarvis:
    return build_jarvis(
        settings.model_copy(update=limits),
        planner=planner,
        store=InMemoryStateStore(),
        audit=InMemoryAuditLog(),
    )


def _limit_event(jarvis: Jarvis, task_id: str) -> str | None:
    return next(
        (
            e.outcome
            for e in jarvis.audit_trail(task_id)
            if e.type is AuditEventType.AGENT_LIMIT_REACHED
        ),
        None,
    )


LISTINGS = [".", "./", "./.", "././", "./././"]


def test_iteration_limit(settings: Settings) -> None:
    planner = ScriptedPlanner(*(act("fs.list_dir", path=p) for p in LISTINGS))
    jarvis = _jarvis(settings, planner, agent_max_iterations=3)
    response = jarvis.ask("explora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.LIMIT_EXCEEDED
    assert response.failure.details["limit"] == "iterations"
    assert len(planner.requests) == 3
    assert _limit_event(jarvis, response.task_id) == "iterations"
    assert jarvis.task(response.task_id).progress.stop_reason is not None


def test_action_limit(settings: Settings) -> None:
    batch = Decision(
        kind=DecisionKind.NEXT_ACTION,
        planner="scripted",
        actions=tuple(
            Action(id=f"s{i}", tool="fs.list_dir", arguments={"path": p})
            for i, p in enumerate(LISTINGS[:3])
        ),
        final=True,
    )
    jarvis = _jarvis(settings, ScriptedPlanner(batch), agent_max_actions=2)
    response = jarvis.ask("explora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.details["limit"] == "actions"
    assert [a.status for a in response.actions] == [
        ActionStatus.EXECUTED,
        ActionStatus.EXECUTED,
        ActionStatus.SKIPPED,
    ]
    assert _limit_event(jarvis, response.task_id) == "actions"


def test_replan_limit(settings: Settings) -> None:
    planner = ScriptedPlanner(*(act("fs.read_file", path=f"missing{i}.txt") for i in range(5)))
    jarvis = _jarvis(settings, planner, agent_max_replans=1)
    response = jarvis.ask("lee algo")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.details["limit"] == "replans"
    assert len(response.actions) == 2  # first failure replanned once, second one hit the limit
    assert jarvis.task(response.task_id).progress.replans_used == 1


def test_global_timeout_stops_before_executing_more(settings: Settings) -> None:
    def slow(request: DecisionRequest) -> Decision:
        time.sleep(0.35)  # planning time counts against the global budget
        return act("fs.list_dir", path=".")

    jarvis = _jarvis(settings, ScriptedPlanner(slow), task_timeout_seconds=0.3)
    started = time.monotonic()
    response = jarvis.ask("lista")
    assert time.monotonic() - started < 5
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.TIMEOUT
    assert response.failure.details == {"scope": "task", "limit": "time"}
    assert response.actions[0].attempts == 0  # decided, never executed
    assert _limit_event(jarvis, response.task_id) == "time"


def test_cycle_a_b_a_b(settings: Settings) -> None:
    planner = ScriptedPlanner(*(act("fs.list_dir", path=p) for p in [".", "./", ".", "./", "."]))
    jarvis = _jarvis(settings, planner)
    response = jarvis.ask("explora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.CYCLE_DETECTED
    assert "alternates" in response.failure.message
    assert len(response.actions) == 3  # the 4th proposal was refused
    assert AuditEventType.AGENT_CYCLE_DETECTED in [
        e.type for e in jarvis.audit_trail(response.task_id)
    ]


def test_cycle_a_a_a(settings: Settings) -> None:
    planner = ScriptedPlanner(*(act("fs.list_dir", path=".") for _ in range(5)))
    response = _jarvis(settings, planner).ask("explora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.CYCLE_DETECTED
    assert len(response.actions) == 2


def test_cycle_detection_is_deterministic() -> None:
    assert detect_cycle(["a", "b", "a"], "b") is not None
    assert detect_cycle(["a", "a"], "a") is not None
    assert detect_cycle(["a", "b", "c"], "d") is None
    assert detect_cycle(["a", "b"], "a") is None


@pytest.mark.parametrize(
    "bad",
    [{"max_iterations": 0}, {"max_actions": 0}, {"max_replans": -1}, {"task_timeout_seconds": 0}],
)
def test_invalid_limits_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AgentLimits(**bad)


# -- bounded context -------------------------------------------------------------------------


def _task_with_history(n: int) -> Task:
    task = Task.from_request(Request(text="explora", user_id="u"))
    for i in range(n):
        record = ActionRecord(
            action=Action(id=f"a{i}", tool="fs.read_file", arguments={"path": f"f{i}"})
        )
        record.status = ActionStatus.FAILED
        record.failure = Failure(kind=ErrorKind.TOOL_ERROR, message="x" * 1000)
        task.actions.append(record)
    return task


def test_history_steps_are_limited_to_the_latest() -> None:
    request = build_request(
        _task_with_history(15), make_context(), [], (), AgentLimits(history_steps=4)
    )
    assert [s.action_id for s in request.history] == ["a11", "a12", "a13", "a14"]
    assert all(len(s.failure_message or "") <= 200 for s in request.history)


def test_observed_windows_are_limited() -> None:
    windows = [{"handle": i, "title": f"w{i}", "process": "x.exe"} for i in range(50)]
    observation = Observation(
        source="windows",
        status="ok",
        data={"windows": windows},
        identifiers=tuple(str(i) for i in range(50)),
    )
    bounded = bound_observation(observation, 30)
    assert len(bounded.data["windows"]) == 30 and bounded.data["truncated"] is True
    assert len(bounded.identifiers) == 30


def test_model_context_is_bounded_valid_and_deterministic() -> None:
    task = _task_with_history(10)
    request = build_request(task, make_context(), [], (), AgentLimits(history_steps=10))
    planner = ModelPlanner(ModelRouter(), max_prompt_chars=2000)
    first = planner._payload(request)
    assert first == planner._payload(request)  # deterministic
    assert len(first) <= 2000
    payload = json.loads(first)  # always valid JSON
    kept = [h["action_id"] for h in payload["untrusted_data"]["history"]]
    assert kept and kept == [f"a{i}" for i in range(10 - len(kept), 10)]  # newest kept


# -- requirement extraction ------------------------------------------------------------------


@pytest.fixture
def extract() -> RequirementExtractor:
    return RequirementExtractor(build_catalog())


def _kinds(extract: RequirementExtractor, goal: str) -> list[tuple[str, str | None]]:
    return [
        (s.requirement.kind.value, s.requirement.application) for s in extract(goal, make_context())
    ]


def test_valid_request_yields_verifiable_requirements(extract: RequirementExtractor) -> None:
    assert _kinds(extract, "abre la calculadora y enfócala") == [
        ("app_open", "calculator"),
        ("app_focused", "calculator"),
    ]
    assert _kinds(extract, "abre el bloc de notas y luego ciérralo") == [
        ("app_open", "notepad"),
        ("app_closed", "notepad"),
    ]
    # Natural sentences (as typed to a model) keep their verifiable requirements.
    assert _kinds(extract, "Abre la calculadora y enfócala.") == [
        ("app_open", "calculator"),
        ("app_focused", "calculator"),
    ]


@pytest.mark.parametrize(
    "goal",
    [
        "enfoca ese",
        "enfoca esa ventana",
        "ciérrala",
        "enfoca photoshop",  # not allowlisted: not verifiable
        "trae la ventana 'Informe' al frente",  # by title: no app identity
        "crea el archivo a.txt",  # no observable state
        "haz magia",
    ],
)
def test_no_requirement_is_invented(extract: RequirementExtractor, goal: str) -> None:
    assert extract(goal, make_context()) == []


def test_reference_resolved_within_the_request(extract: RequirementExtractor) -> None:
    assert _kinds(extract, "abre la calculadora y luego enfoca esa ventana") == [
        ("app_open", "calculator"),
        ("app_focused", "calculator"),
    ]


def test_no_requirements_without_a_desktop() -> None:
    assert RequirementExtractor(None)("abre la calculadora", make_context()) == []
