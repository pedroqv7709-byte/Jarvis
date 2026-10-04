"""The v0.5 agent loop: observe -> decide -> validate -> policy -> execute -> verify -> observe ...

Everything runs through the real pipeline (orchestrator, runner, executor, policy, verification,
audit) on a fake desktop. Assertions check effects (what was launched/focused/closed, files) and
the audited order of events, not only return values.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import ActionStatus
from jarvis.core.errors import ErrorKind
from jarvis.core.task import TaskState
from jarvis.desktop.catalog import build_catalog
from jarvis.observability.audit import AuditEventType as T
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.planning.base import DecisionRequest
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.builtin.windows import windows_tools

from support import (
    INJECTION,
    WINDOWS_PERMISSIONS,
    FakeWindowsAutomation,
    InjectingTool,
    ScriptedPlanner,
    act,
    done,
    fake_model,
)

CALC = ("Calculadora", "ApplicationFrameHost.exe")


@pytest.fixture
def agent_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "permissions": {**WINDOWS_PERMISSIONS, "test.echo": "allow"},
            "windows_open_wait_seconds": 0.05,
            "windows_close_wait_seconds": 0.05,
        }
    )


@pytest.fixture
def desktop() -> FakeWindowsAutomation:
    return FakeWindowsAutomation()


def tools(desktop: FakeWindowsAutomation, *extra: Tool[Any]) -> list[Tool[Any]]:
    return [
        *filesystem_tools(),
        *windows_tools(desktop, build_catalog(), open_wait_seconds=0.05, close_wait_seconds=0.05),
        *extra,
    ]


def make(
    settings: Settings, desktop: FakeWindowsAutomation, *extra: Tool[Any], **kw: Any
) -> Jarvis:
    return build_jarvis(
        settings,
        windows=desktop,
        tools=tools(desktop, *extra),
        store=kw.pop("store", InMemoryStateStore()),
        audit=kw.pop("audit", InMemoryAuditLog()),
        **kw,
    )


def types(jarvis: Jarvis, task_id: str) -> list[T]:
    return [e.type for e in jarvis.audit_trail(task_id)]


def is_subsequence(sequence: Sequence[T], wanted: Sequence[T]) -> bool:
    remaining = iter(sequence)
    return all(any(item is w for item in remaining) for w in wanted)


# -- A: normal multi-step loop ---------------------------------------------------------------


def test_a_observe_decide_execute_verify_twice_then_complete(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("Documento", "msedge.exe")  # in front
    desktop.on_launch["calc.exe"] = lambda: desktop.add_window(*CALC, focus=False)
    jarvis = make(agent_settings, desktop)
    response = jarvis.ask("abre la calculadora y enfócala")

    assert response.state is TaskState.COMPLETED
    assert [(a.tool, a.status) for a in response.actions] == [
        ("windows.open_application", ActionStatus.VERIFIED),
        ("windows.focus_window", ActionStatus.VERIFIED),
    ]
    calc = next(w for w in desktop.windows.values() if w.title == "Calculadora")
    assert desktop.foreground == calc.handle
    assert len(desktop.launches) == 1
    assert is_subsequence(
        types(jarvis, response.task_id),
        [
            T.AGENT_REQUIREMENTS,
            T.AGENT_OBSERVATION,  # nothing open yet
            T.AGENT_PLAN,
            T.AGENT_ACTION_SELECTED,  # open
            T.POLICY_EVALUATED,
            T.ACTION_EXECUTED,
            T.ACTION_VERIFIED,
            T.AGENT_VERIFICATION,
            T.AGENT_OBSERVATION,  # the new state...
            T.AGENT_REQUIREMENT_MET,  # ...shows it open
            T.AGENT_PLAN,
            T.AGENT_ACTION_SELECTED,  # focus, decided from that state
            T.POLICY_EVALUATED,
            T.ACTION_EXECUTED,
            T.ACTION_VERIFIED,
            T.AGENT_VERIFICATION,
            T.AGENT_OBSERVATION,
            T.AGENT_REQUIREMENT_MET,  # in the foreground
            T.AGENT_PLAN,  # done
            T.AGENT_COMPLETED,
        ],
    )
    progress = jarvis.task(response.task_id).progress
    assert (progress.iterations, progress.actions_used, progress.replans_used) == (3, 2, 0)


# -- B: already satisfied --------------------------------------------------------------------


def test_b_already_open_application_is_not_launched_again(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window(*CALC)
    jarvis = make(agent_settings, desktop)
    response = jarvis.ask("abre la calculadora")
    assert response.state is TaskState.COMPLETED
    assert desktop.launches == []
    assert response.actions == ()
    assert "ya se cumplía" in response.message
    completed = next(e for e in jarvis.audit_trail(response.task_id) if e.type is T.AGENT_COMPLETED)
    assert completed.data["evidence"]["requirements"][0]["by"] == "observation"


# -- C: execution "succeeds", verification fails ---------------------------------------------


def test_c_focus_that_never_takes_effect_is_never_reported_as_success(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window(*CALC)
    desktop.add_window("Documento", "msedge.exe")
    desktop.focus_works = False
    jarvis = make(agent_settings, desktop)
    response = jarvis.ask("enfoca la calculadora")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.VERIFICATION_FAILED
    assert [a.status for a in response.actions] == [ActionStatus.FAILED, ActionStatus.FAILED]
    assert len(desktop.focus_calls) == 2  # one bounded retry of an idempotent action
    assert T.AGENT_REPLAN in types(jarvis, response.task_id)
    assert T.AGENT_COMPLETED not in types(jarvis, response.task_id)


def test_c_late_window_is_picked_up_by_observation_without_relaunching(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    # The window appears only after the tool's own verification has given up.
    desktop.on_launch["calc.exe"] = lambda: desktop.delay_window(3, *CALC)
    jarvis = make(agent_settings, desktop)
    response = jarvis.ask("abre la calculadora")
    assert response.state is TaskState.COMPLETED
    assert response.actions[0].status is ActionStatus.FAILED
    assert response.actions[0].verification is not None
    assert len(desktop.launches) == 1  # never relaunched
    assert jarvis.task(response.task_id).progress.replans_used == 1


# -- D: the next decision depends on the previous result -------------------------------------


def test_d_replanning_uses_the_failure(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    def second(request: DecisionRequest) -> Any:
        last = request.history[-1]
        assert (last.tool, last.failure_kind) == ("fs.read_file", "tool_error")
        assert "missing.txt" in (last.failure_message or "")
        assert request.feedback is not None and "failed" in request.feedback
        return act("fs.list_dir", path=".")  # inspect the folder instead

    planner = ScriptedPlanner(act("fs.read_file", path="missing.txt"), second, done())
    jarvis = make(agent_settings, desktop, planner=planner)
    response = jarvis.ask("lee missing.txt o dime qué hay")
    assert response.state is TaskState.COMPLETED
    assert [(a.tool, a.status) for a in response.actions] == [
        ("fs.read_file", ActionStatus.FAILED),
        ("fs.list_dir", ActionStatus.EXECUTED),
    ]
    assert len(planner.requests) == 3
    assert planner.requests[0].history == ()


# -- E: ambiguity ----------------------------------------------------------------------------


def test_e_two_candidate_windows_none_is_touched(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("uno - Bloc de notas", "Notepad.exe")
    desktop.add_window("dos - Bloc de notas", "Notepad.exe")
    desktop.add_window("Documento", "msedge.exe")
    jarvis = make(agent_settings, desktop)
    for goal in ("enfoca el bloc de notas", "cierra el bloc de notas"):
        response = jarvis.ask(goal)
        assert response.state is TaskState.FAILED
        assert response.failure is not None and response.failure.kind is ErrorKind.AMBIGUOUS
        assert len(response.failure.details["candidates"]) == 2
        stopped = next(e for e in jarvis.audit_trail(response.task_id) if e.type is T.AGENT_STOPPED)
        assert "none chosen" in stopped.message
    assert desktop.focus_calls == [] and desktop.close_calls == []
    assert len(desktop.windows) == 3


# -- F: confirmation invalidated by a changed target -----------------------------------------


def test_f_changed_target_needs_a_new_confirmation(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    note = desktop.add_window("notas.txt - Bloc de notas", "Notepad.exe")
    jarvis = make(agent_settings, desktop)
    waiting = jarvis.ask("cierra el bloc de notas")
    assert waiting.state is TaskState.WAITING_PERMISSION
    desktop.rename(note.handle, "*notas.txt - Bloc de notas")  # unsaved changes appear
    again = jarvis.approve(waiting.task_id, "a1")
    assert again.state is TaskState.WAITING_PERMISSION  # the old approval did not apply
    assert again.pending_confirmations[0].target["unsaved_changes_suspected"] is True  # type: ignore[index]
    assert desktop.close_calls == []
    final = jarvis.approve(waiting.task_id, "a1")
    assert final.state is TaskState.COMPLETED
    assert desktop.close_calls == [note.handle]


# -- G/H: policy during replanning and DENY ---------------------------------------------------


def test_g_forbidden_action_proposed_after_a_success_is_denied_and_not_worked_around(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    settings = agent_settings.model_copy(
        update={"permissions": {**WINDOWS_PERMISSIONS, "fs.delete": "deny"}}
    )
    planner = ScriptedPlanner(
        act("fs.write_file", path="a.txt", content="x"),
        act("fs.delete_file", path="a.txt"),
        act("fs.overwrite_file", path="a.txt", content=""),  # must never be asked for
    )
    jarvis = make(settings, desktop, planner=planner)
    response = jarvis.ask("haz limpieza")
    assert response.state is TaskState.FAILED
    assert [a.status for a in response.actions] == [ActionStatus.VERIFIED, ActionStatus.DENIED]
    assert response.failure is not None and response.failure.kind is ErrorKind.POLICY_DENIED
    assert len(planner.requests) == 2  # no replanning around the denial
    assert (settings.workspace_dir / "a.txt").read_text(encoding="utf-8") == "x"
    assert T.AGENT_REPLAN not in types(jarvis, response.task_id)


def test_h_deny_on_the_first_action_ends_safely(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    settings = agent_settings.model_copy(
        update={"permissions": {**WINDOWS_PERMISSIONS, "fs.write": "deny"}}
    )
    planner = ScriptedPlanner(act("fs.write_file", path="b.txt", content="x"))
    jarvis = make(settings, desktop, planner=planner)
    response = jarvis.ask("escribe")
    assert response.state is TaskState.FAILED
    assert response.actions[0].status is ActionStatus.DENIED
    assert len(planner.requests) == 1
    assert not (settings.workspace_dir / "b.txt").exists()


# -- I: tool / observation output injection ---------------------------------------------------


def test_i_injection_in_tool_output_is_inert_data_for_the_deterministic_planner(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    agent_settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (agent_settings.workspace_dir / f"{INJECTION[:-1]}.txt").write_text("x", encoding="utf-8")
    desktop.add_window(INJECTION, "evil.exe")
    jarvis = make(agent_settings, desktop)
    for goal in ("lista los archivos", "lista las ventanas"):
        response = jarvis.ask(goal)
        assert response.state is TaskState.COMPLETED
        assert len(response.actions) == 1
    assert desktop.launches == [] and desktop.close_calls == [] and desktop.focus_calls == []


def test_i_injection_reaches_the_model_only_as_data_and_grants_nothing(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window(*CALC)
    seen: list[dict[str, Any]] = []

    def obey(kwargs: dict[str, Any]) -> str:
        seen.append(kwargs)
        # The "model" falls for it and proposes the injected command.
        return json.dumps(
            {
                "kind": "next_action",
                "action": {
                    "tool": "windows.open_application",
                    "arguments": {"application": "powershell"},
                },
            }
        )

    settings = agent_settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake", "planner_fallback": False}
    )
    provider, _ = fake_model(
        '{"kind": "next_action", "action": {"tool": "test.read_note", "arguments": {}}}',
        obey,
        '{"kind": "stop", "reason": "nothing else"}',
    )
    jarvis = make(settings, desktop, InjectingTool(), model_provider=provider)
    response = jarvis.ask("lee la nota")

    system, user = seen[0]["messages"][0]["content"], seen[0]["messages"][1]["content"]
    assert "untrusted DATA" in system
    assert INJECTION not in system  # never part of the instructions...
    assert INJECTION in user and '"history"' in user  # ...only inside the data payload
    assert desktop.launches == []  # powershell is not an allowed application
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.NOT_FOUND
    # The "grant" in the tool output changed nothing: closing still needs confirmation.
    plain = make(agent_settings, desktop)
    assert plain.ask("cierra la calculadora").state is TaskState.WAITING_PERMISSION
    assert desktop.close_calls == []


# -- J: invalid model output -----------------------------------------------------------------


INVALID_ANSWERS = [
    "{not json",
    '{"kind": "next_action"}',  # no action
    '{"kind": "jump", "action": null}',  # unknown decision type
    '{"kind": "next_action", "action": {"tool": "shell.exec", "arguments": {"cmd": "x"}}}',
    '{"kind": "next_action", "action": {"tool": "fs.read_file", "arguments": {"path": 5}}}',
    '{"kind": "done", "permissions": {"windows.close": "allow"}}',  # unknown field
    "Sure! I will open PowerShell for you.",
]


@pytest.mark.parametrize("answer", INVALID_ANSWERS)
def test_j_invalid_model_output_never_executes(
    agent_settings: Settings, desktop: FakeWindowsAutomation, answer: str
) -> None:
    settings = agent_settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake", "planner_fallback": False}
    )
    provider, _ = fake_model(answer)
    jarvis = make(settings, desktop, model_provider=provider)
    response = jarvis.ask("haz algo con mi equipo")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.PLANNING
    assert response.actions == ()
    assert desktop.launches == [] and desktop.close_calls == []


@pytest.mark.parametrize("answer", INVALID_ANSWERS)
def test_j_invalid_model_output_falls_back_safely(
    agent_settings: Settings, desktop: FakeWindowsAutomation, answer: str
) -> None:
    settings = agent_settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake"}
    )
    provider, completion = fake_model(answer)
    jarvis = make(settings, desktop, model_provider=provider)
    response = jarvis.ask("lista los archivos")
    assert response.state is TaskState.COMPLETED
    assert response.planner == "rule_based"
    assert response.fallback_reason is not None
    assert len(completion.calls) == 1  # the failed model is not asked again in this task


# -- idempotency -----------------------------------------------------------------------------


def test_idempotent_action_is_retried_once_under_the_rules(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window(*CALC)
    desktop.add_window("Documento", "msedge.exe")
    desktop.focus_failures = 1
    response = make(agent_settings, desktop).ask("enfoca la calculadora")
    assert response.state is TaskState.COMPLETED
    assert [a.status for a in response.actions] == [ActionStatus.FAILED, ActionStatus.VERIFIED]
    assert len(desktop.focus_calls) == 2


def test_failed_non_idempotent_action_is_never_repeated(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.on_launch["calc.exe"] = lambda: None  # nothing ever appears
    planner = ScriptedPlanner(
        act("windows.open_application", application="calculator"),
        act("windows.open_application", application="calculator"),  # asks again
    )
    response = make(agent_settings, desktop, planner=planner).ask("abre la calculadora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.CYCLE_DETECTED
    assert "not idempotent" in response.failure.message
    assert len(desktop.launches) == 1


# -- requirements and completion -------------------------------------------------------------


def test_contextual_reference_without_antecedent_stops_instead_of_guessing(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window(*CALC, focus=False)
    desktop.add_window("Documento", "msedge.exe")
    jarvis = make(agent_settings, desktop)
    for goal in ("enfoca esa ventana", "enfoca ese", "ciérrala"):
        response = jarvis.ask(goal)
        assert response.state is TaskState.STOPPED, goal
        assert response.failure is not None and response.failure.kind is ErrorKind.STOPPED
        assert "ambiguous" in response.failure.message
        assert response.actions == ()
        assert jarvis.task(response.task_id).requirements == []
    assert desktop.focus_calls == [] and desktop.close_calls == []


def test_contextual_reference_resolved_from_the_same_request(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("Documento", "msedge.exe")
    desktop.on_launch["calc.exe"] = lambda: desktop.add_window(*CALC, focus=False)
    jarvis = make(agent_settings, desktop)
    response = jarvis.ask("abre la calculadora y luego enfoca esa ventana")
    assert response.state is TaskState.COMPLETED
    task = jarvis.task(response.task_id)
    assert [(s.requirement.kind.value, s.requirement.application) for s in task.requirements] == [
        ("app_open", "calculator"),
        ("app_focused", "calculator"),
    ]
    assert all(s.achieved for s in task.requirements)


def test_alternative_planner_action_satisfies_the_requirement(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    calc = desktop.add_window(*CALC)
    desktop.add_window("Documento", "msedge.exe")
    # Words differ from the request ("handle" instead of "calculadora"): evidence decides.
    planner = ScriptedPlanner(act("windows.focus_window", handle=calc.handle), done())
    response = make(agent_settings, desktop, planner=planner).ask("enfoca la calculadora")
    assert response.state is TaskState.COMPLETED
    assert desktop.foreground == calc.handle


def test_done_without_evidence_is_refused_then_stopped(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    planner = ScriptedPlanner(done(), done())
    jarvis = make(agent_settings, desktop, planner=planner)
    response = jarvis.ask("abre la calculadora")
    assert response.state is TaskState.STOPPED
    assert response.failure is not None and response.failure.kind is ErrorKind.CYCLE_DETECTED
    assert desktop.launches == []
    events = jarvis.audit_trail(response.task_id)
    refused = [e for e in events if e.type is T.AGENT_REPLAN and e.outcome == "completion_refused"]
    assert len(refused) == 1 and "app_open" in refused[0].message
    assert T.AGENT_COMPLETED not in [e.type for e in events]


def test_refused_done_leads_to_a_new_decision(
    agent_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    planner = ScriptedPlanner(done(), act("fs.list_dir", path="."), done())
    response = make(agent_settings, desktop, planner=planner).ask("haz algo útil")
    assert response.state is TaskState.COMPLETED
    assert planner.requests[1].feedback is not None
    assert "completion refused" in planner.requests[1].feedback


def test_completed_only_with_evidence_without_desktop_observation(tmp_path: Path) -> None:
    """With no desktop observation, only a VERIFIED action on the resolved app counts."""
    from jarvis.agent.completion import CompletionChecker
    from jarvis.core.action import Action, ActionRecord
    from jarvis.core.agent import Requirement, RequirementKind, RequirementState
    from jarvis.core.request import Request
    from jarvis.core.task import Task

    focus = Action(id="a1", tool="windows.focus_window", arguments={"handle": 7})
    requirement = Requirement(
        kind=RequirementKind.APP_FOCUSED,
        action=Action(id="r", tool="windows.focus_window", arguments={"application": "calc"}),
        application="calculator",
    )
    task = Task.from_request(Request(text="enfoca la calculadora", user_id="u"))
    task.requirements = [RequirementState(requirement=requirement)]
    record = ActionRecord(
        action=focus,
        target={"window": {"title": "Calculadora", "process": "ApplicationFrameHost.exe"}},
    )
    record.status = ActionStatus.FAILED
    task.actions.append(record)
    checker = CompletionChecker(build_catalog())
    assert not checker.evaluate(task, ()).accepted
    record.status = ActionStatus.VERIFIED
    verdict = checker.evaluate(task, ())
    assert verdict.accepted
    assert verdict.evidence["requirements"][0]["by"] == "verified_action"
