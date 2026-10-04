"""Windows tools through the real pipeline, on a fake desktop (no real Windows needed).

Every hostile case asserts the *effect* on the (fake) desktop: what was launched, focused or
closed - not only the reported status.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionStatus
from jarvis.core.errors import ErrorKind
from jarvis.core.task import TaskState
from jarvis.desktop.catalog import ApplicationConfig, ApplicationEntry, build_catalog
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.state.store import InMemoryStateStore

from support import (
    SYSTEM32,
    WINDOWS_PERMISSIONS,
    FakeWindowsAutomation,
    StaticPlanner,
    fake_model,
    plan_json,
)

NOTEPAD = rf"{SYSTEM32}\notepad.exe"


@pytest.fixture
def win_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"permissions": WINDOWS_PERMISSIONS})


@pytest.fixture
def desktop() -> FakeWindowsAutomation:
    return FakeWindowsAutomation()


@pytest.fixture(autouse=True)
def _system_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")


def _jarvis(
    settings: Settings, desktop: FakeWindowsAutomation, *actions: Action, **kw: Any
) -> Jarvis:
    if actions:
        kw["planner"] = StaticPlanner(*actions)
    return build_jarvis(
        settings, windows=desktop, store=InMemoryStateStore(), audit=InMemoryAuditLog(), **kw
    )


def _open(app: str) -> Action:
    return Action(id="a1", tool="windows.open_application", arguments={"application": app})


def _events(jarvis: Jarvis, task_id: str, kind: AuditEventType) -> list[Any]:
    return [e for e in jarvis.audit_trail(task_id) if e.type is kind]


# -- open -----------------------------------------------------------------------------------


def test_a_open_allowlisted_application(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    jarvis = _jarvis(win_settings, desktop)
    response = jarvis.ask("abre el bloc de notas")
    assert response.state is TaskState.COMPLETED
    (action,) = response.actions
    assert action.status is ActionStatus.VERIFIED
    assert desktop.launches == [NOTEPAD]
    # requested vs resolved vs found are kept apart in the audit trail
    resolved = _events(jarvis, response.task_id, AuditEventType.ACTION_RESOLVED)[0]
    assert resolved.data["requested"] == {"application": "bloc de notas"}
    assert resolved.data["resolved"]["application"]["app_id"] == "notepad"
    assert resolved.data["resolved"]["application"]["executable"] == NOTEPAD
    verified = _events(jarvis, response.task_id, AuditEventType.ACTION_VERIFIED)[0]
    assert verified.outcome == "verified"
    assert verified.data["evidence"]["new_window"] is True
    assert verified.data["evidence"]["windows"][0]["process"] == "Notepad.exe"


def test_open_reusing_an_existing_window_is_reported_as_such(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    """The tool itself still reports a reused window when it is asked to launch anyway."""
    existing = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    desktop.on_launch["calc.exe"] = lambda: None  # single-instance app: no new window
    settings = win_settings.model_copy(update={"windows_open_wait_seconds": 0.2})
    jarvis = _jarvis(settings, desktop, _open("calculadora"))  # static plan: always launches
    response = jarvis.ask("abre la calculadora")
    assert response.state is TaskState.COMPLETED
    verified = _events(jarvis, response.task_id, AuditEventType.ACTION_VERIFIED)[0]
    assert verified.data["evidence"] == {
        "windows": [
            {
                "handle": existing.handle,
                "title": "Calculadora",
                "process": "ApplicationFrameHost.exe",
            }
        ],
        "new_window": False,
    }


@pytest.mark.parametrize("app", ["powershell", "pwsh", "cmd", "regedit", "superapp"])
def test_b_f_unknown_or_forbidden_applications_never_launch(
    win_settings: Settings, desktop: FakeWindowsAutomation, app: str
) -> None:
    jarvis = _jarvis(win_settings, desktop, _open(app))
    response = jarvis.ask(f"abre {app}")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.NOT_FOUND
    assert desktop.launches == []
    failed = _events(jarvis, response.task_id, AuditEventType.ACTION_RESOLUTION_FAILED)[0]
    assert failed.data["requested"] == {"application": app}


@pytest.mark.parametrize(
    "app",
    [
        "cmd.exe /c del /q C:\\*",
        "powershell -enc AAAA",
        "notepad & calc",
        "notepad.exe; rm -rf /",
        "../../Windows/System32/cmd.exe",
        "..\\..\\cmd",
        "C:\\Windows\\System32\\cmd.exe",
        '"notepad"',
    ],
)
def test_c_d_command_lines_and_paths_are_rejected_before_planning_completes(
    win_settings: Settings, desktop: FakeWindowsAutomation, app: str
) -> None:
    jarvis = _jarvis(win_settings, desktop, _open(app))
    response = jarvis.ask("abre algo")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.PLANNING
    assert response.actions == ()  # the plan was rejected by schema validation
    assert desktop.launches == []


def test_f_model_invented_application_does_not_run(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    settings = win_settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake", "planner_fallback": False}
    )
    provider, _ = fake_model(
        plan_json({"tool": "windows.open_application", "arguments": {"application": "hackertool"}})
    )
    jarvis = _jarvis(settings, desktop, model_provider=provider)
    response = jarvis.ask("abre mi herramienta favorita")
    assert response.state is TaskState.FAILED
    assert response.planner == "model"
    assert desktop.launches == []


def test_model_planner_can_drive_windows_tools(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    settings = win_settings.model_copy(update={"model_provider": "litellm", "model": "openai/fake"})
    provider, completion = fake_model(
        plan_json({"tool": "windows.open_application", "arguments": {"application": "notepad"}})
    )
    jarvis = _jarvis(settings, desktop, model_provider=provider)
    response = jarvis.ask("necesito tomar notas")
    assert response.state is TaskState.COMPLETED
    assert desktop.launches == [NOTEPAD]
    assert "windows.open_application" in completion.calls[0]["messages"][0]["content"]


# -- focus ----------------------------------------------------------------------------------


def test_focus_and_verify(win_settings: Settings, desktop: FakeWindowsAutomation) -> None:
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    desktop.add_window("Otra cosa", "msedge.exe")  # currently in front
    jarvis = _jarvis(win_settings, desktop)
    response = jarvis.ask("enfoca la calculadora")
    assert response.state is TaskState.COMPLETED
    assert response.actions[0].status is ActionStatus.VERIFIED
    assert desktop.foreground == calc.handle


def test_g_ambiguous_windows_are_never_picked_arbitrarily(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    a = desktop.add_window("uno - Bloc de notas", "Notepad.exe")
    b = desktop.add_window("dos - Bloc de notas", "Notepad.exe")
    desktop.add_window("Otra cosa", "msedge.exe")  # in front: focusing is really needed
    jarvis = _jarvis(win_settings, desktop)
    for goal in ("enfoca el bloc de notas", "cierra el bloc de notas"):
        response = jarvis.ask(goal)
        assert response.state is TaskState.FAILED
        assert response.failure is not None and response.failure.kind is ErrorKind.AMBIGUOUS
        candidates = {c["handle"] for c in response.failure.details["candidates"]}
        assert candidates == {a.handle, b.handle}
    assert desktop.focus_calls == [] and desktop.close_calls == []

    # Refined criteria resolve to exactly one window.
    refined = _jarvis(
        win_settings,
        desktop,
        Action(id="a1", tool="windows.focus_window", arguments={"handle": a.handle}),
    )
    # "ese" names nothing: no requirement is invented, the handle-based action completes.
    assert refined.ask("enfoca ese").state is TaskState.COMPLETED
    assert desktop.foreground == a.handle


def test_h_focus_refused_by_the_system_fails_verification(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    other = desktop.add_window("Otra cosa", "msedge.exe")
    desktop.focus_works = False  # the call "succeeds" but Windows keeps another window in front
    jarvis = _jarvis(win_settings, desktop)
    response = jarvis.ask("enfoca la calculadora")
    assert response.state is TaskState.FAILED
    assert response.actions[0].verification is not None
    assert response.failure is not None and response.failure.kind is ErrorKind.VERIFICATION_FAILED
    assert desktop.foreground == other.handle


# -- close ----------------------------------------------------------------------------------


def test_e_close_requires_confirmation_showing_the_exact_window(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("cierra la calculadora")
    assert waiting.state is TaskState.WAITING_PERMISSION
    pending = waiting.pending_confirmations[0]
    assert pending.target is not None
    assert pending.target["window"] == {
        "handle": calc.handle,
        "title": "Calculadora",
        "process": "ApplicationFrameHost.exe",
    }
    assert desktop.close_calls == []

    done = jarvis.approve(waiting.task_id, "a1")
    assert done.state is TaskState.COMPLETED
    assert done.actions[0].status is ActionStatus.VERIFIED
    assert desktop.close_calls == [calc.handle]
    assert calc.handle not in desktop.windows


def test_close_confirmation_cannot_be_relaxed_by_grants(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    permissive = win_settings.model_copy(
        update={"permissions": {**WINDOWS_PERMISSIONS, "windows.close": "allow"}}
    )
    desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    response = _jarvis(permissive, desktop).ask("cierra la calculadora")
    assert response.state is TaskState.WAITING_PERMISSION  # category floor
    assert desktop.close_calls == []


def test_close_rejected_leaves_the_window(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("cierra la calculadora")
    assert jarvis.reject(waiting.task_id, "a1").state is TaskState.FAILED
    assert desktop.close_calls == [] and calc.handle in desktop.windows


def test_approval_is_bound_to_the_resolved_window(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    note = desktop.add_window("notas.txt - Bloc de notas", "Notepad.exe")
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("cierra el bloc de notas")
    assert waiting.pending_confirmations[0].target["unsaved_changes_suspected"] is False  # type: ignore[index]

    desktop.rename(note.handle, "*notas.txt - Bloc de notas")  # the user typed meanwhile
    again = jarvis.approve(waiting.task_id, "a1")
    # The target changed (now with unsaved changes): the old approval does not cover it.
    assert again.state is TaskState.WAITING_PERMISSION
    assert again.pending_confirmations[0].target["unsaved_changes_suspected"] is True  # type: ignore[index]
    assert desktop.close_calls == []


def test_close_only_targets_allowlisted_applications(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("Program Manager", "explorer.exe")
    jarvis = _jarvis(
        win_settings,
        desktop,
        Action(
            id="a1",
            tool="windows.close_application",
            arguments={"application": "explorer", "title_contains": "Program Manager"},
        ),
    )
    response = jarvis.ask("cierra el escritorio")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.NOT_FOUND
    missing_app = _jarvis(
        win_settings,
        desktop,
        Action(id="a1", tool="windows.close_application", arguments={"title_contains": "Program"}),
    )
    assert missing_app.ask("cierra eso").state is TaskState.FAILED  # application is required
    assert desktop.close_calls == []


def test_h_close_that_does_not_close_fails_verification_without_forcing(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    note = desktop.add_window("*notas.txt - Bloc de notas", "Notepad.exe")
    desktop.close_behavior[note.handle] = "prompt"  # "Save changes?" dialog appears
    settings = win_settings.model_copy(update={"windows_close_wait_seconds": 0.2})
    jarvis = _jarvis(settings, desktop)
    waiting = jarvis.ask("cierra el bloc de notas")
    assert waiting.pending_confirmations[0].target["unsaved_changes_suspected"] is True  # type: ignore[index]
    response = jarvis.approve(waiting.task_id, "a1")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.VERIFICATION_FAILED
    assert note.handle in desktop.windows  # still open: nothing was forced
    assert desktop.close_calls == [note.handle]  # asked once, never retried


def test_i_backend_failure_is_a_clean_tool_error(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    desktop.close_behavior[calc.handle] = "error"
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("cierra la calculadora")
    response = jarvis.approve(waiting.task_id, "a1")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.TOOL_ERROR
    assert "UI Automation call failed" in response.failure.message
    assert _events(jarvis, response.task_id, AuditEventType.ACTION_FAILED)
    assert desktop.close_calls == [calc.handle]


def test_window_gone_between_resolution_and_execution(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("cierra la calculadora")
    del desktop.windows[calc.handle]  # the user closed it meanwhile
    response = jarvis.approve(waiting.task_id, "a1")
    # The approved window is gone: nothing is closed. Re-observing shows the goal (no
    # calculator window) already holds, so the task completes on that evidence.
    assert response.state is TaskState.COMPLETED
    assert response.actions[0].status is ActionStatus.FAILED
    assert response.actions[0].failure is not None
    assert response.actions[0].failure.kind is ErrorKind.NOT_FOUND
    assert desktop.close_calls == []


# -- open failures --------------------------------------------------------------------------


def test_h_launch_reports_success_but_no_window_appears(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.on_launch["notepad.exe"] = lambda: None  # the launcher "worked", nothing shows up
    settings = win_settings.model_copy(update={"windows_open_wait_seconds": 0.2})
    jarvis = _jarvis(settings, desktop)
    response = jarvis.ask("abre el bloc de notas")
    assert response.state is TaskState.FAILED
    assert response.actions[0].status is ActionStatus.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.VERIFICATION_FAILED
    assert desktop.launches == [NOTEPAD]  # launched once, not retried


# -- listing, chaining, timeouts ------------------------------------------------------------


def test_get_windows(win_settings: Settings, desktop: FakeWindowsAutomation) -> None:
    desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    front = desktop.add_window("notas - Bloc de notas", "Notepad.exe")
    response = _jarvis(win_settings, desktop).ask("lista las ventanas")
    assert response.state is TaskState.COMPLETED
    listed = response.actions[0].output
    assert listed is not None and listed["total"] == 2
    assert [w["foreground"] for w in listed["windows"]] == [False, True]
    assert listed["windows"][1]["handle"] == front.handle


def test_open_focus_close_chain(win_settings: Settings, desktop: FakeWindowsAutomation) -> None:
    jarvis = _jarvis(win_settings, desktop)
    waiting = jarvis.ask("abre la calculadora y luego enfócala y luego ciérrala")
    assert waiting.state is TaskState.WAITING_PERMISSION
    # The new window is already in front after opening: no focus action is needed.
    assert [(a.tool, a.status) for a in waiting.actions] == [
        ("windows.open_application", ActionStatus.VERIFIED),
        ("windows.close_application", ActionStatus.AWAITING_CONFIRMATION),
    ]
    done = jarvis.approve(waiting.task_id, waiting.actions[1].action_id)
    assert done.state is TaskState.COMPLETED
    assert desktop.windows == {}


@pytest.fixture
def hang(desktop: FakeWindowsAutomation) -> Iterator[threading.Event]:
    event = threading.Event()
    desktop.hang = event
    yield event
    event.set()


def test_hung_close_times_out_and_is_never_retried(
    win_settings: Settings, desktop: FakeWindowsAutomation, hang: threading.Event
) -> None:
    desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    settings = win_settings.model_copy(
        update={"tool_timeout_seconds": 0.3, "max_attempts_per_action": 5}
    )
    jarvis = _jarvis(settings, desktop)
    waiting = jarvis.ask("cierra la calculadora")
    response = jarvis.approve(waiting.task_id, "a1")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.TIMEOUT
    assert response.failure.details["outcome_unknown"] is True
    assert len(desktop.close_calls) == 1


def test_hung_focus_is_retried_within_limits(
    win_settings: Settings, desktop: FakeWindowsAutomation, hang: threading.Event
) -> None:
    desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    desktop.add_window("Otra cosa", "msedge.exe")  # in front: focusing is really needed
    settings = win_settings.model_copy(
        update={"tool_timeout_seconds": 0.2, "max_attempts_per_action": 2}
    )
    response = _jarvis(settings, desktop).ask("enfoca la calculadora")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.TIMEOUT
    assert len(desktop.focus_calls) == 2  # idempotent: bounded retry


# -- allowlist configuration ----------------------------------------------------------------


@pytest.mark.parametrize(
    "executable",
    [
        r"C:\Windows\System32\cmd.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\Windows\System32\mshta.exe",
        r"relative\app.exe",
        r"C:\tools\script.bat",
        r"C:\tools\app.exe & calc.exe",
        r"%UNDEFINED_JARVIS_VAR%\app.exe",
    ],
)
def test_user_allowlist_rejects_shells_scripts_and_odd_paths(executable: str) -> None:
    with pytest.raises(ValidationError):
        ApplicationConfig(executable=executable)


def test_user_allowlist_adds_applications() -> None:
    catalog = build_catalog(
        {
            "teams": ApplicationConfig(
                executable=r"C:\Program Files\Teams\ms-teams.exe",
                process_names=["ms-teams.exe"],
                aliases=["microsoft teams"],
            )
        },
        system_root=r"C:\Windows",
    )
    entry = catalog.resolve("Microsoft Teams")
    assert entry.app_id == "teams"
    assert entry.executable == r"C:\Program Files\Teams\ms-teams.exe"
    assert catalog.resolve("notepad").app_id == "notepad"


def test_catalog_refuses_forbidden_entries_even_when_built_directly() -> None:
    from jarvis.desktop.catalog import ApplicationCatalog

    with pytest.raises(ValueError):
        ApplicationCatalog(
            [
                ApplicationEntry(
                    app_id="shell",
                    display_name="shell",
                    executable=r"C:\Windows\System32\cmd.exe",
                    process_names=frozenset({"cmd.exe"}),
                )
            ]
        )


def test_windows_tools_absent_without_backend(settings: Settings) -> None:
    jarvis = build_jarvis(settings, store=InMemoryStateStore(), audit=InMemoryAuditLog())
    assert not any(s.name.startswith("windows.") for s in jarvis.tools())


def test_confirmation_message_names_the_window(
    win_settings: Settings, desktop: FakeWindowsAutomation
) -> None:
    desktop.add_window("*informe.txt - Bloc de notas", "Notepad.exe")
    waiting = _jarvis(win_settings, desktop).ask("cierra el bloc de notas")
    assert "objetivo: ventana '*informe.txt - Bloc de notas' (Notepad.exe)" in waiting.message
    assert "posibles cambios sin guardar" in waiting.message


def test_tools_act_only_on_the_resolved_window(tmp_path: Any) -> None:
    """Defence in depth inside ``run``: the resolved handle must still be the same window
    (same process) at the moment of acting, e.g. not a reused handle of another program."""
    from jarvis.core.errors import ToolError
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.windows import CloseApplicationTool, FocusWindowTool
    from jarvis.tools.workspace import Workspace

    desktop = FakeWindowsAutomation()
    calc = desktop.add_window("Calculadora", "ApplicationFrameHost.exe")
    catalog = build_catalog(system_root=r"C:\Windows")
    target = {"window": {"handle": calc.handle, "title": "Calculadora", "process": "notepad.exe"}}
    ctx = ToolContext(Workspace(tmp_path), "t", "a1", 1, target=target)  # process differs
    close = CloseApplicationTool(desktop, catalog, wait_seconds=0)
    focus = FocusWindowTool(desktop, catalog, wait_seconds=0)
    for tool, args in (
        (close, close.validate_arguments({"application": "calculator"})),
        (focus, focus.validate_arguments({"application": "calculator"})),
    ):
        with pytest.raises(ToolError) as excinfo:
            tool.run(args, ctx)  # type: ignore[arg-type]
        assert excinfo.value.failure.kind is ErrorKind.NOT_FOUND
    assert desktop.close_calls == [] and desktop.focus_calls == []
