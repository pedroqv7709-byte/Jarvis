from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.context.model import ContextItem
from jarvis.core.action import Action
from jarvis.core.errors import ErrorKind
from jarvis.core.request import Request
from jarvis.core.task import TaskState
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.orchestration.orchestrator import OrchestrationError
from jarvis.planning.base import Plan, PlanningRequest
from jarvis.state.store import InMemoryStateStore

from support import CountingTool, StaticPlanner


def _types(jarvis: Jarvis, task_id: str) -> list[AuditEventType]:
    return [e.type for e in jarvis.audit_trail(task_id)]


def test_full_flow_updates_state_and_audit(memory_jarvis: Jarvis) -> None:
    response = memory_jarvis.ask("crea un archivo de prueba")
    assert response.state is TaskState.COMPLETED
    task = memory_jarvis.task(response.task_id)
    assert task.planner == "rule_based"
    assert [t.to_state for t in task.transitions] == [TaskState.RUNNING, TaskState.COMPLETED]
    assert _types(memory_jarvis, task.id) == [
        AuditEventType.REQUEST_RECEIVED,
        AuditEventType.CONTEXT_BUILT,
        AuditEventType.AGENT_REQUIREMENTS,
        AuditEventType.TASK_STATE_CHANGED,
        AuditEventType.AGENT_PLAN,  # decision: next action
        AuditEventType.AGENT_ACTION_SELECTED,
        AuditEventType.ACTION_SELECTED,
        AuditEventType.POLICY_EVALUATED,
        AuditEventType.ACTION_ATTEMPT_STARTED,
        AuditEventType.ACTION_EXECUTED,
        AuditEventType.ACTION_VERIFIED,
        AuditEventType.AGENT_VERIFICATION,
        AuditEventType.AGENT_PLAN,  # decision: done
        AuditEventType.AGENT_COMPLETED,  # accepted on the evidence of the verified action
        AuditEventType.TASK_STATE_CHANGED,
        AuditEventType.RESPONSE_SENT,
    ]


def test_request_guard_rejects_before_planning(memory_jarvis: Jarvis) -> None:
    response = memory_jarvis.ask("   ")
    assert response.state is TaskState.FAILED
    assert response.failure is not None
    assert response.failure.kind is ErrorKind.REQUEST_REJECTED
    types = _types(memory_jarvis, response.task_id)
    assert AuditEventType.REQUEST_REJECTED in types
    assert AuditEventType.PLAN_CREATED not in types


def test_planning_failure_is_propagated(memory_jarvis: Jarvis) -> None:
    response = memory_jarvis.ask("abre la puerta del garaje")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.PLANNING
    assert AuditEventType.PLAN_FAILED in _types(memory_jarvis, response.task_id)


def test_tool_failure_is_propagated_to_response(memory_jarvis: Jarvis) -> None:
    response = memory_jarvis.ask("lee el archivo no-existe.txt")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.TOOL_ERROR
    assert "no-existe.txt" in response.message


def test_reply_only_plan_completes_without_actions(settings: Settings) -> None:
    jarvis = build_jarvis(
        settings,
        planner=StaticPlanner(reply="Hola"),
        store=InMemoryStateStore(),
        audit=InMemoryAuditLog(),
    )
    response = jarvis.ask("hola")
    assert response.state is TaskState.COMPLETED
    assert response.message == "Hola"
    assert response.actions == ()


class _ExplodingPlanner:
    name = "exploding"

    def plan(self, request: PlanningRequest) -> Plan:
        raise RuntimeError("planner bug")


def test_unexpected_errors_never_leave_task_stuck(settings: Settings) -> None:
    store = InMemoryStateStore()
    jarvis = build_jarvis(
        settings, planner=_ExplodingPlanner(), store=store, audit=InMemoryAuditLog()
    )
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.INTERNAL
    assert store.get(response.task_id).state is TaskState.FAILED


class _FailingProvider:
    name = "broken"

    def provide(self, request: Request) -> Iterable[ContextItem]:
        raise OSError("sensor offline")


def test_failing_context_provider_degrades_gracefully(settings: Settings) -> None:
    audit = InMemoryAuditLog()
    jarvis = build_jarvis(
        settings,
        store=InMemoryStateStore(),
        audit=audit,
        context_providers=[_FailingProvider()],
    )
    response = jarvis.ask("lista los archivos")
    assert response.state is TaskState.COMPLETED
    built = next(
        e for e in audit.events(response.task_id) if e.type is AuditEventType.CONTEXT_BUILT
    )
    assert built.data["degraded_sources"] == ["broken"]


def test_commands_on_tasks_in_wrong_state_are_refused(memory_jarvis: Jarvis) -> None:
    done = memory_jarvis.ask("lista los archivos")
    with pytest.raises(OrchestrationError):
        memory_jarvis.approve(done.task_id, "a1")
    with pytest.raises(OrchestrationError):
        memory_jarvis.cancel(done.task_id)


def test_approving_unknown_action_is_refused(settings: Settings) -> None:
    jarvis = build_jarvis(settings, store=InMemoryStateStore(), audit=InMemoryAuditLog())
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    (settings.workspace_dir / "a.txt").write_text("x", encoding="utf-8")
    waiting = jarvis.ask("borra el archivo a.txt")
    with pytest.raises(OrchestrationError):
        jarvis.approve(waiting.task_id, "a99")


def test_audit_never_contains_sensitive_argument_values(settings: Settings) -> None:
    from jarvis.core.categories import ActionCategory
    from jarvis.tools.base import Tool, ToolArguments, ToolContext, ToolSpec

    class LoginArgs(ToolArguments):
        user: str
        password: str

    class LoginTool(Tool[LoginArgs]):
        spec = ToolSpec(
            name="test.login",
            description="x",
            required_permissions=frozenset({"test.echo"}),
            categories=frozenset({ActionCategory.READ}),
            side_effects=False,
            idempotent=True,
        )
        args_model = LoginArgs

        def run(self, args: LoginArgs, ctx: ToolContext) -> dict[str, Any]:
            return {"token": "tok-123"}

    audit = InMemoryAuditLog()
    action = Action(id="a1", tool="test.login", arguments={"user": "ana", "password": "hunter2"})
    jarvis = build_jarvis(
        settings,
        planner=StaticPlanner(action),
        tools=[LoginTool(), CountingTool()],
        store=InMemoryStateStore(),
        audit=audit,
    )
    response = jarvis.ask("login")
    dumped = "".join(e.model_dump_json() for e in audit.events(response.task_id))
    assert "hunter2" not in dumped
    assert "tok-123" not in dumped
    assert "ana" in dumped
