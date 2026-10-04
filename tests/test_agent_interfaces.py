"""API, CLI and audit for the agent loop: STOPPED, persisted progress, reconstructable flow."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from conftest import API_TOKEN
from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.categories import ActionCategory
from jarvis.interface.cli import main
from jarvis.observability.audit import AuditEventType as T
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool, ToolArguments, ToolContext, ToolSpec
from jarvis.tools.builtin.filesystem import filesystem_tools

from support import FakeWindowsAutomation, ScriptedPlanner, act, done, wait_for

MakeApi = Callable[..., tuple[TestClient, JarvisRuntime]]


def _limited(settings: Settings) -> Settings:
    return settings.model_copy(update={"agent_max_iterations": 1})


def _two_listings() -> ScriptedPlanner:
    return ScriptedPlanner(act("fs.list_dir", path="."), act("fs.list_dir", path="./"))


# -- API -------------------------------------------------------------------------------------


def test_api_reports_stopped_and_progress(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api(_limited(settings), planner=_two_listings())
    task_id = client.post("/v1/tasks", json={"text": "explora"}).json()["task_id"]
    view = wait_for(client, task_id)
    assert view["status"] == "STOPPED"
    assert view["error"]["kind"] == "limit_exceeded"
    progress = view["progress"]
    assert progress["phase"] == "finished"
    assert (progress["iterations"], progress["actions_used"], progress["replans_used"]) == (1, 1, 0)
    assert progress["elapsed_seconds"] >= 0
    assert "iteration limit" in progress["stop_reason"]

    listed = client.get("/v1/tasks", params={"status": "STOPPED"}).json()
    assert [t["task_id"] for t in listed] == [task_id]
    events = [e["type"] for e in client.get(f"/v1/tasks/{task_id}/events").json()]
    assert "agent.limit_reached" in events and "agent.stopped" in events

    raw = json.dumps(
        [
            client.get(f"/v1/tasks/{task_id}").json(),
            client.get(f"/v1/tasks/{task_id}/events").json(),
        ]
    )
    assert API_TOKEN not in raw  # the bearer token never appears in responses or events


def test_api_shows_the_current_action_while_waiting(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    client.post("/v1/tasks", json={"text": "crea el archivo x.txt"})
    task_id = client.post("/v1/tasks", json={"text": "borra el archivo x.txt"}).json()["task_id"]
    view = wait_for(client, task_id)
    assert view["status"] == "WAITING_PERMISSION"
    assert view["progress"]["phase"] == "waiting"
    assert view["progress"]["current_action"] == "a1: fs.delete_file"
    client.post(f"/v1/tasks/{task_id}/reject", json={"action_id": "a1"})
    rejected = wait_for(client, task_id, until=("FAILED", "COMPLETED"))
    assert rejected["status"] == "FAILED"
    assert rejected["progress"]["current_action"] is None


def test_api_endpoints_still_require_the_token(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api(_limited(settings), planner=_two_listings())
    task_id = client.post("/v1/tasks", json={"text": "explora"}).json()["task_id"]
    wait_for(client, task_id)
    for method, path, body in [
        ("GET", f"/v1/tasks/{task_id}", None),
        ("GET", "/v1/tasks", None),
        ("GET", f"/v1/tasks/{task_id}/events", None),
        ("POST", f"/v1/tasks/{task_id}/cancel", {}),
    ]:
        assert (
            client.request(method, path, json=body, headers={"Authorization": ""}).status_code
            == 401
        )
        bad = {"Authorization": "Bearer " + "x" * 43}
        assert client.request(method, path, json=body, headers=bad).status_code == 403
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200


# -- CLI -------------------------------------------------------------------------------------


def _cli(settings: Settings, planner: Any, *argv: str) -> tuple[int, str]:
    out = io.StringIO()

    def factory(handler: Any) -> Any:
        return build_jarvis(settings, confirmations=handler, planner=planner)

    code = main(list(argv), stdin=io.StringIO(""), stdout=out, factory=factory)
    return code, out.getvalue()


def test_cli_shows_stopped_reason_and_progress(settings: Settings) -> None:
    code, out = _cli(_limited(settings), _two_listings(), "explora")
    assert code == 1
    assert "Tarea detenida de forma segura" in out
    assert "Progreso: fase finished | iteraciones 1 | acciones 1 | replanificaciones 0" in out
    assert "Motivo de parada: iteration limit reached (1)" in out
    assert ": stopped)" in out


def test_cli_shows_the_current_action_while_waiting(settings: Settings) -> None:
    _cli(settings, None, "crea el archivo y.txt")
    code, out = _cli(settings, None, "--no-input", "borra el archivo y.txt")
    assert code == 0
    assert "Acción actual: a1: fs.delete_file" in out
    assert "fase waiting" in out


def test_cli_json_includes_progress(settings: Settings) -> None:
    code, out = _cli(settings, None, "--json", "run", "lista los archivos")
    assert code == 0
    payload = json.loads(out)
    assert payload["status"] == "COMPLETED"
    assert payload["progress"]["phase"] == "finished"
    assert payload["progress"]["iterations"] == 2  # decide the listing, then decide done


# -- audit -----------------------------------------------------------------------------------


class SecretArgs(ToolArguments):
    user: str
    password: str


class Login(Tool[SecretArgs]):
    spec = ToolSpec(
        name="test.login",
        description="Log in.",
        required_permissions=frozenset({"test.echo"}),
        categories=frozenset({ActionCategory.READ}),
        side_effects=False,
        idempotent=True,
    )
    args_model = SecretArgs

    def run(self, args: SecretArgs, ctx: ToolContext) -> dict[str, Any]:
        return {"token": "tok-SECRET"}


def test_audit_reconstructs_the_loop_without_secrets_or_reasoning(settings: Settings) -> None:
    desktop = FakeWindowsAutomation()
    audit = InMemoryAuditLog()
    planner = ScriptedPlanner(
        act("fs.read_file", path="missing.txt"),
        act("test.login", user="ana", password="hunter2"),
        done(),
    )
    jarvis = build_jarvis(
        settings,
        windows=desktop,
        tools=[*filesystem_tools(), Login()],
        planner=planner,
        store=InMemoryStateStore(),
        audit=audit,
    )
    response = jarvis.ask("entra")
    events = audit.events(response.task_id)
    sequence = [e.type for e in events]
    expected = [
        T.REQUEST_RECEIVED,
        T.AGENT_OBSERVATION,
        T.AGENT_PLAN,
        T.AGENT_ACTION_SELECTED,
        T.POLICY_EVALUATED,
        T.ACTION_FAILED,
        T.AGENT_VERIFICATION,
        T.AGENT_REPLAN,
        T.AGENT_OBSERVATION,
        T.AGENT_PLAN,
        T.AGENT_ACTION_SELECTED,
        T.POLICY_EVALUATED,
        T.ACTION_EXECUTED,
        T.AGENT_VERIFICATION,
        T.AGENT_OBSERVATION,
        T.AGENT_PLAN,
        T.AGENT_COMPLETED,
        T.RESPONSE_SENT,
    ]
    remaining = iter(sequence)
    assert all(any(item is wanted for item in remaining) for wanted in expected), sequence

    dumped = "".join(e.model_dump_json() for e in events)
    assert "hunter2" not in dumped and "tok-SECRET" not in dumped
    assert "ana" in dumped  # non-secret arguments are kept for the audit
    for event in events:
        assert not {"reasoning", "thoughts", "chain_of_thought"} & set(event.data)
        rationale = event.data.get("rationale")
        assert rationale is None or len(rationale) <= 300


def test_a_pending_confirmation_is_not_reported_as_a_verification(settings: Settings) -> None:
    audit = InMemoryAuditLog()
    jarvis = build_jarvis(settings, store=InMemoryStateStore(), audit=audit)
    jarvis.ask("crea el archivo z.txt")
    waiting = jarvis.ask("borra el archivo z.txt")
    assert T.AGENT_VERIFICATION not in [e.type for e in audit.events(waiting.task_id)]
    jarvis.approve(waiting.task_id, "a1")
    verifications = [e for e in audit.events(waiting.task_id) if e.type is T.AGENT_VERIFICATION]
    assert [v.data["verification"] for v in verifications] == ["verified"]
