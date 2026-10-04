"""Local HTTP API over the persistent runtime (TestClient: real runtime, no sockets)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from jarvis.config import Settings
from jarvis.core.action import Action
from jarvis.core.task import Task
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.state.sqlite import SqliteStateStore
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.builtin.filesystem import filesystem_tools

from support import CountingTool, GoalPlanner, HangingTool, fake_model, plan_json, wait_for

MakeApi = Callable[..., tuple[TestClient, JarvisRuntime]]


def _post(client: TestClient, text: str) -> str:
    response = client.post("/v1/tasks", json={"text": text})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "PENDING"
    return str(body["task_id"])


def _run(client: TestClient, text: str, **kw: Any) -> dict[str, Any]:
    return wait_for(client, _post(client, text), **kw)


def _types(client: TestClient, task_id: str) -> list[str]:
    return [e["type"] for e in client.get(f"/v1/tasks/{task_id}/events").json()]


# -- health ---------------------------------------------------------------------------------


def test_health(make_api: MakeApi) -> None:
    client, runtime = make_api()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runtime"] == "ready"
    assert body["state_store"] == "ok"
    assert body["worker"] == "running"
    assert body["planner"] == "rule_based"
    assert runtime.worker.alive


def test_health_reports_a_stopped_runtime(make_api: MakeApi) -> None:
    client, runtime = make_api()
    runtime.stop()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["runtime"] == "stopped"
    assert client.post("/v1/tasks", json={"text": "lista los archivos"}).status_code == 503


# -- task lifecycle -------------------------------------------------------------------------


def test_create_and_complete(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    view = _run(client, 'crea el archivo hola.txt con el contenido "hola" y luego léelo')
    assert view["status"] == "COMPLETED"
    assert [a["status"] for a in view["actions"]] == ["verified", "executed"]
    assert view["actions"][1]["output"]["content"] == "hola"
    assert view["planner"] == "rule_based"
    assert (settings.workspace_dir / "hola.txt").read_text(encoding="utf-8") == "hola"
    listed = client.get("/v1/tasks").json()
    assert listed[0]["task_id"] == view["task_id"]
    assert listed[0]["status"] == "COMPLETED"


def test_failure_is_reported_and_audited(make_api: MakeApi) -> None:
    client, _ = make_api()
    view = _run(client, "lee el archivo no-existe.txt")
    assert view["status"] == "FAILED"
    assert view["error"]["kind"] == "tool_error"
    assert "action.failed" in _types(client, view["task_id"])
    planning = _run(client, "abre la puerta del garaje")
    assert planning["status"] == "FAILED"
    assert planning["error"]["kind"] == "planning"


def test_unknown_task_and_invalid_input(make_api: MakeApi) -> None:
    client, _ = make_api()
    assert client.get("/v1/tasks/task_nope").status_code == 404
    assert client.get("/v1/tasks/task_nope/events").status_code == 404
    assert client.post("/v1/tasks", json={"text": ""}).status_code == 422
    assert client.post("/v1/tasks", json={"text": "x", "sudo": True}).status_code == 422


# -- confirmations --------------------------------------------------------------------------


def _waiting_delete(client: TestClient, name: str = "x.txt") -> dict[str, Any]:
    assert _run(client, f"crea el archivo {name}")["status"] == "COMPLETED"
    view = _run(client, f"borra el archivo {name}")
    assert view["status"] == "WAITING_PERMISSION"
    assert view["pending_confirmations"][0]["action_id"] == "a1"
    assert view["pending_confirmations"][0]["description"] == "Delete a file from the workspace."
    return view


def test_approve_from_another_client(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    view = _waiting_delete(client)
    assert (settings.workspace_dir / "x.txt").exists()
    response = client.post(f"/v1/tasks/{view['task_id']}/approve", json={"action_id": "a1"})
    assert response.status_code == 202
    done = wait_for(client, view["task_id"], until=("COMPLETED", "FAILED"))
    assert done["status"] == "COMPLETED"
    assert done["actions"][0]["status"] == "verified"
    assert not (settings.workspace_dir / "x.txt").exists()


def test_reject(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    view = _waiting_delete(client)
    client.post(f"/v1/tasks/{view['task_id']}/reject", json={"action_id": "a1"})
    done = wait_for(client, view["task_id"], until=("COMPLETED", "FAILED"))
    assert done["status"] == "FAILED"
    assert done["error"]["kind"] == "confirmation_rejected"
    assert (settings.workspace_dir / "x.txt").exists()


def test_approve_is_refused_when_nothing_awaits(make_api: MakeApi) -> None:
    client, _ = make_api()
    done = _run(client, "lista los archivos")
    assert (
        client.post(f"/v1/tasks/{done['task_id']}/approve", json={"action_id": "a1"}).status_code
        == 409
    )
    waiting = _waiting_delete(client)
    unknown = client.post(f"/v1/tasks/{waiting['task_id']}/approve", json={"action_id": "a9"})
    assert unknown.status_code == 409
    assert client.post("/v1/tasks/task_nope/approve", json={"action_id": "a1"}).status_code == 404


def test_deny_stays_deny(make_api: MakeApi, settings: Settings) -> None:
    denied = settings.model_copy(update={"permissions": {"fs.read": "allow", "fs.write": "deny"}})
    client, _ = make_api(denied)
    view = _run(client, "crea el archivo prohibido.txt")
    assert view["status"] == "FAILED"
    assert view["actions"][0]["status"] == "denied"
    assert view["actions"][0]["decision"] == "deny"
    response = client.post(f"/v1/tasks/{view['task_id']}/approve", json={"action_id": "a1"})
    assert response.status_code == 409
    assert not (settings.workspace_dir / "prohibido.txt").exists()


def test_stale_approval_does_not_cover_different_arguments(
    make_api: MakeApi, settings: Settings
) -> None:
    client, _ = make_api()
    assert _run(client, "crea el archivo otro.txt")["status"] == "COMPLETED"
    view = _waiting_delete(client)
    # The planned arguments change after the confirmation was requested.
    store = SqliteStateStore(settings.state_db_path)
    task = store.get(view["task_id"])
    record = task.actions[0]
    record.action = record.action.model_copy(update={"arguments": {"path": "otro.txt"}})
    store.save(task)

    client.post(f"/v1/tasks/{view['task_id']}/approve", json={"action_id": "a1"})
    # Wait until the worker has resumed and issued a *new* confirmation request.
    for _ in range(250):
        if _types(client, view["task_id"]).count("confirmation.requested") == 2:
            break
        threading.Event().wait(0.02)
    after = wait_for(client, view["task_id"], until=("WAITING_PERMISSION", "COMPLETED", "FAILED"))
    assert after["status"] == "WAITING_PERMISSION"
    assert after["pending_confirmations"][0]["arguments"] == {"path": "otro.txt"}
    assert (settings.workspace_dir / "otro.txt").exists()
    assert (settings.workspace_dir / "x.txt").exists()


def test_policy_is_reevaluated_when_another_runtime_resumes(
    make_api: MakeApi, settings: Settings
) -> None:
    client, runtime = make_api()
    view = _waiting_delete(client)
    runtime.stop()

    revoked = settings.model_copy(
        update={"permissions": {"fs.read": "allow", "fs.write": "allow", "fs.delete": "deny"}}
    )
    client2, _ = make_api(revoked)
    client2.post(f"/v1/tasks/{view['task_id']}/approve", json={"action_id": "a1"})
    done = wait_for(client2, view["task_id"], until=("COMPLETED", "FAILED"))
    assert done["status"] == "FAILED"
    assert done["actions"][0]["status"] == "denied"
    assert (settings.workspace_dir / "x.txt").exists()


def test_concurrent_approve_and_cancel_end_consistently(
    make_api: MakeApi, settings: Settings
) -> None:
    client, _ = make_api()
    view = _waiting_delete(client)
    task_id = view["task_id"]
    barrier = threading.Barrier(2)
    codes: dict[str, int] = {}

    def approve() -> None:
        barrier.wait()
        codes["approve"] = client.post(
            f"/v1/tasks/{task_id}/approve", json={"action_id": "a1"}
        ).status_code

    def cancel() -> None:
        barrier.wait()
        codes["cancel"] = client.post(f"/v1/tasks/{task_id}/cancel", json={}).status_code

    threads = [threading.Thread(target=approve), threading.Thread(target=cancel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = wait_for(client, task_id, until=("COMPLETED", "CANCELLED"))
    if final["status"] == "CANCELLED" and final["error"]["details"].get("cancelled_while"):
        # Cancelled mid-run: wait for the runner to record what actually happened.
        for _ in range(250):
            if "task.cancel_observed" in _types(client, task_id):
                break
            threading.Event().wait(0.02)
        final = client.get(f"/v1/tasks/{task_id}").json()
    deleted = not (settings.workspace_dir / "x.txt").exists()
    executed = final["actions"][0]["status"] in ("verified", "executed")
    # Any interleaving is acceptable; a contradiction between the effect and the record is not.
    assert deleted == executed, (final, codes)
    assert final["status"] in ("COMPLETED", "CANCELLED")


# -- cancellation ---------------------------------------------------------------------------


def _blocking_api(make_api: MakeApi, release: threading.Event) -> tuple[TestClient, HangingTool]:
    tool = HangingTool(release)
    planner = GoalPlanner(
        {
            "bloquea": (Action(id="a1", tool="test.hang", arguments={"text": "x"}),),
            "bloquea y escribe": (
                Action(id="a1", tool="test.hang", arguments={"text": "x"}),
                Action(
                    id="a2",
                    tool="fs.write_file",
                    arguments={"path": "despues.txt", "content": "x"},
                    depends_on=("a1",),
                ),
            ),
        }
    )
    client, _ = make_api(planner=planner, tools=[tool, *filesystem_tools()])
    return client, tool


def test_cancel_pending_task(
    make_api: MakeApi, release: threading.Event, settings: Settings
) -> None:
    client, tool = _blocking_api(make_api, release)
    busy = _post(client, "bloquea")
    assert tool.started.wait(5)
    queued = _post(client, "crea un archivo de prueba")
    assert client.get(f"/v1/tasks/{queued}").json()["status"] == "PENDING"

    response = client.post(f"/v1/tasks/{queued}/cancel", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"
    release.set()
    assert wait_for(client, busy, until=("COMPLETED", "FAILED"))["status"] == "COMPLETED"
    assert client.get(f"/v1/tasks/{queued}").json()["status"] == "CANCELLED"
    assert not (settings.workspace_dir / "prueba.txt").exists()


def test_finished_task_cannot_be_cancelled(make_api: MakeApi) -> None:
    client, _ = make_api()
    done = _run(client, "lista los archivos")
    response = client.post(f"/v1/tasks/{done['task_id']}/cancel", json={})
    assert response.status_code == 409
    assert client.get(f"/v1/tasks/{done['task_id']}").json()["status"] == "COMPLETED"


def test_cancel_running_task_records_what_really_happened(
    make_api: MakeApi, release: threading.Event, settings: Settings
) -> None:
    client, tool = _blocking_api(make_api, release)
    task_id = _post(client, "bloquea y escribe")
    assert tool.started.wait(5)
    assert client.get(f"/v1/tasks/{task_id}").json()["status"] == "RUNNING"

    response = client.post(f"/v1/tasks/{task_id}/cancel", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "CANCELLED"
    assert body["error"]["details"]["cancelled_while"] == "running"

    release.set()  # the operation in progress finishes; it is not rolled back
    for _ in range(200):
        if "task.cancel_observed" in _types(client, task_id):
            break
        threading.Event().wait(0.02)
    final = client.get(f"/v1/tasks/{task_id}").json()
    assert final["status"] == "CANCELLED"
    assert final["actions"][0]["status"] == "executed"  # it did run
    assert final["actions"][1]["status"] == "skipped"
    assert not (settings.workspace_dir / "despues.txt").exists()
    assert tool.calls == ["x"]


# -- concurrency ----------------------------------------------------------------------------


def test_concurrent_requests_do_not_corrupt_state(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    ids: list[str] = []
    lock = threading.Lock()

    def submit(i: int) -> None:
        task_id = _post(client, f'crea el archivo f{i}.txt con el contenido "{i}"')
        with lock:
            ids.append(task_id)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 8
    store = SqliteStateStore(settings.state_db_path)
    for task_id in ids:
        assert wait_for(client, task_id)["status"] == "COMPLETED"
        transitions = [
            (t.from_state.value, t.to_state.value) for t in store.get(task_id).transitions
        ]
        assert transitions == [("pending", "running"), ("running", "completed")]
    for i in range(8):
        assert (settings.workspace_dir / f"f{i}.txt").read_text(encoding="utf-8") == str(i)


# -- local only -----------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::"])
def test_server_host_must_be_loopback(host: str, settings: Settings) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, server_host=host)  # type: ignore[call-arg]


def test_loopback_hosts_are_accepted() -> None:
    for host in ("127.0.0.1", "::1", "localhost"):
        assert Settings(_env_file=None, server_host=host).server_host == host  # type: ignore[call-arg]
    assert Settings(_env_file=None, server_host="::1").server_url == "http://[::1]:8765"  # type: ignore[call-arg]


def test_browser_style_requests_are_refused(make_api: MakeApi) -> None:
    client, _ = make_api()
    # DNS rebinding: the browser would send the attacker's host name.
    assert client.get("/health", headers={"host": "evil.example"}).status_code == 400
    # Cross-site request from a web page.
    cross = client.post(
        "/v1/tasks", json={"text": "lista los archivos"}, headers={"origin": "https://evil.example"}
    )
    assert cross.status_code == 403
    # "Simple" (no-preflight) form post.
    form = client.post("/v1/tasks", content="text=borra", headers={"content-type": "text/plain"})
    assert form.status_code == 415
    local = client.post(
        "/v1/tasks",
        json={"text": "lista los archivos"},
        headers={"origin": "http://localhost:3000"},
    )
    assert local.status_code == 202


def test_tools_endpoint(make_api: MakeApi) -> None:
    client, _ = make_api(tools=[CountingTool()])
    assert client.get("/v1/tools").json() == [
        {
            "name": "test.echo",
            "description": "Echo text back.",
            "permissions": ["test.echo"],
            "categories": ["read"],
        }
    ]


# -- end to end -----------------------------------------------------------------------------


def test_end_to_end_over_http(make_api: MakeApi, settings: Settings) -> None:
    """HTTP POST -> FastAPI -> Orchestrator -> Planner (model via LiteLLM adapter, external call
    replaced) -> Policy -> filesystem tool in sandbox -> Verification -> SQLite -> Audit -> GET."""
    model_settings = settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake-model"}
    )
    provider, _ = fake_model(
        plan_json(
            {
                "tool": "fs.write_file",
                "arguments": {"path": "informe.md", "content": "# Hecho\n"},
                "description": "Crear informe",
            }
        )
    )
    client, _ = make_api(model_settings, model_provider=provider)
    assert client.get("/health").json()["planner"] == "model -> rule_based"

    task_id = _post(client, "prepárame un informe")
    view = wait_for(client, task_id)
    assert view["status"] == "COMPLETED"
    assert view["planner"] == "model"
    (action,) = view["actions"]
    assert (action["tool"], action["decision"], action["verification"]) == (
        "fs.write_file",
        "allow",
        "verified",
    )
    assert (settings.workspace_dir / "informe.md").read_text(encoding="utf-8") == "# Hecho\n"

    stored = SqliteStateStore(settings.state_db_path).get(task_id)  # single source of truth
    assert stored.state.value == "completed"
    assert stored.actions[0].verification is not None

    types = _types(client, task_id)
    for expected in (
        "request.received",
        "agent.plan",
        "policy.evaluated",
        "action.executed",
        "action.verified",
        "task.state_changed",
    ):
        assert expected in types
    events = client.get(f"/v1/tasks/{task_id}/events").json()
    created = next(e for e in events if e["type"] == "agent.plan")
    assert created["data"]["metadata"]["model"] == "openai/fake-model"


class CountingStore(InMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.gets = 0

    def get(self, task_id: str) -> Task:
        self.gets += 1
        return super().get(task_id)


def test_a_task_view_is_one_consistent_read(make_api: MakeApi) -> None:
    """Regression: the view was built from two loads, which could mix two versions of a task
    being changed by the worker (seen as an intermittent KeyError)."""
    store = CountingStore()
    client, _ = make_api(store=store, audit=InMemoryAuditLog())
    view = _run(client, "lista los archivos")
    assert view["status"] == "COMPLETED"
    before = store.gets
    client.get(f"/v1/tasks/{view['task_id']}")
    assert store.gets - before == 1
