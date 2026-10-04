"""Local API token: required on every endpoint except /health."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.config import Settings
from jarvis.interface.client import ClientError, HttpBackend
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.runtime.token import issue_token, read_token, revoke_token

from support import WINDOWS_PERMISSIONS, FakeWindowsAutomation

MakeApi = Callable[..., tuple[TestClient, JarvisRuntime]]

PROTECTED = [
    ("GET", "/v1/tasks", None),
    ("POST", "/v1/tasks", {"text": "lista los archivos"}),
    ("GET", "/v1/tasks/task_x", None),
    ("POST", "/v1/tasks/task_x/approve", {"action_id": "a1"}),
    ("POST", "/v1/tasks/task_x/reject", {"action_id": "a1"}),
    ("POST", "/v1/tasks/task_x/cancel", {}),
    ("GET", "/v1/tasks/task_x/events", None),
    ("GET", "/v1/tools", None),
]


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_missing_token_is_401(
    make_api: MakeApi, method: str, path: str, body: dict[str, str] | None
) -> None:
    client, _ = make_api()
    response = client.request(method, path, json=body, headers={"Authorization": ""})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_wrong_token_is_403(
    make_api: MakeApi, method: str, path: str, body: dict[str, str] | None
) -> None:
    client, _ = make_api()
    response = client.request(
        method, path, json=body, headers={"Authorization": "Bearer " + "y" * 43}
    )
    assert response.status_code == 403


def test_wrong_scheme_is_401(make_api: MakeApi) -> None:
    client, _ = make_api()
    response = client.get("/v1/tasks", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401


def test_health_needs_no_token(make_api: MakeApi) -> None:
    client, _ = make_api()
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200


def test_nothing_is_created_without_a_valid_token(make_api: MakeApi) -> None:
    client, runtime = make_api()
    client.post(
        "/v1/tasks", json={"text": "crea un archivo de prueba"}, headers={"Authorization": ""}
    )
    assert runtime.tasks(None, 10) == []


def test_token_file_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "data" / "api_token"
    first = issue_token(path)
    assert len(first) >= 40 and read_token(path) == first
    second = issue_token(path)  # every server start rotates the token
    assert second != first and read_token(path) == second
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    revoke_token(path, first)  # not ours anymore: left untouched
    assert read_token(path) == second
    revoke_token(path, second)
    assert read_token(path) is None


def test_short_tokens_are_refused(make_api: MakeApi) -> None:
    from jarvis.interface.api.security import token_guard

    with pytest.raises(ValueError):
        token_guard("short")


def test_cli_client_reports_a_rejected_token(make_api: MakeApi) -> None:
    client, _ = make_api()

    def send(method: str, path: str, body: dict[str, object] | None, timeout: float) -> object:
        response = client.request(
            method, path, json=body, headers={"Authorization": "Bearer " + "z" * 43}
        )
        return response.status_code, response.json()

    backend = HttpBackend("http://127.0.0.1:8765", transport=send)  # type: ignore[arg-type]
    with pytest.raises(ClientError, match="JARVIS_DATA_DIR"):
        backend.tasks(None, 5)


def test_windows_tools_are_listed(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api(
        settings.model_copy(update={"permissions": WINDOWS_PERMISSIONS}),
        windows=FakeWindowsAutomation(),
    )
    tools = {t["name"]: t for t in client.get("/v1/tools").json()}
    assert {
        "windows.get_windows",
        "windows.open_application",
        "windows.focus_window",
        "windows.close_application",
    } <= set(tools)
    assert tools["windows.close_application"]["categories"] == ["close_application"]
    assert tools["windows.close_application"]["permissions"] == ["windows.close"]
    assert not any("command" in name or "exec" in name for name in tools)
