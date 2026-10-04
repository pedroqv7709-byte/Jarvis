"""How the CLI reaches Jarvis: through the local server, or in-process.

Both backends return the same public views, so the CLI renders one shape regardless of mode.

* :class:`HttpBackend` talks to ``jarvis server`` over loopback HTTP (standard library only, so
  client commands never import FastAPI, uvicorn or LiteLLM).
* :class:`LocalBackend` runs Jarvis inside the CLI process, as before v0.3. Commands that execute
  or change tasks hold the instance lock, so they can never run concurrently with a server on the
  same data directory.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Protocol
from urllib.parse import urlencode

from pydantic import TypeAdapter

from jarvis.core.errors import JarvisError
from jarvis.interface.views import (
    EventView,
    HealthView,
    TaskAccepted,
    TaskSummary,
    TaskView,
    ToolView,
    event_view,
    state_of,
    task_summary,
    task_view,
    tool_view,
)
from jarvis.orchestration.jarvis import Jarvis
from jarvis.runtime.lock import InstanceLock


class ClientError(JarvisError):
    """The server refused the request (unknown task, wrong state, invalid input...)."""


class ServerUnavailableError(JarvisError):
    pass


class Backend(Protocol):
    @property
    def mode(self) -> str: ...

    def submit(self, text: str) -> TaskView: ...
    def get(self, task_id: str) -> TaskView: ...
    def approve(self, task_id: str, action_id: str) -> TaskView: ...
    def reject(self, task_id: str, action_id: str) -> TaskView: ...
    def cancel(self, task_id: str) -> TaskView: ...
    def resume(self, task_id: str) -> TaskView: ...
    def tasks(self, status: str | None, limit: int) -> list[TaskSummary]: ...
    def events(self, task_id: str) -> list[EventView]: ...
    def tools(self) -> list[ToolView]: ...
    def planner(self) -> str: ...


# -- HTTP -----------------------------------------------------------------------------------

Transport = Callable[[str, str, dict[str, Any] | None, float], tuple[int, Any]]
"""(method, path, json body, timeout) -> (status code, decoded JSON)."""


def urllib_transport(base_url: str, token: str | None = None) -> Transport:
    # No proxies: requests to 127.0.0.1 must never be routed through HTTP(S)_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def send(
        method: str, path: str, body: dict[str, Any] | None, timeout: float
    ) -> tuple[int, Any]:
        request = urllib.request.Request(
            base_url + path,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except (urllib.error.URLError, OSError) as exc:
            raise ServerUnavailableError(f"cannot reach Jarvis at {base_url}: {exc}") from exc

    return send


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return {"detail": raw[:200].decode("utf-8", "replace")}


class HttpBackend:
    mode = "server"

    def __init__(
        self,
        base_url: str,
        transport: Transport | None = None,
        timeout: float = 10.0,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url
        self._send = transport or urllib_transport(base_url, token)
        self._timeout = timeout

    def health(self, timeout: float = 1.0) -> HealthView | None:
        """Health if a Jarvis server answers, else None."""
        try:
            status, data = self._send("GET", "/health", None, timeout)
            return HealthView.model_validate(data) if status in (200, 503) else None
        except ServerUnavailableError, ValueError:
            return None

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        status, data = self._send(method, path, body, self._timeout)
        if status in (401, 403):
            raise ClientError(
                f"the server refused the local API token (HTTP {status}); the CLI reads it from "
                "<data_dir>/api_token, so it must use the same JARVIS_DATA_DIR as the server"
            )
        if status >= 400:
            detail = data.get("detail") if isinstance(data, dict) else data
            raise ClientError(f"{detail} (HTTP {status})")
        return data

    def submit(self, text: str) -> TaskView:
        accepted = TaskAccepted.model_validate(self._call("POST", "/v1/tasks", {"text": text}))
        return self.get(accepted.task_id)

    def get(self, task_id: str) -> TaskView:
        return TaskView.model_validate(self._call("GET", f"/v1/tasks/{task_id}"))

    def approve(self, task_id: str, action_id: str) -> TaskView:
        data = self._call("POST", f"/v1/tasks/{task_id}/approve", {"action_id": action_id})
        return TaskView.model_validate(data)

    def reject(self, task_id: str, action_id: str) -> TaskView:
        data = self._call("POST", f"/v1/tasks/{task_id}/reject", {"action_id": action_id})
        return TaskView.model_validate(data)

    def cancel(self, task_id: str) -> TaskView:
        return TaskView.model_validate(self._call("POST", f"/v1/tasks/{task_id}/cancel", {}))

    def resume(self, task_id: str) -> TaskView:
        raise ClientError(
            "with the server running, tasks are resumed automatically "
            "(after approve/reject, and by recovery when the server starts)"
        )

    def tasks(self, status: str | None, limit: int) -> list[TaskSummary]:
        query = urlencode({k: v for k, v in (("status", status), ("limit", limit)) if v})
        data = self._call("GET", f"/v1/tasks?{query}")
        return TypeAdapter(list[TaskSummary]).validate_python(data)

    def events(self, task_id: str) -> list[EventView]:
        data = self._call("GET", f"/v1/tasks/{task_id}/events")
        return TypeAdapter(list[EventView]).validate_python(data)

    def tools(self) -> list[ToolView]:
        return TypeAdapter(list[ToolView]).validate_python(self._call("GET", "/v1/tools"))

    def planner(self) -> str:
        health = self.health(self._timeout)
        if health is None:
            raise ServerUnavailableError(f"cannot reach Jarvis at {self.base_url}")
        return health.planner


# -- in-process -----------------------------------------------------------------------------


class LocalBackend:
    mode = "local"

    def __init__(self, factory: Callable[[], Jarvis], lock: InstanceLock | None) -> None:
        self._factory = factory
        self._lock = lock
        self._jarvis: Jarvis | None = None

    @property
    def jarvis(self) -> Jarvis:
        if self._jarvis is None:
            self._jarvis = self._factory()
        return self._jarvis

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock if self._lock is not None else nullcontext():
            yield

    def _view(self, task_id: str) -> TaskView:
        return task_view(*self.jarvis.snapshot(task_id))

    def submit(self, text: str) -> TaskView:
        with self._exclusive():
            response = self.jarvis.ask(text)
        return self._view(response.task_id)

    def get(self, task_id: str) -> TaskView:
        return self._view(task_id)

    def approve(self, task_id: str, action_id: str) -> TaskView:
        with self._exclusive():
            self.jarvis.approve(task_id, action_id)
        return self._view(task_id)

    def reject(self, task_id: str, action_id: str) -> TaskView:
        with self._exclusive():
            self.jarvis.reject(task_id, action_id)
        return self._view(task_id)

    def cancel(self, task_id: str) -> TaskView:
        with self._exclusive():
            self.jarvis.cancel(task_id)
        return self._view(task_id)

    def resume(self, task_id: str) -> TaskView:
        with self._exclusive():
            self.jarvis.resume(task_id)
        return self._view(task_id)

    def tasks(self, status: str | None, limit: int) -> list[TaskSummary]:
        state = state_of(status) if status else None
        return [task_summary(t) for t in self.jarvis.tasks(state=state, limit=limit)]

    def events(self, task_id: str) -> list[EventView]:
        self.jarvis.task(task_id)
        return [event_view(e) for e in self.jarvis.audit_trail(task_id)]

    def tools(self) -> list[ToolView]:
        return [tool_view(spec) for spec in self.jarvis.tools()]

    def planner(self) -> str:
        return self.jarvis.planner_description
