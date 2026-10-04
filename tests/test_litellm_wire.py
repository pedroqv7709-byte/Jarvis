"""The real LiteLLM library and vendor SDK, over real HTTP, against a loopback fake of the OpenAI
chat completions endpoint. No internet, no real key: this checks what actually goes on the wire
(JSON Schema, no retries, no credentials from Jarvis) and runs the agent loop on top of it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import ActionStatus
from jarvis.core.task import TaskState
from jarvis.desktop.catalog import build_catalog
from jarvis.models.base import ChatMessage, ModelError
from jarvis.models.litellm_provider import LiteLLMProvider
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.planning.model_planner import ModelAnswer, StepProposal
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.builtin.windows import windows_tools

from support import WINDOWS_PERMISSIONS, FakeWindowsAutomation

pytest.importorskip("litellm")

FAKE_KEY = "sk-fake-loopback-key-000000"


class FakeOpenAI:
    """Scripted ``POST /v1/chat/completions``: each item is the answer text, or an HTTP status."""

    def __init__(self) -> None:
        self.script: list[str | int] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                fake.requests.append(json.loads(self.rfile.read(length)))
                fake.headers.append(dict(self.headers))
                item = fake.script.pop(0) if fake.script else 500
                if isinstance(item, int):
                    body = json.dumps({"error": {"message": "boom", "type": "server_error"}})
                    self._send(item, body)
                    return
                self._send(
                    200,
                    json.dumps(
                        {
                            "id": "chatcmpl-1",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "gpt-4o-mini",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": item},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ),
                )

            def _send(self, status: int, body: str) -> None:
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenAI]:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)  # read by LiteLLM/the SDK, never by Jarvis
    fake = FakeOpenAI()
    thread = threading.Thread(target=fake.server.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def provider(openai: FakeOpenAI) -> LiteLLMProvider:
    return LiteLLMProvider("openai/gpt-4o-mini", timeout_seconds=10, api_base=openai.url)


def test_the_request_on_the_wire_carries_the_decision_schema(openai: FakeOpenAI) -> None:
    openai.script = ['{"kind": "done"}']
    model = provider(openai)
    assert model.structured_mode == "json_schema"
    answer = model.structured_output([ChatMessage(role="user", content="hola")], ModelAnswer)
    assert isinstance(answer.root, StepProposal)
    (body,) = openai.requests
    assert body["model"] == "gpt-4o-mini" and body["temperature"] == 0
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == StepProposal.model_json_schema()
    # The credential travels only as the SDK's auth header, set by LiteLLM from the environment.
    assert openai.headers[0].get("Authorization") == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(body)


def test_a_failing_backend_is_called_once_no_hidden_retries(openai: FakeOpenAI) -> None:
    openai.script = [500, 500, 500]
    with pytest.raises(ModelError) as raised:
        provider(openai).structured_output([ChatMessage(role="user", content="x")], ModelAnswer)
    assert len(openai.requests) == 1  # the OpenAI SDK would retry twice by default
    assert FAKE_KEY not in str(raised.value)


def test_agent_loop_over_real_litellm_and_http(settings: Settings, openai: FakeOpenAI) -> None:
    desktop = FakeWindowsAutomation()
    desktop.on_launch["calc.exe"] = lambda: desktop.add_window(
        "Calculadora", "ApplicationFrameHost.exe", focus=False
    )
    openai.script = [
        '{"kind": "next_action", "action": {"tool": "windows.open_application", '
        '"arguments": {"application": "calculator"}}}',
        '{"kind": "next_action", "action": {"tool": "windows.focus_window", '
        '"arguments": {"application": "calculator"}}}',
        '{"kind": "done", "reply": "Hecho."}',
    ]
    configured = settings.model_copy(
        update={
            "permissions": dict(WINDOWS_PERMISSIONS),
            "model_provider": "litellm",
            "model": "openai/gpt-4o-mini",
            "model_api_base": openai.url,
            "planner_fallback": False,
        }
    )
    jarvis = build_jarvis(
        configured,  # the provider comes from settings, exactly as in production
        windows=desktop,
        tools=[
            *filesystem_tools(),
            *windows_tools(desktop, build_catalog(), open_wait_seconds=0.05),
        ],
        store=InMemoryStateStore(),
        audit=InMemoryAuditLog(),
    )
    response = jarvis.ask("Abre la calculadora y enfócala.")
    assert response.state is TaskState.COMPLETED, response.failure
    assert response.planner == "model"
    assert [a.status for a in response.actions] == [ActionStatus.VERIFIED] * 2
    assert len(openai.requests) == 3
    last = json.loads(openai.requests[-1]["messages"][-1]["content"])
    assert [r["achieved"] for r in last["task"]["requirements"]] == [True, True]
