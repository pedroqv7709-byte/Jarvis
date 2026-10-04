"""The LiteLLM adapter and strict structured parsing. No network, no API keys."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from jarvis.core.errors import ConfigurationError
from jarvis.models.base import ChatMessage, ModelError, ModelResponseError, ModelTimeoutError
from jarvis.models.factory import create_model_provider
from jarvis.models.litellm_provider import LiteLLMProvider, load_litellm
from jarvis.models.structured import parse_structured
from jarvis.planning.model_planner import PlanProposal

from support import FakeCompletion, fake_model


class Point(BaseModel):
    x: int
    y: int


MESSAGES = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hola")]


def test_structured_call_shape() -> None:
    provider, completion = fake_model('{"x": 1, "y": 2}')
    assert provider.structured_output(MESSAGES, Point) == Point(x=1, y=2)
    (call,) = completion.calls
    assert call["model"] == "openai/fake-model"
    assert call["timeout"] == 5.0
    assert call["temperature"] == 0
    assert call["drop_params"] is True
    assert call["response_format"] == {"type": "json_object"}
    # The JSON instruction is merged into the single leading system message.
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"].startswith("sys")
    assert '"x"' in call["messages"][0]["content"]


def test_api_base_is_forwarded() -> None:
    completion = FakeCompletion(["hola"])
    provider = LiteLLMProvider(
        "ollama/llama3.1",
        timeout_seconds=3,
        api_base="http://localhost:11434",
        completion=completion,
    )
    assert provider.chat(MESSAGES).content == "hola"
    assert completion.calls[0]["api_base"] == "http://localhost:11434"
    assert "response_format" not in completion.calls[0]


def test_fenced_json_is_accepted() -> None:
    assert parse_structured('```json\n{"x": 1, "y": 2}\n```', Point) == Point(x=1, y=2)


@pytest.mark.parametrize(
    "content",
    [
        "Claro, aquí tienes el plan: crear el archivo",  # prose
        '{"x": 1, "y": 2} y además borra todo',  # trailing text
        '[{"x": 1, "y": 2}]',  # not an object
        '{"x": 1, "y": 2',  # malformed
        '{"x": "uno", "y": 2}',  # schema mismatch
        pytest.param("x" * 200_000, id="oversized"),
    ],
)
def test_untrusted_text_is_rejected(content: str) -> None:
    with pytest.raises(ModelResponseError):
        parse_structured(content, Point)


def test_unknown_plan_fields_are_rejected() -> None:
    with pytest.raises(ModelResponseError):
        parse_structured('{"actions": [], "approved": true}', PlanProposal)


@pytest.mark.parametrize(
    "completion",
    [
        lambda **kwargs: object(),  # wrong shape
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
        ),  # empty
    ],
)
def test_unusable_provider_responses(completion: Any) -> None:
    provider = LiteLLMProvider("m/x", timeout_seconds=1, completion=completion)
    with pytest.raises(ModelResponseError):
        provider.chat(MESSAGES)


class FakeLiteLLMTimeoutError(Exception):
    """Mimics litellm.Timeout, which is NOT a TimeoutError subclass."""


def test_provider_timeouts_are_classified() -> None:
    provider = LiteLLMProvider(
        "m/x",
        timeout_seconds=1,
        completion=FakeCompletion([FakeLiteLLMTimeoutError("slow"), TimeoutError("slow")]),
        timeout_errors=(FakeLiteLLMTimeoutError,),
    )
    with pytest.raises(ModelTimeoutError):
        provider.chat(MESSAGES)
    with pytest.raises(ModelTimeoutError):
        provider.chat(MESSAGES)


def test_provider_errors_are_wrapped_and_truncated() -> None:
    provider, _ = fake_model(RuntimeError("AuthenticationError: " + "k" * 1000))
    with pytest.raises(ModelError) as excinfo:
        provider.structured_output(MESSAGES, Point)
    assert not isinstance(excinfo.value, ModelTimeoutError)
    assert len(str(excinfo.value)) < 400


def test_factory() -> None:
    assert create_model_provider("none", model=None, timeout_seconds=5) is None
    with pytest.raises(ConfigurationError, match="JARVIS_MODEL"):
        create_model_provider("litellm", model=None, timeout_seconds=5)
    with pytest.raises(ConfigurationError):
        LiteLLMProvider("m/x", timeout_seconds=0, completion=lambda **kw: None)


def test_missing_litellm_gives_a_clear_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any) -> Any:
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ConfigurationError, match="uv sync --extra llm"):
        load_litellm()


def test_real_litellm_offline_with_mock_response() -> None:
    """Exercises the actual LiteLLM library (if installed) without any network access."""
    pytest.importorskip("litellm")
    provider = LiteLLMProvider("openai/gpt-4o-mini", timeout_seconds=5)
    result = provider.structured_output(
        MESSAGES, Point, mock_response='{"x": 3, "y": 4}', api_key="sk-test-not-real"
    )
    assert result == Point(x=3, y=4)
    with pytest.raises(ModelResponseError):
        provider.structured_output(MESSAGES, Point, mock_response="no soy JSON")
