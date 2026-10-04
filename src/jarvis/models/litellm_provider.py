"""LiteLLM adapter: one :class:`ModelProvider` for every backend LiteLLM supports.

LiteLLM is an infrastructure detail. It is an optional dependency (``uv sync --extra llm``),
imported lazily here and nowhere else. Credentials are read by LiteLLM from its standard
environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...); Jarvis never handles them.

Structured output: when the backend supports it, the expected JSON Schema is sent as
``response_format={"type": "json_schema", ...}`` (non-strict: tool arguments are free-form
objects); otherwise ``json_object`` mode plus the schema in the system prompt. Either way the
answer is parsed and validated by Jarvis - the backend's compliance is never trusted.

Retries: none here. LiteLLM and the vendor SDKs underneath it (e.g. the OpenAI client retries twice
by default) are told not to retry, so one planning call costs at most one model timeout. Jarvis's
fallback chain is the only "retry", and it is bounded.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from jarvis.core.errors import ConfigurationError
from jarvis.models.base import (
    ChatMessage,
    ChatResponse,
    ModelCapabilities,
    ModelError,
    ModelResponseError,
    ModelTimeoutError,
    StructuredMode,
)
from jarvis.models.structured import parse_structured

CompletionFn = Callable[..., Any]

# Without this LiteLLM downloads its model price map from the internet on import.
_LOCAL_COST_MAP_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"
_MAX_ERROR_CHARS = 300
# Provider error messages sometimes echo (part of) the credential that was rejected.
_SECRET_LIKE = re.compile(
    r"(sk-[A-Za-z0-9_\-*]{6,}|AIza[0-9A-Za-z_\-]{10,}|gsk_[A-Za-z0-9]{10,}|"
    r"(?i:bearer)\s+[A-Za-z0-9._\-]{8,}|(?i:api[_-]?key)\s*[=:]\s*\S+)"
)

JSON_INSTRUCTION = (
    "Respond with exactly one JSON object that validates against the following JSON Schema. "
    "Do not add prose, comments or markdown.\nJSON Schema:\n{schema}"
)


def scrub_secrets(text: str) -> str:
    return _SECRET_LIKE.sub("***", text)


def wire_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The JSON Schema advertised to the backend.

    A schema class may expose ``response_json_schema()`` when what it *accepts* (e.g. a union of
    formats) differs from what the model should be asked to produce.
    """
    custom = getattr(schema, "response_json_schema", None)
    result: dict[str, Any] = custom() if callable(custom) else schema.model_json_schema()
    return result


def _schema_supported(model: str) -> bool:
    """Ask LiteLLM's (local) model map whether the backend accepts a JSON Schema."""
    try:
        module = importlib.import_module("litellm")
        return bool(module.supports_response_schema(model=model))
    except Exception:
        return False


def load_litellm() -> tuple[CompletionFn, tuple[type[BaseException], ...]]:
    """Import LiteLLM. Returns its ``completion`` function and its timeout exception types."""
    os.environ.setdefault(_LOCAL_COST_MAP_ENV, "True")
    try:
        module = importlib.import_module("litellm")
    except ImportError as exc:
        raise ConfigurationError(
            "JARVIS_MODEL_PROVIDER=litellm requires LiteLLM: run `uv sync --extra llm`"
        ) from exc
    timeout_type = getattr(module, "Timeout", None)
    timeout_types = (timeout_type,) if isinstance(timeout_type, type) else ()
    completion: CompletionFn = module.completion
    return completion, timeout_types


class LiteLLMProvider:
    name = "litellm"

    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float,
        api_base: str | None = None,
        completion: CompletionFn | None = None,
        timeout_errors: tuple[type[BaseException], ...] = (),
        structured_mode: StructuredMode = "auto",
    ) -> None:
        if not model:
            raise ConfigurationError("JARVIS_MODEL must name a LiteLLM model, e.g. openai/gpt-4o")
        if timeout_seconds <= 0:
            raise ConfigurationError("the model timeout must be positive")
        if completion is None:
            completion, timeout_errors = load_litellm()
            if structured_mode == "auto":
                structured_mode = "json_schema" if _schema_supported(model) else "json_object"
        elif structured_mode == "auto":
            structured_mode = "json_object"  # an injected backend: assume the least
        self.model = model
        self.structured_mode: StructuredMode = structured_mode
        self._timeout = timeout_seconds
        self._api_base = api_base
        self._completion = completion
        self._timeout_errors = timeout_errors

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(chat=True, structured_output=True)

    def chat(self, messages: Sequence[ChatMessage], **options: Any) -> ChatResponse:
        content = self._complete(list(messages), options)
        return ChatResponse(content=content, model=self.model)

    def structured_output[T: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[T], **options: Any
    ) -> T:
        advertised = wire_schema(schema)
        instruction = JSON_INSTRUCTION.format(schema=json.dumps(advertised))
        content = self._complete(
            _with_system_instruction(list(messages), instruction),
            {"response_format": self._response_format(schema, advertised), **options},
        )
        # Whatever the backend promised, the answer is validated here, against what Jarvis accepts.
        return parse_structured(content, schema)

    def _response_format(self, schema: type[BaseModel], advertised: dict[str, Any]) -> Any:
        # Backends require an object at the root of a response schema.
        if self.structured_mode == "json_schema" and advertised.get("type") == "object":
            name = re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64]
            return {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": advertised, "strict": False},
            }
        return {"type": "json_object"}

    def _complete(self, messages: list[ChatMessage], options: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "timeout": self._timeout,
            "temperature": 0,
            # Silently drop parameters a given backend does not support (e.g. response_format).
            "drop_params": True,
            # No hidden retries (LiteLLM's or the vendor SDK's): one call, one timeout.
            "num_retries": 0,
            "max_retries": 0,
            **options,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        try:
            response = self._completion(**kwargs)
        except Exception as exc:
            if isinstance(exc, (TimeoutError, *self._timeout_errors)):
                raise ModelTimeoutError(
                    f"{self.model} did not answer within {self._timeout:g}s"
                ) from exc
            message = scrub_secrets(str(exc))[:_MAX_ERROR_CHARS]
            raise ModelError(f"{type(exc).__name__}: {message}") from exc
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ModelResponseError("unexpected response shape from the provider") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError("the provider returned an empty response")
        return content


def _with_system_instruction(messages: list[ChatMessage], instruction: str) -> list[ChatMessage]:
    """Merge into the leading system message (some backends accept only one, and only first)."""
    if messages and messages[0].role == "system":
        merged = ChatMessage(role="system", content=f"{messages[0].content}\n\n{instruction}")
        return [merged, *messages[1:]]
    return [ChatMessage(role="system", content=instruction), *messages]
