"""Provider-independent model contract.

Jarvis talks to language models only through :class:`ModelProvider`. Concrete adapters (a LiteLLM
gateway, a local runtime, a specific vendor SDK) implement this protocol; nothing else in Jarvis
imports a vendor SDK. Model output is always treated as untrusted input by the rest of the system.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.errors import JarvisError

StructuredMode = Literal["auto", "json_schema", "json_object"]
"""How an adapter asks a backend for JSON: a JSON Schema, plain JSON mode, or decide by backend."""


class ModelError(JarvisError):
    """A provider failed or returned unusable output."""


class ModelTimeoutError(ModelError):
    """The provider did not answer within its time budget."""


class ModelResponseError(ModelError):
    """The provider answered, but the answer is not usable (malformed, wrong shape, empty)."""


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    chat: bool = True
    structured_output: bool = False
    tool_calling: bool = False
    vision: bool = False
    audio: bool = False
    local: bool = False
    """Runs on this machine (usable offline, data stays local)."""
    max_context_tokens: int | None = None

    def satisfies(self, required: ModelCapabilities) -> bool:
        flags = ("chat", "structured_output", "tool_calling", "vision", "audio", "local")
        if any(getattr(required, f) and not getattr(self, f) for f in flags):
            return False
        return required.max_context_tokens is None or (
            self.max_context_tokens is not None
            and self.max_context_tokens >= required.max_context_tokens
        )


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    def capabilities(self) -> ModelCapabilities: ...

    def chat(self, messages: Sequence[ChatMessage], **options: Any) -> ChatResponse: ...

    def structured_output[T: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[T], **options: Any
    ) -> T:
        """Return an instance of ``schema``. Implementations must validate before returning."""
        ...
