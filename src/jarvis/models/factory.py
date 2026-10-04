"""Builds the configured model provider, if any."""

from __future__ import annotations

from typing import Literal

from jarvis.core.errors import ConfigurationError
from jarvis.models.base import ModelProvider, StructuredMode

ProviderName = Literal["none", "litellm"]


def create_model_provider(
    provider: ProviderName,
    *,
    model: str | None,
    timeout_seconds: float,
    api_base: str | None = None,
    structured_mode: StructuredMode = "auto",
) -> ModelProvider | None:
    """Return a provider, or ``None`` when no model is configured (deterministic planning)."""
    if provider == "none":
        return None
    if provider == "litellm":
        if not model:
            raise ConfigurationError(
                "JARVIS_MODEL_PROVIDER=litellm requires JARVIS_MODEL (e.g. openai/gpt-4o-mini)"
            )
        from jarvis.models.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(
            model,
            timeout_seconds=timeout_seconds,
            api_base=api_base,
            structured_mode=structured_mode,
        )
    raise ConfigurationError(f"unknown model provider {provider!r}")
