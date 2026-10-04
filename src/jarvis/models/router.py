"""Chooses a model provider by required capabilities.

Jarvis -> ModelRouter -> provider adapter -> model. Routing is deterministic: among providers that
satisfy the requirements, the highest priority wins; ties keep registration order.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.models.base import ModelCapabilities, ModelError, ModelProvider


class NoModelAvailableError(ModelError):
    pass


@dataclass(frozen=True, slots=True)
class _Route:
    provider: ModelProvider
    priority: int
    order: int


class ModelRouter:
    def __init__(self) -> None:
        self._routes: list[_Route] = []

    def register(self, provider: ModelProvider, *, priority: int = 0) -> None:
        if any(r.provider.name == provider.name for r in self._routes):
            raise ValueError(f"provider {provider.name!r} is already registered")
        self._routes.append(_Route(provider, priority, len(self._routes)))

    @property
    def providers(self) -> list[str]:
        return [r.provider.name for r in self._routes]

    def select(self, required: ModelCapabilities | None = None) -> ModelProvider:
        required = required or ModelCapabilities()
        candidates = [r for r in self._routes if r.provider.capabilities().satisfies(required)]
        if not candidates:
            raise NoModelAvailableError(
                f"no registered model satisfies {required.model_dump(exclude_defaults=True)}"
            )
        best = max(candidates, key=lambda r: (r.priority, -r.order))
        return best.provider
