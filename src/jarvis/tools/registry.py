"""Registry of available tools, with registration-time safety invariants."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jarvis.core.errors import FailureError, JarvisError
from jarvis.tools.base import Tool, ToolArguments, ToolSpec


class ToolRegistrationError(JarvisError):
    pass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        spec = getattr(type(tool), "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ToolRegistrationError(f"{type(tool).__name__} has no ToolSpec")
        if spec.name in self._tools:
            raise ToolRegistrationError(f"tool {spec.name!r} is already registered")
        # A tool without declared permissions could never be denied by a grant; refuse it.
        if not spec.required_permissions:
            raise ToolRegistrationError(f"tool {spec.name!r} must declare required permissions")
        if not spec.categories:
            raise ToolRegistrationError(f"tool {spec.name!r} must declare at least one category")
        args_model = getattr(type(tool), "args_model", None)
        if not (isinstance(args_model, type) and issubclass(args_model, ToolArguments)):
            raise ToolRegistrationError(f"tool {spec.name!r} args_model must extend ToolArguments")
        if args_model.model_config.get("extra") != "forbid":
            raise ToolRegistrationError(f"tool {spec.name!r} args_model must forbid extra fields")
        self._tools[spec.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def normalize_arguments(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Arguments as the tool would see them (schema defaults applied), for comparisons.
        Unknown tools or invalid arguments are returned unchanged."""
        found = self.get(tool)
        if found is None:
            return arguments
        try:
            normalized: dict[str, Any] = found.validate_arguments(arguments).model_dump(mode="json")
        except FailureError:
            return arguments
        return normalized

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def catalog(self) -> list[dict[str, Any]]:
        """Tool descriptions suitable for a planner (no handles to the tools themselves)."""
        return [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "input_schema": tool.input_schema(),
                "idempotent": tool.spec.idempotent,
                "side_effects": tool.spec.side_effects,
            }
            for tool in self._tools.values()
        ]
