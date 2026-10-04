"""Tool selection: bind a planned action to a registered tool and validated arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.core.action import Action
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.tools.base import Tool, ToolArguments
from jarvis.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class SelectedTool:
    tool: Tool[Any]
    args: ToolArguments


class ToolSelector:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def select(self, action: Action) -> SelectedTool:
        tool = self._registry.get(action.tool)
        if tool is None:
            raise ToolError(ErrorKind.NOT_FOUND, f"unknown tool {action.tool!r}")
        return SelectedTool(tool=tool, args=tool.validate_arguments(action.arguments))
