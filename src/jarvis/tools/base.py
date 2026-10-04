"""Tool contract.

A tool is a *capability*, not an authority. It declares what it needs (permissions) and what kind
of effect it has (categories); whether it may run is decided by the policy engine, outside the
tool. Tools receive a :class:`ToolContext` that deliberately exposes no registry, executor or
policy handle, so a tool cannot invoke other tools or bypass authorization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.core.action import VerificationResult
from jarvis.core.categories import ActionCategory
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.tools.workspace import Workspace

TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$"


class ToolSpec(BaseModel):
    """Static, immutable description of a tool. This is what the policy engine evaluates."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1)
    required_permissions: frozenset[str]
    categories: frozenset[ActionCategory]
    side_effects: bool
    idempotent: bool
    """Safe to repeat after an attempt whose outcome is unknown."""
    metadata: dict[str, str] = Field(default_factory=dict)


class ToolArguments(BaseModel):
    """Base class for tool argument models. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool may know about the run it is part of."""

    workspace: Workspace
    task_id: str
    action_id: str
    attempt: int
    target: Mapping[str, Any] | None = None
    """The target resolved (and, when required, approved) before execution."""


class Tool[ArgsT: ToolArguments](ABC):
    spec: ClassVar[ToolSpec]
    args_model: ClassVar[type[ToolArguments]]

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def validate_arguments(self, raw: dict[str, Any]) -> ArgsT:
        try:
            return self.args_model.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as exc:
            raise ToolError(
                ErrorKind.VALIDATION,
                f"invalid arguments for {self.spec.name}",
                details={"errors": exc.errors(include_url=False, include_context=False)},
            ) from exc

    @abstractmethod
    def run(self, args: ArgsT, ctx: ToolContext) -> dict[str, Any]:
        """Perform the effect and return a JSON-serializable result."""

    def resolve_target(self, args: ArgsT, ctx: ToolContext) -> dict[str, Any] | None:
        """Read-only: resolve the request to a concrete target before authorization.

        The result is audited, shown in confirmations, bound into the approval digest and
        handed to ``run``/``verify`` as ``ctx.target``. Raise ToolError if the target cannot be
        resolved unambiguously. ``None`` means "this tool has no target resolution".
        """
        return None

    @property
    def has_resolver(self) -> bool:
        return type(self).resolve_target is not Tool.resolve_target

    def verify(
        self, args: ArgsT, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult | None:
        """Independently check that the effect happened. ``None`` means "no verifier"."""
        return None

    @property
    def has_verifier(self) -> bool:
        return type(self).verify is not Tool.verify
