"""Permission grants.

A permission *scope* is a dotted string such as ``fs.write`` or ``messages.send``. Tools declare
which scopes they need; the user grants each scope one of three modes. Anything not granted is
denied.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SCOPE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.*")


class GrantMode(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class PermissionSet(BaseModel):
    """Immutable set of grants. Lookup order: exact scope, longest matching ``prefix.*``, deny."""

    model_config = ConfigDict(frozen=True)

    grants: dict[str, GrantMode] = Field(default_factory=dict)

    @field_validator("grants")
    @classmethod
    def _validate_scopes(cls, grants: dict[str, GrantMode]) -> dict[str, GrantMode]:
        for scope in grants:
            if not scope or set(scope) - _SCOPE_CHARS:
                raise ValueError(f"invalid permission scope: {scope!r}")
            if "*" in scope and not (scope == "*" or scope.endswith(".*")):
                raise ValueError(f"wildcards are only allowed as a trailing '.*': {scope!r}")
            if scope.count("*") > 1:
                raise ValueError(f"invalid permission scope: {scope!r}")
        return grants

    @classmethod
    def of(cls, grants: Mapping[str, GrantMode | str]) -> PermissionSet:
        return cls(grants={k: GrantMode(v) for k, v in grants.items()})

    def mode_for(self, scope: str) -> GrantMode:
        if scope in self.grants:
            return self.grants[scope]
        best: tuple[int, GrantMode] | None = None
        for pattern, mode in self.grants.items():
            if not pattern.endswith("*"):
                continue
            prefix = pattern[:-1]  # "fs.*" -> "fs.", "*" -> ""
            if scope.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), mode)
        return best[1] if best else GrantMode.DENY
