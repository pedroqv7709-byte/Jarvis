"""Policy decision types shared by the safety layer and the persisted task state."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.ids import utcnow


class Decision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {Decision.ALLOW: 0, Decision.REQUIRE_CONFIRMATION: 1, Decision.DENY: 2}


def most_restrictive(*decisions: Decision) -> Decision:
    """Combine decisions; DENY beats REQUIRE_CONFIRMATION beats ALLOW."""
    if not decisions:
        raise ValueError("at least one decision is required")
    return max(decisions, key=lambda d: d.severity)


class RuleOutcome(BaseModel):
    """What a single policy rule concluded, and why."""

    model_config = ConfigDict(frozen=True)

    rule: str
    decision: Decision
    reason: str


class PolicyVerdict(BaseModel):
    """Final, auditable result of evaluating an action against the policy."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    outcomes: tuple[RuleOutcome, ...] = ()
    evaluated_permissions: tuple[str, ...] = ()
    evaluated_categories: tuple[str, ...] = ()
    evaluated_at: datetime = Field(default_factory=utcnow)

    @property
    def reasons(self) -> list[str]:
        return [f"{o.rule}: {o.reason}" for o in self.outcomes]
