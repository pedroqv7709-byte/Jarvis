"""Deterministic policy engine.

The policy runs entirely outside any model. It evaluates an action against the *registered* tool
spec (never against claims made by the planner) and the current permission grants.

Composition rules:

* every rule may return an outcome or abstain (``None``);
* the final decision is the most restrictive outcome (DENY > REQUIRE_CONFIRMATION > ALLOW);
* if every rule abstains, the decision is DENY (default deny);
* if a rule raises, the decision is DENY (fail closed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from jarvis.context.model import ExecutionContext
from jarvis.core.action import Action
from jarvis.core.categories import ActionCategory
from jarvis.core.decisions import Decision, PolicyVerdict, RuleOutcome, most_restrictive
from jarvis.permissions.model import GrantMode
from jarvis.tools.base import ToolSpec

# Minimum decision per category. Grants can only make things stricter than these floors.
DEFAULT_CATEGORY_FLOORS: Mapping[ActionCategory, Decision] = {
    ActionCategory.READ: Decision.ALLOW,
    ActionCategory.WRITE: Decision.ALLOW,
    ActionCategory.OPEN_APPLICATION: Decision.ALLOW,
    ActionCategory.WINDOW_MANAGEMENT: Decision.ALLOW,
    ActionCategory.CLOSE_APPLICATION: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.DELETE: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.SEND_MESSAGE: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.PURCHASE: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.TRANSFER_MONEY: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.CHANGE_CREDENTIALS: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.SYSTEM_CONFIGURATION: Decision.REQUIRE_CONFIRMATION,
    ActionCategory.DANGEROUS: Decision.DENY,
}

_GRANT_TO_DECISION = {
    GrantMode.ALLOW: Decision.ALLOW,
    GrantMode.CONFIRM: Decision.REQUIRE_CONFIRMATION,
    GrantMode.DENY: Decision.DENY,
}


@dataclass(frozen=True, slots=True)
class PolicyInput:
    action: Action
    spec: ToolSpec
    context: ExecutionContext


class PolicyRule(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, inp: PolicyInput) -> RuleOutcome | None: ...


class PermissionGrantRule:
    """Each required permission must be granted; the strictest grant mode wins."""

    name = "permission_grant"

    def evaluate(self, inp: PolicyInput) -> RuleOutcome:
        required = sorted(inp.spec.required_permissions)
        if not required:
            return RuleOutcome(
                rule=self.name, decision=Decision.DENY, reason="tool declares no permissions"
            )
        modes = {scope: inp.context.permissions.mode_for(scope) for scope in required}
        decision = most_restrictive(*(_GRANT_TO_DECISION[m] for m in modes.values()))
        detail = ", ".join(f"{scope}={mode}" for scope, mode in modes.items())
        return RuleOutcome(rule=self.name, decision=decision, reason=f"grants: {detail}")


class CategoryFloorRule:
    """Applies the minimum decision for each category the tool declares."""

    name = "category_floor"

    def __init__(self, overrides: Mapping[ActionCategory, Decision] | None = None) -> None:
        floors = dict(DEFAULT_CATEGORY_FLOORS)
        # Overrides may only tighten the defaults, never relax them.
        for category, decision in (overrides or {}).items():
            floors[category] = most_restrictive(floors.get(category, Decision.ALLOW), decision)
        self._floors = floors

    def evaluate(self, inp: PolicyInput) -> RuleOutcome | None:
        strict = {
            category: self._floors.get(category, Decision.REQUIRE_CONFIRMATION)
            for category in inp.spec.categories
        }
        strict = {c: d for c, d in strict.items() if d is not Decision.ALLOW}
        if not strict:
            return None
        decision = most_restrictive(*strict.values())
        detail = ", ".join(f"{c}->{d}" for c, d in sorted(strict.items()))
        return RuleOutcome(rule=self.name, decision=decision, reason=f"category floors: {detail}")


class PolicyEngine:
    def __init__(self, rules: Sequence[PolicyRule]) -> None:
        if not rules:
            raise ValueError("a policy engine needs at least one rule")
        self._rules = tuple(rules)

    @classmethod
    def default(
        cls, category_floors: Mapping[ActionCategory, Decision] | None = None
    ) -> PolicyEngine:
        return cls([PermissionGrantRule(), CategoryFloorRule(category_floors)])

    def evaluate(self, inp: PolicyInput) -> PolicyVerdict:
        outcomes: list[RuleOutcome] = []
        for rule in self._rules:
            try:
                outcome = rule.evaluate(inp)
            except Exception as exc:
                outcome = RuleOutcome(
                    rule=rule.name,
                    decision=Decision.DENY,
                    reason=f"rule raised {type(exc).__name__}; failing closed",
                )
            if outcome is not None:
                outcomes.append(outcome)

        if not outcomes:
            outcomes.append(
                RuleOutcome(rule="default", decision=Decision.DENY, reason="no rule allowed it")
            )
        decision = most_restrictive(*(o.decision for o in outcomes))
        return PolicyVerdict(
            decision=decision,
            outcomes=tuple(outcomes),
            evaluated_permissions=tuple(sorted(inp.spec.required_permissions)),
            evaluated_categories=tuple(sorted(inp.spec.categories)),
        )
