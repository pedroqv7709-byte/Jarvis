"""Structural validation of plans, independent of which planner produced them."""

from __future__ import annotations

from jarvis.core.action import Action
from jarvis.core.errors import FailureError, PlanningError
from jarvis.planning.base import Decision, DecisionKind, Plan
from jarvis.tools.registry import ToolRegistry


class PlanValidator:
    def __init__(self, registry: ToolRegistry, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._registry = registry
        self._max_steps = max_steps

    def validate(self, plan: Plan) -> None:
        if not plan.actions and not plan.reply:
            raise PlanningError("the plan contains no actions and no reply")
        self.validate_actions(plan.actions)

    def validate_decision(self, decision: Decision) -> None:
        """Same structural checks for one incremental decision (and its provisional steps)."""
        if decision.kind is DecisionKind.NEXT_ACTION:
            self.validate_actions(decision.actions)

    def validate_actions(self, actions: tuple[Action, ...]) -> None:
        if len(actions) > self._max_steps:
            raise PlanningError(
                f"the plan has {len(actions)} steps; the limit is {self._max_steps}"
            )
        seen: set[str] = set()
        for action in actions:
            if action.id in seen:
                raise PlanningError(f"duplicate action id {action.id!r}")
            for dep in action.depends_on:
                # Dependencies must point to earlier actions: this keeps plans acyclic and makes
                # the list order a valid execution order.
                if dep not in seen:
                    raise PlanningError(
                        f"action {action.id!r} depends on {dep!r}, which is not an earlier action"
                    )
            tool = self._registry.get(action.tool)
            if tool is None:
                raise PlanningError(
                    f"action {action.id!r} uses unknown tool {action.tool!r}",
                    details={"tool": action.tool},
                )
            try:
                tool.validate_arguments(action.arguments)
            except FailureError as exc:
                raise PlanningError(
                    f"action {action.id!r} has invalid arguments for {action.tool!r}",
                    details=exc.failure.details,
                ) from exc
            seen.add(action.id)
