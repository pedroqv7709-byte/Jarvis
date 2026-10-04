"""Adapter that runs a whole-plan planner (``Planner.plan``) inside the agent loop.

The first decision returns the full plan as a provisional, final batch. Jarvis executes it step by
step (each step still resolved, authorized and verified); if a step fails, this planner cannot
adapt, so it stops - the pre-v0.5 behaviour.
"""

from __future__ import annotations

from jarvis.planning.base import (
    Decision,
    DecisionKind,
    DecisionRequest,
    Planner,
    PlanningRequest,
)


class PlanFollower:
    def __init__(self, inner: Planner) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    def decide(self, request: DecisionRequest) -> Decision:
        if not request.history:
            plan = self._inner.plan(
                PlanningRequest(goal=request.goal, context=request.context, catalog=request.catalog)
            )
            if not plan.actions:
                return Decision(kind=DecisionKind.DONE, planner=plan.planner, reply=plan.reply)
            return Decision(
                kind=DecisionKind.NEXT_ACTION,
                planner=plan.planner,
                actions=plan.actions,
                final=True,
                failed_attempts=plan.failed_attempts,
                metadata=plan.metadata,
            )
        last = request.history[-1]
        if last.failure_kind is not None:
            return Decision(
                kind=DecisionKind.STOP,
                planner=self.name,
                reason=f"action {last.action_id} failed and this planner cannot adapt",
            )
        return Decision(kind=DecisionKind.DONE, planner=self.name)
