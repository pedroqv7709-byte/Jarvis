"""Ordered planner fallback: use the first planner whose decision *validates*.

A primary planner (normally the model) that fails, times out, or produces a decision rejected by
the :class:`PlanValidator` is recorded as a failed :class:`PlanningAttempt`, and the next planner
is tried. Nothing is executed until a decision has validated. The attempts travel with the
decision (or with the final error) so the agent loop can audit and report the fallback.

Within one task, a planner that already failed is skipped for later decisions (except the last
planner of the chain), so a dead model does not cost its timeout at every step.

Falling back is always safe in this design: every action, whichever planner proposed it, still
goes through policy, confirmation and verification. Policy denials never trigger replanning.
"""

from __future__ import annotations

from collections.abc import Sequence

from jarvis.core.errors import ErrorKind, Failure, FailureError, PlanningError
from jarvis.core.task import PlanningAttempt
from jarvis.planning.base import Decision, DecisionRequest, StepPlanner
from jarvis.planning.validation import PlanValidator


class PlanningFailedError(PlanningError):
    """Every planner failed. Carries the individual attempts."""

    def __init__(self, attempts: Sequence[PlanningAttempt]) -> None:
        summary = "; ".join(f"{a.planner}: {a.failure.message}" for a in attempts)
        super().__init__(
            f"no planner produced a valid plan ({summary})",
            details={"attempts": [a.model_dump(mode="json") for a in attempts]},
        )
        self.attempts = tuple(attempts)


class FallbackPlanner:
    name = "fallback"

    def __init__(self, planners: Sequence[StepPlanner], validator: PlanValidator) -> None:
        if not planners:
            raise ValueError("at least one planner is required")
        self._planners = tuple(planners)
        self._validator = validator

    @property
    def planners(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._planners)

    def decide(self, request: DecisionRequest) -> Decision:
        attempts: list[PlanningAttempt] = []
        last = self._planners[-1]
        for planner in self._planners:
            if planner is not last and planner.name in request.failed_planners:
                continue
            try:
                decision = planner.decide(request)
            except FailureError as exc:
                attempts.append(
                    PlanningAttempt(planner=planner.name, stage="plan", failure=exc.failure)
                )
                continue
            except Exception as exc:
                failure = Failure(
                    kind=ErrorKind.INTERNAL, message=f"planner raised {type(exc).__name__}"
                )
                attempts.append(
                    PlanningAttempt(planner=planner.name, stage="plan", failure=failure)
                )
                continue
            try:
                self._validator.validate_decision(decision)
            except FailureError as exc:
                attempts.append(
                    PlanningAttempt(planner=planner.name, stage="validate", failure=exc.failure)
                )
                continue
            return decision.model_copy(update={"failed_attempts": tuple(attempts)})
        raise PlanningFailedError(attempts)
