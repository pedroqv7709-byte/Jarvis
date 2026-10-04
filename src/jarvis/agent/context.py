"""Builds the bounded, redacted context a planner receives for each decision.

Only what is needed to continue: the request, requirement status, the last N steps (status,
verification, failure, small output preview), the current observations (bounded) and the remaining
budget. Everything coming from tools or the desktop is untrusted data.
"""

from __future__ import annotations

import json
from typing import Any

from jarvis.agent.limits import AgentLimits
from jarvis.context.model import ExecutionContext
from jarvis.core.action import ActionRecord, ActionStatus
from jarvis.core.observation import Observation
from jarvis.core.task import Task
from jarvis.observability.redaction import redact
from jarvis.planning.base import Budget, DecisionRequest, StepSummary

MAX_FAILURE_CHARS = 200
MAX_OUTPUT_PREVIEW = 300
MAX_CANDIDATES = 10


def summarize(record: ActionRecord) -> StepSummary:
    failure = record.failure
    candidates: list[dict[str, Any]] = []
    if failure is not None and isinstance(failure.details.get("candidates"), list):
        candidates = [redact(c) for c in failure.details["candidates"][:MAX_CANDIDATES]]
    output = record.output
    preview = None
    if output is not None:
        preview = json.dumps(redact(output), ensure_ascii=False, default=str)[:MAX_OUTPUT_PREVIEW]
    target = None
    if record.target:
        target = {k: redact(v) for k, v in record.target.items() if k in ("application", "window")}
    return StepSummary(
        action_id=record.id,
        tool=record.action.tool,
        arguments=redact(record.action.arguments),
        status=record.status.value,
        verification=record.verification.status.value if record.verification else None,
        failure_kind=failure.kind.value if failure else None,
        failure_message=failure.message[:MAX_FAILURE_CHARS] if failure else None,
        candidates=candidates,
        output_preview=preview,
        target=target,
    )


def bound_observation(observation: Observation, max_windows: int) -> Observation:
    windows = observation.data.get("windows")
    if not isinstance(windows, list) or len(windows) <= max_windows:
        return observation
    data = {**observation.data, "windows": windows[:max_windows], "truncated": True}
    return observation.model_copy(
        update={"data": data, "identifiers": observation.identifiers[:max_windows]}
    )


def build_request(
    task: Task,
    context: ExecutionContext,
    catalog: list[dict[str, Any]],
    observations: tuple[Observation, ...],
    limits: AgentLimits,
) -> DecisionRequest:
    executed = [
        r for r in task.actions if r.status not in (ActionStatus.PLANNED, ActionStatus.SKIPPED)
    ]
    progress = task.progress
    return DecisionRequest(
        goal=task.goal,
        context=context,
        catalog=catalog,
        history=tuple(summarize(r) for r in executed[-limits.history_steps :]),
        observations=tuple(bound_observation(o, limits.max_observed_windows) for o in observations),
        requirements=tuple(s.model_copy(deep=True) for s in task.requirements),
        budget=Budget(
            iterations_left=max(0, limits.max_iterations - progress.iterations),
            actions_left=max(0, limits.max_actions - progress.actions_used),
            replans_left=max(0, limits.max_replans - progress.replans_used),
            seconds_left=max(0.0, limits.task_timeout_seconds - task.elapsed_seconds),
        ),
        feedback=progress.feedback,
        failed_planners=frozenset(a.planner for a in task.planning_attempts),
    )
