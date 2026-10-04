"""The agent loop: observe -> decide -> validate -> execute (authorize, verify) -> observe ...

Owns task-level state transitions. Each iteration:

1. if an action is pending (planned, or awaiting a confirmation that was just decided) it is
   executed through :class:`ActionExecutor` - selection, target resolution, policy, confirmation,
   bounded attempts, independent verification - exactly as before v0.5;
2. otherwise Jarvis observes (desktop, requirement evidence), asks the planner for the next
   decision, validates it, and applies it: queue the next action(s), check a DONE claim against
   verified evidence, or stop safely.

A failed action does not end the task by itself: the loop observes and asks the planner again
(replanning), within hard limits that no planner can change - iterations, actions, replans, global
time, cycle detection and repeat guards. User rejections and policy denials end the task: they are
never "worked around" by replanning.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence

from jarvis.agent.completion import CompletionChecker
from jarvis.agent.context import build_request
from jarvis.agent.limits import AgentLimits, action_signature, detect_cycle
from jarvis.agent.observers import Observer
from jarvis.context.model import ExecutionContext
from jarvis.core.action import Action, ActionRecord, ActionStatus, AttemptStatus
from jarvis.core.agent import MAX_TRACKED_PROPOSALS, AgentPhase
from jarvis.core.errors import ErrorKind, Failure, FailureError, InvalidTransitionError
from jarvis.core.ids import utcnow
from jarvis.core.observation import Observation
from jarvis.core.task import PlanningAttempt, Task, TaskState
from jarvis.core.timeouts import run_with_timeout
from jarvis.execution.executor import ActionExecutor, ExecutionOutcome
from jarvis.observability.audit import AuditEventType, Auditor
from jarvis.planning.base import Decision, DecisionKind, StepPlanner
from jarvis.planning.validation import PlanValidator
from jarvis.state.store import ConcurrencyError, StateStore
from jarvis.tools.registry import ToolRegistry

StopCheck = Callable[[], bool]
"""Returns True when the hosting process is shutting down and the run should pause."""

PENDING_STATUSES = frozenset(
    {
        ActionStatus.PLANNED,
        ActionStatus.SELECTED,
        ActionStatus.AUTHORIZED,
        ActionStatus.AWAITING_CONFIRMATION,
    }
)
FINAL_ACTION_FAILURES = frozenset({ActionStatus.DENIED, ActionStatus.REJECTED})


class _CancelledError(Exception):
    def __init__(self, task: Task) -> None:
        super().__init__(task.id)
        self.task = task


class TaskRunner:
    def __init__(
        self,
        executor: ActionExecutor,
        store: StateStore,
        auditor: Auditor,
        *,
        planner: StepPlanner,
        validator: PlanValidator,
        registry: ToolRegistry,
        completion: CompletionChecker | None = None,
        observers: Sequence[Observer] = (),
        limits: AgentLimits | None = None,
        observe_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._executor = executor
        self._store = store
        self._auditor = auditor
        self._planner = planner
        self._validator = validator
        self._registry = registry
        self._completion = completion or CompletionChecker()
        self._observers = tuple(observers)
        self._limits = limits or AgentLimits()
        self._observe_timeout = observe_timeout
        self._clock = clock

    def run(
        self, task: Task, context: ExecutionContext, should_stop: StopCheck | None = None
    ) -> Task:
        try:
            return self._run(task, context, should_stop)
        except _CancelledError as cancelled:
            return cancelled.task

    # -- the loop -------------------------------------------------------------------------

    def _run(self, task: Task, context: ExecutionContext, should_stop: StopCheck | None) -> Task:
        if task.state in (TaskState.PENDING, TaskState.WAITING_PERMISSION):
            reason = "starting" if task.state is TaskState.PENDING else "resuming"
            self.transition(task, TaskState.RUNNING, reason)
        elif task.state is not TaskState.RUNNING:
            # RUNNING is accepted as-is: it means a previous process stopped mid-run.
            raise InvalidTransitionError(f"task {task.id} cannot run from state {task.state}")

        context = context.model_copy(update={"task_state": task.state})
        limits = self._limits
        while True:
            if should_stop is not None and should_stop():
                # Pause at a safe point: the task stays RUNNING and is resumed by recovery.
                self._save(task)
                return task
            if task.elapsed_seconds >= limits.task_timeout_seconds:
                return self._stop_limit(
                    task,
                    "time",
                    Failure(
                        kind=ErrorKind.TIMEOUT,
                        message=f"task time budget exhausted ({limits.task_timeout_seconds:g}s)",
                        details={"scope": "task", "limit": "time"},
                    ),
                )

            unverified = next(
                (r for r in task.actions if self._executor.needs_verification(r)), None
            )
            if unverified is not None:
                # Resumed after a stop between execution and verification: verify, never re-run.
                task.progress.phase = AgentPhase.EXECUTING
                outcome = self._executor.reverify(task, unverified, lambda: self._save(task))
                self._auditor.emit(
                    AuditEventType.AGENT_VERIFICATION,
                    task,
                    record=unverified,
                    outcome=unverified.status.value,
                    verification=(
                        unverified.verification.status.value if unverified.verification else None
                    ),
                    resumed=True,
                )
                if outcome is ExecutionOutcome.FAILED:
                    finished = self._after_failure(task, unverified)
                    if finished is not None:
                        return finished
                continue

            record = next((r for r in task.actions if r.status in PENDING_STATUSES), None)
            if record is not None:
                # Counted from the records themselves, so a restart never counts twice.
                starting = record.status is ActionStatus.PLANNED and not record.attempts
                if starting and _started_actions(task) >= limits.max_actions:
                    return self._stop_limit(
                        task,
                        "actions",
                        Failure(
                            kind=ErrorKind.LIMIT_EXCEEDED,
                            message=f"action limit reached ({limits.max_actions})",
                            details={"limit": "actions"},
                        ),
                    )
                outcome = self._execute(task, record, context)
                task.progress.actions_used = _started_actions(task)
                if outcome is ExecutionOutcome.WAITING_CONFIRMATION:
                    task.progress.phase = AgentPhase.WAITING
                    self.transition(
                        task, TaskState.WAITING_PERMISSION, f"action {record.id} awaits"
                    )
                    return task
                if outcome is ExecutionOutcome.FAILED:
                    finished = self._after_failure(task, record)
                    if finished is not None:
                        return finished
                continue

            if task.progress.final_batch:
                # The planner declared its batch final: completing it is a DONE claim.
                task.progress.final_batch = False
                observations = self._observe(task)
                decision = Decision(kind=DecisionKind.DONE, planner=task.planner or "planner")
            else:
                if task.progress.iterations >= limits.max_iterations:
                    return self._stop_limit(
                        task,
                        "iterations",
                        Failure(
                            kind=ErrorKind.LIMIT_EXCEEDED,
                            message=f"iteration limit reached ({limits.max_iterations})",
                            details={"limit": "iterations"},
                        ),
                    )
                observations = self._observe(task)
                maybe = self._decide(task, context, observations)
                if maybe is None:
                    return task
                decision = maybe
            finished = self._apply(task, decision, observations)
            if finished is not None:
                return finished

    # -- steps ----------------------------------------------------------------------------

    def _execute(
        self, task: Task, record: ActionRecord, context: ExecutionContext
    ) -> ExecutionOutcome:
        unmet = [
            d
            for d in record.action.depends_on
            if not any(r.id == d and r.is_done for r in task.actions)
        ]
        if unmet:
            record.status = ActionStatus.FAILED
            record.failure = Failure(
                kind=ErrorKind.INTERNAL, message=f"dependencies not satisfied: {', '.join(unmet)}"
            )
            return ExecutionOutcome.FAILED
        task.progress.phase = AgentPhase.EXECUTING
        self._save(task)
        outcome = self._executor.execute(task, record, context, lambda: self._save(task))
        if outcome is ExecutionOutcome.WAITING_CONFIRMATION:
            return outcome  # nothing ran yet, so there is nothing verified to report
        self._auditor.emit(
            AuditEventType.AGENT_VERIFICATION,
            task,
            record=record,
            outcome=record.status.value,
            verification=record.verification.status.value if record.verification else None,
            failure=record.failure.kind.value if record.failure else None,
        )
        return outcome

    def _after_failure(self, task: Task, record: ActionRecord) -> Task | None:
        if record.status in FINAL_ACTION_FAILURES:
            # A denial or the user's rejection is final: never worked around by replanning.
            return self._finish_failed(task, record)
        skip_remaining(task)  # provisional steps planned after it are superseded
        task.progress.final_batch = False
        failure = record.failure
        kind = failure.kind.value if failure else "unknown"
        if task.progress.replans_used >= self._limits.max_replans:
            return self._stop_limit(
                task,
                "replans",
                Failure(
                    kind=ErrorKind.LIMIT_EXCEEDED,
                    message=f"replanning limit reached ({self._limits.max_replans}) after "
                    f"{record.action.tool} failed ({kind})",
                    details={"limit": "replans", "action": record.id, "cause": kind},
                ),
            )
        task.progress.replans_used += 1
        task.progress.phase = AgentPhase.REPLANNING
        task.progress.feedback = f"action {record.id} ({record.action.tool}) failed: {kind}"
        self._auditor.emit(
            AuditEventType.AGENT_REPLAN,
            task,
            record=record,
            outcome=kind,
            message=failure.message if failure else "",
            replans_used=task.progress.replans_used,
        )
        self._save(task)
        return None

    def _observe(self, task: Task) -> tuple[Observation, ...]:
        if task.progress.phase is not AgentPhase.REPLANNING:
            # Observing as part of replanning keeps the REPLANNING phase visible.
            task.progress.phase = AgentPhase.OBSERVING
        observations = tuple(self._safe_observe(o) for o in self._observers)
        if observations:
            self._auditor.emit(
                AuditEventType.AGENT_OBSERVATION,
                task,
                sources={o.source: o.status for o in observations},
                windows={o.source: len(o.identifiers) for o in observations},
                foreground={
                    o.source: o.data.get("foreground") for o in observations if o.status == "ok"
                },
            )
        # What the observation proves, recorded after what was seen.
        for state in self._completion.update(task, observations):
            self._auditor.emit(
                AuditEventType.AGENT_REQUIREMENT_MET,
                task,
                outcome=state.requirement.kind.value,
                application=state.requirement.application,
                evidence=state.evidence,
            )
        return observations

    def _safe_observe(self, observer: Observer) -> Observation:
        try:
            return run_with_timeout(observer.observe, self._observe_timeout, name="observe")
        except Exception as exc:
            return Observation(
                source=observer.name,
                status="error",
                data={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            )

    def _decide(
        self, task: Task, context: ExecutionContext, observations: tuple[Observation, ...]
    ) -> Decision | None:
        if task.progress.phase is not AgentPhase.REPLANNING:
            task.progress.phase = AgentPhase.PLANNING
        self._save(task)
        request = build_request(task, context, self._registry.catalog(), observations, self._limits)
        started = self._clock()
        try:
            decision = self._planner.decide(request)
            self._validator.validate_decision(decision)
        except FailureError as exc:
            task.progress.iterations += 1
            self._record_attempts(task, getattr(exc, "attempts", ()))
            self._auditor.emit(
                AuditEventType.PLAN_FAILED,
                task,
                outcome=exc.failure.kind,
                message=exc.failure.message,
            )
            self._fail_planning(task, exc.failure)
            return None
        finally:
            task.progress.planning_seconds += self._clock() - started
        task.progress.iterations += 1
        self._record_attempts(task, decision.failed_attempts)
        task.planner = decision.planner
        self._auditor.emit(
            AuditEventType.AGENT_PLAN,
            task,
            outcome=decision.kind.value,
            planner=decision.planner,
            fallback_used=bool(decision.failed_attempts),
            metadata=decision.metadata,
            actions=[
                {
                    "id": a.id,
                    "tool": a.tool,
                    "arguments": a.arguments,
                    "depends_on": list(a.depends_on),
                }
                for a in decision.actions
            ],
            rationale=decision.rationale,
            expected_observation=decision.expected_observation,
            reason=decision.reason,
            iteration=task.progress.iterations,
        )
        return decision

    def _apply(
        self, task: Task, decision: Decision, observations: tuple[Observation, ...]
    ) -> Task | None:
        progress = task.progress
        progress.feedback = None
        if decision.kind is DecisionKind.DONE:
            if decision.reply:
                task.reply = decision.reply
            verdict = self._completion.evaluate(task, observations)
            if verdict.accepted:
                progress.phase = AgentPhase.FINISHED
                self._auditor.emit(
                    AuditEventType.AGENT_COMPLETED,
                    task,
                    message=verdict.reason,
                    evidence=verdict.evidence,
                )
                self.transition(task, TaskState.COMPLETED, f"completion verified: {verdict.reason}")
                return task
            progress.done_rejections += 1
            if progress.done_rejections >= 2:
                return self._stop_cycle(
                    task, f"completion claimed again without evidence ({verdict.reason})"
                )
            if progress.replans_used >= self._limits.max_replans:
                return self._stop_limit(
                    task,
                    "replans",
                    Failure(
                        kind=ErrorKind.LIMIT_EXCEEDED,
                        message=f"replanning limit reached ({self._limits.max_replans})",
                        details={"limit": "replans"},
                    ),
                )
            progress.replans_used += 1
            progress.phase = AgentPhase.REPLANNING
            progress.feedback = f"completion refused: {verdict.reason}"
            self._auditor.emit(
                AuditEventType.AGENT_REPLAN,
                task,
                outcome="completion_refused",
                message=verdict.reason,
                replans_used=progress.replans_used,
            )
            self._save(task)
            return None

        if decision.kind is DecisionKind.STOP:
            reason = decision.reason or "the planner stopped"
            progress.phase = AgentPhase.FINISHED
            progress.stop_reason = reason
            self._auditor.emit(AuditEventType.AGENT_STOPPED, task, message=reason, by="planner")
            last = next(
                (
                    r
                    for r in reversed(task.actions)
                    if r.status not in PENDING_STATUSES and r.status is not ActionStatus.SKIPPED
                ),
                None,
            )
            if last is not None and not last.is_done:
                return self._finish_failed(task, last)
            task.failure = Failure(kind=ErrorKind.STOPPED, message=reason)
            skip_remaining(task)
            self.transition(task, TaskState.STOPPED, reason)
            return task

        first = decision.actions[0]
        refusal = self._repeat_guard(task, first)
        if refusal is None:
            refusal = detect_cycle(progress.proposals, action_signature(first))
        if refusal is not None:
            return self._stop_cycle(task, refusal)
        progress.proposals = [*progress.proposals, action_signature(first)][-MAX_TRACKED_PROPOSALS:]
        records = [ActionRecord(action=a) for a in self._materialize(task, decision.actions)]
        task.actions.extend(records)
        progress.final_batch = decision.final
        self._auditor.emit(
            AuditEventType.AGENT_ACTION_SELECTED,
            task,
            record=records[0],
            arguments=records[0].action.arguments,
            provisional_steps=len(records) - 1,
            expected_observation=decision.expected_observation,
        )
        self._save(task)
        return None

    # -- guards ---------------------------------------------------------------------------

    def _repeat_guard(self, task: Task, action: Action) -> str | None:
        """Never re-run a non-idempotent action that already failed (its outcome may be
        unknown), and never retry the same idempotent action a third time."""
        failed = [
            r
            for r in task.actions
            if r.status in (ActionStatus.FAILED, *FINAL_ACTION_FAILURES)
            and r.action.tool == action.tool
            and r.action.arguments == action.arguments
        ]
        if not failed:
            return None
        tool = self._registry.get(action.tool)
        if tool is not None and not tool.spec.idempotent:
            kind = failed[-1].failure.kind.value if failed[-1].failure else "unknown"
            return (
                f"refusing to repeat {action.tool}: it is not idempotent and already failed "
                f"({kind}); its outcome may be unknown"
            )
        if len(failed) >= 2:
            return f"refusing to retry {action.tool} a third time"
        return None

    @staticmethod
    def _materialize(task: Task, actions: tuple[Action, ...]) -> list[Action]:
        """Give new actions task-unique ids; keep dependencies inside the batch only."""
        used = {r.id for r in task.actions}
        mapping: dict[str, str] = {}
        counter = len(task.actions)
        result = []
        for action in actions:
            new_id = action.id
            if new_id in used:
                while True:
                    counter += 1
                    new_id = f"a{counter}"
                    if new_id not in used:
                        break
            used.add(new_id)
            mapping[action.id] = new_id
            deps = tuple(mapping[d] for d in action.depends_on if d in mapping)
            result.append(action.model_copy(update={"id": new_id, "depends_on": deps}))
        return result

    def _record_attempts(self, task: Task, attempts: Sequence[PlanningAttempt]) -> None:
        for attempt in attempts:
            self._auditor.emit(
                AuditEventType.PLAN_ATTEMPT_FAILED,
                task,
                outcome=attempt.failure.kind,
                message=attempt.failure.message,
                planner=attempt.planner,
                stage=attempt.stage,
                details=attempt.failure.details,
            )
        task.planning_attempts = [*task.planning_attempts, *attempts]

    # -- endings --------------------------------------------------------------------------

    def transition(self, task: Task, to_state: TaskState, reason: str) -> None:
        previous = task.state
        task.transition(to_state, reason)
        self._save(task)
        self._auditor.emit(
            AuditEventType.TASK_STATE_CHANGED,
            task,
            outcome=to_state,
            message=reason,
            from_state=previous,
        )

    def _finish_failed(self, task: Task, failed: ActionRecord) -> Task:
        task.failure = failed.failure or Failure(
            kind=ErrorKind.INTERNAL, message=f"action {failed.id} failed"
        )
        task.progress.phase = AgentPhase.FINISHED
        skip_remaining(task)
        self.transition(task, TaskState.FAILED, f"action {failed.id} failed")
        return task

    def _fail_planning(self, task: Task, failure: Failure) -> None:
        last = next(
            (r for r in reversed(task.actions) if r.status is not ActionStatus.SKIPPED), None
        )
        if last is not None and last.failure is not None and not last.is_done:
            # Report the root cause (the failed action), not only that replanning failed.
            failure = last.failure.model_copy(
                update={"details": {**last.failure.details, "replanning": failure.message}}
            )
        task.failure = failure
        task.progress.phase = AgentPhase.FINISHED
        skip_remaining(task)
        self.transition(task, TaskState.FAILED, "planning failed")

    def _stop_limit(self, task: Task, limit: str, failure: Failure) -> Task:
        self._auditor.emit(
            AuditEventType.AGENT_LIMIT_REACHED, task, outcome=limit, message=failure.message
        )
        return self._stop(task, failure, by="limit")

    def _stop_cycle(self, task: Task, reason: str) -> Task:
        self._auditor.emit(AuditEventType.AGENT_CYCLE_DETECTED, task, message=reason)
        return self._stop(task, Failure(kind=ErrorKind.CYCLE_DETECTED, message=reason), by="cycle")

    def _stop(self, task: Task, failure: Failure, *, by: str) -> Task:
        task.failure = failure
        task.progress.stop_reason = failure.message
        task.progress.phase = AgentPhase.FINISHED
        self._auditor.emit(AuditEventType.AGENT_STOPPED, task, message=failure.message, by=by)
        skip_remaining(task)
        self.transition(task, TaskState.STOPPED, failure.message)
        return task

    # -- persistence ----------------------------------------------------------------------

    def _save(self, task: Task) -> None:
        task.touch()
        try:
            self._store.save(task)
        except ConcurrencyError:
            current = self._store.get(task.id)
            if current.state is TaskState.CANCELLED:
                raise _CancelledError(self._merge_into_cancelled(task, current)) from None
            raise

    def _merge_into_cancelled(self, local: Task, cancelled: Task) -> Task:
        """The task was cancelled while running. Keep the cancellation, but record what really
        happened to the actions (e.g. an attempt that completed) instead of the cancel's view."""
        for record in local.actions:
            last = record.attempts[-1] if record.attempts else None
            if last is not None and last.status is AttemptStatus.STARTED:
                # This checkpoint precedes Tool.run, so the attempt never ran.
                last.status = AttemptStatus.FAILED
                last.failure = Failure(
                    kind=ErrorKind.CANCELLED, message="cancelled before the attempt ran"
                )
                last.finished_at = utcnow()
        cancelled.actions = [r.model_copy(deep=True) for r in local.actions]
        skip_remaining(cancelled)
        cancelled.total_attempts = local.total_attempts
        cancelled.execution_seconds = local.execution_seconds
        cancelled.planner = local.planner
        cancelled.planning_attempts = list(local.planning_attempts)
        cancelled.requirements = [s.model_copy(deep=True) for s in local.requirements]
        cancelled.progress = local.progress.model_copy(update={"phase": AgentPhase.FINISHED})
        # Best effort: if this write races too, the audit event below still records the facts.
        with contextlib.suppress(ConcurrencyError):
            self._store.save(cancelled)
        self._auditor.emit(
            AuditEventType.TASK_CANCEL_OBSERVED,
            cancelled,
            message="cancellation observed by the runner; execution stopped",
            actions={r.id: r.status.value for r in cancelled.actions},
        )
        return cancelled


def _started_actions(task: Task) -> int:
    """Actions that began executing (anything past PLANNED, except never-run SKIPPED ones)."""
    return sum(
        1
        for r in task.actions
        if r.attempts or r.status not in (ActionStatus.PLANNED, ActionStatus.SKIPPED)
    )


def skip_remaining(task: Task) -> None:
    """Mark every action that has not finished or stopped as SKIPPED."""
    for record in task.actions:
        if record.status in PENDING_STATUSES:
            record.status = ActionStatus.SKIPPED
