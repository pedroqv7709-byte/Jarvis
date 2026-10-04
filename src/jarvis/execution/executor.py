"""Generic action executor.

For one action it runs, in order and with each step recorded:

1. selection     - registered tool + schema-validated arguments;
2. authorization - policy verdict, plus user confirmation when required;
3. execution     - bounded attempts with classified errors and retries;
4. verification  - independent check of the effect.

This is the only code path that calls ``Tool.run``, and it cannot reach that call without an
ALLOW verdict or a matching, unconsumed user approval.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from jarvis.context.model import ExecutionContext
from jarvis.core.action import (
    ActionRecord,
    ActionResult,
    ActionStatus,
    AttemptStatus,
    ConfirmationDecision,
    ConfirmationRequest,
    VerificationStatus,
)
from jarvis.core.decisions import Decision
from jarvis.core.errors import ErrorKind, Failure, FailureError
from jarvis.core.ids import utcnow
from jarvis.core.task import Task
from jarvis.core.timeouts import OperationTimeoutError, run_with_timeout
from jarvis.execution.confirmation import ConfirmationHandler, ConfirmationOutcome
from jarvis.execution.retry import ClassifiedFailure, RetryPolicy, classify_exception
from jarvis.execution.selection import SelectedTool, ToolSelector
from jarvis.execution.verification import VerificationService
from jarvis.observability.audit import AuditEventType, Auditor
from jarvis.safety.policy import PolicyEngine, PolicyInput
from jarvis.tools.base import ToolContext
from jarvis.tools.workspace import Workspace

Checkpoint = Callable[[], None]


class ExecutionOutcome(StrEnum):
    DONE = "done"
    """Executed, and verified or unverifiable."""
    WAITING_CONFIRMATION = "waiting_confirmation"
    FAILED = "failed"


class ActionExecutor:
    def __init__(
        self,
        *,
        selector: ToolSelector,
        policy: PolicyEngine,
        verification: VerificationService,
        confirmations: ConfirmationHandler,
        retry: RetryPolicy,
        auditor: Auditor,
        workspace: Workspace,
        max_total_attempts: int,
        tool_timeout_seconds: float | None = None,
        task_timeout_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_total_attempts < 1:
            raise ValueError("max_total_attempts must be >= 1")
        for value in (tool_timeout_seconds, task_timeout_seconds):
            if value is not None and value <= 0:
                raise ValueError("timeouts must be positive")
        self._selector = selector
        self._policy = policy
        self._verification = verification
        self._confirmations = confirmations
        self._retry = retry
        self._auditor = auditor
        self._workspace = workspace
        self._max_total_attempts = max_total_attempts
        self._sleep = sleep
        self._clock = clock
        self._tool_timeout = tool_timeout_seconds
        self._task_timeout = task_timeout_seconds

    def execute(
        self,
        task: Task,
        record: ActionRecord,
        context: ExecutionContext,
        checkpoint: Checkpoint,
    ) -> ExecutionOutcome:
        selected = self._select(task, record)
        if selected is None:
            return self._fail(task, record, checkpoint)

        if self._recover_interrupted_attempt(task, record, selected) is False:
            return self._fail(task, record, checkpoint)

        if not self._resolve(task, record, selected):
            return self._fail(task, record, checkpoint)

        authorization = self._authorize(task, record, selected, context, checkpoint)
        if authorization is not None:
            if authorization is ExecutionOutcome.FAILED:
                return self._fail(task, record, checkpoint)
            return authorization

        output = self._run_attempts(task, record, selected, checkpoint)
        if output is None:
            return self._fail(task, record, checkpoint)

        return self._verify(task, record, selected, output, checkpoint)

    # -- 1. selection -------------------------------------------------------------------------

    def _select(self, task: Task, record: ActionRecord) -> SelectedTool | None:
        try:
            selected = self._selector.select(record.action)
        except FailureError as exc:
            record.failure = exc.failure
            record.status = ActionStatus.FAILED
            self._auditor.emit(
                AuditEventType.ACTION_SELECTION_FAILED,
                task,
                record=record,
                outcome=exc.failure.kind,
                message=exc.failure.message,
            )
            return None
        if record.status is ActionStatus.PLANNED:
            record.status = ActionStatus.SELECTED
        self._auditor.emit(
            AuditEventType.ACTION_SELECTED,
            task,
            record=record,
            arguments=record.action.arguments,
            permissions=sorted(selected.tool.spec.required_permissions),
            categories=sorted(selected.tool.spec.categories),
        )
        return selected

    def _recover_interrupted_attempt(
        self, task: Task, record: ActionRecord, selected: SelectedTool
    ) -> bool | None:
        """Handle an attempt left STARTED by a crash. Returns False if the action must fail."""
        if not record.attempts or record.attempts[-1].status is not AttemptStatus.STARTED:
            return None
        attempt = record.attempts[-1]
        failure = Failure(
            kind=ErrorKind.INTERRUPTED,
            message="a previous attempt was interrupted; its outcome is unknown",
            details={"attempt": attempt.attempt},
        )
        attempt.failure = failure
        attempt.status = AttemptStatus.FAILED
        attempt.finished_at = utcnow()
        self._auditor.emit(
            AuditEventType.ACTION_ATTEMPT_FAILED,
            task,
            record=record,
            outcome=failure.kind,
            message=failure.message,
        )
        if selected.tool.spec.idempotent:
            return True
        # Re-running a non-idempotent tool could duplicate its effect; stop and surface it.
        record.failure = failure.model_copy(
            update={"message": failure.message + "; not retried because the tool is not idempotent"}
        )
        return False

    # -- 1b. target resolution ----------------------------------------------------------------

    def _resolve(self, task: Task, record: ActionRecord, selected: SelectedTool) -> bool:
        """Resolve the concrete target (read-only) before authorization. False = must fail."""
        if not selected.tool.has_resolver:
            record.target = None
            return True
        ctx = self._context(task, record, attempt=len(record.attempts) + 1, target=None)
        try:
            raw = run_with_timeout(
                functools.partial(selected.tool.resolve_target, selected.args, ctx),
                self._tool_timeout,
                name=f"{selected.tool.spec.name}.resolve",
            )
            target = _ensure_json_object(raw)
        except OperationTimeoutError as exc:
            failure = Failure(kind=ErrorKind.TIMEOUT, message=str(exc))
        except Exception as exc:
            failure = classify_exception(exc).failure
        else:
            record.target = target
            self._auditor.emit(
                AuditEventType.ACTION_RESOLVED,
                task,
                record=record,
                requested=record.action.arguments,
                resolved=target,
            )
            return True
        record.target = None
        record.failure = failure
        self._auditor.emit(
            AuditEventType.ACTION_RESOLUTION_FAILED,
            task,
            record=record,
            outcome=failure.kind,
            message=failure.message,
            requested=record.action.arguments,
            details=failure.details,
        )
        return False

    def _context(
        self, task: Task, record: ActionRecord, *, attempt: int, target: dict[str, Any] | None
    ) -> ToolContext:
        return ToolContext(
            workspace=self._workspace,
            task_id=task.id,
            action_id=record.id,
            attempt=attempt,
            target=target,
        )

    # -- 2. authorization ---------------------------------------------------------------------

    def _authorize(
        self,
        task: Task,
        record: ActionRecord,
        selected: SelectedTool,
        context: ExecutionContext,
        checkpoint: Checkpoint,
    ) -> ExecutionOutcome | None:
        """Return None when authorized; otherwise the outcome to report."""
        action = record.action
        digest = action.digest(record.target)
        decision = record.confirmation_decision
        if decision is not None and not decision.approved and decision.arguments_digest == digest:
            return self._reject(task, record, decided_by=decision.decided_by)

        verdict = self._policy.evaluate(
            PolicyInput(action=action, spec=selected.tool.spec, context=context)
        )
        record.verdict = verdict
        self._auditor.emit(
            AuditEventType.POLICY_EVALUATED,
            task,
            record=record,
            decision=verdict.decision,
            reasons=verdict.reasons,
            permissions=list(verdict.evaluated_permissions),
            categories=list(verdict.evaluated_categories),
        )

        if verdict.decision is Decision.DENY:
            record.status = ActionStatus.DENIED
            record.failure = Failure(
                kind=ErrorKind.POLICY_DENIED,
                message=f"the policy denied this action ({'; '.join(verdict.reasons)})",
                details={"reasons": verdict.reasons},
            )
            return ExecutionOutcome.FAILED

        if verdict.decision is Decision.REQUIRE_CONFIRMATION:
            approved_for_these_arguments = (
                decision is not None
                and decision.approved
                and not decision.consumed
                and decision.arguments_digest == digest
            )
            if approved_for_these_arguments:
                assert decision is not None
                record.confirmation_decision = decision.model_copy(update={"consumed": True})
                self._mark_authorized(record, checkpoint)
                return None
            outcome = self._ask(
                task, record, selected, context, digest, verdict.reasons, checkpoint
            )
            if outcome is ConfirmationOutcome.DEFERRED:
                return ExecutionOutcome.WAITING_CONFIRMATION
            if outcome is ConfirmationOutcome.REJECTED:
                return self._reject(task, record, decided_by=self._confirmations.name)

        self._mark_authorized(record, checkpoint)
        return None

    @staticmethod
    def _mark_authorized(record: ActionRecord, checkpoint: Checkpoint) -> None:
        record.status = ActionStatus.AUTHORIZED
        checkpoint()

    def _ask(
        self,
        task: Task,
        record: ActionRecord,
        selected: SelectedTool,
        context: ExecutionContext,
        digest: str,
        reasons: list[str],
        checkpoint: Checkpoint,
    ) -> ConfirmationOutcome:
        request = ConfirmationRequest(
            action_id=record.id,
            tool=record.action.tool,
            # From the registered tool, never from the plan: a planner (possibly a model) must not
            # be able to word the confirmation prompt the user decides on.
            description=selected.tool.spec.description,
            arguments=record.action.arguments,
            arguments_digest=digest,
            reasons=tuple(reasons),
            target=record.target,
        )
        record.confirmation = request
        record.confirmation_decision = None
        record.status = ActionStatus.AWAITING_CONFIRMATION
        self._auditor.emit(
            AuditEventType.CONFIRMATION_REQUESTED, task, record=record, reasons=reasons
        )
        checkpoint()

        outcome = self._confirmations.request(request, context)
        if outcome is not ConfirmationOutcome.DEFERRED:
            approved = outcome is ConfirmationOutcome.APPROVED
            record.confirmation_decision = ConfirmationDecision(
                approved=approved,
                decided_by=self._confirmations.name,
                arguments_digest=digest,
                consumed=approved,
            )
            self._auditor.emit(
                AuditEventType.CONFIRMATION_DECIDED,
                task,
                record=record,
                outcome=outcome,
                decided_by=self._confirmations.name,
            )
        return outcome

    @staticmethod
    def _reject(task: Task, record: ActionRecord, *, decided_by: str) -> ExecutionOutcome:
        record.status = ActionStatus.REJECTED
        record.failure = Failure(
            kind=ErrorKind.CONFIRMATION_REJECTED,
            message="the action was rejected by the user",
            details={"decided_by": decided_by},
        )
        return ExecutionOutcome.FAILED

    # -- 3. execution -------------------------------------------------------------------------

    def _run_attempts(
        self, task: Task, record: ActionRecord, selected: SelectedTool, checkpoint: Checkpoint
    ) -> dict[str, Any] | None:
        spec = selected.tool.spec
        while True:
            if task.total_attempts >= self._max_total_attempts:
                record.failure = Failure(
                    kind=ErrorKind.LIMIT_EXCEEDED,
                    message=f"task attempt budget exhausted ({self._max_total_attempts})",
                )
                return None
            remaining = self._remaining_budget(task)
            if remaining is not None and remaining <= 0:
                record.failure = Failure(
                    kind=ErrorKind.TIMEOUT,
                    message=f"task execution time budget exhausted ({self._task_timeout:g}s)",
                    details={"scope": "task"},
                )
                return None
            timeout = _min_optional(self._tool_timeout, remaining)

            attempt = ActionResult(action_id=record.id, attempt=len(record.attempts) + 1)
            record.attempts.append(attempt)
            task.total_attempts += 1
            self._auditor.emit(
                AuditEventType.ACTION_ATTEMPT_STARTED, task, record=record, attempt=attempt.attempt
            )
            # Persist "started" before running, so a crash is detectable on resume.
            checkpoint()

            ctx = self._context(task, record, attempt=attempt.attempt, target=record.target)
            started = self._clock()
            try:
                raw = run_with_timeout(
                    functools.partial(selected.tool.run, selected.args, ctx),
                    timeout,
                    name=spec.name,
                )
                output = _ensure_json_object(raw)
            except OperationTimeoutError as exc:
                classified = _timeout_failure(exc, spec.side_effects)
            except Exception as exc:
                classified = classify_exception(exc)
            else:
                task.execution_seconds += self._clock() - started
                attempt.output = output
                attempt.status = AttemptStatus.SUCCEEDED
                attempt.finished_at = utcnow()
                record.status = ActionStatus.EXECUTED
                self._auditor.emit(
                    AuditEventType.ACTION_EXECUTED,
                    task,
                    record=record,
                    outcome=AttemptStatus.SUCCEEDED,
                    attempt=attempt.attempt,
                    output=output,
                )
                checkpoint()
                return output

            task.execution_seconds += self._clock() - started
            if not self._handle_attempt_failure(task, record, attempt, classified, spec.idempotent):
                return None
            checkpoint()

    def _handle_attempt_failure(
        self,
        task: Task,
        record: ActionRecord,
        attempt: ActionResult,
        classified: ClassifiedFailure,
        idempotent: bool,
    ) -> bool:
        """Record a failed attempt. Returns True if another attempt should be made."""
        failure = classified.failure
        attempt.failure = failure
        attempt.status = AttemptStatus.FAILED
        attempt.finished_at = utcnow()
        self._auditor.emit(
            AuditEventType.ACTION_ATTEMPT_FAILED,
            task,
            record=record,
            outcome=failure.kind,
            message=failure.message,
            attempt=attempt.attempt,
            retryable=failure.retryable,
        )
        if self._retry.should_retry(classified, attempt.attempt, idempotent=idempotent):
            delay = _min_optional(self._retry.delay(attempt.attempt), self._remaining_budget(task))
            delay = max(0.0, delay or 0.0)
            self._auditor.emit(
                AuditEventType.ACTION_RETRY_SCHEDULED,
                task,
                record=record,
                attempt=attempt.attempt,
                delay_seconds=delay,
            )
            self._sleep(delay)
            task.execution_seconds += delay
            return True
        record.failure = failure
        return False

    def _remaining_budget(self, task: Task) -> float | None:
        if self._task_timeout is None:
            return None
        return self._task_timeout - task.elapsed_seconds

    def reverify(
        self, task: Task, record: ActionRecord, checkpoint: Checkpoint
    ) -> ExecutionOutcome:
        """Verify an action that executed but was never verified (the process stopped in
        between). The action is not executed again."""
        selected = self._select(task, record)
        output = record.output
        if selected is None or output is None:
            return self._fail(task, record, checkpoint)
        return self._verify(task, record, selected, output, checkpoint)

    def needs_verification(self, record: ActionRecord) -> bool:
        """Executed, with a verifier available, but no verification recorded yet."""
        if record.status is not ActionStatus.EXECUTED or record.verification is not None:
            return False
        try:
            return self._selector.select(record.action).tool.has_verifier
        except FailureError:
            return False

    # -- 4. verification ----------------------------------------------------------------------

    def _verify(
        self,
        task: Task,
        record: ActionRecord,
        selected: SelectedTool,
        output: dict[str, Any],
        checkpoint: Checkpoint,
    ) -> ExecutionOutcome:
        ctx = self._context(task, record, attempt=len(record.attempts), target=record.target)
        result = self._verification.verify(selected.tool, selected.args, output, ctx)
        record.verification = result
        self._auditor.emit(
            AuditEventType.ACTION_VERIFIED,
            task,
            record=record,
            outcome=result.status,
            message=result.message,
            evidence=result.evidence,
        )
        if result.status is VerificationStatus.FAILED:
            record.failure = Failure(kind=ErrorKind.VERIFICATION_FAILED, message=result.message)
            return self._fail(task, record, checkpoint)
        if result.status is VerificationStatus.VERIFIED:
            record.status = ActionStatus.VERIFIED
        checkpoint()
        return ExecutionOutcome.DONE

    # -- failure ------------------------------------------------------------------------------

    def _fail(self, task: Task, record: ActionRecord, checkpoint: Checkpoint) -> ExecutionOutcome:
        if record.status not in (ActionStatus.DENIED, ActionStatus.REJECTED):
            record.status = ActionStatus.FAILED
        failure = record.failure
        self._auditor.emit(
            AuditEventType.ACTION_FAILED,
            task,
            record=record,
            outcome=failure.kind if failure else None,
            message=failure.message if failure else "",
        )
        checkpoint()
        return ExecutionOutcome.FAILED


def _min_optional(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _timeout_failure(exc: OperationTimeoutError, side_effects: bool) -> ClassifiedFailure:
    """A hard timeout. Not explicit: non-idempotent tools are therefore never retried after it."""
    return ClassifiedFailure(
        Failure(
            kind=ErrorKind.TIMEOUT,
            message=str(exc),
            retryable=True,
            details={"timeout_seconds": exc.seconds, "outcome_unknown": side_effects},
        ),
        explicit=False,
    )


def _ensure_json_object(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise FailureError(
            Failure(kind=ErrorKind.INTERNAL, message="tool returned a non-object result")
        )
    try:
        normalized: dict[str, Any] = json.loads(json.dumps(output))
    except (TypeError, ValueError) as exc:
        raise FailureError(
            Failure(kind=ErrorKind.INTERNAL, message=f"tool returned non-JSON output: {exc}")
        ) from exc
    return normalized
