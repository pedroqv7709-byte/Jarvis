"""Executor behaviour, exercised through the real pipeline with controlled tools."""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import Action, ActionStatus, AttemptStatus, VerificationStatus
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.core.task import TaskState
from jarvis.execution.retry import RetryPolicy, classify_exception
from jarvis.observability.audit import AuditEventType, InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.state.store import InMemoryStateStore
from jarvis.tools.base import Tool

from support import (
    CountingTool,
    FlakyTool,
    LyingTool,
    NonIdempotentFlakyTool,
    RaisingVerifierTool,
    StaticPlanner,
)


def _jarvis(
    settings: Settings, tools: list[Tool[Any]], *actions: Action, **kwargs: Any
) -> tuple[Jarvis, list[float]]:
    sleeps: list[float] = []
    jarvis = build_jarvis(
        settings,
        planner=StaticPlanner(*actions),
        tools=tools,
        store=InMemoryStateStore(),
        audit=InMemoryAuditLog(),
        sleep=sleeps.append,
        **kwargs,
    )
    return jarvis, sleeps


def _echo(action_id: str = "a1", tool: str = "test.echo", **kw: Any) -> Action:
    return Action(id=action_id, tool=tool, arguments={"text": "hi"}, **kw)


def test_successful_flow_selects_authorizes_executes_verifies(settings: Settings) -> None:
    tool = CountingTool()
    jarvis, _ = _jarvis(settings, [tool], _echo())
    response = jarvis.ask("echo")
    assert response.state is TaskState.COMPLETED
    record = jarvis.task(response.task_id).actions[0]
    assert record.status is ActionStatus.VERIFIED
    assert record.verdict is not None and record.verdict.decision == "allow"
    assert record.verification is not None
    assert record.verification.status is VerificationStatus.VERIFIED
    assert [a.status for a in record.attempts] == [AttemptStatus.SUCCEEDED]
    assert tool.calls == ["hi"]

    types = [e.type for e in jarvis.audit_trail(response.task_id)]
    selected = types.index(AuditEventType.ACTION_SELECTED)
    evaluated = types.index(AuditEventType.POLICY_EVALUATED)
    executed = types.index(AuditEventType.ACTION_EXECUTED)
    verified = types.index(AuditEventType.ACTION_VERIFIED)
    assert selected < evaluated < executed < verified


def test_non_retryable_failure_fails_task_without_retry(settings: Settings) -> None:
    tool = FlakyTool(failures=5, exc=ToolError(ErrorKind.TOOL_ERROR, "bad input file"))
    jarvis, sleeps = _jarvis(settings, [tool], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert tool.calls == 1
    assert sleeps == []
    assert response.failure is not None and response.failure.kind is ErrorKind.TOOL_ERROR


def test_transient_failure_is_retried_then_succeeds(settings: Settings) -> None:
    tool = FlakyTool(failures=2)
    jarvis, sleeps = _jarvis(settings, [tool], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.COMPLETED
    record = jarvis.task(response.task_id).actions[0]
    assert [a.status for a in record.attempts] == [
        AttemptStatus.FAILED,
        AttemptStatus.FAILED,
        AttemptStatus.SUCCEEDED,
    ]
    assert len(sleeps) == 2
    events = jarvis.audit_trail(response.task_id)
    assert sum(e.type is AuditEventType.ACTION_RETRY_SCHEDULED for e in events) == 2


def test_retry_limit_is_enforced(settings: Settings) -> None:
    tool = FlakyTool(failures=100)
    jarvis, _ = _jarvis(settings, [tool], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert tool.calls == settings.max_attempts_per_action
    assert response.failure is not None and response.failure.kind is ErrorKind.TRANSIENT


def test_task_wide_attempt_budget_stops_runaway_retries(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"max_attempts_per_action": 10, "max_total_attempts_per_task": 4}
    )
    tool = FlakyTool(failures=100)
    jarvis, _ = _jarvis(settings, [tool], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert tool.calls == 4
    assert response.failure is not None and response.failure.kind is ErrorKind.LIMIT_EXCEEDED


def test_non_idempotent_tool_not_retried_on_unclassified_transient_error(
    settings: Settings,
) -> None:
    tool = NonIdempotentFlakyTool(failures=1, exc=ConnectionError("reset"))
    jarvis, _ = _jarvis(settings, [tool], _echo(tool="test.flaky_writer"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert tool.calls == 1


def test_non_idempotent_tool_retried_when_tool_declares_retryable(settings: Settings) -> None:
    tool = NonIdempotentFlakyTool(failures=1)  # raises ToolError(TRANSIENT, retryable=True)
    jarvis, _ = _jarvis(settings, [tool], _echo(tool="test.flaky_writer"))
    assert jarvis.ask("x").state is TaskState.COMPLETED
    assert tool.calls == 2


def test_unexpected_exception_is_captured_as_internal_failure(settings: Settings) -> None:
    tool = FlakyTool(failures=1, exc=ZeroDivisionError("oops"))
    jarvis, _ = _jarvis(settings, [tool], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.kind is ErrorKind.INTERNAL
    task = jarvis.task(response.task_id)
    assert task.actions[0].attempts[0].finished_at is not None


def test_failure_skips_remaining_actions_and_keeps_completed_ones(settings: Settings) -> None:
    ok, bad = CountingTool(), FlakyTool(failures=9, exc=ToolError(ErrorKind.TOOL_ERROR, "no"))
    jarvis, _ = _jarvis(
        settings,
        [ok, bad],
        _echo("a1"),
        _echo("a2", tool="test.flaky", depends_on=("a1",)),
        _echo("a3", depends_on=("a2",)),
    )
    response = jarvis.ask("x")
    statuses = [a.status for a in response.actions]
    assert statuses == [ActionStatus.VERIFIED, ActionStatus.FAILED, ActionStatus.SKIPPED]
    assert ok.calls == ["hi"]


# -- verification ---------------------------------------------------------------------------


def test_executed_but_unverifiable(settings: Settings) -> None:
    jarvis, _ = _jarvis(settings, [FlakyTool(failures=0)], _echo(tool="test.flaky"))
    response = jarvis.ask("x")
    assert response.state is TaskState.COMPLETED
    action = response.actions[0]
    assert action.status is ActionStatus.EXECUTED
    assert action.verification is VerificationStatus.UNVERIFIABLE
    assert "no se verificaron" in response.message


def test_verification_failure_fails_the_task(settings: Settings) -> None:
    jarvis, _ = _jarvis(settings, [LyingTool()], _echo(tool="test.liar"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert response.actions[0].status is ActionStatus.FAILED
    assert response.actions[0].verification is VerificationStatus.FAILED
    assert response.failure is not None
    assert response.failure.kind is ErrorKind.VERIFICATION_FAILED


def test_crashing_verifier_counts_as_verification_failure(settings: Settings) -> None:
    jarvis, _ = _jarvis(settings, [RaisingVerifierTool()], _echo(tool="test.raising_verifier"))
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert response.failure is not None
    assert response.failure.kind is ErrorKind.VERIFICATION_FAILED
    assert "verifier crashed" in response.failure.message


def test_external_verifier_can_be_plugged_in(settings: Settings) -> None:
    from jarvis.core.action import VerificationResult
    from jarvis.execution.verification import ToolSelfVerifier, VerificationService

    class Rejecting:
        def verify(self, tool: Any, args: Any, output: Any, ctx: Any) -> VerificationResult:
            return VerificationResult(status=VerificationStatus.FAILED, message="external no")

    service = VerificationService([ToolSelfVerifier(), Rejecting()])
    jarvis, _ = _jarvis(settings, [CountingTool()], _echo(), verification=service)
    response = jarvis.ask("x")
    assert response.state is TaskState.FAILED
    assert response.failure is not None and response.failure.message == "external no"


# -- retry policy units ---------------------------------------------------------------------


def test_retry_policy_backoff_is_bounded() -> None:
    policy = RetryPolicy(
        max_attempts=5, backoff_seconds=1, backoff_multiplier=3, max_backoff_seconds=5
    )
    assert [policy.delay(n) for n in (1, 2, 3)] == [1, 3, 5]


@pytest.mark.parametrize("bad", [{"max_attempts": 0}, {"backoff_multiplier": 0.5}])
def test_retry_policy_rejects_invalid_limits(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**bad)


def test_exception_classification() -> None:
    assert classify_exception(TimeoutError()).failure.kind is ErrorKind.TIMEOUT
    assert classify_exception(ConnectionError()).failure.retryable
    assert classify_exception(FileNotFoundError("x")).failure.kind is ErrorKind.TOOL_ERROR
    assert classify_exception(ValueError("x")).failure.kind is ErrorKind.INTERNAL
    explicit = classify_exception(ToolError(ErrorKind.TRANSIENT, "x", retryable=True))
    assert explicit.explicit and explicit.failure.retryable
