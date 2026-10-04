"""Failure model.

Failures are data (:class:`Failure`) so they can be persisted, audited and returned to the user.
Exceptions (:class:`JarvisError` subclasses) are only used to move failures across call
boundaries inside the process.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorKind(StrEnum):
    VALIDATION = "validation"
    """Arguments or inputs did not satisfy their schema."""
    NOT_FOUND = "not_found"
    """A referenced tool, task, action or target (application, window) does not exist."""
    AMBIGUOUS = "ambiguous"
    """Several targets match; Jarvis never picks one arbitrarily."""
    REQUEST_REJECTED = "request_rejected"
    """The request guard refused the request before planning."""
    PLANNING = "planning"
    """No valid plan could be produced for the goal."""
    POLICY_DENIED = "policy_denied"
    """The policy engine (or a sandbox boundary) denied the action."""
    CONFIRMATION_REJECTED = "confirmation_rejected"
    """The user rejected an action that required confirmation."""
    TOOL_ERROR = "tool_error"
    """The tool ran and reported a non-transient failure."""
    TRANSIENT = "transient"
    """A temporary failure; retrying may succeed."""
    TIMEOUT = "timeout"
    """The operation exceeded its time budget."""
    VERIFICATION_FAILED = "verification_failed"
    """The tool reported success but verification showed the effect did not happen."""
    INTERRUPTED = "interrupted"
    """The process stopped mid-attempt; the outcome of that attempt is unknown."""
    LIMIT_EXCEEDED = "limit_exceeded"
    """A configured safety limit (attempts, steps, iterations, replans) was reached."""
    CYCLE_DETECTED = "cycle_detected"
    """The planner kept proposing the same behaviour."""
    STOPPED = "stopped"
    """The planner decided it cannot safely continue."""
    CANCELLED = "cancelled"
    INTERNAL = "internal"
    """Unexpected error inside Jarvis or a tool."""


class Failure(BaseModel):
    """Structured, serializable description of something that went wrong."""

    model_config = ConfigDict(frozen=True)

    kind: ErrorKind
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class JarvisError(Exception):
    """Base class for exceptions raised inside Jarvis."""


class InvalidTransitionError(JarvisError):
    """A task was asked to move to a state that its state machine does not allow."""


class FailureError(JarvisError):
    """Carries a :class:`Failure` across a call boundary."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ToolError(FailureError):
    """Raised by tools to report a classified failure.

    ``retryable=True`` is the tool's explicit statement that the attempt had no side effects
    and may be repeated; Jarvis only retries non-idempotent tools on that basis.
    """

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            Failure(kind=kind, message=message, retryable=retryable, details=details or {})
        )


class PlanningError(FailureError):
    """Raised by planners and plan validation.

    ``kind`` defaults to PLANNING; a planner may report TIMEOUT when its model did not answer.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        kind: ErrorKind = ErrorKind.PLANNING,
    ) -> None:
        super().__init__(Failure(kind=kind, message=message, details=details or {}))


class ConfigurationError(JarvisError):
    """The configured settings cannot be turned into a working system."""
