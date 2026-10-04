"""Error classification and bounded retry policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.core.errors import ErrorKind, Failure, FailureError

DEFAULT_RETRYABLE = frozenset({ErrorKind.TRANSIENT, ErrorKind.TIMEOUT})


@dataclass(frozen=True, slots=True)
class ClassifiedFailure:
    failure: Failure
    explicit: bool
    """True when the tool itself classified the failure (raised a ToolError)."""


def classify_exception(exc: BaseException) -> ClassifiedFailure:
    if isinstance(exc, FailureError):
        return ClassifiedFailure(exc.failure, explicit=True)
    name = type(exc).__name__
    if isinstance(exc, TimeoutError):
        failure = Failure(kind=ErrorKind.TIMEOUT, message=str(exc) or name, retryable=True)
    elif isinstance(exc, ConnectionError):
        failure = Failure(kind=ErrorKind.TRANSIENT, message=str(exc) or name, retryable=True)
    elif isinstance(exc, OSError):
        failure = Failure(kind=ErrorKind.TOOL_ERROR, message=f"{name}: {exc}")
    else:
        failure = Failure(kind=ErrorKind.INTERNAL, message=f"{name}: {exc}")
    return ClassifiedFailure(failure, explicit=False)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 5.0
    retryable_kinds: frozenset[ErrorKind] = field(default=DEFAULT_RETRYABLE)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")

    def should_retry(self, failure: ClassifiedFailure, attempt: int, *, idempotent: bool) -> bool:
        """Retry only bounded, retryable failures.

        Non-idempotent tools are retried only when the tool itself declared the failure
        retryable, i.e. asserted that the failed attempt had no side effects.
        """
        if attempt >= self.max_attempts:
            return False
        f = failure.failure
        if not f.retryable or f.kind not in self.retryable_kinds:
            return False
        return idempotent or failure.explicit

    def delay(self, attempt: int) -> float:
        raw = self.backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        return float(min(self.max_backoff_seconds, raw))
