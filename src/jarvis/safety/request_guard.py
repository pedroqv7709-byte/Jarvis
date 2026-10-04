"""Request-level checks that run before any planning happens."""

from __future__ import annotations

from jarvis.core.errors import ErrorKind, Failure
from jarvis.core.request import Request


class RequestGuard:
    def __init__(self, max_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars

    def check(self, request: Request) -> Failure | None:
        text = request.text
        if not text.strip():
            return Failure(kind=ErrorKind.REQUEST_REJECTED, message="the request is empty")
        if len(text) > self._max_chars:
            return Failure(
                kind=ErrorKind.REQUEST_REJECTED,
                message=f"the request exceeds {self._max_chars} characters",
                details={"length": len(text)},
            )
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in text):
            return Failure(
                kind=ErrorKind.REQUEST_REJECTED, message="the request contains control characters"
            )
        return None
