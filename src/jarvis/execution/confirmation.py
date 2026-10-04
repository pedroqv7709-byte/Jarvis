"""User confirmation for actions the policy marks REQUIRE_CONFIRMATION.

Handlers can approve or reject immediately (an interactive prompt) or defer, in which case the
task waits in WAITING_PERMISSION until an explicit decision arrives through the interface.
No handler is allowed to approve silently on the user's behalf; the default defers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from jarvis.context.model import ExecutionContext
from jarvis.core.action import ConfirmationRequest


class ConfirmationOutcome(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ConfirmationHandler(Protocol):
    @property
    def name(self) -> str: ...

    def request(
        self, request: ConfirmationRequest, context: ExecutionContext
    ) -> ConfirmationOutcome: ...


class DeferringConfirmationHandler:
    """Never decides; the task waits for an explicit approve/reject."""

    name = "deferred"

    def request(
        self, request: ConfirmationRequest, context: ExecutionContext
    ) -> ConfirmationOutcome:
        return ConfirmationOutcome.DEFERRED
