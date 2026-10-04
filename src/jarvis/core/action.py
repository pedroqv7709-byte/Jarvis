"""Actions and their execution records.

An :class:`Action` is an *intent* produced by a planner: "call tool X with arguments Y". It carries
no authority. Everything that happens to it afterwards is recorded on its :class:`ActionRecord`,
whose lifecycle keeps four concepts separate:

* selected   - the tool exists and the arguments validated against its schema;
* authorized - the policy engine (and the user, when required) allowed it;
* executed   - the tool ran and reported success;
* verified   - an independent check confirmed the effect actually happened.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.decisions import PolicyVerdict
from jarvis.core.errors import Failure
from jarvis.core.ids import utcnow


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    depends_on: tuple[str, ...] = ()

    def digest(self, target: dict[str, Any] | None = None) -> str:
        """Stable hash of what would be executed (tool + arguments [+ resolved target]).

        Confirmations are bound to this digest so an approval cannot be reused for different
        arguments, nor for a different concrete target (e.g. another window).
        """
        return arguments_digest(self.tool, self.arguments, target)


def arguments_digest(
    tool: str, arguments: dict[str, Any], target: dict[str, Any] | None = None
) -> str:
    payload: dict[str, Any] = {"tool": tool, "arguments": arguments}
    if target is not None:  # absent for tools without target resolution: digests unchanged
        payload["target"] = target
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ActionStatus(StrEnum):
    PLANNED = "planned"
    SELECTED = "selected"
    AUTHORIZED = "authorized"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTED = "executed"
    """Ran successfully but no verifier exists for this tool."""
    VERIFIED = "verified"
    DENIED = "denied"
    REJECTED = "rejected"
    """The user rejected the confirmation request."""
    FAILED = "failed"
    SKIPPED = "skipped"
    """Not run because the task stopped before reaching it."""


DONE_STATUSES = frozenset({ActionStatus.EXECUTED, ActionStatus.VERIFIED})
STOPPED_STATUSES = frozenset(
    {ActionStatus.DENIED, ActionStatus.REJECTED, ActionStatus.FAILED, ActionStatus.SKIPPED}
)


class AttemptStatus(StrEnum):
    STARTED = "started"
    """Checkpointed before the tool runs. Left in this state only if the process died."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActionResult(BaseModel):
    """Outcome of one execution attempt of an action."""

    model_config = ConfigDict(validate_assignment=True)

    action_id: str
    attempt: int = Field(ge=1)
    status: AttemptStatus = AttemptStatus.STARTED
    output: dict[str, Any] | None = None
    failure: Failure | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"
    """The tool has no verifier; the execution result is reported as-is."""


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: VerificationStatus
    message: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=utcnow)


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: str
    tool: str
    description: str
    arguments: dict[str, Any]
    arguments_digest: str
    reasons: tuple[str, ...]
    target: dict[str, Any] | None = None
    """The concrete target resolved by Jarvis (e.g. the exact window) that the user approves."""
    requested_at: datetime = Field(default_factory=utcnow)


class ConfirmationDecision(BaseModel):
    """A user's answer to a :class:`ConfirmationRequest`. Approvals are single-use."""

    model_config = ConfigDict(validate_assignment=True)

    approved: bool
    decided_by: str
    arguments_digest: str
    decided_at: datetime = Field(default_factory=utcnow)
    consumed: bool = False


class ActionRecord(BaseModel):
    """Everything Jarvis knows about one planned action. Persisted with the task."""

    model_config = ConfigDict(validate_assignment=True)

    action: Action
    status: ActionStatus = ActionStatus.PLANNED
    verdict: PolicyVerdict | None = None
    confirmation: ConfirmationRequest | None = None
    confirmation_decision: ConfirmationDecision | None = None
    target: dict[str, Any] | None = None
    """What Jarvis resolved the request to (kept apart from what the planner asked for)."""
    attempts: list[ActionResult] = Field(default_factory=list)
    verification: VerificationResult | None = None
    failure: Failure | None = None

    @property
    def id(self) -> str:
        return self.action.id

    @property
    def is_done(self) -> bool:
        return self.status in DONE_STATUSES

    @property
    def output(self) -> dict[str, Any] | None:
        for attempt in reversed(self.attempts):
            if attempt.status is AttemptStatus.SUCCEEDED:
                return attempt.output
        return None
