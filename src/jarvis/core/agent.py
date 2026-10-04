"""Persisted state of the agent loop: progress counters and the task's verifiable requirements.

Requirements are derived deterministically from the user's request (never from a model). They are
what lets Jarvis tell "the planner says DONE" apart from "Jarvis has verified evidence that the
task is done".
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.action import Action
from jarvis.core.ids import utcnow

MAX_TRACKED_PROPOSALS = 50


class AgentPhase(StrEnum):
    PENDING = "pending"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    """An action failed (or completion was refused); the next decision must react to it."""
    WAITING = "waiting"
    FINISHED = "finished"


class AgentProgress(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    phase: AgentPhase = AgentPhase.PENDING
    iterations: int = 0
    """Planner decisions requested."""
    actions_used: int = 0
    """Actions that started executing."""
    replans_used: int = 0
    planning_seconds: float = 0.0
    proposals: list[str] = Field(default_factory=list)
    """Signatures (tool + arguments digest) of proposed actions, for cycle detection."""
    done_rejections: int = 0
    final_batch: bool = False
    """The current planned batch was declared final: when it completes, completion is checked."""
    feedback: str | None = None
    """Operational feedback for the next decision (e.g. why completion was refused)."""
    stop_reason: str | None = None


class RequirementKind(StrEnum):
    APP_OPEN = "app_open"
    APP_FOCUSED = "app_focused"
    APP_CLOSED = "app_closed"
    ACTION_DONE = "action_done"
    """No external condition to observe: satisfied by a successful (verified) matching action."""


class Requirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: RequirementKind
    action: Action
    """The action that achieves it (template from the deterministic grammar)."""
    application: str | None = None


class RequirementState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    requirement: Requirement
    achieved: bool = False
    evidence: dict[str, Any] | None = None
    achieved_at: datetime | None = None

    def mark(self, evidence: dict[str, Any]) -> None:
        self.achieved = True
        self.evidence = evidence
        self.achieved_at = utcnow()
