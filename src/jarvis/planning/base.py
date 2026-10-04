"""Planner contract.

A planner turns a goal into a :class:`Plan`: an ordered list of actions with explicit
dependencies. Plans are proposals; they confer no authority. Every action still goes through
selection, policy, execution and verification.

The plan shape (ids + ``depends_on``) is what later enables multi-step execution graphs, retries,
replanning, priorities and delegation to sub-agents without changing this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis.context.model import ExecutionContext
from jarvis.core.action import Action
from jarvis.core.agent import RequirementState
from jarvis.core.observation import Observation
from jarvis.core.task import PlanningAttempt


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    planner: str
    actions: tuple[Action, ...] = ()
    reply: str | None = None
    """Direct answer when the goal requires no actions."""
    failed_attempts: tuple[PlanningAttempt, ...] = ()
    """Planners that failed before this plan was produced (set by fallback planning)."""
    metadata: dict[str, str] = Field(default_factory=dict)
    """Operational facts for the audit log (e.g. which model). Never model reasoning."""


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    goal: str
    context: ExecutionContext
    catalog: list[dict[str, Any]] = field(default_factory=list)
    """Descriptions of the tools available (name, description, input_schema)."""


class Planner(Protocol):
    @property
    def name(self) -> str: ...

    def plan(self, request: PlanningRequest) -> Plan:
        """Return a plan or raise :class:`jarvis.core.errors.PlanningError`."""
        ...


# -- incremental (agent loop) planning ------------------------------------------------------


class DecisionKind(StrEnum):
    NEXT_ACTION = "next_action"
    DONE = "done"
    STOP = "stop"


class Decision(BaseModel):
    """A planner's next step. Typed protocol only: never free text.

    ``NEXT_ACTION``: ``actions[0]`` runs next; any further actions are a provisional plan that
    Jarvis executes in order unless something fails (then it asks again). ``final`` declares that
    the task should be done once they all succeed (Jarvis still checks the evidence).
    ``DONE`` is a *claim* that Jarvis verifies before completing. ``STOP`` ends the task safely.
    ``rationale``/``expected_observation`` are short audit summaries, never chain-of-thought.
    """

    model_config = ConfigDict(frozen=True)

    kind: DecisionKind
    planner: str
    actions: tuple[Action, ...] = ()
    final: bool = False
    reply: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=500)
    rationale: str | None = Field(default=None, max_length=300)
    expected_observation: str | None = Field(default=None, max_length=300)
    failed_attempts: tuple[PlanningAttempt, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.kind is DecisionKind.NEXT_ACTION and not self.actions:
            raise ValueError("next_action requires at least one action")
        if self.kind is not DecisionKind.NEXT_ACTION and self.actions:
            raise ValueError(f"{self.kind} must not carry actions")
        return self


class StepSummary(BaseModel):
    """What a planner may know about one earlier action (bounded, redacted, untrusted data)."""

    model_config = ConfigDict(frozen=True)

    action_id: str
    tool: str
    arguments: dict[str, Any]
    status: str
    verification: str | None = None
    failure_kind: str | None = None
    failure_message: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    output_preview: str | None = None
    target: dict[str, Any] | None = None


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    iterations_left: int
    actions_left: int
    replans_left: int
    seconds_left: float


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    goal: str
    context: ExecutionContext
    catalog: list[dict[str, Any]]
    history: tuple[StepSummary, ...] = ()
    observations: tuple[Observation, ...] = ()
    requirements: tuple[RequirementState, ...] = ()
    budget: Budget | None = None
    feedback: str | None = None
    failed_planners: frozenset[str] = frozenset()
    """Planners that already failed in this task (a fallback chain may skip them)."""


@runtime_checkable
class StepPlanner(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, request: DecisionRequest) -> Decision:
        """Return the next decision or raise :class:`jarvis.core.errors.PlanningError`."""
        ...
