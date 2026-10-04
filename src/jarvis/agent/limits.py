"""Hard limits of the agent loop and cycle detection. Independent of any planner or model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.core.action import Action, arguments_digest


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_iterations: int = 20
    """Planner decisions per task."""
    max_actions: int = 20
    """Actions that may start executing per task."""
    max_replans: int = 5
    """Decisions made in reaction to a failure (or a refused completion)."""
    task_timeout_seconds: float = 300.0
    """Global budget: tool execution + backoff + planning. Waiting for the user is excluded."""
    history_steps: int = 10
    """Earlier actions shown to the planner."""
    max_observed_windows: int = 30
    max_prompt_chars: int = 12_000
    """Upper bound for the serialized context sent to a model."""

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_actions", "history_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        if self.task_timeout_seconds <= 0 or self.max_prompt_chars < 1000:
            raise ValueError("invalid agent limits")


def action_signature(action: Action) -> str:
    """What an action would do, independent of its id (tool + exact arguments)."""
    return arguments_digest(action.tool, action.arguments)


def detect_cycle(previous: Sequence[str], new: str) -> str | None:
    """Deterministic detection of obvious loops in the sequence of proposed actions."""
    sequence = [*previous, new]
    repeats = sequence.count(new)
    if repeats >= 3:
        return f"the same action was proposed {repeats} times"
    alternating = (
        len(sequence) >= 4
        and sequence[-1] == sequence[-3]
        and sequence[-2] == sequence[-4]
        and sequence[-1] != sequence[-2]
    )
    if alternating:
        return "the planner alternates between the same two actions (A -> B -> A -> B)"
    return None
