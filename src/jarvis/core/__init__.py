"""Domain model: requests, tasks, actions, decisions and failures.

This package depends on nothing else inside Jarvis; every other package depends on it.
"""

from jarvis.core.action import (
    Action,
    ActionRecord,
    ActionResult,
    ActionStatus,
    AttemptStatus,
    ConfirmationDecision,
    ConfirmationRequest,
    VerificationResult,
    VerificationStatus,
)
from jarvis.core.categories import ActionCategory
from jarvis.core.decisions import Decision, PolicyVerdict, RuleOutcome
from jarvis.core.errors import (
    ConfigurationError,
    ErrorKind,
    Failure,
    FailureError,
    InvalidTransitionError,
    JarvisError,
    PlanningError,
    ToolError,
)
from jarvis.core.request import Request
from jarvis.core.response import ActionSummary, Response
from jarvis.core.task import PlanningAttempt, Task, TaskState, TaskTransition

__all__ = [
    "Action",
    "ActionCategory",
    "ActionRecord",
    "ActionResult",
    "ActionStatus",
    "ActionSummary",
    "AttemptStatus",
    "ConfigurationError",
    "ConfirmationDecision",
    "ConfirmationRequest",
    "Decision",
    "ErrorKind",
    "Failure",
    "FailureError",
    "InvalidTransitionError",
    "JarvisError",
    "PlanningAttempt",
    "PlanningError",
    "PolicyVerdict",
    "Request",
    "Response",
    "RuleOutcome",
    "Task",
    "TaskState",
    "TaskTransition",
    "ToolError",
    "VerificationResult",
    "VerificationStatus",
]
