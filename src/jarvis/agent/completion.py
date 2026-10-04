"""Completion: "the planner says DONE" versus "Jarvis has verified evidence".

A task may only complete when:

* it has requirements (verifiable conditions derived with confidence from the request, see
  :mod:`jarvis.agent.requirements`): every one was achieved, in order, with evidence - a fresh
  observation of the desktop (window open / in the foreground / gone) or a VERIFIED action whose
  *resolved target* is that application (whatever words its arguments used) - and no action
  failed without that failure being moot (its application's condition holds anyway);
* it has none: the last action attempted succeeded (an earlier failure must be followed by a
  successful step), or no action was needed and the planner gave a reply.

Requirements are checked in order, so "open X and then close it" cannot be satisfied by the
initial state of the desktop alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.action import DONE_STATUSES, ActionRecord, ActionStatus
from jarvis.core.agent import Requirement, RequirementKind, RequirementState
from jarvis.core.errors import FailureError
from jarvis.core.observation import Observation
from jarvis.core.task import Task
from jarvis.desktop.automation import WindowInfo
from jarvis.desktop.catalog import ApplicationCatalog, ApplicationEntry

Normalizer = Callable[[str, dict[str, Any]], dict[str, Any]]

_TOOL_FOR = {
    RequirementKind.APP_OPEN: "windows.open_application",
    RequirementKind.APP_FOCUSED: "windows.focus_window",
    RequirementKind.APP_CLOSED: "windows.close_application",
}


def _identity(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return arguments


@dataclass(frozen=True, slots=True)
class CompletionVerdict:
    accepted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Desktop:
    windows: list[WindowInfo]
    foreground: int | None


class CompletionChecker:
    def __init__(
        self, catalog: ApplicationCatalog | None = None, normalize: Normalizer = _identity
    ) -> None:
        self._catalog = catalog
        # Legacy ACTION_DONE requirements compare arguments as tools see them.
        self._normalize = normalize

    def update(self, task: Task, observations: tuple[Observation, ...]) -> list[RequirementState]:
        """Mark requirements achieved, in order, from fresh evidence. Returns the newly met."""
        desktop = _desktop(observations)
        newly: list[RequirementState] = []
        for state in task.requirements:
            if state.achieved:
                continue
            evidence = self._evidence(state.requirement, task, desktop)
            if evidence is None:
                break
            state.mark(evidence)
            newly.append(state)
        return newly

    def evaluate(self, task: Task, observations: tuple[Observation, ...]) -> CompletionVerdict:
        self.update(task, observations)
        records = [r for r in task.actions if r.status is not ActionStatus.SKIPPED]
        if any(r.status is ActionStatus.AWAITING_CONFIRMATION for r in records):
            return CompletionVerdict(False, "an action is still awaiting confirmation")
        if task.requirements:
            pending = [s.requirement for s in task.requirements if not s.achieved]
            if pending:
                first = pending[0]
                return CompletionVerdict(
                    False, f"requirement not met yet: {first.kind.value} ({first.application})"
                )
            if records and not records[-1].is_done and not self._moot(records[-1], task):
                return CompletionVerdict(False, f"the last action ({records[-1].id}) failed")
            return CompletionVerdict(
                True,
                "all requirements verified",
                {"requirements": [s.evidence for s in task.requirements]},
            )
        if not records:
            if task.reply:
                return CompletionVerdict(True, "no action was needed", {"reply": True})
            return CompletionVerdict(False, "no action was executed and no answer was given")
        last = records[-1]
        if not last.is_done:
            return CompletionVerdict(False, f"the last action ({last.id}) did not succeed")
        return CompletionVerdict(
            True, "the last action succeeded", {"action": last.id, "status": last.status.value}
        )

    # -- evidence -------------------------------------------------------------------------

    def _evidence(
        self, requirement: Requirement, task: Task, desktop: _Desktop | None
    ) -> dict[str, Any] | None:
        if requirement.kind is RequirementKind.ACTION_DONE:  # pre-v0.5 persisted tasks
            record = self._legacy_done_action(requirement, task.actions)
            return None if record is None else {"by": "action", "action_id": record.id}
        entry = self._entry(requirement.application)
        if entry is None:
            return None
        if desktop is not None:
            matching = [w for w in desktop.windows if entry.matches(w)]
            handles = [w.handle for w in matching]
            if requirement.kind is RequirementKind.APP_OPEN and matching:
                return {"by": "observation", "app": entry.app_id, "windows": handles}
            if requirement.kind is RequirementKind.APP_CLOSED and not matching:
                return {"by": "observation", "app": entry.app_id, "windows": []}
            if requirement.kind is RequirementKind.APP_FOCUSED and desktop.foreground in handles:
                return {"by": "observation", "app": entry.app_id, "foreground": desktop.foreground}
            return None
        # Desktop not observable: a VERIFIED action on the same resolved application counts.
        for record in reversed(task.actions):
            if (
                record.status is ActionStatus.VERIFIED
                and record.action.tool == _TOOL_FOR[requirement.kind]
                and self._targets_app(record, entry)
            ):
                return {"by": "verified_action", "app": entry.app_id, "action_id": record.id}
        return None

    @staticmethod
    def _targets_app(record: ActionRecord, entry: ApplicationEntry) -> bool:
        target = record.target or {}
        application = target.get("application")
        if isinstance(application, dict) and application.get("app_id") == entry.app_id:
            return True
        window = target.get("window")
        if isinstance(window, dict):
            info = WindowInfo(0, str(window.get("title", "")), 0, str(window.get("process", "")))
            return entry.matches(info)
        return False

    def _moot(self, record: ActionRecord, task: Task) -> bool:
        """A failed action whose application's condition is verified anyway."""
        achieved = {s.requirement.application for s in task.requirements if s.achieved}
        target = record.target or {}
        application = target.get("application")
        app_id = application.get("app_id") if isinstance(application, dict) else None
        if app_id is None:
            name = record.action.arguments.get("application")
            entry = self._entry(name) if isinstance(name, str) else None
            app_id = entry.app_id if entry is not None else None
        return app_id is not None and app_id in achieved

    def _entry(self, name: str | None) -> ApplicationEntry | None:
        if self._catalog is None or name is None:
            return None
        try:
            return self._catalog.resolve(name)
        except FailureError:
            return None

    def _legacy_done_action(
        self, requirement: Requirement, records: list[ActionRecord]
    ) -> ActionRecord | None:
        template = requirement.action
        wanted = self._normalize(template.tool, template.arguments)
        for record in reversed(records):
            if (
                record.status in DONE_STATUSES
                and record.action.tool == template.tool
                and self._normalize(record.action.tool, record.action.arguments) == wanted
            ):
                return record
        return None


def _desktop(observations: tuple[Observation, ...]) -> _Desktop | None:
    """The latest successful Windows observation, or None if the desktop is not observable."""
    for observation in observations:
        if observation.source == "windows" and observation.status == "ok":
            windows = [
                WindowInfo(int(w["handle"]), str(w["title"]), 0, str(w["process"]))
                for w in observation.data.get("windows", [])
            ]
            foreground = observation.data.get("foreground")
            return _Desktop(windows, int(foreground) if foreground is not None else None)
    return None
