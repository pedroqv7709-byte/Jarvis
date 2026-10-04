"""Requirement extraction: only conditions Jarvis can verify with confidence.

The user's *intent* is what a planner works towards; a *requirement* is a condition derived from
the request that Jarvis can check against evidence before accepting completion. Not every request
produces requirements. One is created only when the request names, unambiguously, a target Jarvis
can resolve and observe: an allowlisted application that must be open, in the foreground, or
closed. Anything else ("enfoca ese", unknown names, titles, files) yields no requirement, so a
mis-reading of the language can never become an obligation that later blocks completion; those
tasks complete on the evidence of their verified actions instead.
"""

from __future__ import annotations

import re

from jarvis.context.model import ExecutionContext
from jarvis.core.agent import Requirement, RequirementKind, RequirementState
from jarvis.core.errors import FailureError, PlanningError
from jarvis.desktop.catalog import ApplicationCatalog
from jarvis.planning.base import PlanningRequest
from jarvis.planning.rule_based import RuleBasedPlanner

# A sentence's closing full stop ("Abre la calculadora y enfócala.") never names anything, but the
# rule-based parser would not recognise the last clause with it attached.
_FINAL_STOP = re.compile(r"[.\s]+$")

_KINDS = {
    "windows.open_application": RequirementKind.APP_OPEN,
    "windows.focus_window": RequirementKind.APP_FOCUSED,
    "windows.close_application": RequirementKind.APP_CLOSED,
}


class RequirementExtractor:
    def __init__(self, catalog: ApplicationCatalog | None) -> None:
        self._catalog = catalog

    def __call__(self, goal: str, context: ExecutionContext) -> list[RequirementState]:
        if self._catalog is None:
            return []
        try:
            goal = _FINAL_STOP.sub("", goal)
            plan = RuleBasedPlanner().plan(PlanningRequest(goal=goal, context=context))
        except PlanningError:
            return []  # not understood, or an ambiguous reference: nothing is invented
        states = []
        for step in plan.actions:
            kind = _KINDS.get(step.tool)
            name = step.arguments.get("application")
            if kind is None or not isinstance(name, str) or set(step.arguments) != {"application"}:
                continue
            try:
                entry = self._catalog.resolve(name)
            except FailureError:
                continue  # not an allowlisted application: not verifiable
            states.append(
                RequirementState(
                    requirement=Requirement(kind=kind, action=step, application=entry.app_id)
                )
            )
        return states
