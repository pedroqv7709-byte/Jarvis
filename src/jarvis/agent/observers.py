"""Observers: what the agent loop looks at before each decision.

Observers are read-only. The Windows observer reuses the v0.4 ``WindowsAutomation`` query methods
(the same ones verification uses); there is no second desktop API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from jarvis.core.observation import Observation
from jarvis.desktop.automation import WindowsAutomation


class Observer(Protocol):
    @property
    def name(self) -> str: ...

    def observe(self) -> Observation: ...


class WindowsObserver:
    name = "windows"

    def __init__(self, automation: WindowsAutomation) -> None:
        self._automation = automation

    def observe(self) -> Observation:
        try:
            windows = self._automation.list_windows()
            foreground = self._automation.foreground_window()
        except Exception as exc:
            return Observation(
                source=self.name,
                status="error",
                data={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            )
        return Observation(
            source=self.name,
            status="ok",
            data={
                "windows": [
                    {"handle": w.handle, "title": w.title, "process": w.process_name}
                    for w in windows
                ],
                "foreground": foreground,
                "total": len(windows),
            },
            identifiers=tuple(str(w.handle) for w in windows),
        )


def observe_all(observers: Sequence[Observer]) -> tuple[Observation, ...]:
    return tuple(observer.observe() for observer in observers)
