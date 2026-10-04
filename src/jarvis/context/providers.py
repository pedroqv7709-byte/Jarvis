"""Context providers contribute information to the execution context.

This is the extension point for perception (screen, active window, camera, microphone) and memory
(recalled facts, preferences): each becomes a provider. Only a system provider exists today.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from jarvis.context.model import ContextItem
from jarvis.core.ids import utcnow
from jarvis.core.request import Request


@runtime_checkable
class ContextProvider(Protocol):
    @property
    def name(self) -> str: ...

    def provide(self, request: Request) -> Iterable[ContextItem]: ...


class SystemContextProvider:
    """Current time and host platform."""

    name = "system"

    def __init__(self, clock: Callable[[], datetime] = utcnow) -> None:
        self._clock = clock

    def provide(self, request: Request) -> Iterable[ContextItem]:
        now = self._clock()
        return (
            ContextItem(source=self.name, key="current_time", value=now.isoformat()),
            ContextItem(source=self.name, key="platform", value=platform.system()),
        )
