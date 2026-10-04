"""The desktop automation contract that Windows tools depend on.

Jarvis -> WindowsAutomation (this protocol) -> concrete backend (UI Automation via pywinauto,
or a fake in tests). Nothing outside ``jarvis.desktop.uia`` imports pywinauto.

The protocol separates *actions* (launch, focus, request_close) from *queries* (list_windows,
foreground_window, window_exists). Verification only ever uses queries, so "the action said it
worked" and "the system shows it worked" stay distinct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from jarvis.core.errors import JarvisError


class DesktopAutomationError(JarvisError):
    """The desktop backend failed or is unavailable."""


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    title: str
    process_id: int
    process_name: str
    """Executable file name of the owning process, e.g. ``Notepad.exe``."""
    class_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WindowsAutomation(Protocol):
    # -- queries (used by resolution and verification) ----------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        """Visible, titled top-level windows, as the system reports them right now."""
        ...

    def foreground_window(self) -> int | None:
        """Handle of the window that currently has the foreground, if any."""
        ...

    def window_exists(self, handle: int) -> bool:
        """True while the window still exists and is shown."""
        ...

    # -- actions ------------------------------------------------------------------------------

    def launch(self, executable: str) -> int:
        """Start an allowlisted executable (absolute path, no arguments, no shell). Returns pid."""
        ...

    def focus(self, handle: int) -> None: ...

    def request_close(self, handle: int) -> None:
        """Ask the window to close normally (as the user would). Never terminates a process."""
        ...
