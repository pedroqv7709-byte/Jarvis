"""Desktop automation behind a replaceable interface (Windows UI Automation today)."""

import importlib.util
import sys
from typing import Literal

from jarvis.desktop.automation import DesktopAutomationError, WindowInfo, WindowsAutomation
from jarvis.desktop.catalog import ApplicationCatalog, ApplicationConfig, build_catalog

WindowsMode = Literal["auto", "off"]


def windows_automation_available() -> bool:
    """Windows and the optional ``windows`` extra installed. Does not import pywinauto."""
    return sys.platform == "win32" and importlib.util.find_spec("pywinauto") is not None


def create_windows_automation(mode: WindowsMode) -> WindowsAutomation | None:
    if mode == "off" or not windows_automation_available():
        return None
    from jarvis.desktop.uia import UiaWindowsAutomation

    return UiaWindowsAutomation()


__all__ = [
    "ApplicationCatalog",
    "ApplicationConfig",
    "DesktopAutomationError",
    "WindowInfo",
    "WindowsAutomation",
    "WindowsMode",
    "build_catalog",
    "create_windows_automation",
    "windows_automation_available",
]
