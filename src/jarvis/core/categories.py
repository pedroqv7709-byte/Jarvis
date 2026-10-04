"""Categories describe *what kind of effect* an action has, independently of the tool.

Tools declare their categories; the safety layer maps categories to minimum decisions (floors).
New capabilities (browser, Windows, messaging) reuse these categories, so their safety treatment
is decided by policy, not by each tool's author.
"""

from __future__ import annotations

from enum import StrEnum


class ActionCategory(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    OPEN_APPLICATION = "open_application"
    WINDOW_MANAGEMENT = "window_management"
    """Changes which window is focused/visible; no data is modified."""
    CLOSE_APPLICATION = "close_application"
    """May discard unsaved work in the application."""
    SEND_MESSAGE = "send_message"
    PURCHASE = "purchase"
    TRANSFER_MONEY = "transfer_money"
    CHANGE_CREDENTIALS = "change_credentials"
    SYSTEM_CONFIGURATION = "system_configuration"
    DANGEROUS = "dangerous"
    """Irreversible or destructive at system level; never executed."""
