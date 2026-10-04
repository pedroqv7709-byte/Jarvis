"""Windows desktop tools: explicit, limited capabilities over an allowlist.

* ``windows.get_windows``       - list visible windows (read only).
* ``windows.open_application``  - launch an *allowlisted* application by name.
* ``windows.focus_window``      - bring exactly one matching window to the foreground.
* ``windows.close_application`` - ask exactly one window of an allowlisted app to close normally.

There is deliberately no tool that runs a command line. Every tool resolves its concrete target
first (``resolve_target``: read only, audited, shown in confirmations and bound into approvals);
``run`` acts only on that target, and ``verify`` re-queries the system instead of trusting the
action's own report. Ambiguous criteria are an error, never a guess. Closing never forces: if the
application asks to save changes, the window stays and verification reports it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Self

from pydantic import Field, model_validator

from jarvis.core.action import VerificationResult, VerificationStatus
from jarvis.core.categories import ActionCategory
from jarvis.core.errors import ErrorKind, ToolError
from jarvis.desktop.automation import DesktopAutomationError, WindowInfo, WindowsAutomation
from jarvis.desktop.catalog import (
    APPLICATION_NAME_MAX,
    APPLICATION_NAME_PATTERN,
    ApplicationCatalog,
    ApplicationEntry,
)
from jarvis.tools.base import Tool, ToolArguments, ToolContext, ToolSpec

MAX_LISTED_WINDOWS = 50
_POLL_SECONDS = 0.2

Clock = Callable[[], float]
Sleep = Callable[[float], None]


class _WindowsTool[ArgsT: ToolArguments](Tool[ArgsT]):
    def __init__(
        self,
        automation: WindowsAutomation,
        catalog: ApplicationCatalog,
        *,
        wait_seconds: float = 10.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._automation = automation
        self._catalog = catalog
        self._wait = wait_seconds
        self._clock = clock
        self._sleep = sleep

    def input_schema(self) -> dict[str, Any]:
        """Tell planners which names the allowlist accepts (catalog data only; the tool still
        refuses anything that does not resolve)."""
        schema = super().input_schema()
        properties = schema.get("properties", {})
        field = properties.get("application")
        if not isinstance(field, dict):
            return schema
        names = ", ".join(sorted(entry.app_id for entry in self._catalog.entries()))
        described = f"{field.get('description', 'Allowlisted application name.')} Allowed: {names}."
        return {
            **schema,
            "properties": {**properties, "application": {**field, "description": described}},
        }

    def _poll[T](self, probe: Callable[[], T | None]) -> T | None:
        """Re-query until ``probe`` returns a value or the wait budget runs out."""
        deadline = self._clock() + self._wait
        while True:
            value = probe()
            if value is not None or self._clock() >= deadline:
                return value
            self._sleep(_POLL_SECONDS)


def _act[T](action: Callable[[], T], what: str) -> T:
    """Backend failures become classified, non-retryable tool errors."""
    try:
        return action()
    except DesktopAutomationError as exc:
        raise ToolError(ErrorKind.TOOL_ERROR, f"{what} failed: {exc}") from exc


def _summary(window: WindowInfo) -> dict[str, Any]:
    return {"handle": window.handle, "title": window.title, "process": window.process_name}


def _target_window(target: Mapping[str, Any] | None) -> dict[str, Any]:
    if not target or "window" not in target:
        raise ToolError(ErrorKind.INTERNAL, "no resolved window target")
    window: dict[str, Any] = target["window"]
    return window


# -- window selection -----------------------------------------------------------------------


class WindowCriteria(ToolArguments):
    application: str | None = Field(
        default=None,
        pattern=APPLICATION_NAME_PATTERN,
        max_length=APPLICATION_NAME_MAX,
        description="Allowlisted application name (e.g. 'notepad', 'calculator').",
    )
    title_contains: str | None = Field(default=None, min_length=1, max_length=200)
    process: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]{1,64}\.exe$",
        description="Executable name of the owning process, e.g. 'Notepad.exe'.",
    )
    handle: int | None = Field(default=None, ge=1, description="Exact window handle.")

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if not any((self.application, self.title_contains, self.process, self.handle)):
            raise ValueError("give at least one of application, title_contains, process, handle")
        return self


def select_windows(
    windows: list[WindowInfo], criteria: WindowCriteria, catalog: ApplicationCatalog
) -> list[WindowInfo]:
    entry = catalog.resolve(criteria.application) if criteria.application else None
    selected = []
    for window in windows:
        if entry is not None and not entry.matches(window):
            continue
        if criteria.title_contains and criteria.title_contains.casefold() not in (
            window.title.casefold()
        ):
            continue
        if criteria.process and criteria.process.casefold() != window.process_name.casefold():
            continue
        if criteria.handle is not None and criteria.handle != window.handle:
            continue
        selected.append(window)
    return selected


def resolve_single_window(
    automation: WindowsAutomation, criteria: WindowCriteria, catalog: ApplicationCatalog
) -> WindowInfo:
    candidates = select_windows(automation.list_windows(), criteria, catalog)
    requested = criteria.model_dump(exclude_none=True)
    if not candidates:
        raise ToolError(
            ErrorKind.NOT_FOUND, "no window matches the criteria", details={"requested": requested}
        )
    if len(candidates) > 1:
        raise ToolError(
            ErrorKind.AMBIGUOUS,
            f"{len(candidates)} windows match; refine the criteria (e.g. with 'handle')",
            details={
                "requested": requested,
                "candidates": [_summary(w) for w in candidates[:MAX_LISTED_WINDOWS]],
            },
        )
    return candidates[0]


def _still_same_window(automation: WindowsAutomation, window: Mapping[str, Any]) -> WindowInfo:
    """Act only on the window that was resolved (and approved): same handle, same process."""
    for current in automation.list_windows():
        if current.handle == window["handle"]:
            if current.process_name.casefold() != str(window["process"]).casefold():
                break
            return current
    raise ToolError(
        ErrorKind.NOT_FOUND,
        "the resolved window no longer exists (or now belongs to another process)",
        details={"window": dict(window)},
    )


# -- get_windows ----------------------------------------------------------------------------


class GetWindowsArgs(ToolArguments):
    application: str | None = Field(
        default=None, pattern=APPLICATION_NAME_PATTERN, max_length=APPLICATION_NAME_MAX
    )
    title_contains: str | None = Field(default=None, min_length=1, max_length=200)


class GetWindowsTool(_WindowsTool[GetWindowsArgs]):
    spec = ToolSpec(
        name="windows.get_windows",
        description="List visible desktop windows (title, process, handle). Read only.",
        required_permissions=frozenset({"windows.read"}),
        categories=frozenset({ActionCategory.READ}),
        side_effects=False,
        idempotent=True,
    )
    args_model = GetWindowsArgs

    def run(self, args: GetWindowsArgs, ctx: ToolContext) -> dict[str, Any]:
        windows = self._automation.list_windows()
        if args.application or args.title_contains:
            criteria = WindowCriteria(
                application=args.application, title_contains=args.title_contains
            )
            windows = select_windows(windows, criteria, self._catalog)
        foreground = self._automation.foreground_window()
        return {
            "windows": [
                {**_summary(w), "foreground": w.handle == foreground}
                for w in windows[:MAX_LISTED_WINDOWS]
            ],
            "total": len(windows),
        }


# -- open_application -----------------------------------------------------------------------


class OpenApplicationArgs(ToolArguments):
    application: str = Field(
        pattern=APPLICATION_NAME_PATTERN,
        max_length=APPLICATION_NAME_MAX,
        description="Name of an allowlisted application (e.g. 'notepad', 'calculator'). "
        "Never a path or a command.",
    )


class OpenApplicationTool(_WindowsTool[OpenApplicationArgs]):
    spec = ToolSpec(
        name="windows.open_application",
        description="Open an allowlisted Windows application by name. Unknown names are refused.",
        required_permissions=frozenset({"windows.open"}),
        categories=frozenset({ActionCategory.OPEN_APPLICATION}),
        side_effects=True,
        idempotent=False,
    )
    args_model = OpenApplicationArgs

    def resolve_target(self, args: OpenApplicationArgs, ctx: ToolContext) -> dict[str, Any]:
        entry = self._catalog.resolve(args.application)
        existing = [w.handle for w in self._automation.list_windows() if entry.matches(w)]
        return {"application": entry.describe(), "preexisting_windows": existing}

    def _entry(self, ctx: ToolContext) -> ApplicationEntry:
        if not ctx.target:
            raise ToolError(ErrorKind.INTERNAL, "no resolved application target")
        # Re-resolve by id: only catalog data is ever launched, never text from the plan.
        return self._catalog.resolve(ctx.target["application"]["app_id"])

    def run(self, args: OpenApplicationArgs, ctx: ToolContext) -> dict[str, Any]:
        entry = self._entry(ctx)
        assert ctx.target is not None
        preexisting = set(ctx.target["preexisting_windows"])
        pid = _act(lambda: self._automation.launch(entry.executable), f"launching {entry.app_id}")

        def new_window() -> WindowInfo | None:
            return next(
                (
                    w
                    for w in self._automation.list_windows()
                    if entry.matches(w) and w.handle not in preexisting
                ),
                None,
            )

        window = self._poll(new_window)
        return {
            "launched_pid": pid,
            "window": _summary(window) if window else None,
            "new_window": window is not None,
        }

    def verify(
        self, args: OpenApplicationArgs, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        entry = self._entry(ctx)
        assert ctx.target is not None
        preexisting = set(ctx.target["preexisting_windows"])
        matching = [w for w in self._automation.list_windows() if entry.matches(w)]
        if not matching:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=f"no {entry.display_name} window is open after launching it",
                evidence={"launched_pid": output.get("launched_pid")},
            )
        new = [w for w in matching if w.handle not in preexisting]
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            message=(
                f"{entry.display_name} window found"
                if new
                else f"{entry.display_name} is open (an existing window was reused)"
            ),
            evidence={
                "windows": [_summary(w) for w in (new or matching)],
                "new_window": bool(new),
            },
        )


# -- focus_window ---------------------------------------------------------------------------


class FocusWindowTool(_WindowsTool[WindowCriteria]):
    spec = ToolSpec(
        name="windows.focus_window",
        description="Bring one window to the foreground. Criteria must match exactly one window.",
        required_permissions=frozenset({"windows.focus"}),
        categories=frozenset({ActionCategory.WINDOW_MANAGEMENT}),
        side_effects=True,
        idempotent=True,
    )
    args_model = WindowCriteria

    def resolve_target(self, args: WindowCriteria, ctx: ToolContext) -> dict[str, Any]:
        window = resolve_single_window(self._automation, args, self._catalog)
        return {"window": _summary(window)}

    def run(self, args: WindowCriteria, ctx: ToolContext) -> dict[str, Any]:
        window = _target_window(ctx.target)
        _still_same_window(self._automation, window)
        _act(lambda: self._automation.focus(window["handle"]), "focusing the window")
        return {"focus_requested": window["handle"]}

    def verify(
        self, args: WindowCriteria, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        window = _target_window(ctx.target)
        foreground = self._poll(
            lambda: h if (h := self._automation.foreground_window()) == window["handle"] else None
        )
        if foreground == window["handle"]:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                message="the window is in the foreground",
                evidence={"foreground": window["handle"]},
            )
        actual = self._automation.foreground_window()
        current = next((w for w in self._automation.list_windows() if w.handle == actual), None)
        return VerificationResult(
            status=VerificationStatus.FAILED,
            message="another window is in the foreground",
            evidence={
                "expected": window["handle"],
                "foreground": _summary(current) if current else actual,
            },
        )


# -- close_application ----------------------------------------------------------------------


class CloseApplicationArgs(WindowCriteria):
    application: str = Field(
        pattern=APPLICATION_NAME_PATTERN,
        max_length=APPLICATION_NAME_MAX,
        description="Allowlisted application whose window to close (required).",
    )


class CloseApplicationTool(_WindowsTool[CloseApplicationArgs]):
    spec = ToolSpec(
        name="windows.close_application",
        description="Ask one window of an allowlisted application to close normally (like "
        "clicking X). Never forces; the application may ask to save changes.",
        required_permissions=frozenset({"windows.close"}),
        categories=frozenset({ActionCategory.CLOSE_APPLICATION}),
        side_effects=True,
        idempotent=False,
    )
    args_model = CloseApplicationArgs

    def resolve_target(self, args: CloseApplicationArgs, ctx: ToolContext) -> dict[str, Any]:
        window = resolve_single_window(self._automation, args, self._catalog)
        entry = self._catalog.resolve(args.application)
        return {
            "application": entry.describe(),
            "window": _summary(window),
            # Many editors mark unsaved documents with a leading '*'. A hint for the user only;
            # closing is always confirmed and never forced.
            "unsaved_changes_suspected": window.title.lstrip().startswith("*"),
        }

    def run(self, args: CloseApplicationArgs, ctx: ToolContext) -> dict[str, Any]:
        window = _target_window(ctx.target)
        _still_same_window(self._automation, window)
        _act(lambda: self._automation.request_close(window["handle"]), "closing the window")
        gone = self._poll(
            lambda: True if not self._automation.window_exists(window["handle"]) else None
        )
        return {"close_requested": window["handle"], "closed_within_wait": bool(gone)}

    def verify(
        self, args: CloseApplicationArgs, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        window = _target_window(ctx.target)
        if not self._automation.window_exists(window["handle"]):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                message="the window no longer exists",
                evidence={"closed": window["handle"]},
            )
        same_process = [
            _summary(w)
            for w in self._automation.list_windows()
            if w.process_name.casefold() == str(window["process"]).casefold()
        ]
        return VerificationResult(
            status=VerificationStatus.FAILED,
            message="the window is still open (it may be asking to save changes); "
            "nothing was forced",
            evidence={"window": window, "windows_of_process": same_process[:10]},
        )


def windows_tools(
    automation: WindowsAutomation,
    catalog: ApplicationCatalog,
    *,
    open_wait_seconds: float = 10.0,
    close_wait_seconds: float = 5.0,
    clock: Clock = time.monotonic,
    sleep: Sleep = time.sleep,
) -> list[Tool[Any]]:
    common: dict[str, Any] = {"clock": clock, "sleep": sleep}
    return [
        GetWindowsTool(automation, catalog, wait_seconds=0, **common),
        OpenApplicationTool(automation, catalog, wait_seconds=open_wait_seconds, **common),
        FocusWindowTool(automation, catalog, wait_seconds=min(2.0, open_wait_seconds), **common),
        CloseApplicationTool(automation, catalog, wait_seconds=close_wait_seconds, **common),
    ]
