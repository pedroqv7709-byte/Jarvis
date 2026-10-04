"""Windows backend: actions through UI Automation (pywinauto), queries through Win32.

Actions (focus, close) use UI Automation via pywinauto. Queries (window list, foreground window,
existence) use plain Win32 calls through ctypes: a different path from the one that performed the
action, which is what verification relies on.

This is the only module that imports pywinauto, lazily, so non-Windows installs and the core never
need it. Each call initialises COM for the current thread because Jarvis runs tool calls on
short-lived worker threads (for timeouts).
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import subprocess
import sys
import threading
import warnings
from ctypes import wintypes
from pathlib import Path
from typing import Any

from jarvis.desktop.automation import DesktopAutomationError, WindowInfo
from jarvis.desktop.catalog import FORBIDDEN_EXECUTABLES

assert sys.platform == "win32"

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi")

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_DWMWA_CLOAKED = 14
_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

_user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetForegroundWindow.restype = wintypes.HWND
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_dwmapi.DwmGetWindowAttribute.argtypes = [
    wintypes.HWND,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
]


def _process_name(pid: int) -> str:
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return Path(buffer.value).name
    finally:
        _kernel32.CloseHandle(handle)


def _is_cloaked(hwnd: int) -> bool:
    """Suspended Store apps keep 'visible' but cloaked (not actually shown) windows."""
    cloaked = wintypes.DWORD(0)
    result = _dwmapi.DwmGetWindowAttribute(
        hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
    )
    return result == 0 and cloaked.value != 0


def _window_info(hwnd: int) -> WindowInfo | None:
    if not _user32.IsWindowVisible(hwnd) or _is_cloaked(hwnd):
        return None
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return None
    title = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, title, length + 1)
    class_name = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, class_name, 256)
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return WindowInfo(
        handle=int(hwnd),
        title=title.value,
        process_id=int(pid.value),
        process_name=_process_name(pid.value),
        class_name=class_name.value,
    )


def _pywinauto() -> Any:
    """Import pywinauto on first use, in STA mode, with its import-time warnings silenced."""
    sys.coinit_flags = 2  # type: ignore[attr-defined]  # COINIT_APARTMENTTHREADED, read by comtypes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("pywinauto")


def _init_com_for_this_thread() -> None:
    comtypes = importlib.import_module("comtypes")
    # Already initialised on this thread (possibly in another mode) is fine: usable as is.
    with contextlib.suppress(OSError):
        comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)


class UiaWindowsAutomation:
    def __init__(self) -> None:
        self._children: list[subprocess.Popen[bytes]] = []
        self._lock = threading.Lock()

    # -- queries (Win32) ----------------------------------------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        found: list[WindowInfo] = []

        def collect(hwnd: int, _: int) -> bool:
            info = _window_info(hwnd)
            if info is not None:
                found.append(info)
            return True

        if not _user32.EnumWindows(_EnumWindowsProc(collect), 0):
            raise DesktopAutomationError(f"EnumWindows failed ({ctypes.get_last_error()})")
        return found

    def foreground_window(self) -> int | None:
        hwnd = _user32.GetForegroundWindow()
        return int(hwnd) if hwnd else None

    def window_exists(self, handle: int) -> bool:
        return bool(_user32.IsWindow(handle)) and _window_info(handle) is not None

    # -- actions ------------------------------------------------------------------------------

    def launch(self, executable: str) -> int:
        path = Path(executable)
        # Defence in depth: the catalog already guarantees all of this.
        if not path.is_absolute() or path.suffix.casefold() != ".exe":
            raise DesktopAutomationError(f"refusing to launch {executable!r}: not an absolute .exe")
        if path.name.casefold() in FORBIDDEN_EXECUTABLES:
            raise DesktopAutomationError(f"refusing to launch {path.name}: forbidden executable")
        if not path.is_file():
            raise DesktopAutomationError(f"{executable} does not exist")
        process = subprocess.Popen(
            [str(path)],  # no arguments, no shell
            cwd=str(path.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            shell=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        with self._lock:
            # Keep references to reap launcher processes that exit (Store apps redirect).
            self._children = [p for p in self._children if p.poll() is None]
            self._children.append(process)
        return process.pid

    def focus(self, handle: int) -> None:
        self._element(handle).set_focus()

    def request_close(self, handle: int) -> None:
        # UI Automation WindowPattern.Close: the same as the user clicking the window's X.
        self._element(handle).close()

    def _element(self, handle: int) -> Any:
        _init_com_for_this_thread()
        desktop = _pywinauto().Desktop(backend="uia")
        try:
            return desktop.window(handle=handle).wrapper_object()
        except Exception as exc:
            raise DesktopAutomationError(f"window {handle} is not available: {exc}") from exc
