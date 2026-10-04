"""Allowlist of applications Jarvis may open, close or identify.

A request names an application ("notepad", "Calculadora"); Jarvis maps it through this catalog to
an executable it already knows. Nothing the planner (or a model) writes is ever executed as a
command: unknown names are refused, there are no arguments, no shell, and shells / script hosts can
never be allowlisted, not even by user configuration.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.core.errors import ErrorKind, ToolError
from jarvis.desktop.automation import WindowInfo

# What a request may look like: a short name, never a path or a command line. Words separated by
# single spaces, each starting with a letter or digit: no path separators, quotes, shell
# operators, and nothing shaped like a command-line flag (" -x", " /c").
APPLICATION_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(?: [A-Za-z0-9][A-Za-z0-9._-]*)*$"
APPLICATION_NAME_MAX = 40

# Interpreters, shells and script/binary hosts: never launchable through Jarvis.
FORBIDDEN_EXECUTABLES = frozenset(
    {
        "cmd.exe",
        "powershell.exe",
        "powershell_ise.exe",
        "pwsh.exe",
        "wt.exe",
        "conhost.exe",
        "bash.exe",
        "wsl.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        "rundll32.exe",
        "regsvr32.exe",
        "regedit.exe",
        "reg.exe",
        "wmic.exe",
        "msiexec.exe",
        "schtasks.exe",
        "sc.exe",
        "taskkill.exe",
        "certutil.exe",
        "bitsadmin.exe",
        "installutil.exe",
        "msbuild.exe",
        "python.exe",
        "pythonw.exe",
        "py.exe",
        "node.exe",
        "java.exe",
        "javaw.exe",
    }
)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


@dataclass(frozen=True, slots=True)
class ApplicationEntry:
    app_id: str
    display_name: str
    executable: str
    """Absolute path of the executable to launch."""
    process_names: frozenset[str]
    """Lower-case executable names whose windows belong to this application."""
    title_pattern: str | None = None
    """Required for hosts shared by many apps (e.g. ApplicationFrameHost.exe)."""
    aliases: frozenset[str] = field(default_factory=frozenset)

    def matches(self, window: WindowInfo) -> bool:
        if window.process_name.casefold() not in self.process_names:
            return False
        if self.title_pattern is None:
            return True
        return re.search(self.title_pattern, window.title, re.IGNORECASE) is not None

    def describe(self) -> dict[str, str]:
        return {
            "app_id": self.app_id,
            "display_name": self.display_name,
            "executable": self.executable,
        }


class ApplicationConfig(BaseModel):
    """User-provided allowlist entry (``JARVIS_WINDOWS_APPLICATIONS``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: str = Field(min_length=4, max_length=260)
    display_name: str | None = Field(default=None, max_length=60)
    process_names: list[str] = Field(default_factory=list, max_length=10)
    title_pattern: str | None = Field(default=None, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("executable")
    @classmethod
    def _safe_executable(cls, value: str) -> str:
        expanded = os.path.expandvars(value)
        if any(ch in expanded for ch in '"<>|&^%*?'):
            raise ValueError("executable contains shell, wildcard or unexpanded variable chars")
        path = PureWindowsPath(expanded)
        if not path.is_absolute() or path.suffix.casefold() != ".exe":
            raise ValueError("executable must be an absolute path to an .exe file")
        if path.name.casefold() in FORBIDDEN_EXECUTABLES:
            raise ValueError(f"{path.name} can never be allowlisted (shell or script host)")
        return value

    @field_validator("title_pattern")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class ApplicationCatalog:
    def __init__(self, entries: Iterable[ApplicationEntry]) -> None:
        self._entries: dict[str, ApplicationEntry] = {}
        self._names: dict[str, str] = {}
        for entry in entries:
            if PureWindowsPath(entry.executable).name.casefold() in FORBIDDEN_EXECUTABLES:
                raise ValueError(f"{entry.executable} can never be allowlisted")
            self._entries[entry.app_id] = entry
            for name in {entry.app_id, entry.display_name, *entry.aliases}:
                self._names[normalize_name(name)] = entry.app_id

    def entries(self) -> list[ApplicationEntry]:
        return list(self._entries.values())

    def resolve(self, requested: str) -> ApplicationEntry:
        """Map a requested name to an allowlisted entry, or refuse. Never guesses."""
        app_id = self._names.get(normalize_name(requested))
        if app_id is None:
            raise ToolError(
                ErrorKind.NOT_FOUND,
                f"{requested!r} is not an allowed application",
                details={"requested": requested, "allowed": sorted(self._entries)},
            )
        return self._entries[app_id]


def default_entries(system_root: str | None = None) -> list[ApplicationEntry]:
    root = system_root or os.environ.get("SYSTEMROOT") or r"C:\Windows"
    system32 = str(PureWindowsPath(root) / "System32")
    return [
        ApplicationEntry(
            app_id="notepad",
            display_name="Notepad",
            executable=str(PureWindowsPath(system32) / "notepad.exe"),
            process_names=frozenset({"notepad.exe"}),
            aliases=frozenset({"bloc de notas", "notepad.exe", "blocdenotas"}),
        ),
        ApplicationEntry(
            app_id="calculator",
            display_name="Calculator",
            executable=str(PureWindowsPath(system32) / "calc.exe"),
            # The Store calculator draws its window from ApplicationFrameHost.exe, which hosts
            # many apps: the title is what tells them apart.
            process_names=frozenset({"calculatorapp.exe", "applicationframehost.exe"}),
            title_pattern=r"^(Calculator|Calculadora)$",
            aliases=frozenset({"calculadora", "calc", "calc.exe"}),
        ),
    ]


def build_catalog(
    user_entries: Mapping[str, ApplicationConfig] | None = None, system_root: str | None = None
) -> ApplicationCatalog:
    entries = {e.app_id: e for e in default_entries(system_root)}
    for app_id, config in (user_entries or {}).items():
        normalized = normalize_name(app_id)
        if len(normalized) > APPLICATION_NAME_MAX or not re.match(
            APPLICATION_NAME_PATTERN, normalized
        ):
            raise ValueError(f"invalid application id {app_id!r}")
        executable = os.path.expandvars(config.executable)
        entries[normalized] = ApplicationEntry(
            app_id=normalized,
            display_name=config.display_name or app_id,
            executable=executable,
            process_names=frozenset(
                n.casefold() for n in (config.process_names or [PureWindowsPath(executable).name])
            ),
            title_pattern=config.title_pattern,
            aliases=frozenset(config.aliases),
        )
    return ApplicationCatalog(entries.values())
