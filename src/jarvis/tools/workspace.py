"""Filesystem confinement for tools.

Every path a tool touches is resolved through :class:`Workspace`, which rejects absolute paths and
anything that resolves (including through symlinks or ``..``) outside the workspace root.
"""

from __future__ import annotations

from pathlib import Path, PurePath

from jarvis.core.errors import ErrorKind, ToolError

# Names Windows maps to devices regardless of extension or directory ("NUL", "con.txt", ...).
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{i}" for i in range(10)}
    | {f"lpt{i}" for i in range(10)}
    | {"com¹", "com²", "com³", "lpt¹", "lpt²", "lpt³"}
)


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        pure = PurePath(relative)
        if pure.is_absolute() or pure.anchor or relative.startswith(("/", "\\")):
            raise self._violation(relative, "absolute paths are not allowed")
        # Checked on every platform so behaviour does not depend on where Jarvis runs.
        for part in pure.parts:
            self._check_component(relative, part)
        candidate = (self.root / pure).resolve()
        if not candidate.is_relative_to(self.root):
            raise self._violation(relative, "path escapes the workspace")
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix() or "."

    def _check_component(self, relative: str, part: str) -> None:
        if part in (".", ".."):
            return
        if any(ord(ch) < 32 for ch in part):
            raise self._violation(relative, "control characters are not allowed")
        if ":" in part:
            raise self._violation(relative, "':' (drives, alternate data streams) is not allowed")
        if part.endswith((" ", ".")):
            raise self._violation(relative, "names may not end with a dot or a space")
        if part.split(".", 1)[0].rstrip(" ").lower() in _RESERVED_NAMES:
            raise self._violation(relative, "reserved device name")

    @staticmethod
    def _violation(path: str, reason: str) -> ToolError:
        return ToolError(
            ErrorKind.POLICY_DENIED,
            f"sandbox violation for {path!r}: {reason}",
            details={"sandbox_violation": True, "path": path},
        )
