"""Single-runtime guarantee per data directory.

Only one Jarvis runtime (the server, or a CLI running in local mode) may execute tasks against a
given state database at a time. Otherwise a starting server could "recover" a task that a local
CLI is still executing and run its actions twice.

The lock is an OS-level file lock, so it is released automatically if the process dies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Self

from jarvis.core.errors import JarvisError


class InstanceLockedError(JarvisError):
    pass


class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            _lock(handle)
        except OSError as exc:
            handle.close()
            raise InstanceLockedError(
                f"another Jarvis runtime is using {self.path.parent} "
                "(a running `jarvis server`, or a CLI command in local mode)"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


if sys.platform == "win32":
    import msvcrt

    def _lock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
