"""Local API token.

``jarvis server`` generates a fresh random token at every start and writes it to
``<data_dir>/api_token``; clients on the same machine (the CLI) read it from there. Only processes
that can read that file can use the API. The token is removed on clean shutdown, and a restart
invalidates any previous token.

On Windows the file inherits the ACL of the data directory (under the user profile by default,
which other standard users cannot read). On POSIX it is created with mode 0600. The token is never
logged, never sent anywhere but the loopback API, and the data directory is git-ignored.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path

TOKEN_BYTES = 32


def issue_token(path: Path) -> str:
    """Create a new token, replacing any previous one atomically."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(token)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    return token


def read_token(path: Path) -> str | None:
    try:
        token = path.read_text(encoding="ascii").strip()
    except FileNotFoundError, UnicodeDecodeError:
        return None
    return token or None


def revoke_token(path: Path, token: str) -> None:
    """Remove the token file, but only if it still holds this server's token."""
    if read_token(path) == token:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
