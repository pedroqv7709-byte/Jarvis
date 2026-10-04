"""Shared fixtures."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.interface.api.app import create_app
from jarvis.observability.audit import InMemoryAuditLog
from jarvis.orchestration.jarvis import Jarvis
from jarvis.runtime.lock import InstanceLock
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.state.store import InMemoryStateStore

API_TOKEN = "test-token-" + "x" * 32
"""Injected by tests; real servers generate a random one at start."""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
        user_id="tester",
        permissions={
            "fs.read": "allow",
            "fs.write": "allow",
            "fs.delete": "confirm",
            "test.echo": "allow",
            "messages.send": "allow",
        },
        retry_backoff_seconds=0,
        # Unit tests never touch the real desktop; Windows tools get a fake when needed.
        windows_automation="off",
    )


@pytest.fixture
def workspace_dir(settings: Settings) -> Path:
    return settings.workspace_dir


@pytest.fixture
def memory_jarvis(settings: Settings) -> Jarvis:
    """Jarvis wired with the real components, in-memory state and audit."""
    return build_jarvis(settings, store=InMemoryStateStore(), audit=InMemoryAuditLog())


@pytest.fixture
def make_api(settings: Settings) -> Iterator[Callable[..., tuple[TestClient, JarvisRuntime]]]:
    """Start real runtimes (worker thread, recovery, lock) behind TestClient; no sockets.

    Everything is real except the transport. Each call wires a new runtime; all are stopped at
    teardown (stopping twice is harmless).
    """
    stack = ExitStack()

    def make(
        custom: Settings | None = None, *, grace: float = 2.0, **build_kwargs: Any
    ) -> tuple[TestClient, JarvisRuntime]:
        current = custom or settings
        runtime = JarvisRuntime(
            build_jarvis(current, **build_kwargs),
            lock=InstanceLock(current.lock_path),
            shutdown_grace_seconds=grace,
        )
        client = TestClient(
            create_app(runtime, token=API_TOKEN),
            base_url="http://127.0.0.1:8765",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        stack.enter_context(client)
        return client, runtime

    yield make
    stack.close()


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Gate for HangingTool; always opened at teardown so no thread is left blocked."""
    event = threading.Event()
    yield event
    event.set()
