"""``jarvis server``: build the runtime once and serve the local API until interrupted."""

from __future__ import annotations

import contextlib
import signal
import sys
from typing import TextIO

import uvicorn

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.interface.api.app import create_app
from jarvis.runtime.lock import InstanceLock
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.runtime.token import issue_token, revoke_token


def build_runtime(settings: Settings) -> JarvisRuntime:
    # The server never prompts: confirmations wait in WAITING_PERMISSION until a client decides.
    jarvis = build_jarvis(settings)
    return JarvisRuntime(
        jarvis,
        lock=InstanceLock(settings.lock_path),
        shutdown_grace_seconds=settings.shutdown_grace_seconds,
    )


def make_server(settings: Settings, runtime: JarvisRuntime, token: str) -> uvicorn.Server:
    """A uvicorn server for an already started runtime (it does not manage the runtime)."""
    config = uvicorn.Config(
        create_app(runtime, token=token, manage_lifecycle=False),
        host=settings.server_host,
        port=settings.server_port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)


def serve(settings: Settings, out: TextIO) -> int:
    runtime = build_runtime(settings)
    # Started before uvicorn so that a held lock or a broken database fails fast and clearly.
    runtime.start()
    # Issued after the instance lock is held, so two servers can never overwrite each other's.
    token = issue_token(settings.api_token_path)
    try:
        out.write(
            f"Jarvis escuchando en {settings.server_url} (solo local). Ctrl+C para detener.\n"
        )
        out.flush()
        server = make_server(settings, runtime, token)
        if sys.platform == "win32":
            # uvicorn handles SIGINT (Ctrl+C) itself. Windows has no catchable SIGTERM, so also
            # route Ctrl+Break through the same graceful path instead of killing the process.
            signal.signal(signal.SIGBREAK, server.handle_exit)
        # uvicorn re-raises Ctrl+C after its own graceful shutdown; that is a normal stop.
        with contextlib.suppress(KeyboardInterrupt):
            server.run()
    finally:
        revoke_token(settings.api_token_path, token)
        clean = runtime.stop()
        out.write(
            "Jarvis detenido.\n"
            if clean
            else "Jarvis detenido; la tarea en curso se reanudará al volver a arrancar.\n"
        )
    return 0
