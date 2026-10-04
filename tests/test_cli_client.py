"""The CLI as a client of the persistent server, and its behaviour when no server runs."""

from __future__ import annotations

import io
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from jarvis.config import Settings
from jarvis.interface.api.server import build_runtime, make_server
from jarvis.interface.cli import main
from jarvis.interface.client import HttpBackend
from jarvis.runtime.runtime import JarvisRuntime
from jarvis.runtime.token import issue_token

MakeApi = Callable[..., tuple[TestClient, JarvisRuntime]]


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _backend(client: TestClient) -> HttpBackend:
    def send(method: str, path: str, body: dict[str, Any] | None, timeout: float) -> Any:
        response = (
            client.request(method, path, json=body)
            if body is not None
            else (client.request(method, path))
        )
        return response.status_code, response.json() if response.content else None

    return HttpBackend("http://127.0.0.1:8765", transport=send)


def _cli(backend: HttpBackend, *argv: str, stdin: io.StringIO | None = None) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), stdin=stdin or io.StringIO(""), stdout=out, backend=backend)
    return code, out.getvalue()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_run_through_the_server(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    backend = _backend(client)
    code, out = _cli(backend, "crea el archivo nota.txt con el contenido hola")
    assert code == 0
    assert "verificado" in out
    assert "(tarea" in out and "completed" in out
    assert (settings.workspace_dir / "nota.txt").read_text(encoding="utf-8") == "hola"

    code, out = _cli(backend, "status")
    assert "Modo: servidor (http://127.0.0.1:8765)" in out
    assert "Planificador: determinista" in out


def test_confirmation_from_a_second_command(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    backend = _backend(client)
    _cli(backend, "crea el archivo x.txt")
    code, out = _cli(backend, "--no-input", "borra el archivo x.txt")
    assert code == 0
    assert "jarvis approve" in out
    task_id = out.rsplit("(tarea ", 1)[1].split(":")[0]

    code, out = _cli(backend, "approve", task_id, "a1")
    assert code == 0
    assert "verificado" in out
    assert not (settings.workspace_dir / "x.txt").exists()


def test_interactive_confirmation_in_server_mode(make_api: MakeApi, settings: Settings) -> None:
    client, _ = make_api()
    backend = _backend(client)
    _cli(backend, "crea el archivo a.txt")
    _cli(backend, "crea el archivo b.txt")

    code, out = _cli(backend, "borra el archivo a.txt", stdin=Tty("s\n"))
    assert "¿Continuar? [s/N]" in out
    assert code == 0 and not (settings.workspace_dir / "a.txt").exists()

    code, out = _cli(backend, "borra el archivo b.txt", stdin=Tty("n\n"))
    assert code == 1
    assert (settings.workspace_dir / "b.txt").exists()


def test_resume_is_not_offered_through_the_server(make_api: MakeApi) -> None:
    client, _ = make_api()
    code, out = _cli(_backend(client), "resume", "task_x")
    assert code == 2
    assert "automatically" in out


def test_without_a_server_the_cli_runs_locally_and_says_so(settings: Settings) -> None:
    settings = settings.model_copy(update={"server_port": _free_port()})
    out = io.StringIO()
    code = main(["lista los archivos"], stdin=io.StringIO(""), stdout=out, settings=settings)
    assert code == 0
    text = out.getvalue()
    assert "servidor no disponible" in text and "jarvis server" in text
    assert "Planificador: determinista" in text
    assert settings.lock_path.exists()  # local mode took the instance lock


def test_local_mode_refuses_to_run_next_to_a_live_runtime(
    make_api: MakeApi, settings: Settings
) -> None:
    make_api()  # holds the instance lock (TestClient opens no port, so the CLI sees no server)
    settings = settings.model_copy(update={"server_port": _free_port()})
    out = io.StringIO()
    code = main(["crea un archivo de prueba"], stdin=io.StringIO(""), stdout=out, settings=settings)
    assert code == 2
    assert "another Jarvis runtime" in out.getvalue()
    assert not (settings.workspace_dir / "prueba.txt").exists()


def test_real_socket_end_to_end(settings: Settings) -> None:
    """One test over a real loopback socket: uvicorn + the CLI's urllib transport."""
    settings = settings.model_copy(update={"server_port": _free_port()})
    runtime = build_runtime(settings)
    runtime.start()
    token = issue_token(settings.api_token_path)  # what `jarvis server` does; the CLI reads it
    server = make_server(settings, runtime, token)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.05)
        out = io.StringIO()
        code = main(
            ["crea un archivo de prueba"], stdin=io.StringIO(""), stdout=out, settings=settings
        )
        assert code == 0, out.getvalue()
        assert "servidor no disponible" not in out.getvalue()
        assert "verificado" in out.getvalue()
        assert (settings.workspace_dir / "prueba.txt").exists()
    finally:
        server.should_exit = True
        thread.join(10)
        assert runtime.stop() is True
    assert not thread.is_alive()
