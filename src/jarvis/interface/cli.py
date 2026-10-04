"""Command-line interface.

    jarvis server                             # start the persistent local process
    jarvis "crea un archivo de prueba"        # same as: jarvis run "..."
    jarvis tasks | show ID | audit ID | tools | status
    jarvis approve ID ACTION | reject ID ACTION | resume ID | cancel ID

Every command except ``server`` goes to the running server if one answers on the configured
loopback address. Otherwise it runs in-process ("local mode", the pre-v0.3 behaviour) and says so;
it never starts a server on its own. ``--local`` forces local mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from pydantic import ValidationError

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.context.model import ExecutionContext
from jarvis.core.action import ConfirmationRequest
from jarvis.core.errors import JarvisError
from jarvis.core.task import TaskState
from jarvis.execution.confirmation import ConfirmationHandler, ConfirmationOutcome
from jarvis.interface.client import Backend, HttpBackend, LocalBackend
from jarvis.interface.views import TaskView, is_settled
from jarvis.orchestration.jarvis import Jarvis
from jarvis.orchestration.response import describe_target
from jarvis.runtime.lock import InstanceLock
from jarvis.runtime.token import read_token

COMMANDS = (
    "run",
    "server",
    "tasks",
    "show",
    "audit",
    "tools",
    "status",
    "approve",
    "reject",
    "resume",
    "cancel",
)

_PLANNER_NAMES = {"model": "modelo", "rule_based": "determinista"}
POLL_SECONDS = 0.1
Factory = Callable[[ConfirmationHandler | None], Jarvis]


def _ask_yes_no(
    stdin: TextIO,
    stdout: TextIO,
    *,
    description: str,
    tool: str,
    arguments: dict[str, Any],
    reasons: Sequence[str],
    target: dict[str, Any] | None = None,
) -> bool:
    """Anything other than an explicit yes is a rejection."""
    described = describe_target(target)
    stdout.write(
        f"\nConfirmación requerida: {description}\n"
        f"  herramienta: {tool}\n"
        f"  argumentos: {json.dumps(arguments, ensure_ascii=False)}\n"
        + (f"  objetivo: {described}\n" if described else "")
        + f"  motivo: {'; '.join(reasons)}\n"
        "¿Continuar? [s/N]: "
    )
    stdout.flush()
    return stdin.readline().strip().lower() in {"s", "si", "sí", "y", "yes"}


class InteractiveConfirmationHandler:
    """Local mode: asks on the terminal while the task runs in this process."""

    name = "cli-prompt"

    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        self._stdin = stdin
        self._stdout = stdout

    def request(
        self, request: ConfirmationRequest, context: ExecutionContext
    ) -> ConfirmationOutcome:
        approved = _ask_yes_no(
            self._stdin,
            self._stdout,
            description=request.description,
            tool=request.tool,
            arguments=request.arguments,
            reasons=request.reasons,
            target=request.target,
        )
        return ConfirmationOutcome.APPROVED if approved else ConfirmationOutcome.REJECTED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description="Jarvis - asistente personal")
    parser.add_argument("--json", action="store_true", help="salida en JSON")
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="no preguntar en la terminal; las confirmaciones quedan pendientes",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="ejecutar en este proceso aunque haya un servidor en marcha",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ejecutar una solicitud")
    run.add_argument("text", nargs="+")

    sub.add_parser("server", help="iniciar Jarvis como proceso persistente (API local)")

    tasks = sub.add_parser("tasks", help="listar tareas")
    tasks.add_argument("--state", choices=[s.value for s in TaskState])
    tasks.add_argument("--limit", type=int, default=20)

    for name, help_text in (
        ("show", "ver una tarea"),
        ("audit", "ver la auditoría de una tarea"),
        ("resume", "reanudar una tarea (solo modo local)"),
        ("cancel", "cancelar una tarea"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("task_id")

    for name, help_text in (("approve", "aprobar una acción"), ("reject", "rechazar una acción")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("task_id")
        cmd.add_argument("action_id")

    sub.add_parser("tools", help="listar herramientas registradas")
    sub.add_parser("status", help="ver modo y planificador activos")
    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Allow ``jarvis "haz X"`` as shorthand for ``jarvis run "haz X"``."""
    args = list(argv)
    for i, arg in enumerate(args):
        if arg.startswith("-"):
            continue
        if arg not in COMMANDS:
            args.insert(i, "run")
        break
    return args


# -- rendering ------------------------------------------------------------------------------


def _planner_line(view: TaskView) -> str | None:
    if view.planner is None:
        return None
    name = _PLANNER_NAMES.get(view.planner, view.planner)
    if view.fallback_reason:
        return f"Planificador: {name} (fallback: {view.fallback_reason})"
    return f"Planificador: {name}"


def _render(view: TaskView, out: TextIO) -> None:
    out.write(view.message + "\n")
    planner_line = _planner_line(view)
    if planner_line:
        out.write(planner_line + "\n")
    for action in view.actions:
        output = action.output or {}
        if "content" in output:
            out.write(f"\n--- {output.get('path', action.action_id)} ---\n{output['content']}\n")
        elif "entries" in output:
            out.write(f"\n{output.get('path', '.')}:\n")
            for entry in output["entries"]:
                suffix = "/" if entry.get("type") == "dir" else ""
                out.write(f"  {entry.get('name')}{suffix}\n")
    if view.progress is not None:
        p = view.progress
        out.write(
            f"Progreso: fase {p.phase} | iteraciones {p.iterations} | acciones {p.actions_used}"
            f" | replanificaciones {p.replans_used} | {p.elapsed_seconds:.1f}s\n"
        )
        if p.current_action and view.status in ("PENDING", "RUNNING", "WAITING_PERMISSION"):
            out.write(f"Acción actual: {p.current_action}\n")
        if view.status == "STOPPED" and p.stop_reason:
            out.write(f"Motivo de parada: {p.stop_reason}\n")
    for pending in view.pending_confirmations:
        out.write(
            f"\nPara aprobar:  jarvis approve {view.task_id} {pending.action_id}"
            f"\nPara rechazar: jarvis reject {view.task_id} {pending.action_id}\n"
        )
    out.write(f"\n(tarea {view.task_id}: {view.status.lower()})\n")


def _dump(data: Any, out: TextIO) -> None:
    out.write(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


# -- entry point ----------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    factory: Factory | None = None,
    settings: Settings | None = None,
    backend: Backend | None = None,
) -> int:
    """``factory`` forces local mode with that runtime; ``backend`` injects a client directly."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if stdout is sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parser().parse_args(_normalize_argv(sys.argv[1:] if argv is None else argv))
    interactive = args.command == "run" and not args.no_input and stdin.isatty()

    try:
        settings_given = settings is not None
        settings = settings or Settings()
        if args.command == "server":
            from jarvis.interface.api.server import serve  # web stack only for the server

            return serve(settings, stdout)
        if backend is None:
            handler = InteractiveConfirmationHandler(stdin, stdout) if interactive else None
            backend = _select_backend(args, settings, factory, settings_given, handler, stdout)
        return _dispatch(backend, args, stdin, stdout, interactive)
    except ValidationError as exc:
        stdout.write(f"Error de configuración: {exc}\n")
        return 2
    except JarvisError as exc:
        stdout.write(f"Error: {exc}\n")
        return 2


def _select_backend(
    args: argparse.Namespace,
    settings: Settings,
    factory: Factory | None,
    settings_given: bool,
    handler: ConfirmationHandler | None,
    out: TextIO,
) -> Backend:
    if factory is None and not args.local:
        http = HttpBackend(settings.server_url, token=read_token(settings.api_token_path))
        if http.health() is not None:
            return http
        if args.command != "status":
            out.write(
                f"(servidor no disponible en {settings.server_url}: ejecutando en modo local; "
                "para un proceso persistente usa `jarvis server`)\n"
            )

    def build() -> Jarvis:
        if factory is not None:
            return factory(handler)
        return build_jarvis(settings, confirmations=handler)

    # A caller-provided runtime without explicit settings manages its own data directory.
    lock = InstanceLock(settings.lock_path) if (factory is None or settings_given) else None
    return LocalBackend(build, lock)


def _wait(backend: Backend, view: TaskView) -> TaskView:
    while not is_settled(view):
        time.sleep(POLL_SECONDS)
        view = backend.get(view.task_id)
    return view


def _settle(
    backend: Backend, view: TaskView, stdin: TextIO, out: TextIO, interactive: bool
) -> TaskView:
    """Wait for a server-side task; in a terminal, resolve its confirmations interactively."""
    view = _wait(backend, view)
    while interactive and backend.mode == "server" and view.status == "WAITING_PERMISSION":
        for pending in view.pending_confirmations:
            approved = _ask_yes_no(
                stdin,
                out,
                description=pending.description,
                tool=pending.tool,
                arguments=pending.arguments,
                reasons=pending.reasons,
                target=pending.target,
            )
            decide = backend.approve if approved else backend.reject
            view = decide(view.task_id, pending.action_id)
        view = _wait(backend, view)
    return view


def _dispatch(
    backend: Backend, args: argparse.Namespace, stdin: TextIO, out: TextIO, interactive: bool
) -> int:
    view: TaskView | None = None
    try:
        match args.command:
            case "run":
                view = backend.submit(" ".join(args.text))
            case "approve":
                view = backend.approve(args.task_id, args.action_id)
            case "reject":
                view = backend.reject(args.task_id, args.action_id)
            case "resume":
                view = backend.resume(args.task_id)
            case "cancel":
                view = backend.cancel(args.task_id)
            case "show":
                _dump(backend.get(args.task_id).model_dump(mode="json"), out)
                return 0
            case "audit":
                return _audit(backend, args, out)
            case "tasks":
                return _tasks(backend, args, out)
            case "tools":
                for tool in backend.tools():
                    perms = ",".join(tool.permissions)
                    cats = ",".join(tool.categories)
                    out.write(f"{tool.name:<18} [{perms}] ({cats}) {tool.description}\n")
                return 0
            case "status":
                chain = [_PLANNER_NAMES.get(n, n) for n in backend.planner().split(" -> ")]
                out.write(f"Planificador: {' -> '.join(chain)}\n")
                where = (
                    f"servidor ({backend.base_url})"
                    if isinstance(backend, HttpBackend)
                    else "local"
                )
                out.write(f"Modo: {where}\n")
                return 0
        assert view is not None
        view = _settle(backend, view, stdin, out, interactive)
    except KeyboardInterrupt:
        if view is not None:
            out.write(f"\nLa tarea sigue en el servidor: jarvis show {view.task_id}\n")
        return 130

    if args.json:
        _dump(view.model_dump(mode="json"), out)
    else:
        _render(view, out)
    return 0 if view.status in ("COMPLETED", "WAITING_PERMISSION") else 1


def _audit(backend: Backend, args: argparse.Namespace, out: TextIO) -> int:
    events = backend.events(args.task_id)
    if args.json:
        _dump([e.model_dump(mode="json") for e in events], out)
        return 0
    for e in events:
        parts = [e.at.strftime("%H:%M:%S"), e.type]
        parts += [p for p in (e.action_id, e.tool, e.decision, e.outcome) if p]
        out.write(" | ".join(parts) + (f" - {e.message}" if e.message else "") + "\n")
    return 0


def _tasks(backend: Backend, args: argparse.Namespace, out: TextIO) -> int:
    status = args.state.upper() if args.state else None
    tasks = backend.tasks(status, args.limit)
    if args.json:
        _dump([t.model_dump(mode="json") for t in tasks], out)
        return 0
    for t in tasks:
        created = t.created_at.strftime("%Y-%m-%d %H:%M")
        out.write(f"{t.task_id}  {t.status.lower():<18} {created}  {t.goal}\n")
    return 0
