"""Deterministic, model-free planner for workspace file operations.

It understands a small, explicit grammar (Spanish and English) and fails loudly on anything else;
it never guesses. Requests may chain several steps with ``;``, "y luego", "y después" or
"and then"; each step depends on the previous one. Quoted text is never split.

Examples::

    crea un archivo de prueba
    crea el archivo notas.txt con el contenido "hola mundo"
    lee el archivo notas.txt
    lista los archivos
    crea el archivo a.txt con contenido hola y luego léelo
    borra el archivo notas.txt
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jarvis.core.action import Action
from jarvis.core.agent import RequirementKind
from jarvis.core.errors import PlanningError
from jarvis.planning.base import Decision, DecisionKind, DecisionRequest, Plan, PlanningRequest

_FLAGS = re.IGNORECASE | re.DOTALL
_PATH = r"(?P<path>\"[^\"]+\"|'[^']+'|(?!(?:con|with|que)\b)[^\s\"']+)"
_CONTENT_INTRO = (
    r"(?:con(?:\s+el)?\s+(?:contenido|texto)|que\s+diga|with(?:\s+the)?\s+(?:content|text)"
    r"|containing)\s*:?\s+(?P<content>.+)"
)
_ARTICLE = r"(?:(?:el|la|un|una|a|an|the)\s+)?"
_FILE = r"(?:archivo|fichero|file)"

_WRITE = re.compile(
    r"^(?:crea|crear|create|escribe|escribir|write|genera|generar|make)\s+"
    + _ARTICLE
    + r"(?:(?:nuevo|nueva|new)\s+)?(?P<test_pre>(?:test|prueba)\s+)?"
    + _FILE
    + r"(?P<test>\s+(?:de\s+prueba|de\s+test|test))?"
    + r"(?:\s+(?:llamado|llamada|named|called))?"
    + rf"(?:\s+{_PATH})?"
    + rf"(?:\s+{_CONTENT_INTRO})?$",
    _FLAGS,
)
_OVERWRITE = re.compile(
    r"^(?:sobrescribe|sobreescribe|overwrite)\s+" + _ARTICLE + _FILE + rf"\s+{_PATH}\s+"
    rf"{_CONTENT_INTRO}$",
    _FLAGS,
)
_READ = re.compile(
    r"^(?:lee|leer|read|muestra|mostrar|show|cat)\s+"
    + _ARTICLE
    + r"(?:contenido\s+del\s+|content\s+of\s+(?:the\s+)?)?"
    + _FILE
    + rf"\s+{_PATH}$",
    _FLAGS,
)
_READ_IT = re.compile(r"^(?:l[eé]elo|l[eé]ela|read\s+it|mu[eé]stralo|show\s+it)$", _FLAGS)
_LIST = re.compile(
    r"^(?:lista|listar|list|muestra|mostrar|show|enumera)\s+(?:(?:los|las|the)\s+)?"
    r"(?:archivos|ficheros|files)(?:\s+(?:en|de|in|of)\s+"
    r"(?:(?:la\s+carpeta|el\s+directorio|the\s+folder|the\s+directory)\s+)?" + _PATH + r")?$",
    _FLAGS,
)
_DELETE = re.compile(
    r"^(?:borra|borrar|elimina|eliminar|delete|remove|rm)\s+" + _ARTICLE + _FILE + rf"\s+{_PATH}$",
    _FLAGS,
)
_DELETE_IT = re.compile(r"^(?:b[oó]rralo|elim[ií]nalo|delete\s+it|remove\s+it)$", _FLAGS)

# Windows applications. The name is only a lookup key into the allowlist; it is never executed.
_APP = (
    r"(?!(?:(?:el|la|un|una|the|a|an)\s+)?(?:archivo|fichero|file|ventana|window)\b)"
    r"(?P<app>[A-Za-z0-9][A-Za-z0-9 ._-]{0,39})"
)
_APP_NOUN = r"(?:(?:aplicaci[oó]n|programa|app|application)\s+)?"
_OPEN_APP = re.compile(
    r"^(?:abre|abrir|open|inicia|iniciar|lanza|lanzar|launch|start)\s+"
    + _ARTICLE
    + _APP_NOUN
    + _APP
    + "$",
    _FLAGS,
)
_FOCUS_WINDOW = re.compile(
    r"^(?:enfoca|enfocar|focus|activa|activate|cambia\s+a|switch\s+to|trae)\s+"
    r"(?:(?:la\s+ventana(?:\s+de)?|the\s+window(?:\s+of)?|el|la|the)\s+)?"
    + _APP_NOUN
    + r"(?:(?P<title>\"[^\"]+\"|'[^']+')|"
    + _APP
    + r")(?:\s+al\s+frente)?$",
    _FLAGS,
)
_CLOSE_APP = re.compile(
    r"^(?:cierra|cerrar|close|quit)\s+"
    + _ARTICLE
    + r"(?:(?:aplicaci[oó]n|programa|app|application|ventana\s+de|window\s+of)\s+)?"
    + _APP
    + "$",
    _FLAGS,
)
_CLOSE_IT = re.compile(r"^(?:ci[eé]rrala|ci[eé]rralo|close\s+it)$", _FLAGS)
_FOCUS_IT = re.compile(r"^(?:enf[oó]cala|enf[oó]calo|focus\s+it)$", _FLAGS)
_LIST_WINDOWS = re.compile(
    r"^(?:(?:lista|listar|list|muestra|mostrar|show|enumera)\s+(?:(?:las|the)\s+)?"
    r"(?:ventanas|windows)(?:\s+abiertas|\s+open)?|qu[eé]\s+ventanas\s+(?:hay\s+)?abiertas)$",
    _FLAGS,
)

_QUOTED = re.compile(r"\"[^\"]*\"|(?<!\w)'[^']*'(?!\w)")
# A plain "y"/"and" separates steps only when a known verb or pronoun follows it, so content
# such as "pan y vino" is never split.
_NEXT_VERB = (
    r"(?=(?:enf[oó]c|ci[eé]rr|l[eé]e|b[oó]rr|abre|abrir|cierra|cerrar|enfoca|lista|listar|crea|"
    r"crear|sobrescribe|muestra|trae|activa|focus|close|open|read|delete|list|create)\w*\b)"
)
_SEPARATOR = re.compile(
    r"\s*(?:;|,?\s+y\s+luego\s+|,?\s+y\s+despu[eé]s\s+|,?\s+and\s+then\s+|"
    r",?\s+y\s+" + _NEXT_VERB + r"|,?\s+and\s+" + _NEXT_VERB + r")\s*",
    _FLAGS,
)
_LEADING_NOISE = re.compile(r"^(?:jarvis[,:]?\s+)?(?:por\s+favor[,]?\s+|please[,]?\s+)?", _FLAGS)
_TRAILING_NOISE = re.compile(r"(?:[,]?\s+(?:por\s+favor|please))?[\s!?¡¿]*$", _FLAGS)

DEFAULT_TEST_CONTENT = "Archivo de prueba creado por Jarvis.\n"

SUPPORTED = (
    "crear archivo (crea el archivo X con el contenido Y), sobrescribir archivo, "
    "leer archivo, listar archivos, borrar archivo, abrir/enfocar/cerrar una aplicación, "
    "listar ventanas; encadenables con 'y luego'"
)


def split_clauses(text: str) -> list[str]:
    stash: list[str] = []

    def hide(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\x00{len(stash) - 1}\x00"

    masked = _QUOTED.sub(hide, text)
    parts = _SEPARATOR.split(masked)

    def restore(part: str) -> str:
        return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], part)

    return [restore(p).strip() for p in parts if p.strip()]


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _clean_path(raw: str) -> str:
    if raw[:1] in "\"'":
        return _unquote(raw)
    return raw.rstrip(".,;:")


@dataclass(frozen=True, slots=True)
class _Step:
    tool: str
    arguments: dict[str, Any]
    description: str


class AmbiguousReferenceError(PlanningError):
    """The request refers to something ("ese", "esa ventana") that nothing in it identifies."""

    def __init__(self, reference: str) -> None:
        super().__init__(
            f"'{reference}' does not identify an application; name it (e.g. 'la calculadora')",
            details={"ambiguous_reference": reference},
        )


# Words that point at something instead of naming it.
_DEMONSTRATIVES = frozenset(
    {
        "ese",
        "esa",
        "eso",
        "este",
        "esta",
        "esto",
        "aquel",
        "aquella",
        "aquello",
        "that",
        "this",
        "it",
    }
)


def _app_or_reference(raw: str, memory: Mapping[str, str]) -> str:
    """A named application, or the one an earlier step of the same request named.

    References are resolved only from the request itself; nothing is guessed from the desktop.
    """
    words = raw.casefold().split()
    if words and words[0] in _DEMONSTRATIVES:
        previous = memory.get("application")
        if previous is None:
            raise AmbiguousReferenceError(raw)
        return previous
    return raw


_EXPECTED = {
    RequirementKind.APP_OPEN: "a window of {app} exists",
    RequirementKind.APP_FOCUSED: "a window of {app} is in the foreground",
    RequirementKind.APP_CLOSED: "no window of {app} remains",
}


class RuleBasedPlanner:
    """Deterministic planner.

    ``plan`` parses the request grammar into steps. ``decide`` walks those steps in order: a step
    whose verifiable requirement is already achieved (e.g. the application is already open)
    costs nothing; a step already done is skipped; after a failure it retries at most once an
    idempotent action whose verification failed, and otherwise stops with an explanation. It
    never invents targets or alternatives.
    """

    name = "rule_based"

    def decide(self, request: DecisionRequest) -> Decision:
        try:
            plan = self.plan(PlanningRequest(goal=request.goal, context=request.context))
        except AmbiguousReferenceError as exc:
            return self._stop(f"ambiguous request: {exc.failure.message}")
        except PlanningError:
            if not request.history:
                raise  # not understood and nothing was done: a planning failure
            # Steps came from another planner: judge only by the outcome of the last one.
            last = request.history[-1]
            if last.failure_kind is not None:
                return self._stop(
                    f"{last.tool} failed ({last.failure_kind}): {last.failure_message}"
                )
            return Decision(kind=DecisionKind.DONE, planner=self.name)

        idempotent = {t["name"]: bool(t.get("idempotent")) for t in request.catalog}
        achieved = [s.requirement.action for s in request.requirements if s.achieved]
        for step in plan.actions:
            if step in achieved:
                continue  # the condition this step aims at already holds (verified)
            attempts = [
                h for h in request.history if h.tool == step.tool and h.arguments == step.arguments
            ]
            if attempts and attempts[-1].failure_kind is None:
                continue  # done earlier in this task
            if attempts:
                last = attempts[-1]
                failed = [h for h in attempts if h.failure_kind is not None]
                retry = (
                    last.failure_kind == "verification_failed"
                    and idempotent.get(step.tool, False)
                    and len(failed) < 2
                )
                if not retry:
                    detail = f"{step.tool} failed ({last.failure_kind}): {last.failure_message}"
                    if last.candidates:
                        detail += f"; {len(last.candidates)} candidate windows, none chosen"
                    return self._stop(detail)
            requirement = next(
                (s.requirement for s in request.requirements if s.requirement.action == step),
                None,
            )
            expected = (
                _EXPECTED[requirement.kind].format(app=requirement.application)
                if requirement is not None and requirement.kind in _EXPECTED
                else f"{step.tool} succeeds"
            )
            return Decision(
                kind=DecisionKind.NEXT_ACTION,
                planner=self.name,
                actions=(step.model_copy(update={"depends_on": ()}),),
                rationale="retry after a failed verification" if attempts else "next step",
                expected_observation=expected,
            )
        return Decision(kind=DecisionKind.DONE, planner=self.name)

    def _stop(self, reason: str) -> Decision:
        return Decision(kind=DecisionKind.STOP, planner=self.name, reason=reason[:500])

    def plan(self, request: PlanningRequest) -> Plan:
        clauses = split_clauses(request.goal)
        if not clauses:
            raise PlanningError("the request is empty")
        actions: list[Action] = []
        memory: dict[str, str] = {}
        for index, clause in enumerate(clauses, start=1):
            step = self._parse(self._normalize(clause), memory)
            for key in ("path", "application"):
                if isinstance(step.arguments.get(key), str):
                    memory[key] = step.arguments[key]
            actions.append(
                Action(
                    id=f"a{index}",
                    tool=step.tool,
                    arguments=step.arguments,
                    description=step.description,
                    depends_on=(actions[-1].id,) if actions else (),
                )
            )
        return Plan(planner=self.name, actions=tuple(actions))

    @staticmethod
    def _normalize(clause: str) -> str:
        clause = _LEADING_NOISE.sub("", clause.strip())
        return _TRAILING_NOISE.sub("", clause).strip()

    def _parse(self, clause: str, memory: Mapping[str, str]) -> _Step:
        handlers: list[
            tuple[re.Pattern[str], Callable[[re.Match[str], Mapping[str, str]], _Step]]
        ] = [
            (_OVERWRITE, self._overwrite),
            (_WRITE, self._write),
            (_READ_IT, self._read_it),
            (_READ, self._read),
            (_LIST, self._list),
            (_DELETE_IT, self._delete_it),
            (_DELETE, self._delete),
            (_LIST_WINDOWS, self._list_windows),
            (_OPEN_APP, self._open_app),
            (_FOCUS_WINDOW, self._focus_window),
            (_CLOSE_APP, self._close_app),
            (_CLOSE_IT, self._close_it),
            (_FOCUS_IT, self._focus_it),
        ]
        for pattern, handler in handlers:
            match = pattern.match(clause)
            if match:
                return handler(match, memory)
        raise PlanningError(
            f"no sé cómo planificar: {clause!r}. Soportado: {SUPPORTED}",
            details={"clause": clause},
        )

    @staticmethod
    def _content(match: re.Match[str]) -> str | None:
        content = match.group("content")
        return None if content is None else _unquote(content.strip())

    def _write(self, match: re.Match[str], _: Mapping[str, str]) -> _Step:
        is_test = bool(match.group("test") or match.group("test_pre"))
        raw_path = match.group("path")
        content = self._content(match)
        if raw_path:
            path = _clean_path(raw_path)
        elif is_test:
            marker = (match.group("test") or match.group("test_pre") or "").lower()
            path = "prueba.txt" if "prueba" in marker else "test.txt"
        else:
            raise PlanningError("indica el nombre del archivo a crear")
        if content is None:
            content = DEFAULT_TEST_CONTENT if is_test else ""
        return _Step(
            "fs.write_file",
            {"path": path, "content": content},
            f"Crear el archivo '{path}'",
        )

    def _overwrite(self, match: re.Match[str], _: Mapping[str, str]) -> _Step:
        path = _clean_path(match.group("path"))
        return _Step(
            "fs.overwrite_file",
            {"path": path, "content": self._content(match) or ""},
            f"Sobrescribir el archivo '{path}'",
        )

    @staticmethod
    def _read(match: re.Match[str], _: Mapping[str, str]) -> _Step:
        path = _clean_path(match.group("path"))
        return _Step("fs.read_file", {"path": path}, f"Leer el archivo '{path}'")

    @staticmethod
    def _read_it(_: re.Match[str], memory: Mapping[str, str]) -> _Step:
        last_path = memory.get("path")
        if last_path is None:
            raise PlanningError("'léelo' no se refiere a ningún archivo anterior")
        return _Step("fs.read_file", {"path": last_path}, f"Leer el archivo '{last_path}'")

    @staticmethod
    def _list(match: re.Match[str], _: Mapping[str, str]) -> _Step:
        raw = match.group("path")
        path = _clean_path(raw) if raw else "."
        return _Step("fs.list_dir", {"path": path}, f"Listar los archivos de '{path}'")

    @staticmethod
    def _delete(match: re.Match[str], _: Mapping[str, str]) -> _Step:
        path = _clean_path(match.group("path"))
        return _Step("fs.delete_file", {"path": path}, f"Borrar el archivo '{path}'")

    @staticmethod
    def _delete_it(_: re.Match[str], memory: Mapping[str, str]) -> _Step:
        last_path = memory.get("path")
        if last_path is None:
            raise PlanningError("'bórralo' no se refiere a ningún archivo anterior")
        return _Step("fs.delete_file", {"path": last_path}, f"Borrar el archivo '{last_path}'")

    @staticmethod
    def _list_windows(_: re.Match[str], __: Mapping[str, str]) -> _Step:
        return _Step("windows.get_windows", {}, "Listar las ventanas abiertas")

    @staticmethod
    def _open_app(match: re.Match[str], memory: Mapping[str, str]) -> _Step:
        app = _app_or_reference(match.group("app").strip(), memory)
        return _Step("windows.open_application", {"application": app}, f"Abrir {app}")

    @staticmethod
    def _focus_window(match: re.Match[str], memory: Mapping[str, str]) -> _Step:
        if match.group("title"):
            title = _unquote(match.group("title"))
            return _Step(
                "windows.focus_window",
                {"title_contains": title},
                f"Enfocar la ventana '{title}'",
            )
        app = _app_or_reference(match.group("app").strip(), memory)
        return _Step("windows.focus_window", {"application": app}, f"Enfocar {app}")

    @staticmethod
    def _close_app(match: re.Match[str], memory: Mapping[str, str]) -> _Step:
        app = _app_or_reference(match.group("app").strip(), memory)
        return _Step("windows.close_application", {"application": app}, f"Cerrar {app}")

    @staticmethod
    def _close_it(_: re.Match[str], memory: Mapping[str, str]) -> _Step:
        app = memory.get("application")
        if app is None:
            raise AmbiguousReferenceError("ciérrala")
        return _Step("windows.close_application", {"application": app}, f"Cerrar {app}")

    @staticmethod
    def _focus_it(_: re.Match[str], memory: Mapping[str, str]) -> _Step:
        app = memory.get("application")
        if app is None:
            raise AmbiguousReferenceError("enfócala")
        return _Step("windows.focus_window", {"application": app}, f"Enfocar {app}")
