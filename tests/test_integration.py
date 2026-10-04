"""End-to-end scenarios using the production wiring (SQLite state, JSONL audit, real tools)."""

from __future__ import annotations

import io
import json

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.core.action import ActionStatus, VerificationStatus
from jarvis.core.decisions import Decision
from jarvis.core.task import TaskState
from jarvis.execution.confirmation import ConfirmationHandler
from jarvis.interface.cli import main
from jarvis.observability.audit import AuditEventType, JsonlAuditLog
from jarvis.orchestration.jarvis import Jarvis

from support import fake_model, plan_json


def test_create_test_file_end_to_end(settings: Settings) -> None:
    """Usuario: "crea un archivo de prueba" -> plan -> policy -> execute -> verify -> audit."""
    jarvis = build_jarvis(settings)
    response = jarvis.ask("crea un archivo de prueba")

    # Response
    assert response.state is TaskState.COMPLETED
    assert "Listo." in response.message
    (summary,) = response.actions
    assert summary.tool == "fs.write_file"
    assert summary.decision is Decision.ALLOW
    assert summary.verification is VerificationStatus.VERIFIED

    # Real effect on disk, confined to the workspace
    created = settings.workspace_dir / "prueba.txt"
    assert created.read_text(encoding="utf-8") == "Archivo de prueba creado por Jarvis.\n"

    # Persisted state (read back from SQLite by a fresh instance)
    task = build_jarvis(settings).task(response.task_id)
    assert task.state is TaskState.COMPLETED
    assert task.actions[0].status is ActionStatus.VERIFIED
    assert task.version >= 3

    # Audit trail on disk
    log = JsonlAuditLog(settings.audit_log_path)
    events = log.events(response.task_id)
    types = [e.type for e in events]
    for expected in (
        AuditEventType.REQUEST_RECEIVED,
        AuditEventType.PLAN_CREATED,
        AuditEventType.POLICY_EVALUATED,
        AuditEventType.ACTION_EXECUTED,
        AuditEventType.ACTION_VERIFIED,
        AuditEventType.RESPONSE_SENT,
    ):
        assert expected in types
    policy = next(e for e in events if e.type is AuditEventType.POLICY_EVALUATED)
    assert policy.tool == "fs.write_file" and policy.decision == "allow"
    assert policy.data["permissions"] == ["fs.write"]


def test_multi_step_with_confirmation_end_to_end(settings: Settings) -> None:
    jarvis = build_jarvis(settings)
    first = jarvis.ask(
        'crea el archivo tmp.txt con el contenido "borrador" y luego léelo y luego bórralo'
    )
    assert first.state is TaskState.WAITING_PERMISSION
    assert [a.status for a in first.actions] == [
        ActionStatus.VERIFIED,
        ActionStatus.EXECUTED,
        ActionStatus.AWAITING_CONFIRMATION,
    ]
    assert first.actions[1].output is not None
    assert first.actions[1].output["content"] == "borrador"
    assert (settings.workspace_dir / "tmp.txt").exists()

    final = build_jarvis(settings).approve(first.task_id, "a3")
    assert final.state is TaskState.COMPLETED
    assert final.actions[2].status is ActionStatus.VERIFIED
    assert not (settings.workspace_dir / "tmp.txt").exists()


def _cli(settings: Settings, *argv: str) -> tuple[int, str]:
    out = io.StringIO()

    def factory(handler: ConfirmationHandler | None) -> Jarvis:
        return build_jarvis(settings, confirmations=handler)

    code = main(list(argv), stdin=io.StringIO(""), stdout=out, factory=factory)
    return code, out.getvalue()


def test_cli_runs_the_pipeline(settings: Settings) -> None:
    code, out = _cli(settings, "crea el archivo hola.txt con el contenido hola")
    assert code == 0
    assert "verificado" in out
    assert (settings.workspace_dir / "hola.txt").read_text(encoding="utf-8") == "hola"

    code, out = _cli(settings, "lee el archivo hola.txt")
    assert code == 0 and "hola" in out


def test_cli_confirmation_flow(settings: Settings) -> None:
    _cli(settings, "crea el archivo x.txt")
    code, out = _cli(settings, "borra el archivo x.txt")
    assert code == 0
    assert "jarvis approve" in out

    code, listing = _cli(settings, "--json", "tasks", "--state", "waiting_permission")
    task_id = json.loads(listing)[0]["task_id"]
    code, out = _cli(settings, "approve", task_id, "a1")
    assert code == 0
    assert not (settings.workspace_dir / "x.txt").exists()

    code, out = _cli(settings, "audit", task_id)
    assert "confirmation.decided" in out and "action.verified" in out


def test_cli_reports_failures_with_exit_code(settings: Settings) -> None:
    code, out = _cli(settings, "haz magia")
    assert code == 1
    assert "No se pudo completar" in out
    code, out = _cli(settings, "approve", "task_missing", "a1")
    assert code == 2


def test_model_planned_request_end_to_end(settings: Settings) -> None:
    """request -> ModelPlanner -> LiteLLM adapter (fake completion) -> PlanValidator -> TaskRunner
    -> Policy -> filesystem tool -> verification -> SQLite state -> JSONL audit."""
    settings = settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake-model"}
    )
    provider, completion = fake_model(
        plan_json(
            {
                "tool": "fs.write_file",
                "arguments": {"path": "informe.md", "content": "# Informe\n"},
                "description": "Crear el informe",
            },
            {"tool": "fs.read_file", "arguments": {"path": "informe.md"}, "depends_on": [0]},
        )
    )
    jarvis = build_jarvis(settings, model_provider=provider)
    response = jarvis.ask("prepárame un informe vacío y enséñamelo")

    assert response.state is TaskState.COMPLETED
    assert response.planner == "model"
    assert [a.status for a in response.actions] == [ActionStatus.VERIFIED, ActionStatus.EXECUTED]
    assert response.actions[1].output is not None
    assert response.actions[1].output["content"] == "# Informe\n"
    assert (settings.workspace_dir / "informe.md").read_text(encoding="utf-8") == "# Informe\n"

    # The model saw the tool catalog and the goal; nothing else was required.
    sent = completion.calls[0]["messages"]
    assert "fs.write_file" in sent[0]["content"]
    assert '"goal": "prepárame un informe vacío y enséñamelo"' in sent[-1]["content"]

    stored = build_jarvis(settings, model_provider=provider).task(response.task_id)
    assert stored.planner == "model"
    assert stored.state is TaskState.COMPLETED

    events = JsonlAuditLog(settings.audit_log_path).events(response.task_id)
    created = next(e for e in events if e.type is AuditEventType.PLAN_CREATED)
    assert created.data["planner"] == "model"
    assert created.data["metadata"]["model"] == "openai/fake-model"
    policies = [e for e in events if e.type is AuditEventType.POLICY_EVALUATED]
    assert [p.decision for p in policies] == ["allow", "allow"]
    assert AuditEventType.ACTION_VERIFIED in [e.type for e in events]


def test_cli_reports_planner_and_fallback(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"model_provider": "litellm", "model": "openai/fake-model"}
    )
    provider, _ = fake_model("esto no es un plan", plan_json({"tool": "fs.list_dir"}))

    def factory(handler: ConfirmationHandler | None) -> Jarvis:
        return build_jarvis(settings, confirmations=handler, model_provider=provider)

    out = io.StringIO()
    main(["crea un archivo de prueba"], stdin=io.StringIO(""), stdout=out, factory=factory)
    assert "Planificador: determinista (fallback: el modelo no devolvió un plan utilizable)" in (
        out.getvalue()
    )
    out = io.StringIO()
    main(["lista los archivos"], stdin=io.StringIO(""), stdout=out, factory=factory)
    assert "Planificador: modelo" in out.getvalue()
    out = io.StringIO()
    main(["status"], stdin=io.StringIO(""), stdout=out, factory=factory)
    assert out.getvalue().splitlines() == ["Planificador: modelo -> determinista", "Modo: local"]


def test_cli_without_model_reports_deterministic(settings: Settings) -> None:
    code, out = _cli(settings, "lista los archivos")
    assert code == 0
    assert "Planificador: determinista" in out
    assert "fallback" not in out


def test_cli_reports_configuration_errors(settings: Settings) -> None:
    broken = settings.model_copy(update={"model_provider": "litellm", "model": None})

    def factory(handler: ConfirmationHandler | None) -> Jarvis:
        return build_jarvis(broken, confirmations=handler)

    out = io.StringIO()
    code = main(["lista los archivos"], stdin=io.StringIO(""), stdout=out, factory=factory)
    assert code == 2
    assert "JARVIS_MODEL" in out.getvalue()
