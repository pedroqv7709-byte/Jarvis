"""Turns a task into a user-facing :class:`Response`."""

from __future__ import annotations

import json
from typing import Any

from jarvis.core.action import ActionRecord, ActionStatus
from jarvis.core.errors import ErrorKind
from jarvis.core.response import ActionSummary, Response
from jarvis.core.task import PlanningAttempt, Task, TaskState

_STATUS_TEXT = {
    ActionStatus.VERIFIED: "verificado",
    ActionStatus.EXECUTED: "ejecutado (sin verificación disponible)",
    ActionStatus.AWAITING_CONFIRMATION: "esperando confirmación",
    ActionStatus.DENIED: "denegado por la política",
    ActionStatus.REJECTED: "rechazado por el usuario",
    ActionStatus.FAILED: "falló",
    ActionStatus.SKIPPED: "omitido",
}


class ResponseBuilder:
    def build(self, task: Task) -> Response:
        actions = tuple(self._summary(r) for r in task.actions)
        pending = tuple(
            r.confirmation
            for r in task.actions
            if r.status is ActionStatus.AWAITING_CONFIRMATION and r.confirmation is not None
        )
        return Response(
            request_id=task.request.id,
            task_id=task.id,
            state=task.state,
            message=self._message(task),
            planner=task.planner,
            fallback_reason=self._fallback_reason(task),
            actions=actions,
            pending_confirmations=pending,
            failure=task.failure,
        )

    @staticmethod
    def _summary(record: ActionRecord) -> ActionSummary:
        return ActionSummary(
            action_id=record.id,
            tool=record.action.tool,
            description=record.action.description,
            status=record.status,
            decision=record.verdict.decision if record.verdict else None,
            verification=record.verification.status if record.verification else None,
            attempts=len(record.attempts),
            output=record.output,
            failure=record.failure,
        )

    @staticmethod
    def _fallback_reason(task: Task) -> str | None:
        if task.planner is None or not task.planning_attempts:
            return None
        return "; ".join(_describe_attempt(a) for a in task.planning_attempts)

    def _message(self, task: Task) -> str:
        # The tool name is always shown: descriptions may come from a model and are not trusted.
        lines = [
            f"- [{r.id}] {r.action.description or r.action.tool} ({r.action.tool}): "
            f"{_STATUS_TEXT.get(r.status, r.status.value)}"
            for r in task.actions
        ]
        match task.state:
            case TaskState.COMPLETED:
                if not task.actions:
                    if task.reply:
                        return task.reply
                    if task.requirements:
                        return "Listo: ya se cumplía (verificado); no hizo falta ninguna acción."
                    return "Listo."
                # An action that ends EXECUTED (not VERIFIED) had no verifier available.
                unverified = any(r.status is ActionStatus.EXECUTED for r in task.actions)
                head = "Listo (algunas acciones no se verificaron)." if unverified else "Listo."
                return "\n".join([head, *lines])
            case TaskState.WAITING_PERMISSION:
                head = "Necesito tu confirmación para continuar:"
                for r in task.actions:
                    if r.status is ActionStatus.AWAITING_CONFIRMATION and r.confirmation:
                        args = json.dumps(r.confirmation.arguments, ensure_ascii=False)
                        head += (
                            f"\n  {r.confirmation.tool} {args} (acción {r.id}): "
                            f"{r.confirmation.description} - motivo: "
                            f"{'; '.join(r.confirmation.reasons)}"
                        )
                        target = describe_target(r.confirmation.target)
                        if target:
                            head += f"\n    objetivo: {target}"
                return "\n".join([head, *lines])
            case TaskState.FAILED:
                reason = task.failure.message if task.failure else "error desconocido"
                return "\n".join([f"No se pudo completar la tarea: {reason}", *lines])
            case TaskState.CANCELLED:
                return "\n".join(["Tarea cancelada.", *lines])
            case TaskState.STOPPED:
                reason = task.failure.message if task.failure else "límite alcanzado"
                return "\n".join([f"Tarea detenida de forma segura: {reason}", *lines])
            case _:
                return "\n".join([f"Estado de la tarea: {task.state.value}", *lines])


def describe_target(target: dict[str, Any] | None) -> str | None:
    """Human description of a resolved target, for confirmation prompts."""
    if not target:
        return None
    parts = []
    window = target.get("window")
    if isinstance(window, dict):
        parts.append(f"ventana '{window.get('title')}' ({window.get('process')})")
    application = target.get("application")
    if isinstance(application, dict) and not parts:
        parts.append(f"{application.get('display_name')} ({application.get('executable')})")
    if target.get("unsaved_changes_suspected"):
        parts.append("posibles cambios sin guardar")
    return "; ".join(parts) or None


def _describe_attempt(attempt: PlanningAttempt) -> str:
    who = "el modelo" if attempt.planner == "model" else f"el planner {attempt.planner}"
    if attempt.failure.kind is ErrorKind.TIMEOUT:
        return f"{who} no respondió a tiempo"
    if attempt.stage == "validate":
        return f"el plan de {who} no pasó la validación"
    if attempt.failure.details.get("cause") == "unavailable":
        return f"{who} no está disponible"
    return f"{who} no devolvió un plan utilizable"
