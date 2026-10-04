"""Verification: confirm that an executed action actually had its intended effect.

"The tool returned success" (EXECUTED) is not the same as "the effect is observable" (VERIFIED).
Verifiers are pluggable: a tool may verify itself, and external verifiers (e.g. one that checks a
window really opened via UI Automation, or a screenshot-based check) can be registered per tool
without modifying the tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from jarvis.core.action import VerificationResult, VerificationStatus
from jarvis.tools.base import Tool, ToolArguments, ToolContext


class Verifier(Protocol):
    def verify(
        self, tool: Tool[Any], args: ToolArguments, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult | None:
        """Return a result, or ``None`` when this verifier does not apply to the tool."""
        ...


class ToolSelfVerifier:
    """Delegates to the tool's own ``verify`` method, if it defines one."""

    def verify(
        self, tool: Tool[Any], args: ToolArguments, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult | None:
        if not tool.has_verifier:
            return None
        return tool.verify(args, output, ctx)


class VerificationService:
    def __init__(self, verifiers: Sequence[Verifier] | None = None) -> None:
        self._verifiers: tuple[Verifier, ...] = (
            tuple(verifiers) if verifiers is not None else (ToolSelfVerifier(),)
        )

    def verify(
        self, tool: Tool[Any], args: ToolArguments, output: dict[str, Any], ctx: ToolContext
    ) -> VerificationResult:
        results: list[VerificationResult] = []
        for verifier in self._verifiers:
            try:
                result = verifier.verify(tool, args, output, ctx)
            except Exception as exc:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    message=f"verifier {type(verifier).__name__} raised {type(exc).__name__}: "
                    f"{exc}",
                )
            if result is not None:
                results.append(result)

        if not results:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                message=f"no verifier available for {tool.spec.name}",
            )
        for result in results:
            if result.status is not VerificationStatus.VERIFIED:
                return result
        evidence: dict[str, Any] = {}
        for result in results:
            evidence.update(result.evidence)
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            message="; ".join(r.message for r in results if r.message),
            evidence=evidence,
        )
