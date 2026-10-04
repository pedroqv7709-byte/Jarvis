"""Composition root: the one place where concrete implementations are chosen and wired."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from jarvis.agent.completion import CompletionChecker
from jarvis.agent.limits import AgentLimits
from jarvis.agent.observers import WindowsObserver
from jarvis.agent.requirements import RequirementExtractor
from jarvis.config import Settings
from jarvis.context.builder import ContextBuilder
from jarvis.context.providers import ContextProvider, SystemContextProvider
from jarvis.desktop import WindowsAutomation, build_catalog, create_windows_automation
from jarvis.desktop.catalog import ApplicationCatalog
from jarvis.execution.confirmation import ConfirmationHandler, DeferringConfirmationHandler
from jarvis.execution.executor import ActionExecutor
from jarvis.execution.retry import RetryPolicy
from jarvis.execution.runner import TaskRunner
from jarvis.execution.selection import ToolSelector
from jarvis.execution.verification import VerificationService
from jarvis.models.base import ModelProvider
from jarvis.models.factory import create_model_provider
from jarvis.models.router import ModelRouter
from jarvis.observability.audit import Auditor, AuditSink, JsonlAuditLog
from jarvis.observability.tracing import Tracer
from jarvis.orchestration.jarvis import Jarvis
from jarvis.orchestration.orchestrator import Orchestrator
from jarvis.permissions.model import PermissionSet
from jarvis.planning.base import Planner, StepPlanner
from jarvis.planning.fallback import FallbackPlanner
from jarvis.planning.follower import PlanFollower
from jarvis.planning.model_planner import ModelPlanner
from jarvis.planning.rule_based import RuleBasedPlanner
from jarvis.planning.validation import PlanValidator
from jarvis.safety.policy import PolicyEngine
from jarvis.safety.request_guard import RequestGuard
from jarvis.state.sqlite import SqliteStateStore
from jarvis.state.store import StateStore
from jarvis.tools.base import Tool
from jarvis.tools.builtin.filesystem import filesystem_tools
from jarvis.tools.builtin.windows import windows_tools
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.workspace import Workspace


def build_jarvis(
    settings: Settings | None = None,
    *,
    planner: Planner | StepPlanner | None = None,
    tools: Sequence[Tool[Any]] | None = None,
    confirmations: ConfirmationHandler | None = None,
    policy: PolicyEngine | None = None,
    store: StateStore | None = None,
    audit: AuditSink | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    verification: VerificationService | None = None,
    tracer: Tracer | None = None,
    model_provider: ModelProvider | None = None,
    windows: WindowsAutomation | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Jarvis:
    """Wire a Jarvis instance.

    ``planner`` replaces planning entirely. Otherwise the planner is derived from settings; a
    ``model_provider`` given here takes the place of the configured one (used by tests to swap
    only the external model while keeping the real pipeline).
    """
    settings = settings or Settings()

    workspace = Workspace(settings.workspace_dir)
    workspace.ensure()
    automation = (
        windows if windows is not None else create_windows_automation(settings.windows_automation)
    )
    catalog = build_catalog(settings.windows_applications)
    registry = ToolRegistry(
        tools if tools is not None else default_tools(settings, automation, catalog)
    )
    store = store if store is not None else SqliteStateStore(settings.state_db_path)
    audit = audit if audit is not None else JsonlAuditLog(settings.audit_log_path)
    auditor = Auditor(audit)

    executor = ActionExecutor(
        selector=ToolSelector(registry),
        policy=policy or PolicyEngine.default(),
        verification=verification or VerificationService(),
        confirmations=confirmations or DeferringConfirmationHandler(),
        retry=RetryPolicy(
            max_attempts=settings.max_attempts_per_action,
            backoff_seconds=settings.retry_backoff_seconds,
            backoff_multiplier=settings.retry_backoff_multiplier,
            max_backoff_seconds=settings.retry_max_backoff_seconds,
        ),
        auditor=auditor,
        workspace=workspace,
        max_total_attempts=settings.max_total_attempts_per_task,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        task_timeout_seconds=settings.task_timeout_seconds,
        sleep=sleep,
    )
    validator = PlanValidator(registry, max_steps=settings.max_plan_steps)
    step_planner: StepPlanner
    if planner is None:
        step_planner = build_planner(settings, validator, model_provider)
    elif isinstance(planner, StepPlanner):
        step_planner = planner
    else:
        step_planner = PlanFollower(planner)  # whole-plan planner: follow its plan step by step
    runner = TaskRunner(
        executor,
        store,
        auditor,
        planner=step_planner,
        validator=validator,
        registry=registry,
        completion=CompletionChecker(
            catalog if automation is not None else None, registry.normalize_arguments
        ),
        observers=[WindowsObserver(automation)] if automation is not None else [],
        limits=agent_limits(settings),
        observe_timeout=settings.tool_timeout_seconds,
    )
    orchestrator = Orchestrator(
        contexts=ContextBuilder(
            permissions=PermissionSet.of(settings.permissions),
            providers=(
                context_providers if context_providers is not None else [SystemContextProvider()]
            ),
            workspace=str(workspace.root),
        ),
        guard=RequestGuard(max_chars=settings.max_request_chars),
        runner=runner,
        store=store,
        auditor=auditor,
        tracer=tracer,
        requirements=RequirementExtractor(catalog if automation is not None else None),
    )
    return Jarvis(
        orchestrator=orchestrator,
        store=store,
        audit=audit,
        registry=registry,
        user_id=settings.user_id,
        planner_description=describe_planner(step_planner),
    )


def agent_limits(settings: Settings) -> AgentLimits:
    return AgentLimits(
        max_iterations=settings.agent_max_iterations,
        max_actions=settings.agent_max_actions,
        max_replans=settings.agent_max_replans,
        task_timeout_seconds=settings.task_timeout_seconds,
        history_steps=settings.agent_history_steps,
        max_observed_windows=settings.agent_max_observed_windows,
        max_prompt_chars=settings.agent_max_prompt_chars,
    )


def default_tools(
    settings: Settings,
    automation: WindowsAutomation | None = None,
    catalog: ApplicationCatalog | None = None,
) -> list[Tool[Any]]:
    """Filesystem tools, plus Windows tools when a desktop backend is available."""
    tools = filesystem_tools()
    if automation is not None:
        tools += windows_tools(
            automation,
            catalog or build_catalog(settings.windows_applications),
            # Leave room inside the per-attempt tool timeout for the launch itself.
            open_wait_seconds=min(
                settings.windows_open_wait_seconds, settings.tool_timeout_seconds * 0.8
            ),
            close_wait_seconds=min(
                settings.windows_close_wait_seconds, settings.tool_timeout_seconds * 0.8
            ),
        )
    return tools


def build_planner(
    settings: Settings, validator: PlanValidator, model_provider: ModelProvider | None = None
) -> StepPlanner:
    """Deterministic planner, or model planner (+ deterministic fallback) when configured."""
    provider = (
        model_provider
        if model_provider is not None
        else create_model_provider(
            settings.model_provider,
            model=settings.model,
            timeout_seconds=settings.model_timeout_seconds,
            api_base=settings.model_api_base,
            structured_mode=settings.model_structured_output,
        )
    )
    if provider is None:
        return RuleBasedPlanner()
    router = ModelRouter()
    router.register(provider)
    planners: list[StepPlanner] = [
        ModelPlanner(
            router,
            timeout_seconds=settings.model_timeout_seconds,
            max_prompt_chars=settings.agent_max_prompt_chars,
        )
    ]
    if settings.planner_fallback:
        planners.append(RuleBasedPlanner())
    return FallbackPlanner(planners, validator)


def describe_planner(planner: StepPlanner) -> str:
    if isinstance(planner, FallbackPlanner):
        return " -> ".join(planner.planners)
    return planner.name
