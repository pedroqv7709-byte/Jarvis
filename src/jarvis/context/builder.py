"""Builds the :class:`ExecutionContext` for a task."""

from __future__ import annotations

from collections.abc import Sequence

from jarvis.context.model import ApplicationContext, ContextItem, ExecutionContext, UserProfile
from jarvis.context.providers import ContextProvider
from jarvis.core.task import Task
from jarvis.permissions.model import PermissionSet


class ContextBuilder:
    def __init__(
        self,
        permissions: PermissionSet,
        providers: Sequence[ContextProvider] = (),
        workspace: str | None = None,
    ) -> None:
        self._permissions = permissions
        self._providers = tuple(providers)
        self._workspace = workspace

    def build(self, task: Task) -> ExecutionContext:
        request = task.request
        items: list[ContextItem] = []
        degraded: list[str] = []
        for provider in self._providers:
            # A failing provider degrades the context; it must not take the pipeline down.
            try:
                items.extend(provider.provide(request))
            except Exception:
                degraded.append(provider.name)
        active_app = request.metadata.get("active_application")
        return ExecutionContext(
            task_id=task.id,
            request_id=request.id,
            goal=request.text,
            user=UserProfile(id=request.user_id),
            application=ApplicationContext(
                interface=request.interface,
                active_application=active_app if isinstance(active_app, str) else None,
                workspace=self._workspace,
            ),
            permissions=self._permissions,
            task_state=task.state,
            items=tuple(items),
            degraded_sources=tuple(degraded),
        )
