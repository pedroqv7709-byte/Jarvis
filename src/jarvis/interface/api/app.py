"""Local HTTP API over a :class:`JarvisRuntime`.

The API is a thin adapter: it validates request shapes, calls the runtime, and maps results to
public views. It holds no state of its own and contains no policy or permission logic; approvals
are handed to the domain, which keeps all its guarantees (single use, bound to exact arguments,
policy re-evaluated on resume).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from jarvis import __version__
from jarvis.core.errors import InvalidTransitionError
from jarvis.interface.api.security import LOOPBACK_HOSTS, local_only_guard, token_guard
from jarvis.interface.views import (
    CreateTaskRequest,
    DecisionRequest,
    EventView,
    HealthView,
    TaskAccepted,
    TaskStatus,
    TaskSummary,
    TaskView,
    ToolView,
    event_view,
    state_of,
    status_of,
    task_summary,
    task_view,
    tool_view,
)
from jarvis.orchestration.orchestrator import OrchestrationError
from jarvis.runtime.runtime import JarvisRuntime, RuntimeNotReadyError
from jarvis.state.store import TaskNotFoundError


def create_app(
    runtime: JarvisRuntime,
    *,
    token: str,
    manage_lifecycle: bool = True,
    allowed_hosts: Iterable[str] = LOOPBACK_HOSTS,
) -> FastAPI:
    """Build the app. With ``manage_lifecycle`` the app starts/stops the runtime itself."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_lifecycle:
            await anyio.to_thread.run_sync(runtime.start)
        try:
            yield
        finally:
            if manage_lifecycle:
                await anyio.to_thread.run_sync(runtime.stop)

    app = FastAPI(
        title="Jarvis local API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Registered last = runs first: reject foreign Host/Origin before looking at the token.
    app.middleware("http")(token_guard(token))
    app.middleware("http")(local_only_guard(allowed_hosts))

    @app.exception_handler(TaskNotFoundError)
    async def _not_found(_: Request, exc: TaskNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": f"task not found: {exc}"}, status_code=404)

    @app.exception_handler(OrchestrationError)
    @app.exception_handler(InvalidTransitionError)
    async def _conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(RuntimeNotReadyError)
    async def _not_ready(_: Request, exc: RuntimeNotReadyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    def view(task_id: str) -> TaskView:
        return task_view(*runtime.snapshot(task_id))

    @app.get("/health", response_model=HealthView)
    def health() -> JSONResponse:
        h = runtime.health()
        body = HealthView(
            status="ok" if h.status == "ok" else "error",
            version=h.version,
            runtime=h.runtime.value,
            state_store=h.state_store,
            worker=h.worker,
            queued=h.queued,
            planner=h.planner,
        )
        return JSONResponse(body.model_dump(), status_code=200 if h.status == "ok" else 503)

    @app.post("/v1/tasks", response_model=TaskAccepted, status_code=202)
    def create_task(body: CreateTaskRequest) -> TaskAccepted:
        task = runtime.create_task(body.text, body.metadata)
        return TaskAccepted(task_id=task.id, status=status_of(task.state))

    @app.get("/v1/tasks", response_model=list[TaskSummary])
    def list_tasks(
        status: TaskStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 20,
    ) -> list[TaskSummary]:
        state = state_of(status) if status else None
        return [task_summary(t) for t in runtime.tasks(state, limit)]

    @app.get("/v1/tasks/{task_id}", response_model=TaskView)
    def get_task(task_id: str) -> TaskView:
        return view(task_id)

    @app.post("/v1/tasks/{task_id}/approve", response_model=TaskView, status_code=202)
    def approve(task_id: str, body: DecisionRequest) -> TaskView:
        runtime.decide(task_id, body.action_id, approved=True)
        return view(task_id)

    @app.post("/v1/tasks/{task_id}/reject", response_model=TaskView, status_code=202)
    def reject(task_id: str, body: DecisionRequest) -> TaskView:
        runtime.decide(task_id, body.action_id, approved=False)
        return view(task_id)

    @app.post("/v1/tasks/{task_id}/cancel", response_model=TaskView)
    def cancel(task_id: str) -> TaskView:
        runtime.cancel(task_id)
        return view(task_id)

    @app.get("/v1/tasks/{task_id}/events", response_model=list[EventView])
    def events(task_id: str) -> list[EventView]:
        return [event_view(e) for e in runtime.events(task_id)]

    @app.get("/v1/tools", response_model=list[ToolView])
    def tools() -> list[ToolView]:
        return [tool_view(spec) for spec in runtime.tools()]

    return app
