"""Tracing seam.

Pipeline stages are wrapped in spans through this protocol. The default is a no-op; an
OpenTelemetry-backed implementation can be dropped in later without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol


class Tracer(Protocol):
    def span(self, name: str, **attributes: Any) -> AbstractContextManager[None]: ...


class NoopTracer:
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        yield
