"""Protections for an unauthenticated API that listens on loopback only.

Binding to 127.0.0.1 keeps other machines out, but any web page open in the user's browser can
still send requests to 127.0.0.1. Without these checks a malicious site could, for instance,
POST to ``/v1/tasks/{id}/approve`` (CSRF) or reach the API through DNS rebinding. So:

* ``Host`` must be a loopback name: defeats DNS rebinding (the browser would send the attacker's
  host name).
* A present ``Origin`` must be a loopback origin: browsers attach it to cross-site requests.
* State-changing requests must be ``application/json``: browsers cannot send that cross-origin
  without a CORS preflight, and this API never grants one.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse, Response

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

Middleware = Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]


def _hostname(host_header: str) -> str:
    # urlsplit handles "127.0.0.1:8765", "[::1]:8765" and "localhost".
    return (urlsplit(f"//{host_header}").hostname or "").lower()


def local_only_guard(allowed_hosts: Iterable[str] = LOOPBACK_HOSTS) -> Middleware:
    allowed = frozenset(h.lower() for h in allowed_hosts)

    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if _hostname(request.headers.get("host", "")) not in allowed:
            return JSONResponse({"detail": "invalid Host header"}, status_code=400)
        origin = request.headers.get("origin")
        if origin is not None and (urlsplit(origin).hostname or "").lower() not in allowed:
            return JSONResponse({"detail": "cross-origin requests are not allowed"}, 403)
        if request.method in _UNSAFE_METHODS:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            if content_type != "application/json":
                return JSONResponse({"detail": "Content-Type must be application/json"}, 415)
        return await call_next(request)

    return guard


PUBLIC_PATHS = frozenset({"/health"})


def token_guard(token: str) -> Middleware:
    """Require ``Authorization: Bearer <token>`` on everything except ``/health``."""
    if len(token) < 32:
        raise ValueError("the API token is too short")
    expected = token.encode("utf-8")

    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
        if scheme.casefold() != "bearer" or not credentials:
            return JSONResponse(
                {"detail": "missing API token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(credentials.strip().encode("utf-8"), expected):
            return JSONResponse({"detail": "invalid API token"}, status_code=403)
        return await call_next(request)

    return guard
