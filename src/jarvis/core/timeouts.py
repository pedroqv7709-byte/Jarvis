"""Hard wall-clock timeouts for blocking calls (tools, model providers).

The call runs in a *daemon* thread. If it does not finish in time the caller gets
:class:`OperationTimeoutError` and continues; Python cannot kill the thread, so it is abandoned
(daemon threads never block interpreter exit). Callers must therefore treat the outcome of a
timed-out call as unknown.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class OperationTimeoutError(TimeoutError):
    def __init__(self, operation: str, seconds: float) -> None:
        super().__init__(f"{operation} did not finish within {seconds:g}s")
        self.operation = operation
        self.seconds = seconds


def run_with_timeout[T](fn: Callable[[], T], timeout: float | None, *, name: str) -> T:
    """Run ``fn`` and return its result, or raise :class:`OperationTimeoutError`."""
    if timeout is None:
        return fn()
    if timeout <= 0:
        raise OperationTimeoutError(name, 0)

    result: list[T] = []
    error: list[BaseException] = []
    done = threading.Event()

    def target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=target, name=f"jarvis-{name}", daemon=True)
    worker.start()
    if not done.wait(timeout):
        raise OperationTimeoutError(name, timeout)
    if error:
        raise error[0]
    return result[0]
