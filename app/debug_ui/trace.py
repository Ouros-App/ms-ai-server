from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

_TRACE: ContextVar[dict[str, Any] | None] = ContextVar("debug_trace", default=None)
_MAX_STRING = 20_000
_MAX_DEPTH = 5
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
    "api_key",
    "apikey",
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _safe_keyed(key: object, value: Any, depth: int) -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"
    return _safe(value, depth)


def _safe(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return repr(value)[:_MAX_STRING]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "…"
    if isinstance(value, dict):
        return {
            str(key): _safe_keyed(key, item, depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item, depth + 1) for item in list(value)[:100]]
    return repr(value)[:_MAX_STRING]


@contextmanager
def capture_debug_trace() -> Iterator[list[dict[str, Any]]]:
    """Capture request-local diagnostic events without affecting normal traffic."""

    state: dict[str, Any] = {"started_at": perf_counter(), "events": []}
    marker = _TRACE.set(state)
    try:
        yield state["events"]
    finally:
        _TRACE.reset(marker)


def trace_event(event: str, **payload: Any) -> None:
    """Append one sanitized event when a debug capture is active."""

    state = _TRACE.get()
    if state is None:
        return
    elapsed_ms = (perf_counter() - state["started_at"]) * 1000
    state["events"].append(
        {
            "t_ms": round(elapsed_ms, 1),
            "event": event,
            **{
                key: _safe_keyed(key, value, 0)
                for key, value in payload.items()
            },
        }
    )
