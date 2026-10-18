"""Retry with exponential backoff for transient broker/data errors."""
from __future__ import annotations

import logging
import time
from functools import wraps

log = logging.getLogger("trading_agent.execution")


def is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "temporar"))


def with_retries(max_retries: int = 4, base_delay: float = 1.0):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt >= max_retries or not is_transient(exc):
                        raise
                    delay = base_delay * 2 ** attempt
                    log.warning("%s failed (%s); retry %d/%d in %.0fs", fn.__name__, exc,
                                attempt + 1, max_retries, delay)
                    time.sleep(delay)
        return wrapper
    return deco
