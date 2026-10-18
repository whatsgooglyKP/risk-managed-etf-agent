"""Tiny cross-process lock so the daily run and the intraday monitor never edit state at once."""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

log = logging.getLogger("trading_agent.lock")


@contextlib.contextmanager
def state_lock(state_dir: str | Path, timeout_sec: float = 300.0):
    path = Path(state_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows: no locking (use WSL for scheduling)
        yield
        return
    with open(path / ".lock", "w") as fh:
        deadline = time.time() + timeout_sec
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"state lock busy for {timeout_sec:.0f}s ({path / '.lock'})")
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
