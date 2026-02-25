"""Retry helpers for transient LLM errors."""

from __future__ import annotations

import random
import time
from typing import Any, Callable


def retry_call(
    fn: Callable[[], Any],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter_seconds: float,
    retry_exceptions: tuple[type[Exception], ...],
) -> Any:
    """Retry a callable with exponential backoff and jitter."""

    attempt = 1
    while True:
        try:
            return fn()
        except retry_exceptions:
            if attempt >= max_attempts:
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            delay += random.uniform(0, jitter_seconds)
            time.sleep(delay)
            attempt += 1
