from __future__ import annotations

import pytest

from adminlineage.llm_retry import retry_call


def test_retry_call_retries_then_succeeds():
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("try again")
        return "ok"

    result = retry_call(
        flaky,
        max_attempts=3,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        retry_exceptions=(RuntimeError,),
    )

    assert result == "ok"
    assert calls["count"] == 3


def test_retry_call_raises_after_max_attempts():
    calls = {"count": 0}

    def always_fail() -> None:
        calls["count"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        retry_call(
            always_fail,
            max_attempts=2,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_seconds=0.0,
            retry_exceptions=(RuntimeError,),
        )

    assert calls["count"] == 2
