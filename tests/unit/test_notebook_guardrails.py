from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "execute_notebook_with_guardrails.py"
)
_SPEC = importlib.util.spec_from_file_location("execute_notebook_with_guardrails", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_no_ai_batches_expected_detects_zero_pending_resume(tmp_path: Path):
    run_log_path = tmp_path / "run.log"
    run_log_path.write_text(
        "\n".join(
            [
                "2026-04-09T00:00:00 | INFO | run_id=abc stage=start",
                (
                    "2026-04-09T00:00:01 | INFO | run_id=abc "
                    "stage=resume exact_mode=to completed=3 pending=0"
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert _MODULE._no_ai_batches_expected(run_log_path) is True


def test_no_ai_batches_expected_is_false_when_pending_ai_work_remains(tmp_path: Path):
    run_log_path = tmp_path / "run.log"
    run_log_path.write_text(
        "2026-04-09T00:00:01 | INFO | run_id=abc stage=resume exact_mode=to completed=2 pending=1",
        encoding="utf-8",
    )

    assert _MODULE._no_ai_batches_expected(run_log_path) is False
