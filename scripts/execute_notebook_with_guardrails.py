from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _executor_code(notebook_path: Path, output_path: Path) -> str:
    return f"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

notebook_path = Path(r"{notebook_path}")
output_path = Path(r"{output_path}")

with notebook_path.open("r", encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

client = NotebookClient(notebook, timeout=None, kernel_name="python3", allow_errors=False)
client.execute()

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)
"""


def _has_processed_batch(links_raw_path: Path) -> bool:
    if not links_raw_path.exists():
        return False

    for line in links_raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok" and record.get("match_stage") in {"ai", "grounded"}:
            return True
    return False


def _observed_progress_tokens(run_log_path: Path) -> set[str]:
    if not run_log_path.exists():
        return set()

    tokens: set[str] = set()
    for line in run_log_path.read_text(encoding="utf-8").splitlines():
        if "stage=adjudication" in line and " batch=" in line:
            batch_part = line.split(" batch=", maxsplit=1)[1]
            batch_label = batch_part.split(" ", maxsplit=1)[0].strip()
            if batch_label:
                tokens.add(f"batch:{batch_label}")
        elif "stage=second_stage" in line and " primary_key=" in line and "| INFO |" in line:
            primary_key = (
                line.split(" primary_key=", maxsplit=1)[1]
                .split(" ", maxsplit=1)[0]
                .strip()
            )
            step = ""
            if " step=" in line:
                step = line.split(" step=", maxsplit=1)[1].split(" ", maxsplit=1)[0].strip()
            if primary_key:
                tokens.add(f"second_stage:{step}:{primary_key}")
        elif "stage=grounding" in line and " from_key=" in line and "| INFO |" in line:
            from_key = line.split(" from_key=", maxsplit=1)[1].split(" ", maxsplit=1)[0].strip()
            if from_key:
                tokens.add(f"grounding:{from_key}")
    return tokens


def _no_ai_batches_expected(run_log_path: Path) -> bool:
    if not run_log_path.exists():
        return False

    for line in run_log_path.read_text(encoding="utf-8").splitlines():
        if "stage=finish" in line:
            return True
        if "stage=resume" in line and " pending=0" in line:
            return True
    return False


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--first-batch-minutes", type=float, default=5.0)
    parser.add_argument("--batch-stall-minutes", type=float, default=20.0)
    parser.add_argument("--cwd", required=False)
    args = parser.parse_args()

    notebook_path = Path(args.notebook).resolve()
    output_path = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    cwd = Path(args.cwd).resolve() if args.cwd else notebook_path.parent.resolve()
    run_log_path = run_dir / "run.log"
    links_raw_path = run_dir / "links_raw.jsonl"

    proc = subprocess.Popen(
        [sys.executable, "-c", _executor_code(notebook_path, output_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    start = time.monotonic()
    first_batch_deadline = start + args.first_batch_minutes * 60.0
    batch_stall_seconds = args.batch_stall_minutes * 60.0
    seen_first_batch = False
    seen_tokens = _observed_progress_tokens(run_log_path)
    last_batch_progress_at = start

    while proc.poll() is None:
        current_tokens = _observed_progress_tokens(run_log_path)
        if current_tokens - seen_tokens:
            seen_tokens = current_tokens
            last_batch_progress_at = time.monotonic()

        if not seen_first_batch and _has_processed_batch(links_raw_path):
            seen_first_batch = True
            last_batch_progress_at = time.monotonic()
        elif not seen_first_batch and _no_ai_batches_expected(run_log_path):
            seen_first_batch = True
            last_batch_progress_at = time.monotonic()

        now = time.monotonic()
        if not seen_first_batch and now > first_batch_deadline:
            _terminate_process(proc)
            raise RuntimeError(
                f"First batch was not processed within {args.first_batch_minutes} minutes."
            )

        if seen_tokens and now - last_batch_progress_at > batch_stall_seconds:
            _terminate_process(proc)
            raise RuntimeError(
                f"Notebook batch progress stalled for more than {args.batch_stall_minutes} minutes."
            )

        time.sleep(5)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        raise RuntimeError(f"Notebook execution failed with exit code {proc.returncode}.")

    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
