"""Shared helper utilities."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


def stable_hash(payload: Any) -> str:
    """Create a deterministic SHA256 hash for a JSON-serializable payload."""

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_run_id(request_dict: dict[str, Any], seed: int) -> str:
    """Build deterministic run_id from request and seed."""

    base = {"request": request_dict, "seed": seed}
    return stable_hash(base)[:16]


def now_iso() -> str:
    """Current UTC timestamp in ISO format."""

    return datetime.now(tz=UTC).isoformat()


def sanitize_name(value: str) -> str:
    """Convert arbitrary names into filesystem-safe slug fragments."""

    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "run"


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield list chunks of fixed maximum size."""

    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def safe_git_sha(cwd: str | Path | None = None) -> str | None:
    """Best-effort git SHA lookup for reproducibility metadata."""

    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return output or None
