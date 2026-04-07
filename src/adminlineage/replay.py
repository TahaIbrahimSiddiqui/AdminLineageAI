"""Helpers for exact replay of completed evolution-key runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import normalize_nullable_output_columns
from .utils import ensure_dir, now_iso, stable_hash

_CROSSWALK_JSON = "crosswalk.records.json"
_REVIEW_QUEUE_JSON = "review_queue.records.json"
_LINKS_RAW_JSONL = "links_raw.jsonl"
_GROUNDING_NOTES_JSONL = "grounding_notes.jsonl"
_MANIFEST_JSON = "replay_manifest.json"


def _unique_columns(columns: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for col in columns:
        if col is None or col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def replay_columns(
    *,
    map_col: str,
    exact_match: list[str],
    id_col: str | None,
    extra_context_cols: list[str],
) -> list[str]:
    """Columns whose values can change the final output for one side of the run."""

    return _unique_columns([map_col, *exact_match, id_col, *extra_context_cols])


def _frame_records(df: pd.DataFrame, *, columns: list[str]) -> list[dict[str, Any]]:
    materialized = {
        col: (df[col] if col in df.columns else [None] * len(df))
        for col in columns
    }
    selected = pd.DataFrame(materialized, columns=columns)
    return json.loads(selected.to_json(orient="records", date_format="iso"))


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[str, str]:
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=True)
        return stable_hash(record), serialized

    return sorted(records, key=sort_key)


def frame_fingerprint(df: pd.DataFrame, *, columns: list[str]) -> str:
    """Hash a semantic view of a dataframe while ignoring row order."""

    payload = {
        "columns": columns,
        "records": _sorted_records(_frame_records(df, columns=columns)),
    }
    return stable_hash(payload)


def build_replay_identity(
    *,
    request_payload: dict[str, Any],
    llm_backend: str,
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    map_col_from: str,
    map_col_to: str,
    exact_match: list[str],
    id_col_from: str | None,
    id_col_to: str | None,
    extra_context_cols: list[str],
) -> dict[str, Any]:
    """Build the exact-replay identity from semantic inputs only."""

    from_columns = replay_columns(
        map_col=map_col_from,
        exact_match=exact_match,
        id_col=id_col_from,
        extra_context_cols=extra_context_cols,
    )
    to_columns = replay_columns(
        map_col=map_col_to,
        exact_match=exact_match,
        id_col=id_col_to,
        extra_context_cols=extra_context_cols,
    )
    from_fingerprint = frame_fingerprint(df_from, columns=from_columns)
    to_fingerprint = frame_fingerprint(df_to, columns=to_columns)

    identity = {
        "request": request_payload,
        "llm_backend": llm_backend,
        "extra_context_cols": extra_context_cols,
        "from_columns": from_columns,
        "to_columns": to_columns,
        "from_fingerprint": from_fingerprint,
        "to_fingerprint": to_fingerprint,
    }
    return {
        "replay_key": stable_hash(identity)[:24],
        "identity": identity,
        "from_fingerprint": from_fingerprint,
        "to_fingerprint": to_fingerprint,
    }


def resolve_replay_store_dir(replay_store_dir: str | Path | None) -> Path:
    """Resolve the replay store path when replay mode is enabled."""

    return ensure_dir(replay_store_dir or ".adminlineage_replay")


def _serialize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": list(df.columns),
        "records": json.loads(df.to_json(orient="records", date_format="iso")),
    }


def _deserialize_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    columns = list(payload.get("columns", []))
    records = list(payload.get("records", []))
    if not records:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    for col in columns:
        if col not in frame.columns:
            frame[col] = None
    return normalize_nullable_output_columns(frame[columns])


def publish_replay_bundle(
    *,
    replay_dir: str | Path,
    replay_key: str,
    source_run_id: str,
    request_payload: dict[str, Any],
    counts: dict[str, Any],
    llm_backend: str,
    identity: dict[str, Any],
    crosswalk: pd.DataFrame,
    review_queue: pd.DataFrame,
    links_raw_path: str | Path,
    grounding_notes_path: str | Path | None = None,
) -> None:
    """Persist a canonical replay bundle after a clean live run."""

    replay_dir = ensure_dir(replay_dir)
    manifest = {
        "replay_key": replay_key,
        "source_run_id": source_run_id,
        "request": request_payload,
        "counts": counts,
        "llm_backend": llm_backend,
        "identity": identity,
        "created_at": now_iso(),
    }

    (replay_dir / _CROSSWALK_JSON).write_text(
        json.dumps(_serialize_dataframe(crosswalk), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    (replay_dir / _REVIEW_QUEUE_JSON).write_text(
        json.dumps(_serialize_dataframe(review_queue), indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    shutil.copy2(links_raw_path, replay_dir / _LINKS_RAW_JSONL)
    if grounding_notes_path is not None:
        grounding_notes_path = Path(grounding_notes_path)
        if grounding_notes_path.exists():
            shutil.copy2(grounding_notes_path, replay_dir / _GROUNDING_NOTES_JSONL)
    (replay_dir / _MANIFEST_JSON).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def load_replay_bundle(replay_dir: str | Path) -> dict[str, Any] | None:
    """Load a replay bundle; return None when no bundle exists."""

    replay_dir = Path(replay_dir)
    manifest_path = replay_dir / _MANIFEST_JSON
    if not manifest_path.exists():
        return None

    crosswalk_path = replay_dir / _CROSSWALK_JSON
    review_queue_path = replay_dir / _REVIEW_QUEUE_JSON
    links_raw_path = replay_dir / _LINKS_RAW_JSONL
    grounding_notes_path = replay_dir / _GROUNDING_NOTES_JSONL
    for required_path in (crosswalk_path, review_queue_path, links_raw_path):
        if not required_path.exists():
            raise ValueError(f"Replay bundle is missing {required_path.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crosswalk = _deserialize_dataframe(json.loads(crosswalk_path.read_text(encoding="utf-8")))
    review_queue = _deserialize_dataframe(
        json.loads(review_queue_path.read_text(encoding="utf-8"))
    )

    return {
        "manifest": manifest,
        "crosswalk": crosswalk,
        "review_queue": review_queue,
        "links_raw_path": links_raw_path,
        "grounding_notes_path": grounding_notes_path if grounding_notes_path.exists() else None,
    }
