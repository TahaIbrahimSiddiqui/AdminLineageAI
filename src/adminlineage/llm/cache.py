"""SQLite cache for LLM responses."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..utils import ensure_dir, stable_hash


class SQLiteCache:
    """Simple local SQLite cache keyed by model/prompt/schema version."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def make_key(model: str, prompt: str, schema_version: str) -> tuple[str, str]:
        prompt_hash = stable_hash(prompt)
        key = stable_hash({"model": model, "prompt_hash": prompt_hash, "schema": schema_version})
        return key, prompt_hash

    def get(self, *, model: str, prompt: str, schema_version: str) -> dict[str, Any] | None:
        key, _ = self.make_key(model=model, prompt=prompt, schema_version=schema_version)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(
        self,
        *,
        model: str,
        prompt: str,
        schema_version: str,
        response_json: dict[str, Any],
        created_at: str,
    ) -> None:
        key, prompt_hash = self.make_key(model=model, prompt=prompt, schema_version=schema_version)
        payload = json.dumps(response_json, ensure_ascii=True, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    key, model, prompt_hash, schema_version, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, model, prompt_hash, schema_version, payload, created_at),
            )
            conn.commit()
