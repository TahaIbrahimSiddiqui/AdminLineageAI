"""Data loading and output IO helpers."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LoadedFrames, RunConfig
from .utils import ensure_dir, safe_git_sha


def _resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute() or base_dir is None:
        return path_obj
    return Path(base_dir) / path_obj


def read_dataframe(path: str | Path, *, base_dir: str | Path | None = None) -> pd.DataFrame:
    """Read CSV or Parquet based on file extension."""

    path_obj = _resolve_path(path, base_dir=base_dir)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path_obj)
    raise ValueError(f"Unsupported input format for {path_obj}. Use .csv or .parquet")


@contextmanager
def _temporary_sys_path(path: str | Path | None):
    path_str = str(path) if path is not None else None
    inserted = False
    if path_str and path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path = [item for item in sys.path if item != path_str]


def _load_via_hook(
    callable_path: str,
    config_payload: dict[str, Any],
    *,
    import_base_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load dataframes through module:function hook contract."""

    if ":" not in callable_path:
        raise ValueError("Hook callable must use module:function format")

    module_name, func_name = callable_path.split(":", maxsplit=1)
    with _temporary_sys_path(import_base_dir):
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        result = func(config_payload)

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Hook must return (df_from, df_to)")

    return result[0], result[1]


def load_frames(config: RunConfig, *, cwd: str | Path | None = None) -> LoadedFrames:
    """Load input DataFrames according to config.data mode."""

    data_cfg = config.data
    base_dir = config.source_dir or (Path(cwd) if cwd is not None else Path.cwd())
    if data_cfg.mode == "files":
        from_path = _resolve_path(data_cfg.from_path, base_dir=base_dir)
        to_path = _resolve_path(data_cfg.to_path, base_dir=base_dir)
        df_from = read_dataframe(from_path)
        df_to = read_dataframe(to_path)
        return LoadedFrames(
            df_from=df_from,
            df_to=df_to,
            loader_metadata={
                "mode": "files",
                "from_path": str(from_path),
                "to_path": str(to_path),
            },
        )

    params = dict(data_cfg.params)
    df_from, df_to = _load_via_hook(
        data_cfg.callable,
        params,
        import_base_dir=base_dir,
    )
    loader_metadata = {
        "mode": "python_hook",
        "callable": data_cfg.callable,
        "params": params,
        "git_sha": safe_git_sha(cwd=base_dir),
    }
    return LoadedFrames(df_from=df_from, df_to=df_to, loader_metadata=loader_metadata)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record to JSONL file."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records; return empty list if file missing."""

    path_obj = Path(path)
    if not path_obj.exists():
        return []

    records: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with deterministic formatting."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
