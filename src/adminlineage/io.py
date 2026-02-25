"""Data loading and output IO helpers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LoadedFrames, RunConfig
from .utils import ensure_dir, safe_git_sha


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet based on file extension."""

    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path_obj)
    raise ValueError(f"Unsupported input format for {path_obj}. Use .csv or .parquet")


def _load_via_hook(callable_path: str, config_payload: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Load dataframes through module:function hook contract."""

    if ":" not in callable_path:
        raise ValueError("Hook callable must use module:function format")

    module_name, func_name = callable_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    result = func(config_payload)

    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise ValueError("Hook must return (df_from, df_to) or (df_from, df_to, aliases_df)")

    if len(result) == 2:
        return result[0], result[1], None
    return result[0], result[1], result[2]


def load_frames(config: RunConfig, *, cwd: str | Path | None = None) -> LoadedFrames:
    """Load input DataFrames according to config.data mode."""

    data_cfg = config.data
    if data_cfg.mode == "files":
        df_from = read_dataframe(data_cfg.from_path)
        df_to = read_dataframe(data_cfg.to_path)
        aliases = read_dataframe(data_cfg.aliases_path) if data_cfg.aliases_path else None
        return LoadedFrames(
            df_from=df_from,
            df_to=df_to,
            aliases=aliases,
            loader_metadata={"mode": "files", "from_path": data_cfg.from_path, "to_path": data_cfg.to_path},
        )

    params = dict(data_cfg.params)
    df_from, df_to, aliases = _load_via_hook(data_cfg.callable, params)
    loader_metadata = {
        "mode": "python_hook",
        "callable": data_cfg.callable,
        "params": params,
        "git_sha": safe_git_sha(cwd=cwd),
    }
    return LoadedFrames(df_from=df_from, df_to=df_to, aliases=aliases, loader_metadata=loader_metadata)


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
