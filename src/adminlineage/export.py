"""Crosswalk export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_crosswalk_file(
    input_path: str | Path,
    output_format: str,
    output_path: str | Path | None = None,
) -> Path:
    """Convert crosswalk artifact to target format."""

    in_path = Path(input_path)
    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        raise ValueError("Unsupported input format; expected csv or parquet")

    fmt = output_format.lower()
    if fmt not in {"csv", "parquet", "jsonl"}:
        raise ValueError("output_format must be one of: csv, parquet, jsonl")

    if output_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_export.{fmt}")
    else:
        out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_json(out_path, orient="records", lines=True)

    return out_path
