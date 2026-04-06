"""Input validation utilities for public API and CLI."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .normalize import normalized_key_frame


def _match_key_columns(map_col: str, exact_match: list[str] | None) -> list[str]:
    return [*(exact_match or []), map_col]


def _normalize_key_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _sample_duplicate_keys(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    normalized = normalized_key_frame(df, key_cols)
    duplicate_rows = normalized.loc[normalized.duplicated(keep=False), key_cols]
    if duplicate_rows.empty:
        return []

    samples: list[dict[str, Any]] = []
    for row in duplicate_rows.drop_duplicates().head(limit).itertuples(index=False, name=None):
        sample = {
            col: _normalize_key_value(value)
            for col, value in zip(key_cols, row, strict=False)
        }
        samples.append(sample)
    return samples


def collapse_duplicate_match_keys(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    side_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse duplicate match keys with first-row-wins semantics."""

    if not key_cols:
        collapsed = df.reset_index(drop=True).copy()
        return collapsed, {
            "side": side_label,
            "key_columns": [],
            "input_rows": int(len(df)),
            "effective_rows": int(len(collapsed)),
            "collapsed_rows": 0,
            "collapsed": False,
            "sample_keys": [],
        }

    # We keep the first row for a repeated match key so matching stays deterministic.
    duplicate_mask = normalized_key_frame(df, key_cols).duplicated(keep="first")
    collapsed = df.loc[~duplicate_mask].reset_index(drop=True).copy()
    report = {
        "side": side_label,
        "key_columns": key_cols,
        "input_rows": int(len(df)),
        "effective_rows": int(len(collapsed)),
        "collapsed_rows": int(duplicate_mask.sum()),
        "collapsed": bool(duplicate_mask.any()),
        "sample_keys": _sample_duplicate_keys(df, key_cols=key_cols),
    }
    return collapsed, report


def _format_duplicate_key_sample(sample_keys: list[dict[str, Any]]) -> str:
    if not sample_keys:
        return "n/a"

    rendered = []
    for sample in sample_keys:
        pairs = ", ".join(f"{col}={value!r}" for col, value in sample.items())
        rendered.append(f"({pairs})")
    return "; ".join(rendered)


def build_duplicate_collapse_warning(report: dict[str, Any]) -> str | None:
    """Render a warning message for duplicate match-key collapse."""

    if not report.get("collapsed"):
        return None

    key_label = ", ".join(report["key_columns"])
    sample_text = _format_duplicate_key_sample(report["sample_keys"])
    return (
        f"Collapsed {report['collapsed_rows']} duplicate {report['side']} rows using match key "
        f"[{key_label}]; {report['effective_rows']} unique rows remain. Duplicate values were "
        "ignored and only the first row per key was kept. If expected, ignore this warning; "
        f"otherwise inspect or fix the data. Sample duplicate keys: {sample_text}."
    )


def validate_inputs_data(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    map_col_from: str,
    map_col_to: str | None,
    exact_match: list[str] | None,
    id_col_from: str | None,
    id_col_to: str | None,
) -> dict[str, Any]:
    """Validate inputs and return diagnostics for callers."""

    errors: list[str] = []
    warnings: list[str] = []
    effective_map_col_to = map_col_to or map_col_from
    from_rows_input = int(len(df_from))
    to_rows_input = int(len(df_to))
    collapse_reports = {
        "from": {
            "side": "df_from",
            "key_columns": _match_key_columns(map_col_from, exact_match),
            "input_rows": from_rows_input,
            "effective_rows": from_rows_input,
            "collapsed_rows": 0,
            "collapsed": False,
            "sample_keys": [],
        },
        "to": {
            "side": "df_to",
            "key_columns": _match_key_columns(effective_map_col_to, exact_match),
            "input_rows": to_rows_input,
            "effective_rows": to_rows_input,
            "collapsed_rows": 0,
            "collapsed": False,
            "sample_keys": [],
        },
    }

    if not country or not str(country).strip():
        errors.append("country is required and must be non-empty")

    if map_col_from not in df_from.columns:
        errors.append(f"df_from missing map_col_from column '{map_col_from}'")

    if effective_map_col_to not in df_to.columns:
        errors.append(f"df_to missing map_col_to column '{effective_map_col_to}'")

    exact_match = exact_match or []
    for col in exact_match:
        if col not in df_from.columns:
            errors.append(f"df_from missing exact_match column '{col}'")
        if col not in df_to.columns:
            errors.append(f"df_to missing exact_match column '{col}'")

    if id_col_from and id_col_from not in df_from.columns:
        errors.append(f"df_from missing id_col_from '{id_col_from}'")
    if id_col_to and id_col_to not in df_to.columns:
        errors.append(f"df_to missing id_col_to '{id_col_to}'")

    if not exact_match:
        warnings.append(
            "No exact_match columns supplied; candidate generation will run globally "
            "and may increase false matches."
        )

    if exact_match and not errors:
        left = set(
            normalized_key_frame(df_from, exact_match).drop_duplicates().itertuples(
                index=False,
                name=None,
            )
        )
        right = set(
            normalized_key_frame(df_to, exact_match).drop_duplicates().itertuples(
                index=False,
                name=None,
            )
        )
        overlap = left & right
        if not overlap:
            warnings.append(
                "exact_match columns were provided but no exact-match tuples overlap "
                "between df_from and df_to"
            )

    if not errors:
        _, collapse_reports["from"] = collapse_duplicate_match_keys(
            df_from,
            key_cols=_match_key_columns(map_col_from, exact_match),
            side_label="df_from",
        )
        _, collapse_reports["to"] = collapse_duplicate_match_keys(
            df_to,
            key_cols=_match_key_columns(effective_map_col_to, exact_match),
            side_label="df_to",
        )
        for report in collapse_reports.values():
            warning = build_duplicate_collapse_warning(report)
            if warning:
                warnings.append(warning)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "map_col_to": effective_map_col_to,
        "from_rows_input": from_rows_input,
        "from_rows_effective": collapse_reports["from"]["effective_rows"],
        "to_rows_input": to_rows_input,
        "to_rows_effective": collapse_reports["to"]["effective_rows"],
        "collapse_reports": collapse_reports,
    }
