"""Input validation utilities for public API and CLI."""

from __future__ import annotations

from typing import Any

import pandas as pd


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

    if not country or not str(country).strip():
        errors.append("country is required and must be non-empty")

    if map_col_from not in df_from.columns:
        errors.append(f"df_from missing map_col_from column '{map_col_from}'")

    effective_map_col_to = map_col_to or map_col_from
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
        left = set(df_from[exact_match].drop_duplicates().itertuples(index=False, name=None))
        right = set(df_to[exact_match].drop_duplicates().itertuples(index=False, name=None))
        overlap = left & right
        if not overlap:
            warnings.append(
                "exact_match columns were provided but no exact-match tuples overlap "
                "between df_from and df_to"
            )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "map_col_to": effective_map_col_to,
    }
