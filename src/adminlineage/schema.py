"""Schema constants and output definitions."""

from __future__ import annotations

import pandas as pd

OUTPUT_SCHEMA_VERSION = "2.1.0"
PROMPT_SCHEMA_VERSION = "2.3.0"

LINK_TYPES = (
    "rename",
    "split",
    "merge",
    "transfer",
    "no_match",
    "unknown",
)

RELATIONSHIP_TYPES = (
    "father_to_father",
    "father_to_child",
    "child_to_father",
    "child_to_child",
    "unknown",
)

REQUEST_RELATIONSHIP_TYPES = (
    "auto",
    "father_to_father",
    "father_to_child",
    "child_to_father",
    "child_to_child",
)

_CROSSWALK_COLUMNS_BEFORE_EXPLANATIONS = [
    "from_name",
    "to_name",
    "from_canonical_name",
    "to_canonical_name",
    "from_id",
    "to_id",
    "score",
    "link_type",
    "relationship",
]

_CROSSWALK_COLUMNS_AFTER_EXPLANATIONS = [
    "country",
    "year_from",
    "year_to",
    "run_id",
    "from_key",
    "to_key",
    "constraints_passed",
]


def get_crosswalk_base_columns(*, include_evidence: bool) -> list[str]:
    """Return the ordered base crosswalk columns for one run."""

    columns = list(_CROSSWALK_COLUMNS_BEFORE_EXPLANATIONS)
    if include_evidence:
        columns.append("evidence")
    columns.append("reason")
    columns.extend(_CROSSWALK_COLUMNS_AFTER_EXPLANATIONS)
    return columns


def normalize_nullable_output_columns(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Normalize target-side output columns to plain object dtype with Python None."""

    if crosswalk.empty:
        return crosswalk.copy()

    normalized = crosswalk.copy()
    for column in ["to_key", "to_name", "to_canonical_name", "to_id"]:
        if column not in normalized.columns:
            continue
        series = normalized[column].astype(object)
        normalized[column] = series.where(series.notna(), None)
    return normalized


def get_output_schema_definition(*, include_evidence: bool = False) -> dict:
    """Return a machine-readable output schema definition."""
    crosswalk_columns = get_crosswalk_base_columns(include_evidence=include_evidence)
    required_columns = [
        "from_name",
        "from_canonical_name",
        "from_id",
        "score",
        "link_type",
        "relationship",
        "reason",
        "country",
        "year_from",
        "year_to",
        "run_id",
        "from_key",
        "constraints_passed",
    ]
    if include_evidence:
        required_columns.insert(6, "evidence")

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "crosswalk_columns": crosswalk_columns,
        "link_type_enum": list(LINK_TYPES),
        "relationship_enum": list(RELATIONSHIP_TYPES),
        "conditional_columns": {
            "evidence": "Included only when request.evidence is true.",
            "reason": "Always present; empty unless request.reason is true.",
        },
        "required_output_columns": required_columns,
        "notes": {
            "exact_match_columns": "Appended dynamically from request.exact_match",
            "constraints_passed": "JSON object serialized as dict in DataFrame cells",
        },
    }
