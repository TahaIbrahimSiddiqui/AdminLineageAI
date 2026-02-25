"""Schema constants and output definitions."""

from __future__ import annotations

OUTPUT_SCHEMA_VERSION = "1.0.0"
PROMPT_SCHEMA_VERSION = "1.0.0"

LINK_TYPES = (
    "rename",
    "split",
    "merge",
    "transfer",
    "no_match",
    "unknown",
)

CROSSWALK_BASE_COLUMNS = [
    "from_name",
    "to_name",
    "from_canonical_name",
    "to_canonical_name",
    "from_id",
    "to_id",
    "score",
    "link_type",
    "evidence",
    "country",
    "year_from",
    "year_to",
    "run_id",
    "from_key",
    "to_key",
    "constraints_passed",
]


def get_output_schema_definition() -> dict:
    """Return a machine-readable output schema definition."""
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "crosswalk_columns": CROSSWALK_BASE_COLUMNS,
        "link_type_enum": list(LINK_TYPES),
        "notes": {
            "anchor_columns": "Appended dynamically from request.anchor_cols",
            "constraints_passed": "JSON object serialized as dict in DataFrame cells",
        },
    }
