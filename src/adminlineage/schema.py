"""Schema constants and output definitions."""

from __future__ import annotations

OUTPUT_SCHEMA_VERSION = "2.0.0"
PROMPT_SCHEMA_VERSION = "2.0.0"

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

CROSSWALK_BASE_COLUMNS = [
    "from_name",
    "to_name",
    "from_canonical_name",
    "to_canonical_name",
    "from_id",
    "to_id",
    "score",
    "link_type",
    "relationship",
    "evidence",
    "reason",
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
        "relationship_enum": list(RELATIONSHIP_TYPES),
        "notes": {
            "exact_match_columns": "Appended dynamically from request.exact_match",
            "constraints_passed": "JSON object serialized as dict in DataFrame cells",
        },
    }
