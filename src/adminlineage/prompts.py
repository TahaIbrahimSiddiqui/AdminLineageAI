"""Prompt builders for Gemini adjudication."""

from __future__ import annotations

import json
from typing import Any

from .schema import PROMPT_SCHEMA_VERSION


def build_batch_prompt(
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    anchor_cols: list[str],
    batch_items: list[dict[str, Any]],
) -> str:
    """Create strict JSON-only adjudication prompt for one batch."""

    payload = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "anchor_columns": anchor_cols,
        "items": batch_items,
    }

    return (
        "You are an administrative evolution key adjudication engine.\n"
        "Task: choose mappings from period A units to period B candidates.\n"
        "Rules:\n"
        "1) Use only the provided names and hierarchical context.\n"
        "2) Respect anchor context exactly.\n"
        "3) Return strict JSON only (no markdown, no prose).\n"
        "4) Many-to-many links are allowed (split/merge/transfer/rename/no_match/unknown).\n"
        "5) Evidence must be short factual summaries (1-3 short bullet-like sentences), no chain-of-thought.\n\n"
        "Required response JSON:\n"
        "{\"decisions\":[{\"from_key\":\"...\",\"links\":[{\"to_key\":\"...\",\"link_type\":\"rename|split|merge|transfer|no_match|unknown\",\"score\":0.0,\"evidence\":\"...\"}]}]}\n\n"
        "INPUT_PAYLOAD_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_repair_prompt(original_prompt: str, invalid_output: str, error_message: str) -> str:
    """Build repair prompt when first model response is malformed."""

    return (
        "Your previous response was invalid JSON for the required schema.\n"
        "Return ONLY valid JSON matching the required schema.\n"
        f"Validation error: {error_message}\n"
        "Do not include markdown or explanation.\n\n"
        "ORIGINAL_PROMPT:\n"
        f"{original_prompt}\n\n"
        "PREVIOUS_INVALID_OUTPUT:\n"
        f"{invalid_output}"
    )
