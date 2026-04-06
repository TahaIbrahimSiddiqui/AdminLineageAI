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
    exact_match: list[str],
    relationship: str,
    include_reason: bool,
    batch_items: list[dict[str, Any]],
    allow_external_grounding: bool = False,
) -> str:
    """Create strict JSON-only adjudication prompt for one batch."""

    payload = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "exact_match": exact_match,
        "requested_relationship": relationship,
        "include_reason": include_reason,
        "items": batch_items,
    }

    response_fields = (
        '{"decisions":[{"from_key":"...","links":[{"to_key":"candidate_to_key_or_null",'
        '"link_type":"rename|split|merge|transfer|no_match|unknown",'
        '"relationship":"father_to_father|father_to_child|child_to_father|child_to_child|unknown",'
        '"score":0.0,"evidence":"..."}]}]}'
    )
    if include_reason:
        response_fields = (
            '{"decisions":[{"from_key":"...","links":[{"to_key":"candidate_to_key_or_null",'
            '"link_type":"rename|split|merge|transfer|no_match|unknown",'
            '"relationship":"father_to_father|father_to_child|child_to_father|child_to_child|unknown",'
            '"score":0.0,"evidence":"...","reason":"..."}]}]}'
        )

    reason_rule = (
        "8) Include a short `reason` field with 1-3 factual sentences. "
        "It must explain the match without chain-of-thought.\n"
        if include_reason
        else "8) Do not include a `reason` field in the JSON.\n"
    )

    grounding_rule = (
        "1) Candidate selection is limited to the supplied shortlist and exact-match context. "
        "If grounding tools are available, you may use them only to verify names, geography, "
        "dates, and lineage facts before choosing among the supplied candidates.\n"
        if allow_external_grounding
        else "1) Use only the provided names and hierarchical context.\n"
    )

    return (
        "You are an administrative evolution key adjudication engine.\n"
        "Task: choose mappings from period A units to period B candidates.\n"
        "Rules:\n"
        f"{grounding_rule}"
        "2) Respect the exact_match context exactly.\n"
        "3) Return strict JSON only (no markdown, no prose).\n"
        "4) Return one decision for every input from_key.\n"
        "5) `to_key` must be one of the supplied candidates for that from_key or null.\n"
        "6) Many-to-many links are allowed. "
        "Use rename, split, merge, or transfer only when supported by the "
        "candidate list and context. Prefer unknown or no_match instead of guessing.\n"
        "7) `relationship` is separate from `link_type`. "
        "Use one of father_to_father, father_to_child, child_to_father, "
        "child_to_child, or unknown. If requested_relationship is not auto, use that value "
        "for matched links. For unknown or no_match, use relationship=unknown.\n"
        f"{reason_rule}"
        "Evidence must be short factual summaries, no chain-of-thought.\n\n"
        "Required response JSON:\n"
        f"{response_fields}\n\n"
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
