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
    include_evidence: bool,
    include_reason: bool,
    batch_items: list[dict[str, Any]],
    allow_external_grounding: bool = False,
) -> str:
    """Create strict JSON-only adjudication prompt for one or more shortlist items."""

    payload = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "exact_match": exact_match,
        "requested_relationship": relationship,
        "include_evidence": include_evidence,
        "include_reason": include_reason,
        "items": batch_items,
    }

    response_fields = (
        '{"decisions":[{"from_key":"...","links":[{"to_key":"candidate_to_key_or_null",'
        '"link_type":"rename|split|merge|transfer|no_match|unknown",'
        '"relationship":"father_to_father|father_to_child|child_to_father|child_to_child|unknown",'
        '"score":0.0'
    )
    if include_evidence:
        response_fields += ',"evidence":"..."'
    if include_reason:
        response_fields += ',"reason":"..."'
    response_fields += "}]}]}"

    evidence_rule = (
        "10) Include a short `evidence` field with factual summaries only. "
        "No chain-of-thought.\n"
        if include_evidence
        else "10) Do not include an `evidence` field in the JSON.\n"
    )
    reason_rule = (
        "11) Include a short `reason` field with 1-3 factual sentences. "
        "It must explain the match without chain-of-thought.\n"
        if include_reason
        else "11) Do not include a `reason` field in the JSON.\n"
    )

    grounding_rule = (
        "1) Candidate selection is limited to the supplied shortlist and exact-match context. "
        "If grounding tools are available, you may use them only to verify names, geography, "
        "dates, and lineage facts before choosing among the supplied candidates.\n"
        "2) Search grounding is verification only. Never introduce a new unit, new to_key, or "
        "off-shortlist candidate.\n"
        if allow_external_grounding
        else "1) Use only the provided names and hierarchical context.\n"
    )

    return (
        "You are an administrative evolution key adjudication engine.\n"
        "Task: choose mappings from period A units to period B candidates.\n"
        "Rules:\n"
        f"{grounding_rule}"
        "3) Respect the exact_match context exactly.\n"
        "4) Return strict JSON only (no markdown, no prose).\n"
        "5) Return one decision for every input from_key.\n"
        "6) `to_key` must be one of the supplied candidates for that from_key or null.\n"
        "7) Many-to-many links are allowed. "
        "Use rename, split, merge, or transfer only when supported by the "
        "candidate list and context. Prefer unknown or no_match instead of guessing.\n"
        "8) Never use `exact_match` as a `link_type`. "
        "For a direct same-unit match, use `rename`.\n"
        "9) `relationship` is separate from `link_type`. "
        "Use one of father_to_father, father_to_child, child_to_father, "
        "child_to_child, or unknown. If requested_relationship is not auto, use that value "
        "for matched links. For unknown or no_match, use relationship=unknown.\n"
        f"{evidence_rule}"
        f"{reason_rule}"
        "12) If search is inconclusive, prefer unknown or no_match.\n\n"
        "Required response JSON:\n"
        f"{response_fields}\n\n"
        "INPUT_PAYLOAD_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_second_stage_research_prompt(
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    primary_side: str,
    primary_item: dict[str, Any],
) -> str:
    """Create strict JSON prompt for bounded second-stage lineage research."""

    payload = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "primary_side": primary_side,
        "primary_item": primary_item,
    }

    return (
        "You are researching one unmatched administrative unit between two dates.\n"
        "Use Google Search to determine whether the unit was renamed, split, merged, "
        "transferred, dissolved, or remains unclear between year_from and year_to.\n"
        "Return strict JSON only.\n"
        "Rules:\n"
        "1) Focus on one best predecessor or successor name to try next.\n"
        "2) If evidence is weak, use event_type=unknown and lineage_hint=\"\".\n"
        "3) Keep notes short and factual.\n\n"
        "Required response JSON:\n"
        '{"event_type":"rename|split|merge|transfer|dissolved|unknown",'
        '"lineage_hint":"...",'
        '"notes":"..."}\n\n'
        "INPUT_PAYLOAD_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_second_stage_decision_prompt(
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    primary_side: str,
    relationship: str,
    include_evidence: bool,
    include_reason: bool,
    primary_item: dict[str, Any],
    lineage_research: dict[str, Any],
    candidate_subset: list[dict[str, Any]],
) -> str:
    """Create strict JSON prompt for second-stage shortlist adjudication."""

    payload = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "primary_side": primary_side,
        "requested_relationship": relationship,
        "include_evidence": include_evidence,
        "include_reason": include_reason,
        "primary_item": primary_item,
        "lineage_research": lineage_research,
        "candidate_subset": candidate_subset,
    }

    evidence_rule = (
        ',"evidence":"..."' if include_evidence else ""
    )
    reason_rule = (
        ',"reason":"..."' if include_reason else ""
    )

    return (
        "You are adjudicating one unmatched administrative unit "
        "using a refreshed global shortlist.\n"
        "Use only the supplied lineage research and refreshed shortlist candidates.\n"
        "Rules:\n"
        "1) Candidate selection is limited to the supplied `candidate_subset`.\n"
        "2) `selected_secondary_keys` must be drawn only from the shortlist or be an empty list.\n"
        "3) Many-to-many is allowed when the shortlist supports it.\n"
        "4) If evidence is weak, return an empty list with link_type=no_match or unknown.\n"
        "5) Return strict JSON only.\n\n"
        "Required response JSON:\n"
        '{"primary_key":"...",'
        '"selected_secondary_keys":["candidate_key"],'
        '"link_type":"rename|split|merge|transfer|no_match|unknown",'
        '"relationship":"father_to_father|father_to_child|child_to_father|child_to_child|unknown",'
        '"score":0.0'
        f"{evidence_rule}"
        f"{reason_rule}"
        "}\n\n"
        "INPUT_PAYLOAD_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_repair_prompt(original_prompt: str, invalid_output: str, error_message: str) -> str:
    """Build repair prompt when first model response is malformed."""

    return (
        "Your previous response was invalid JSON for the required schema.\n"
        "Return ONLY valid JSON matching the required schema.\n"
        f"Validation error: {error_message}\n"
        "Use only these link_type values: rename, split, merge, transfer, no_match, unknown.\n"
        "Never use exact_match as link_type; use rename instead.\n"
        "Do not include markdown or explanation.\n\n"
        "ORIGINAL_PROMPT:\n"
        f"{original_prompt}\n\n"
        "PREVIOUS_INVALID_OUTPUT:\n"
        f"{invalid_output}"
    )
