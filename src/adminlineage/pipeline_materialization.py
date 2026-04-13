"""Crosswalk row materialization and final table shaping helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .review import apply_global_flags, build_review_queue
from .schema import get_crosswalk_base_columns, normalize_nullable_output_columns

_MATCHED_LINK_TYPES = {"rename", "split", "merge", "transfer"}


def finalize_crosswalk_table(
    crosswalk: pd.DataFrame,
    *,
    evidence: bool,
    exact_match: list[str],
    review_score_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize, order, and flag the final crosswalk table."""

    crosswalk = crosswalk.copy()
    crosswalk_base_columns = get_crosswalk_base_columns(include_evidence=evidence)
    for col in crosswalk_base_columns:
        if col not in crosswalk.columns:
            if col in {"reason", "lineage_hint"}:
                crosswalk[col] = ""
            else:
                crosswalk[col] = None

    crosswalk = apply_global_flags(crosswalk, low_score_threshold=review_score_threshold)
    crosswalk = normalize_nullable_output_columns(crosswalk)
    exact_match_order = exact_match.copy()
    preferred_order = crosswalk_base_columns + exact_match_order + [
        "review_flags",
        "review_reason",
    ]
    ordered_cols = [col for col in preferred_order if col in crosswalk.columns] + [
        col for col in crosswalk.columns if col not in preferred_order
    ]
    crosswalk = crosswalk[ordered_cols]
    review_queue = build_review_queue(crosswalk)
    return crosswalk, review_queue


def materialize_rows(
    latest_success: dict[str, dict[str, Any]],
    latest_error: dict[str, dict[str, Any]],
    *,
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    exact_match: list[str],
    candidate_map: dict[str, list[dict[str, Any]]],
    evidence: bool,
    reason: bool,
    relationship: str,
    country: str,
    year_from: int | str,
    year_to: int | str,
    run_id: str,
    final_relationship_fn: Callable[..., str],
    merge_indicator_fn: Callable[..., str],
    normalized_scope_col_fn: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """Build row-per-link crosswalk records from adjudication outputs."""

    rows: list[dict[str, Any]] = []
    for from_key in df_from_work["_from_key"].tolist():
        from_row = from_lookup.loc[from_key]
        exact_match_payload = {col: from_row[col] for col in exact_match}

        record = latest_success.get(from_key)
        error_record = latest_error.get(from_key)
        links = record.get("links", []) if record else []
        match_stage = record.get("match_stage") if record else None
        if not links:
            error_text = (
                str(error_record.get("error", "")).strip()
                if error_record is not None
                else "No completed adjudication record."
            )
            links = [
                {
                    "to_key": None,
                    "link_type": "unknown",
                    "relationship": "unknown",
                    "score": 0.0,
                }
            ]
            if evidence:
                links[0]["evidence"] = (
                    f"Adjudication failed after retries: {error_text}"
                    if error_record is not None
                    else error_text
                )
            if reason:
                links[0]["reason"] = ""

        allowed_to_keys = {item["to_key"] for item in candidate_map.get(from_key, [])}

        for link in links:
            raw_to_key = link.get("to_key")
            candidate_membership = (
                True
                if match_stage == "exact"
                else raw_to_key is None or raw_to_key in allowed_to_keys
            )
            to_key = raw_to_key if candidate_membership else None
            to_row = to_lookup.loc[to_key] if to_key in to_lookup.index else None

            link_type = str(link.get("link_type", "unknown"))
            score = float(link.get("score", 0.0))
            evidence_text = str(link.get("evidence", "")).strip() if evidence else ""
            reason_text = str(link.get("reason", "")).strip() if reason else ""

            if not candidate_membership:
                link_type = "unknown"
                score = 0.0
                if evidence:
                    evidence_text = "Model selected a target outside the provided candidates."
                if reason and not reason_text:
                    reason_text = "The chosen target was not present in the candidate list."

            if to_key is None and link_type in _MATCHED_LINK_TYPES:
                link_type = "unknown"

            exact_match_passed = True
            if to_row is not None and exact_match:
                exact_match_passed = all(
                    from_row[normalized_scope_col_fn("from", col)]
                    == to_row[normalized_scope_col_fn("to", col)]
                    for col in exact_match
                )

            row = {
                "from_name": from_row["_from_name_raw"],
                "to_name": to_row["_to_name_raw"] if to_row is not None else None,
                "from_canonical_name": from_row["_from_canonical_name"],
                "to_canonical_name": to_row["_to_canonical_name"] if to_row is not None else None,
                "from_id": from_row["_from_id"],
                "to_id": to_row["_to_id"] if to_row is not None else None,
                "score": score,
                "link_type": link_type,
                "relationship": final_relationship_fn(
                    link.get("relationship"),
                    requested_relationship=relationship,
                    link_type=link_type,
                    to_key=to_key,
                ),
                "merge": merge_indicator_fn(link_type=link_type, to_key=to_key),
                "country": country,
                "year_from": year_from,
                "year_to": year_to,
                "run_id": run_id,
                "from_key": from_key,
                "to_key": to_key,
                "constraints_passed": {
                    "candidate_membership": candidate_membership,
                    "exact_match": exact_match_passed,
                },
            }
            if evidence:
                row["evidence"] = evidence_text[:400]
            row["reason"] = reason_text[:800] if reason else ""
            row.update(exact_match_payload)
            rows.append(row)

    matched_to_keys = {
        row["to_key"] for row in rows if row["merge"] == "both" and row["to_key"] is not None
    }
    for to_key in df_to_work["_to_key"].tolist():
        if to_key in matched_to_keys:
            continue

        to_row = to_lookup.loc[to_key]
        row = {
            "from_name": None,
            "to_name": to_row["_to_name_raw"],
            "from_canonical_name": None,
            "to_canonical_name": to_row["_to_canonical_name"],
            "from_id": None,
            "to_id": to_row["_to_id"],
            "score": 0.0,
            "link_type": "unknown",
            "relationship": "unknown",
            "merge": "only_in_to",
            "country": country,
            "year_from": year_from,
            "year_to": year_to,
            "run_id": run_id,
            "from_key": None,
            "to_key": to_key,
            "constraints_passed": {"target_only_row": True},
        }
        if evidence:
            row["evidence"] = (
                "Target unit appears in the later-period table but was not used by any "
                "matched source row."
            )
        row["reason"] = (
            "No earlier-period source row was linked to this target unit." if reason else ""
        )
        row.update({col: to_row[col] for col in exact_match})
        rows.append(row)
    return rows
