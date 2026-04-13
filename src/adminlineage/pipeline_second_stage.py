"""Second-stage rescue helpers extracted from the main pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import combined_similarity, ngram_cosine, token_jaccard
from .io import append_jsonl
from .llm import BaseLLMClient
from .models import (
    ExactStringPruneMode,
    RequestRelationshipType,
    SecondStageDecision,
    SecondStageResearch,
)
from .normalize import canonicalize_name, char_ngram_counter, tokenize
from .prompts import (
    build_second_stage_decision_prompt,
    build_second_stage_research_prompt,
)
from .utils import now_iso

_MATCHED_LINK_TYPES = {"rename", "split", "merge", "transfer"}


def _build_global_secondary_records(
    *,
    side: str,
    frame: pd.DataFrame,
    exact_match: list[str],
    extra_context_cols: list[str],
) -> list[dict[str, Any]]:
    key_col = "_to_key" if side == "to" else "_from_key"
    id_col = "_to_id" if side == "to" else "_from_id"
    raw_name_col = "_to_name_raw" if side == "to" else "_from_name_raw"
    canonical_name_col = "_to_canonical_name" if side == "to" else "_from_canonical_name"
    tokens_col = "_to_tokens" if side == "to" else "_from_tokens"
    ngrams_col = "_to_char_ngrams" if side == "to" else "_from_char_ngrams"

    records: list[dict[str, Any]] = []
    for row_dict in frame.to_dict(orient="records"):
        records.append(
            {
                "secondary_key": row_dict[key_col],
                "secondary_id": row_dict[id_col],
                "secondary_name": row_dict[raw_name_col],
                "secondary_canonical_name": row_dict[canonical_name_col],
                "secondary_tokens": row_dict[tokens_col],
                "secondary_char_ngrams": row_dict[ngrams_col],
                "exact_match_context": {col: row_dict[col] for col in exact_match},
                "extra_context": {col: row_dict[col] for col in extra_context_cols},
            }
        )
    return records


def _rank_global_secondary_candidates(
    *,
    primary_side: str,
    search_terms: list[str],
    secondary_records_by_side: dict[str, list[dict[str, Any]]],
    unique_search_terms_fn: Callable[[list[str]], list[str]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    secondary_side = "to" if primary_side == "from" else "from"
    secondary_records = secondary_records_by_side[secondary_side]
    unique_terms = unique_search_terms_fn(search_terms)
    if not secondary_records or not unique_terms:
        return []

    encoded_terms = [
        (
            term,
            tokenize(canonicalize_name(term)),
            char_ngram_counter(canonicalize_name(term)),
        )
        for term in unique_terms
    ]

    ranked: list[dict[str, Any]] = []
    for record in secondary_records:
        best_score = 0.0
        best_term = ""
        for term, term_tokens, term_ngrams in encoded_terms:
            score = combined_similarity(
                token_jaccard(term_tokens, record["secondary_tokens"]),
                ngram_cosine(term_ngrams, record["secondary_char_ngrams"]),
            )
            if score > best_score:
                best_score = score
                best_term = term
        ranked.append(
            {
                "secondary_key": record["secondary_key"],
                "secondary_name": record["secondary_name"],
                "secondary_canonical_name": record["secondary_canonical_name"],
                "secondary_id": record["secondary_id"],
                "score": float(round(best_score, 6)),
                "matched_term": best_term,
                "exact_match_context": record["exact_match_context"],
                "extra_context": record["extra_context"],
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["secondary_canonical_name"]),
            str(item["secondary_key"]),
        )
    )
    return ranked[:max_candidates]


def _build_primary_item_from_row(
    row: pd.Series,
    *,
    primary_side: str,
    exact_match: list[str],
) -> dict[str, Any]:
    if primary_side == "from":
        return {
            "primary_key": row["from_key"],
            "primary_id": row["from_id"],
            "primary_name": row["from_name"],
            "primary_canonical_name": row["from_canonical_name"],
            "merge": row["merge"],
            "exact_match_context": {col: row[col] for col in exact_match},
        }
    return {
        "primary_key": row["to_key"],
        "primary_id": row["to_id"],
        "primary_name": row["to_name"],
        "primary_canonical_name": row["to_canonical_name"],
        "merge": row["merge"],
        "exact_match_context": {col: row[col] for col in exact_match},
    }


def _second_stage_candidate_payloads(
    candidates: list[dict[str, Any]],
    *,
    extra_context_cols: list[str],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = {
            "secondary_key": candidate["secondary_key"],
            "secondary_name": candidate["secondary_name"],
            "secondary_canonical_name": candidate["secondary_canonical_name"],
            "score": candidate["score"],
            "exact_match_context": candidate["exact_match_context"],
        }
        if extra_context_cols:
            payload["extra_context"] = candidate["extra_context"]
        payloads.append(payload)
    return payloads


def _second_stage_link_is_matched(link_type: str, selected_secondary_keys: list[str]) -> bool:
    return bool(selected_secondary_keys) and link_type in _MATCHED_LINK_TYPES


def _build_second_stage_link_row(
    *,
    from_key: str,
    to_key: str,
    link_type: str,
    relationship_value: str,
    score: float,
    lineage_hint: str,
    evidence_text: str,
    reason_text: str,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    exact_match: list[str],
    relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
    country: str,
    year_from: int | str,
    year_to: int | str,
    run_id: str,
    final_relationship_fn: Callable[..., str],
    merge_indicator_fn: Callable[..., str],
    normalized_scope_col_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    from_row = from_lookup.loc[from_key]
    to_row = to_lookup.loc[to_key]
    exact_match_passed = True
    if exact_match:
        exact_match_passed = all(
            from_row[normalized_scope_col_fn("from", col)]
            == to_row[normalized_scope_col_fn("to", col)]
            for col in exact_match
        )

    row = {
        "from_name": from_row["_from_name_raw"],
        "to_name": to_row["_to_name_raw"],
        "from_canonical_name": from_row["_from_canonical_name"],
        "to_canonical_name": to_row["_to_canonical_name"],
        "from_id": from_row["_from_id"],
        "to_id": to_row["_to_id"],
        "score": float(score),
        "link_type": link_type,
        "relationship": final_relationship_fn(
            relationship_value,
            requested_relationship=relationship,
            link_type=link_type,
            to_key=to_key,
        ),
        "merge": merge_indicator_fn(link_type=link_type, to_key=to_key),
        "lineage_hint": lineage_hint,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "run_id": run_id,
        "from_key": from_key,
        "to_key": to_key,
        "constraints_passed": {
            "candidate_membership": True,
            "exact_match": exact_match_passed,
            "second_stage": True,
            "global_secondary_search": True,
        },
    }
    if evidence:
        row["evidence"] = evidence_text[:400]
    row["reason"] = reason_text[:800] if reason else ""
    row.update({col: from_row[col] for col in exact_match})
    return row


def _apply_second_stage_record(
    crosswalk: pd.DataFrame,
    *,
    record: dict[str, Any],
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    exact_match: list[str],
    relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
    country: str,
    year_from: int | str,
    year_to: int | str,
    run_id: str,
    final_relationship_fn: Callable[..., str],
    merge_indicator_fn: Callable[..., str],
    normalized_scope_col_fn: Callable[[str, str], str],
) -> pd.DataFrame:
    updated = crosswalk.copy()
    primary_side = str(record.get("primary_side", "")).strip()
    primary_key = str(record.get("primary_key", "")).strip()
    if primary_side not in {"from", "to"} or not primary_key:
        return updated

    research_payload = record.get("lineage_research") or {}
    decision_payload = record.get("decision") or {}
    lineage_hint = str(
        research_payload.get("lineage_hint") or record.get("lineage_hint") or ""
    ).strip()
    selected_secondary_keys = [
        str(value).strip()
        for value in decision_payload.get("selected_secondary_keys", [])
        if str(value).strip()
    ]
    selected_secondary_keys = list(dict.fromkeys(selected_secondary_keys))
    link_type = str(decision_payload.get("link_type", "unknown") or "unknown")
    relationship_value = str(decision_payload.get("relationship", "unknown") or "unknown")
    score = float(decision_payload.get("score", 0.0) or 0.0)
    evidence_text = str(decision_payload.get("evidence", "") or "").strip()
    reason_text = str(decision_payload.get("reason", "") or "").strip()

    if lineage_hint:
        if primary_side == "from":
            lineage_mask = updated["from_key"].eq(primary_key) & updated["merge"].eq("only_in_from")
        else:
            lineage_mask = updated["to_key"].eq(primary_key) & updated["merge"].eq("only_in_to")
        if lineage_mask.any():
            updated.loc[lineage_mask, "lineage_hint"] = lineage_hint

    if record.get("status") != "ok" or not _second_stage_link_is_matched(
        link_type,
        selected_secondary_keys,
    ):
        return updated

    if primary_side == "from":
        valid_selected_keys = [
            to_key for to_key in selected_secondary_keys if to_key in to_lookup.index
        ]
        if not valid_selected_keys:
            return updated

        updated = updated.loc[
            ~(updated["from_key"].eq(primary_key) & updated["merge"].eq("only_in_from"))
        ].copy()
        for to_key in valid_selected_keys:
            updated = updated.loc[
                ~(
                    (updated["to_key"].eq(to_key) & updated["merge"].eq("only_in_to"))
                    | (updated["from_key"].eq(primary_key) & updated["to_key"].eq(to_key))
                )
            ].copy()
            new_row = _build_second_stage_link_row(
                from_key=primary_key,
                to_key=to_key,
                link_type=link_type,
                relationship_value=relationship_value,
                score=score,
                lineage_hint=lineage_hint,
                evidence_text=evidence_text or str(research_payload.get("notes", "") or ""),
                reason_text=reason_text,
                from_lookup=from_lookup,
                to_lookup=to_lookup,
                exact_match=exact_match,
                relationship=relationship,
                evidence=evidence,
                reason=reason,
                country=country,
                year_from=year_from,
                year_to=year_to,
                run_id=run_id,
                final_relationship_fn=final_relationship_fn,
                merge_indicator_fn=merge_indicator_fn,
                normalized_scope_col_fn=normalized_scope_col_fn,
            )
            updated = pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)
        return updated

    valid_selected_keys = [
        from_key for from_key in selected_secondary_keys if from_key in from_lookup.index
    ]
    if not valid_selected_keys:
        return updated

    updated = updated.loc[
        ~(updated["to_key"].eq(primary_key) & updated["merge"].eq("only_in_to"))
    ].copy()
    for from_key in valid_selected_keys:
        updated = updated.loc[
            ~(
                (updated["from_key"].eq(from_key) & updated["merge"].eq("only_in_from"))
                | (updated["from_key"].eq(from_key) & updated["to_key"].eq(primary_key))
            )
        ].copy()
        new_row = _build_second_stage_link_row(
            from_key=from_key,
            to_key=primary_key,
            link_type=link_type,
            relationship_value=relationship_value,
            score=score,
            lineage_hint=lineage_hint,
            evidence_text=evidence_text or str(research_payload.get("notes", "") or ""),
            reason_text=reason_text,
            from_lookup=from_lookup,
            to_lookup=to_lookup,
            exact_match=exact_match,
            relationship=relationship,
            evidence=evidence,
            reason=reason,
            country=country,
            year_from=year_from,
            year_to=year_to,
            run_id=run_id,
            final_relationship_fn=final_relationship_fn,
            merge_indicator_fn=merge_indicator_fn,
            normalized_scope_col_fn=normalized_scope_col_fn,
        )
        updated = pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)
    return updated


def run_second_stage(
    crosswalk: pd.DataFrame,
    *,
    string_exact_match_prune: ExactStringPruneMode,
    grounding_enabled: bool,
    second_stage_results_path: Path,
    run_id: str,
    logger: Any,
    warnings: list[str],
    llm_client: BaseLLMClient,
    model: str,
    temperature: float,
    seed: int,
    max_candidates: int,
    country: str,
    year_from: int | str,
    year_to: int | str,
    relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
    exact_match: list[str],
    extra_context_cols: list[str],
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    prepare_resume_records_fn: Callable[..., list[dict[str, Any]]],
    collect_latest_second_stage_records_fn: Callable[
        [list[dict[str, Any]]],
        tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]],
    ],
    second_stage_primary_side_fn: Callable[[ExactStringPruneMode], str | None],
    unique_search_terms_fn: Callable[[list[str]], list[str]],
    final_relationship_fn: Callable[..., str],
    merge_indicator_fn: Callable[..., str],
    normalized_scope_col_fn: Callable[[str, str], str],
) -> pd.DataFrame:
    primary_side = second_stage_primary_side_fn(string_exact_match_prune)
    if primary_side is None:
        if second_stage_results_path.exists():
            second_stage_results_path.unlink()
        return crosswalk

    if not grounding_enabled:
        warning = "Second-stage rescue is disabled because enable_google_search is false."
        warnings.append(warning)
        logger.warning("run_id=%s stage=second_stage disabled=%s", run_id, warning)
        if second_stage_results_path.exists():
            second_stage_results_path.unlink()
        return crosswalk

    existing_second_stage_records = prepare_resume_records_fn(
        records_path=second_stage_results_path,
        run_id=run_id,
        logger=logger,
        warnings=warnings,
        record_label="second_stage_results",
    )
    latest_second_stage_success, _latest_second_stage_error = (
        collect_latest_second_stage_records_fn(existing_second_stage_records)
    )

    updated = crosswalk.copy()
    for record in latest_second_stage_success.values():
        updated = _apply_second_stage_record(
            updated,
            record=record,
            from_lookup=from_lookup,
            to_lookup=to_lookup,
            exact_match=exact_match,
            relationship=relationship,
            evidence=evidence,
            reason=reason,
            country=country,
            year_from=year_from,
            year_to=year_to,
            run_id=run_id,
            final_relationship_fn=final_relationship_fn,
            merge_indicator_fn=merge_indicator_fn,
            normalized_scope_col_fn=normalized_scope_col_fn,
        )

    merge_value = "only_in_from" if primary_side == "from" else "only_in_to"
    key_col = "from_key" if primary_side == "from" else "to_key"
    pending_rows = (
        updated.loc[(updated["merge"] == merge_value) & updated[key_col].notna()]
        .drop_duplicates(subset=[key_col], keep="first")
        .copy()
    )
    pending_rows = pending_rows.loc[
        ~pending_rows[key_col].astype(str).map(
            lambda value: (primary_side, value) in latest_second_stage_success
        )
    ]

    secondary_records_by_side = {
        "from": _build_global_secondary_records(
            side="from",
            frame=df_from_work,
            exact_match=exact_match,
            extra_context_cols=extra_context_cols,
        ),
        "to": _build_global_secondary_records(
            side="to",
            frame=df_to_work,
            exact_match=exact_match,
            extra_context_cols=extra_context_cols,
        ),
    }

    for _, row in pending_rows.iterrows():
        primary_key = str(row[key_col])
        primary_item = _build_primary_item_from_row(
            row,
            primary_side=primary_side,
            exact_match=exact_match,
        )
        lineage_research_payload: dict[str, Any] = {}
        refreshed_candidates: list[dict[str, Any]] = []
        try:
            logger.info(
                "run_id=%s stage=second_stage step=start primary_side=%s primary_key=%s",
                run_id,
                primary_side,
                primary_key,
            )
            research_prompt = build_second_stage_research_prompt(
                country=country,
                year_from=year_from,
                year_to=year_to,
                primary_side=primary_side,
                primary_item=primary_item,
            )
            raw_research = llm_client.generate_json(
                prompt=research_prompt,
                schema=SecondStageResearch,
                model=model,
                temperature=temperature,
                seed=seed,
                enable_google_search=True,
            )
            research = SecondStageResearch.model_validate(raw_research)
            lineage_research_payload = research.model_dump()
            logger.info(
                "run_id=%s stage=second_stage step=research_ok primary_side=%s "
                "primary_key=%s event_type=%s lineage_hint=%s",
                run_id,
                primary_side,
                primary_key,
                research.event_type,
                research.lineage_hint or "",
            )

            if research.event_type == "unknown" and not (research.lineage_hint or "").strip():
                decision_payload = {
                    "primary_key": primary_key,
                    "selected_secondary_keys": [],
                    "link_type": "no_match",
                    "relationship": "unknown",
                    "score": 0.0,
                }
                if evidence:
                    decision_payload["evidence"] = ""
                if reason:
                    decision_payload["reason"] = ""
                record = {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "primary_side": primary_side,
                    "primary_key": primary_key,
                    "status": "ok",
                    "lineage_research": lineage_research_payload,
                    "lineage_hint": "",
                    "candidate_secondary_keys": [],
                    "decision": decision_payload,
                    "rewrite_applied": False,
                }
                append_jsonl(second_stage_results_path, record)
                updated = _apply_second_stage_record(
                    updated,
                    record=record,
                    from_lookup=from_lookup,
                    to_lookup=to_lookup,
                    exact_match=exact_match,
                    relationship=relationship,
                    evidence=evidence,
                    reason=reason,
                    country=country,
                    year_from=year_from,
                    year_to=year_to,
                    run_id=run_id,
                    final_relationship_fn=final_relationship_fn,
                    merge_indicator_fn=merge_indicator_fn,
                    normalized_scope_col_fn=normalized_scope_col_fn,
                )
                logger.info(
                    "run_id=%s stage=second_stage step=match_ok primary_side=%s "
                    "primary_key=%s selected=0 rewrite=false skipped_decision=true",
                    run_id,
                    primary_side,
                    primary_key,
                )
                continue

            refreshed_candidates = _rank_global_secondary_candidates(
                primary_side=primary_side,
                search_terms=[
                    primary_item["primary_name"],
                    lineage_research_payload.get("lineage_hint", ""),
                ],
                secondary_records_by_side=secondary_records_by_side,
                unique_search_terms_fn=unique_search_terms_fn,
                max_candidates=max_candidates,
            )
            logger.info(
                "run_id=%s stage=second_stage step=shortlist primary_side=%s primary_key=%s "
                "candidates=%d",
                run_id,
                primary_side,
                primary_key,
                len(refreshed_candidates),
            )

            if refreshed_candidates:
                decision_prompt = build_second_stage_decision_prompt(
                    country=country,
                    year_from=year_from,
                    year_to=year_to,
                    primary_side=primary_side,
                    relationship=relationship,
                    include_evidence=evidence,
                    include_reason=reason,
                    primary_item=primary_item,
                    lineage_research=lineage_research_payload,
                    candidate_subset=_second_stage_candidate_payloads(
                        refreshed_candidates,
                        extra_context_cols=extra_context_cols,
                    ),
                )
                raw_decision = llm_client.generate_json(
                    prompt=decision_prompt,
                    schema=SecondStageDecision,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    enable_google_search=False,
                )
                decision = SecondStageDecision.model_validate(raw_decision)
            else:
                decision = SecondStageDecision(
                    primary_key=primary_key,
                    selected_secondary_keys=[],
                    link_type="no_match",
                    relationship="unknown",
                    score=0.0,
                )

            shortlisted_keys = {candidate["secondary_key"] for candidate in refreshed_candidates}
            filtered_secondary_keys = [
                secondary_key
                for secondary_key in decision.selected_secondary_keys
                if secondary_key in shortlisted_keys
            ]
            selected_secondary_keys = list(dict.fromkeys(filtered_secondary_keys))
            decision_payload = decision.model_dump()
            decision_payload["selected_secondary_keys"] = selected_secondary_keys
            record = {
                "run_id": run_id,
                "timestamp": now_iso(),
                "primary_side": primary_side,
                "primary_key": primary_key,
                "status": "ok",
                "lineage_research": lineage_research_payload,
                "lineage_hint": lineage_research_payload.get("lineage_hint", ""),
                "candidate_secondary_keys": sorted(shortlisted_keys),
                "decision": decision_payload,
                "rewrite_applied": _second_stage_link_is_matched(
                    str(decision_payload.get("link_type", "unknown")),
                    selected_secondary_keys,
                ),
            }
            append_jsonl(second_stage_results_path, record)
            updated = _apply_second_stage_record(
                updated,
                record=record,
                from_lookup=from_lookup,
                to_lookup=to_lookup,
                exact_match=exact_match,
                relationship=relationship,
                evidence=evidence,
                reason=reason,
                country=country,
                year_from=year_from,
                year_to=year_to,
                run_id=run_id,
                final_relationship_fn=final_relationship_fn,
                merge_indicator_fn=merge_indicator_fn,
                normalized_scope_col_fn=normalized_scope_col_fn,
            )
            logger.info(
                "run_id=%s stage=second_stage step=match_ok primary_side=%s "
                "primary_key=%s selected=%d rewrite=%s",
                run_id,
                primary_side,
                primary_key,
                len(selected_secondary_keys),
                record["rewrite_applied"],
            )
        except Exception as exc:
            append_jsonl(
                second_stage_results_path,
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "primary_side": primary_side,
                    "primary_key": primary_key,
                    "status": "error",
                    "lineage_research": lineage_research_payload,
                    "candidate_secondary_keys": [
                        candidate["secondary_key"] for candidate in refreshed_candidates
                    ],
                    "error": str(exc),
                },
            )
            logger.warning(
                "run_id=%s stage=second_stage step=error primary_side=%s primary_key=%s "
                "error=%s",
                run_id,
                primary_side,
                primary_key,
                exc,
            )
            warnings.append(
                f"Second-stage rescue failed for {primary_side}_key={primary_key}: {exc}"
            )

    return updated
