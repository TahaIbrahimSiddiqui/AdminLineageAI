"""Pipeline stage helpers used by the main run orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .candidates import combined_similarity, ngram_cosine, token_jaccard
from .io import append_jsonl
from .llm_types import (
    BaseLLMClient,
    LLMServiceError,
    QuotaExceededLLMError,
    TransientLLMError,
)
from .models import RequestRelationshipType, SecondStageDecision, SecondStageResearch
from .normalize import canonicalize_name, char_ngram_counter, tokenize
from .prompts import (
    build_batch_prompt,
    build_second_stage_decision_prompt,
    build_second_stage_research_prompt,
)
from .review import apply_global_flags, build_review_queue
from .schema import get_crosswalk_base_columns, normalize_nullable_output_columns
from .utils import chunked, now_iso

_MATCHED_LINK_TYPES = {"rename", "split", "merge", "transfer"}


def _candidate_payloads_for_key(
    from_key: str,
    *,
    candidate_map: dict[str, list[dict[str, Any]]],
    to_lookup: pd.DataFrame,
    exact_match: list[str],
    extra_context_cols: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    raw_candidates = candidate_map.get(from_key, [])
    if limit is not None:
        raw_candidates = raw_candidates[:limit]

    for candidate in raw_candidates:
        to_row = to_lookup.loc[candidate["to_key"]]
        candidate_payload = {
            "to_key": candidate["to_key"],
            "to_name": candidate["to_name"],
            "to_canonical_name": candidate["to_canonical_name"],
            "score": candidate["score"],
            "exact_match_context": {col: to_row[col] for col in exact_match},
        }
        if extra_context_cols:
            candidate_payload["extra_context"] = {col: to_row[col] for col in extra_context_cols}
        candidates.append(candidate_payload)
    return candidates


def _build_batch_item(
    from_key: str,
    *,
    from_lookup: pd.DataFrame,
    exact_match: list[str],
    extra_context_cols: list[str],
    candidate_map: dict[str, list[dict[str, Any]]],
    to_lookup: pd.DataFrame,
) -> dict[str, Any]:
    from_row = from_lookup.loc[from_key]
    return {
        "from_key": from_key,
        "from_name": from_row["_from_name_raw"],
        "from_canonical_name": from_row["_from_canonical_name"],
        "exact_match_context": {col: from_row[col] for col in exact_match},
        "extra_context": {col: from_row[col] for col in extra_context_cols},
        "candidates": _candidate_payloads_for_key(
            from_key,
            candidate_map=candidate_map,
            to_lookup=to_lookup,
            exact_match=exact_match,
            extra_context_cols=extra_context_cols,
        ),
    }


def _record_grounding_note(
    *,
    run_id: str,
    from_key: str,
    candidate_keys: list[str],
    status: str,
    grounding_enabled: bool,
    grounding_notes_path: Path,
    notes: str = "",
    error: str = "",
) -> None:
    if not grounding_enabled:
        return
    append_jsonl(
        grounding_notes_path,
        {
            "run_id": run_id,
            "timestamp": now_iso(),
            "from_key": from_key,
            "candidate_keys": candidate_keys,
            "status": status,
            "notes": notes[:4000],
            "error": error,
        },
    )


def _split_failed_batch(
    *,
    batch_keys: list[str],
    batch_label: str,
    exc: Exception,
    warnings: list[str],
    logger: Any,
    run_id: str,
    match_stage: str,
) -> list[list[str]]:
    if len(batch_keys) <= 1:
        return []

    midpoint = max(1, len(batch_keys) // 2)
    split_batches = [batch_keys[:midpoint], batch_keys[midpoint:]]
    warning = (
        f"Batch {batch_label} failed: {exc}. Retrying in smaller batches "
        f"({len(split_batches[0])} + {len(split_batches[1])})."
    )
    warnings.append(warning)
    logger.warning(
        "run_id=%s stage=adjudication match_stage=%s batch=%s failed=%s retrying_split=%s",
        run_id,
        match_stage,
        batch_label,
        exc,
        [len(keys) for keys in split_batches],
    )
    return split_batches


def _run_adjudication_batch(
    batch_keys: list[str],
    *,
    batch_label: str,
    match_stage: str,
    run_id: str,
    country: str,
    year_from: int | str,
    year_to: int | str,
    exact_match: list[str],
    relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
    grounding_enabled: bool,
    response_model: type[BaseModel],
    llm_client: BaseLLMClient,
    model: str,
    temperature: float,
    seed: int,
    links_raw_path: Path,
    grounding_notes_path: Path,
    warnings: list[str],
    logger: Any,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    candidate_map: dict[str, list[dict[str, Any]]],
    extra_context_cols: list[str],
) -> None:
    batch_items = [
        _build_batch_item(
            from_key,
            from_lookup=from_lookup,
            exact_match=exact_match,
            extra_context_cols=extra_context_cols,
            candidate_map=candidate_map,
            to_lookup=to_lookup,
        )
        for from_key in batch_keys
    ]

    if len(batch_items) == 1 and not batch_items[0]["candidates"]:
        from_key = batch_items[0]["from_key"]
        links = [
            {
                "to_key": None,
                "link_type": "no_match",
                "relationship": "unknown",
                "score": 0.0,
            }
        ]
        if evidence:
            links[0]["evidence"] = "No shortlist candidates available in the constrained group."
        if reason:
            links[0]["reason"] = ""
        append_jsonl(
            links_raw_path,
            {
                "run_id": run_id,
                "timestamp": now_iso(),
                "batch_index": batch_label,
                "match_stage": match_stage,
                "status": "ok",
                "from_key": from_key,
                "links": links,
            },
        )
        _record_grounding_note(
            run_id=run_id,
            from_key=from_key,
            candidate_keys=[],
            status="skipped",
            grounding_enabled=grounding_enabled,
            grounding_notes_path=grounding_notes_path,
            error="No shortlist candidates available for structured adjudication.",
        )
        return

    prompt = build_batch_prompt(
        country=country,
        year_from=year_from,
        year_to=year_to,
        exact_match=exact_match,
        relationship=relationship,
        include_evidence=evidence,
        include_reason=reason,
        batch_items=batch_items,
        allow_external_grounding=grounding_enabled,
    )

    logger.info(
        "run_id=%s stage=adjudication match_stage=%s batch=%s size=%d",
        run_id,
        match_stage,
        batch_label,
        len(batch_items),
    )

    try:
        raw_response = llm_client.generate_json(
            prompt=prompt,
            schema=response_model,
            model=model,
            temperature=temperature,
            seed=seed,
            enable_google_search=grounding_enabled,
        )
        parsed = response_model.model_validate(raw_response)
        by_from = {decision.from_key: decision for decision in parsed.decisions}

        for item in batch_items:
            from_key = item["from_key"]
            decision = by_from.get(from_key)
            if decision is None:
                links = [
                    {
                        "to_key": None,
                        "link_type": "unknown",
                        "relationship": "unknown",
                        "score": 0.0,
                    }
                ]
                if evidence:
                    links[0]["evidence"] = "LLM omitted this item in batch response."
                if reason:
                    links[0]["reason"] = ""
            else:
                links = [link.model_dump() for link in decision.links]
                if not links:
                    links = [
                        {
                            "to_key": None,
                            "link_type": "no_match",
                            "relationship": "unknown",
                            "score": 0.0,
                        }
                    ]
                    if evidence:
                        links[0]["evidence"] = "No valid link selected by model."
                    if reason:
                        links[0]["reason"] = ""

            append_jsonl(
                links_raw_path,
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "batch_index": batch_label,
                    "match_stage": match_stage,
                    "status": "ok",
                    "from_key": from_key,
                    "links": links,
                },
            )
            _record_grounding_note(
                run_id=run_id,
                from_key=from_key,
                candidate_keys=[candidate["to_key"] for candidate in item["candidates"]],
                status="ok",
                grounding_enabled=grounding_enabled,
                grounding_notes_path=grounding_notes_path,
                notes="Structured grounded JSON adjudication completed."
                if grounding_enabled
                else "",
            )
    except QuotaExceededLLMError as exc:
        logger.exception(
            (
                "run_id=%s stage=adjudication match_stage=%s "
                "batch=%s spending_cap_reached=%s"
            ),
            run_id,
            match_stage,
            batch_label,
            exc,
        )
        for item in batch_items:
            _record_grounding_note(
                run_id=run_id,
                from_key=item["from_key"],
                candidate_keys=[candidate["to_key"] for candidate in item["candidates"]],
                status="error",
                grounding_enabled=grounding_enabled,
                grounding_notes_path=grounding_notes_path,
                error=str(exc),
            )
        raise
    except LLMServiceError as exc:
        if isinstance(exc, TransientLLMError):
            split_batches = _split_failed_batch(
                batch_keys=batch_keys,
                batch_label=batch_label,
                exc=exc,
                warnings=warnings,
                logger=logger,
                run_id=run_id,
                match_stage=match_stage,
            )
            if split_batches:
                for split_index, split_keys in enumerate(split_batches, start=1):
                    _run_adjudication_batch(
                        split_keys,
                        batch_label=f"{batch_label}.{split_index}",
                        match_stage=match_stage,
                        run_id=run_id,
                        country=country,
                        year_from=year_from,
                        year_to=year_to,
                        exact_match=exact_match,
                        relationship=relationship,
                        evidence=evidence,
                        reason=reason,
                        grounding_enabled=grounding_enabled,
                        response_model=response_model,
                        llm_client=llm_client,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                        links_raw_path=links_raw_path,
                        grounding_notes_path=grounding_notes_path,
                        warnings=warnings,
                        logger=logger,
                        from_lookup=from_lookup,
                        to_lookup=to_lookup,
                        candidate_map=candidate_map,
                        extra_context_cols=extra_context_cols,
                    )
                return
            logger.warning(
                "run_id=%s stage=adjudication match_stage=%s batch=%s transient_error=%s",
                run_id,
                match_stage,
                batch_label,
                exc,
            )
            warnings.append(f"Batch {batch_label} failed: {exc}")
            for item in batch_items:
                append_jsonl(
                    links_raw_path,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "batch_index": batch_label,
                        "match_stage": match_stage,
                        "status": "error",
                        "from_key": item["from_key"],
                        "error": str(exc),
                    },
                )
                _record_grounding_note(
                    run_id=run_id,
                    from_key=item["from_key"],
                    candidate_keys=[candidate["to_key"] for candidate in item["candidates"]],
                    status="error",
                    grounding_enabled=grounding_enabled,
                    grounding_notes_path=grounding_notes_path,
                    error=str(exc),
                )
        else:
            logger.exception(
                "run_id=%s stage=adjudication match_stage=%s batch=%s provider_error=%s",
                run_id,
                match_stage,
                batch_label,
                exc,
            )
            for item in batch_items:
                _record_grounding_note(
                    run_id=run_id,
                    from_key=item["from_key"],
                    candidate_keys=[candidate["to_key"] for candidate in item["candidates"]],
                    status="error",
                    grounding_enabled=grounding_enabled,
                    grounding_notes_path=grounding_notes_path,
                    error=str(exc),
                )
            raise
    except Exception as exc:
        split_batches = _split_failed_batch(
            batch_keys=batch_keys,
            batch_label=batch_label,
            exc=exc,
            warnings=warnings,
            logger=logger,
            run_id=run_id,
            match_stage=match_stage,
        )
        if split_batches:
            for split_index, split_keys in enumerate(split_batches, start=1):
                _run_adjudication_batch(
                    split_keys,
                    batch_label=f"{batch_label}.{split_index}",
                    match_stage=match_stage,
                    run_id=run_id,
                    country=country,
                    year_from=year_from,
                    year_to=year_to,
                    exact_match=exact_match,
                    relationship=relationship,
                    evidence=evidence,
                    reason=reason,
                    grounding_enabled=grounding_enabled,
                    response_model=response_model,
                    llm_client=llm_client,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    links_raw_path=links_raw_path,
                    grounding_notes_path=grounding_notes_path,
                    warnings=warnings,
                    logger=logger,
                    from_lookup=from_lookup,
                    to_lookup=to_lookup,
                    candidate_map=candidate_map,
                    extra_context_cols=extra_context_cols,
                )
            return
        logger.warning(
            "run_id=%s stage=adjudication match_stage=%s batch=%s failed=%s",
            run_id,
            match_stage,
            batch_label,
            exc,
        )
        warnings.append(f"Batch {batch_label} failed: {exc}")
        for item in batch_items:
            append_jsonl(
                links_raw_path,
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "batch_index": batch_label,
                    "match_stage": match_stage,
                    "status": "error",
                    "from_key": item["from_key"],
                    "error": str(exc),
                },
            )
            _record_grounding_note(
                run_id=run_id,
                from_key=item["from_key"],
                candidate_keys=[candidate["to_key"] for candidate in item["candidates"]],
                status="error",
                grounding_enabled=grounding_enabled,
                grounding_notes_path=grounding_notes_path,
                error=str(exc),
            )


def run_adjudication_stage(
    *,
    pending_from_keys: list[str],
    batch_size: int,
    run_id: str,
    country: str,
    year_from: int | str,
    year_to: int | str,
    exact_match: list[str],
    relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
    grounding_enabled: bool,
    response_model: type[BaseModel],
    llm_client: BaseLLMClient,
    model: str,
    temperature: float,
    seed: int,
    links_raw_path: Path,
    grounding_notes_path: Path,
    warnings: list[str],
    logger: Any,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    candidate_map: dict[str, list[dict[str, Any]]],
    extra_context_cols: list[str],
) -> None:
    batch_index = 0
    for batch_keys in chunked(pending_from_keys, batch_size):
        batch_index += 1
        _run_adjudication_batch(
            batch_keys,
            batch_label=str(batch_index),
            match_stage="ai",
            run_id=run_id,
            country=country,
            year_from=year_from,
            year_to=year_to,
            exact_match=exact_match,
            relationship=relationship,
            evidence=evidence,
            reason=reason,
            grounding_enabled=grounding_enabled,
            response_model=response_model,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            seed=seed,
            links_raw_path=links_raw_path,
            grounding_notes_path=grounding_notes_path,
            warnings=warnings,
            logger=logger,
            from_lookup=from_lookup,
            to_lookup=to_lookup,
            candidate_map=candidate_map,
            extra_context_cols=extra_context_cols,
        )


def materialize_rows(
    *,
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    from_lookup: pd.DataFrame,
    to_lookup: pd.DataFrame,
    latest_success: dict[str, dict[str, Any]],
    latest_error: dict[str, dict[str, Any]],
    candidate_map: dict[str, list[dict[str, Any]]],
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
) -> list[dict[str, Any]]:
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
            candidate_membership = True if match_stage == "exact" else (
                raw_to_key is None or raw_to_key in allowed_to_keys
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


def finalize_crosswalk_table(
    crosswalk: pd.DataFrame,
    *,
    exact_match: list[str],
    evidence: bool,
    review_score_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    preferred_order = crosswalk_base_columns + exact_match.copy() + [
        "review_flags",
        "review_reason",
    ]
    ordered_cols = [col for col in preferred_order if col in crosswalk.columns] + [
        col for col in crosswalk.columns if col not in preferred_order
    ]
    crosswalk = crosswalk[ordered_cols]
    review_queue = build_review_queue(crosswalk)
    return crosswalk, review_queue


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
    # The rescue pass deliberately searches the full opposite table instead of the
    # first-pass scoped shortlist so it can recover renames and lineage changes.
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


def _second_stage_candidate_payloads(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        payloads.append(
            {
                "secondary_key": candidate["secondary_key"],
                "secondary_name": candidate["secondary_name"],
                "secondary_canonical_name": candidate["secondary_canonical_name"],
                "score": candidate["score"],
                "exact_match_context": candidate["exact_match_context"],
                "extra_context": candidate["extra_context"],
            }
        )
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
    string_exact_match_prune: str,
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
    second_stage_primary_side_fn: Callable[[str], str | None],
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
                    candidate_subset=_second_stage_candidate_payloads(refreshed_candidates),
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
            decision_payload = decision.model_dump()
            decision_payload["selected_secondary_keys"] = list(
                dict.fromkeys(filtered_secondary_keys)
            )
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
                    decision_payload["selected_secondary_keys"],
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
                len(decision_payload["selected_secondary_keys"]),
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
