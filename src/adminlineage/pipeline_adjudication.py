"""Adjudication-stage helpers extracted from the main pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from .io import append_jsonl
from .llm import (
    BaseLLMClient,
    LLMServiceError,
    QuotaExceededLLMError,
    TransientLLMError,
)
from .models import BatchResponse, BatchResponseModel, RequestRelationshipType
from .prompts import build_batch_prompt
from .utils import chunked, now_iso


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
    response_model: BatchResponseModel,
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
        parsed = cast(BatchResponse, response_model.model_validate(raw_response))
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
            "run_id=%s stage=adjudication match_stage=%s batch=%s spending_cap_reached=%s",
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
    response_model: BatchResponseModel,
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
    for batch_index, batch_keys in enumerate(chunked(pending_from_keys, batch_size), start=1):
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
