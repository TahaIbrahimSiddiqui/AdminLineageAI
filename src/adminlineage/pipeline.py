"""Core evolution-key pipeline implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import build_alias_lookup, generate_shortlist
from .llm import BaseLLMClient, GeminiClient, SQLiteCache
from .logging_utils import setup_logger
from .models import LLMBatchResponse, MappingRequest
from .normalize import add_normalized_columns
from .prompts import build_batch_prompt
from .review import apply_global_flags, build_review_queue, coverage_summary, summarize_counts
from .schema import CROSSWALK_BASE_COLUMNS
from .utils import build_run_id, chunked, ensure_dir, now_iso, sanitize_name
from .validation import validate_inputs_data
from .io import append_jsonl, read_jsonl, write_json


def _default_run_name(country: str, year_from: int | str, year_to: int | str, map_col: str) -> str:
    return f"{sanitize_name(country)}_{year_from}_{year_to}_{sanitize_name(map_col)}"


def _build_llm_client(
    *,
    gemini_api_key_env: str,
    cache_enabled: bool,
    cache_path: Path,
    retry_max_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter: float,
) -> BaseLLMClient:
    cache = SQLiteCache(cache_path) if cache_enabled else None
    return GeminiClient(
        api_key_env=gemini_api_key_env,
        cache=cache,
        max_attempts=retry_max_attempts,
        base_delay_seconds=retry_base_delay,
        max_delay_seconds=retry_max_delay,
        jitter_seconds=retry_jitter,
    )


def _prepare_workframes(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    map_col_from: str,
    map_col_to: str,
    id_col_from: str | None,
    id_col_to: str | None,
    anchor_cols: list[str],
    extra_context_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_from_work = df_from.reset_index(drop=True).copy()
    df_to_work = df_to.reset_index(drop=True).copy()

    df_from_work["_from_key"] = [f"from_{idx}" for idx in range(len(df_from_work))]
    df_to_work["_to_key"] = [f"to_{idx}" for idx in range(len(df_to_work))]

    df_from_work["_from_id"] = (
        df_from_work[id_col_from].astype(str) if id_col_from else df_from_work["_from_key"]
    )
    df_to_work["_to_id"] = df_to_work[id_col_to].astype(str) if id_col_to else df_to_work["_to_key"]

    for col in anchor_cols:
        if col not in df_from_work.columns:
            df_from_work[col] = None
        if col not in df_to_work.columns:
            df_to_work[col] = None

    for col in extra_context_cols:
        if col not in df_from_work.columns:
            df_from_work[col] = None
        if col not in df_to_work.columns:
            df_to_work[col] = None

    df_from_work = add_normalized_columns(df_from_work, name_col=map_col_from, prefix="from")
    df_to_work = add_normalized_columns(df_to_work, name_col=map_col_to, prefix="to")
    return df_from_work, df_to_work


def _group_key_from_row(row: pd.Series, anchor_cols: list[str]) -> tuple[Any, ...] | str:
    if not anchor_cols:
        return "__all__"
    return tuple(row[col] for col in anchor_cols)


def _build_candidate_maps(
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    *,
    anchor_cols: list[str],
    max_candidates: int,
    alias_lookup: dict,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[Any, ...] | str]]:
    to_groups: dict[tuple[Any, ...] | str, pd.DataFrame] = {}
    if anchor_cols:
        for key, group in df_to_work.groupby(anchor_cols, dropna=False):
            to_groups[key if isinstance(key, tuple) else (key,)] = group
    else:
        to_groups["__all__"] = df_to_work

    candidate_map: dict[str, list[dict[str, Any]]] = {}
    group_map: dict[str, tuple[Any, ...] | str] = {}

    for _, from_row in df_from_work.iterrows():
        if anchor_cols:
            raw_key = tuple(from_row[col] for col in anchor_cols)
            group_key: tuple[Any, ...] | str = raw_key
        else:
            group_key = "__all__"

        to_group = to_groups.get(group_key, df_to_work.iloc[0:0])
        candidates = generate_shortlist(
            from_row,
            to_group,
            max_candidates=max_candidates,
            alias_lookup=alias_lookup,
            anchor_cols=anchor_cols,
        )
        candidate_map[from_row["_from_key"]] = candidates
        group_map[from_row["_from_key"]] = group_key

    return candidate_map, group_map


def preview_pipeline_plan(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    map_col_from: str,
    map_col_to: str | None = None,
    anchor_cols: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    extra_context_cols: list[str] | None = None,
    aliases: pd.DataFrame | None = None,
    max_candidates: int = 15,
) -> dict[str, Any]:
    """Preview group sizes and candidate budgets without calling LLM."""

    anchor_cols = anchor_cols or []
    extra_context_cols = extra_context_cols or []
    validation = validate_inputs_data(
        df_from,
        df_to,
        country=country,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        anchor_cols=anchor_cols,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        aliases=aliases,
    )
    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }

    effective_map_col_to = validation["map_col_to"]
    df_from_work, df_to_work = _prepare_workframes(
        df_from,
        df_to,
        map_col_from=map_col_from,
        map_col_to=effective_map_col_to,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        anchor_cols=anchor_cols,
        extra_context_cols=extra_context_cols,
    )

    alias_lookup = build_alias_lookup(aliases, anchor_cols)
    candidate_map, group_map = _build_candidate_maps(
        df_from_work,
        df_to_work,
        anchor_cols=anchor_cols,
        max_candidates=max_candidates,
        alias_lookup=alias_lookup,
    )

    group_counts: dict[str, int] = {}
    for key in group_map.values():
        label = str(key)
        group_counts[label] = group_counts.get(label, 0) + 1

    avg_candidates = 0.0
    if candidate_map:
        avg_candidates = sum(len(c) for c in candidate_map.values()) / len(candidate_map)

    return {
        "valid": True,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "from_rows": int(len(df_from_work)),
        "to_rows": int(len(df_to_work)),
        "anchor_cols": anchor_cols,
        "groups": group_counts,
        "max_candidates": max_candidates,
        "avg_candidates": round(avg_candidates, 4),
        "warnings": validation["warnings"],
    }


def run_pipeline(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    map_col_from: str,
    map_col_to: str | None = None,
    anchor_cols: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    extra_context_cols: list[str] | None = None,
    aliases: pd.DataFrame | None = None,
    model: str = "gemini-2.5-pro",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 25,
    max_candidates: int = 15,
    resume_dir: str | Path = "outputs",
    run_name: str | None = None,
    seed: int = 42,
    llm_client: BaseLLMClient | None = None,
    temperature: float = 0.0,
    cache_enabled: bool = True,
    cache_path: str = "llm_cache.sqlite",
    retry_max_attempts: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 20.0,
    retry_jitter: float = 0.2,
    review_score_threshold: float = 0.6,
    output_write_csv: bool = True,
    output_write_parquet: bool = True,
    output_write_jsonl: bool = True,
    loader_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete evolution-key pipeline and materialize output artifacts."""

    start_time = now_iso()
    anchor_cols = anchor_cols or []
    extra_context_cols = extra_context_cols or []
    loader_metadata = loader_metadata or {}

    validation = validate_inputs_data(
        df_from,
        df_to,
        country=country,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        anchor_cols=anchor_cols,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        aliases=aliases,
    )
    if not validation["valid"]:
        raise ValueError("Invalid inputs: " + "; ".join(validation["errors"]))

    map_col_to_effective = validation["map_col_to"]
    request = MappingRequest(
        country=country,
        year_from=year_from,
        year_to=year_to,
        anchor_cols=anchor_cols,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        model=model,
        batch_size=batch_size,
        max_candidates=max_candidates,
        seed=seed,
    )

    run_name_effective = run_name or _default_run_name(country, year_from, year_to, map_col_from)
    base_dir = ensure_dir(resume_dir)
    run_dir = ensure_dir(base_dir / run_name_effective)
    logger = setup_logger(run_dir)

    warnings = list(validation["warnings"])
    run_id = build_run_id(request.model_dump(), seed=seed)
    logger.info("run_id=%s stage=start country=%s years=%s->%s", run_id, country, year_from, year_to)

    if llm_client is None:
        cache_path_obj = Path(cache_path)
        if not cache_path_obj.is_absolute():
            cache_path_obj = run_dir / cache_path_obj
        llm_client = _build_llm_client(
            gemini_api_key_env=gemini_api_key_env,
            cache_enabled=cache_enabled,
            cache_path=cache_path_obj,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retry_jitter=retry_jitter,
        )

    df_from_work, df_to_work = _prepare_workframes(
        df_from,
        df_to,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        anchor_cols=anchor_cols,
        extra_context_cols=extra_context_cols,
    )
    from_lookup = df_from_work.set_index("_from_key", drop=False)
    to_lookup = df_to_work.set_index("_to_key", drop=False)

    alias_lookup = build_alias_lookup(aliases, anchor_cols)
    candidate_map, _group_map = _build_candidate_maps(
        df_from_work,
        df_to_work,
        anchor_cols=anchor_cols,
        max_candidates=max_candidates,
        alias_lookup=alias_lookup,
    )

    links_raw_path = run_dir / "links_raw.jsonl"
    existing_records = read_jsonl(links_raw_path)
    completed_from_keys = {
        record["from_key"]
        for record in existing_records
        if record.get("status") == "ok" and "from_key" in record
    }
    pending_from_keys = [
        key for key in df_from_work["_from_key"].tolist() if key not in completed_from_keys
    ]
    logger.info(
        "run_id=%s stage=resume completed=%d pending=%d",
        run_id,
        len(completed_from_keys),
        len(pending_from_keys),
    )

    if not output_write_jsonl and links_raw_path.exists() and not pending_from_keys:
        logger.info("run_id=%s stage=resume using existing raw links file", run_id)

    batch_index = 0
    for batch_keys in chunked(pending_from_keys, batch_size):
        batch_index += 1
        batch_items: list[dict[str, Any]] = []
        for from_key in batch_keys:
            from_row = from_lookup.loc[from_key]
            anchors = {col: from_row[col] for col in anchor_cols}
            extras = {col: from_row[col] for col in extra_context_cols}

            candidates = []
            for candidate in candidate_map.get(from_key, []):
                to_row = to_lookup.loc[candidate["to_key"]]
                candidate_payload = {
                    "to_key": candidate["to_key"],
                    "to_name": candidate["to_name"],
                    "to_canonical_name": candidate["to_canonical_name"],
                    "score": candidate["score"],
                    "anchor_context": {col: to_row[col] for col in anchor_cols},
                }
                if extra_context_cols:
                    candidate_payload["extra_context"] = {
                        col: to_row[col] for col in extra_context_cols
                    }
                candidates.append(candidate_payload)

            batch_item = {
                "from_key": from_key,
                "from_name": from_row["_from_name_raw"],
                "from_canonical_name": from_row["_from_canonical_name"],
                "anchor_context": anchors,
                "extra_context": extras,
                "candidates": candidates,
            }
            batch_items.append(batch_item)

        prompt = build_batch_prompt(
            country=country,
            year_from=year_from,
            year_to=year_to,
            anchor_cols=anchor_cols,
            batch_items=batch_items,
        )

        logger.info(
            "run_id=%s stage=adjudication batch=%d size=%d",
            run_id,
            batch_index,
            len(batch_items),
        )

        try:
            raw_response = llm_client.generate_json(
                prompt=prompt,
                schema=LLMBatchResponse,
                model=model,
                temperature=temperature,
                seed=seed,
            )
            parsed = LLMBatchResponse.model_validate(raw_response)
            by_from = {decision.from_key: decision for decision in parsed.decisions}

            for from_key in batch_keys:
                decision = by_from.get(from_key)
                links = []
                if decision is None:
                    links = [
                        {
                            "to_key": None,
                            "link_type": "unknown",
                            "score": 0.0,
                            "evidence": "LLM omitted this item in batch response.",
                        }
                    ]
                else:
                    links = [link.model_dump() for link in decision.links]
                    if not links:
                        links = [
                            {
                                "to_key": None,
                                "link_type": "no_match",
                                "score": 0.0,
                                "evidence": "No valid link selected by model.",
                            }
                        ]

                append_jsonl(
                    links_raw_path,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "batch_index": batch_index,
                        "status": "ok",
                        "from_key": from_key,
                        "links": links,
                    },
                )
        except Exception as exc:
            logger.exception(
                "run_id=%s stage=adjudication batch=%d failed=%s",
                run_id,
                batch_index,
                exc,
            )
            warnings.append(f"Batch {batch_index} failed: {exc}")
            for from_key in batch_keys:
                append_jsonl(
                    links_raw_path,
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "batch_index": batch_index,
                        "status": "error",
                        "from_key": from_key,
                        "error": str(exc),
                    },
                )

    final_records = read_jsonl(links_raw_path)
    latest_success: dict[str, dict[str, Any]] = {}
    for record in final_records:
        if record.get("status") == "ok":
            latest_success[record["from_key"]] = record

    rows: list[dict[str, Any]] = []
    for from_key in df_from_work["_from_key"].tolist():
        from_row = from_lookup.loc[from_key]
        anchor_payload = {col: from_row[col] for col in anchor_cols}

        record = latest_success.get(from_key)
        links = record.get("links", []) if record else []
        if not links:
            links = [
                {
                    "to_key": None,
                    "link_type": "unknown",
                    "score": 0.0,
                    "evidence": "No completed adjudication record.",
                }
            ]

        allowed_to_keys = {item["to_key"] for item in candidate_map.get(from_key, [])}

        for link in links:
            to_key = link.get("to_key")
            to_row = to_lookup.loc[to_key] if to_key in to_lookup.index else None
            candidate_membership = to_key is None or to_key in allowed_to_keys
            anchor_match = True
            if to_row is not None and anchor_cols:
                anchor_match = all(from_row[col] == to_row[col] for col in anchor_cols)

            row = {
                "from_name": from_row["_from_name_raw"],
                "to_name": to_row["_to_name_raw"] if to_row is not None else None,
                "from_canonical_name": from_row["_from_canonical_name"],
                "to_canonical_name": to_row["_to_canonical_name"] if to_row is not None else None,
                "from_id": from_row["_from_id"],
                "to_id": to_row["_to_id"] if to_row is not None else None,
                "score": float(link.get("score", 0.0)),
                "link_type": str(link.get("link_type", "unknown")),
                "evidence": str(link.get("evidence", "")).strip()[:400],
                "country": country,
                "year_from": year_from,
                "year_to": year_to,
                "run_id": run_id,
                "from_key": from_key,
                "to_key": to_key,
                "constraints_passed": {
                    "candidate_membership": candidate_membership,
                    "anchor_match": anchor_match,
                },
            }
            row.update(anchor_payload)
            rows.append(row)

    crosswalk = pd.DataFrame(rows)
    for col in CROSSWALK_BASE_COLUMNS:
        if col not in crosswalk.columns:
            crosswalk[col] = None

    crosswalk = apply_global_flags(crosswalk, low_score_threshold=review_score_threshold)
    review_queue = build_review_queue(crosswalk)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": request.schema_version,
        "output_schema_version": "1.0.0",
        "request": request.model_dump(),
        "counts": summarize_counts(crosswalk),
        "coverage_by_group": coverage_summary(crosswalk, anchor_cols),
        "warnings": warnings,
        "artifacts": {
            "run_dir": str(run_dir),
            "links_raw_jsonl": str(links_raw_path),
        },
        "loader_metadata": loader_metadata,
        "started_at": start_time,
        "finished_at": now_iso(),
    }

    if output_write_csv:
        csv_path = run_dir / "evolution_key.csv"
        crosswalk.to_csv(csv_path, index=False)
        metadata["artifacts"]["evolution_key_csv"] = str(csv_path)

    if output_write_parquet:
        parquet_path = run_dir / "evolution_key.parquet"
        try:
            crosswalk.to_parquet(parquet_path, index=False)
            metadata["artifacts"]["evolution_key_parquet"] = str(parquet_path)
        except Exception as exc:
            warning = f"Parquet write failed ({exc}); CSV output still written."
            warnings.append(warning)
            logger.warning("run_id=%s stage=output %s", run_id, warning)

    review_path = run_dir / "review_queue.csv"
    review_queue.to_csv(review_path, index=False)
    metadata["artifacts"]["review_queue_csv"] = str(review_path)

    metadata_path = run_dir / "run_metadata.json"
    write_json(metadata_path, metadata)
    metadata["artifacts"]["run_metadata_json"] = str(metadata_path)

    logger.info("run_id=%s stage=finish rows=%d", run_id, len(crosswalk))

    anchor_cols_order = anchor_cols.copy()
    preferred_order = CROSSWALK_BASE_COLUMNS + anchor_cols_order + ["review_flags", "review_reason"]
    ordered_cols = [col for col in preferred_order if col in crosswalk.columns] + [
        col for col in crosswalk.columns if col not in preferred_order
    ]
    crosswalk = crosswalk[ordered_cols]

    return crosswalk, metadata
