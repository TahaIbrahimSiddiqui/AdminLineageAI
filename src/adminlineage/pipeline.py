"""Core evolution-key pipeline implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from .candidates import (
    generate_shortlist_from_records,
    prepare_target_records,
)
from .io import append_jsonl, read_jsonl, write_json, write_jsonl
from .llm import (
    BaseLLMClient,
    GeminiClient,
    SQLiteCache,
)
from .logging_utils import setup_logger
from .models import (
    BatchResponseModel,
    ExactStringPruneMode,
    MappingRequest,
    RequestRelationshipType,
    get_batch_response_model,
)
from .normalize import (
    add_normalized_columns,
    canonicalize_name,
    normalized_key_frame,
)
from .pipeline_adjudication import run_adjudication_stage as run_adjudication_stage_helper
from .pipeline_materialization import (
    finalize_crosswalk_table as finalize_crosswalk_table_helper,
)
from .pipeline_materialization import (
    materialize_rows as materialize_rows_helper,
)
from .pipeline_second_stage import run_second_stage as run_second_stage_helper
from .replay import (
    build_replay_identity,
    load_replay_bundle,
    publish_replay_bundle,
    resolve_replay_store_dir,
)
from .review import coverage_summary, summarize_counts
from .schema import (
    OUTPUT_SCHEMA_VERSION,
    RELATIONSHIP_TYPES,
)
from .utils import build_run_id, ensure_dir, now_iso, sanitize_name
from .validation import collapse_duplicate_match_keys, validate_inputs_data

_MATCHED_LINK_TYPES = {"rename", "split", "merge", "transfer"}
_VALID_RELATIONSHIPS = set(RELATIONSHIP_TYPES)
_GLOBAL_SCOPE_KEY = "__all__"
ScopeKey: TypeAlias = tuple[Any, ...] | str


def _default_run_name(country: str, year_from: int | str, year_to: int | str, map_col: str) -> str:
    return f"{sanitize_name(country)}_{year_from}_{year_to}_{sanitize_name(map_col)}"


def _archive_resume_file(
    records_path: Path,
    *,
    logger: Any,
    current_run_id: str,
    existing_records: list[dict[str, Any]],
    record_label: str,
) -> Path:
    existing_run_ids = sorted(
        {
            str(record.get("run_id"))
            for record in existing_records
            if record.get("run_id") not in {None, current_run_id}
        }
    )
    suffix = existing_run_ids[0] if len(existing_run_ids) == 1 else "mixed"
    archive_path = records_path.with_name(f"{records_path.stem}.archive-{suffix}.jsonl")
    counter = 1
    while archive_path.exists():
        archive_path = records_path.with_name(
            f"{records_path.stem}.archive-{suffix}-{counter}.jsonl"
        )
        counter += 1

    records_path.replace(archive_path)
    logger.info(
        "run_id=%s stage=resume archived_stale_%s=%s",
        current_run_id,
        record_label,
        archive_path.name,
    )
    return archive_path


def _prepare_resume_records(
    *,
    records_path: Path,
    run_id: str,
    logger: Any,
    warnings: list[str],
    record_label: str,
) -> list[dict[str, Any]]:
    existing_records = read_jsonl(records_path)
    if not existing_records:
        return []

    existing_run_ids = {record.get("run_id") for record in existing_records if record.get("run_id")}
    same_request = existing_run_ids == {run_id}
    legacy_records = not existing_run_ids and bool(existing_records)
    if same_request and not legacy_records:
        return existing_records

    # Raw link files are tied to the original request shape.
    # Mixing them across runs gets messy fast.
    archive_path = _archive_resume_file(
        records_path,
        logger=logger,
        current_run_id=run_id,
        existing_records=existing_records,
        record_label=record_label,
    )
    warnings.append(
        f"Existing {record_label.replace('_', ' ')} did not match the current request and "
        "were archived to "
        f"{archive_path.name}."
    )
    return []


def _build_llm_client(
    *,
    gemini_api_key_env: str,
    cache_enabled: bool,
    cache_path: Path,
    retry_max_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter: float,
    request_timeout_seconds: int | None,
    env_search_dir: str | Path | None,
) -> BaseLLMClient:
    cache = SQLiteCache(cache_path) if cache_enabled else None
    return GeminiClient(
        api_key_env=gemini_api_key_env,
        cache=cache,
        max_attempts=retry_max_attempts,
        base_delay_seconds=retry_base_delay,
        max_delay_seconds=retry_max_delay,
        jitter_seconds=retry_jitter,
        request_timeout_seconds=request_timeout_seconds,
        env_search_dir=env_search_dir,
    )


def _prepare_workframes(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    map_col_from: str,
    map_col_to: str,
    id_col_from: str | None,
    id_col_to: str | None,
    exact_match: list[str],
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

    for col in exact_match:
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
    for col in exact_match:
        df_from_work[_normalized_scope_col("from", col)] = normalized_key_frame(
            df_from_work,
            [col],
        )[col]
        df_to_work[_normalized_scope_col("to", col)] = normalized_key_frame(
            df_to_work,
            [col],
        )[col]
    return df_from_work, df_to_work


def _normalized_scope_col(prefix: str, col: str) -> str:
    return f"_{prefix}_scope_{col}"


def _normalized_scope_cols(prefix: str, exact_match: list[str]) -> list[str]:
    return [_normalized_scope_col(prefix, col) for col in exact_match]


def _normalize_group_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _normalize_group_key(key: Any) -> tuple[Any, ...]:
    values = key if isinstance(key, tuple) else (key,)
    return tuple(_normalize_group_value(value) for value in values)


def _collapse_input_frames(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    map_col_from: str,
    map_col_to: str,
    exact_match: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_from_effective, _ = collapse_duplicate_match_keys(
        df_from,
        key_cols=[*exact_match, map_col_from],
        side_label="df_from",
    )
    df_to_effective, _ = collapse_duplicate_match_keys(
        df_to,
        key_cols=[*exact_match, map_col_to],
        side_label="df_to",
    )
    return df_from_effective, df_to_effective


def _build_candidate_maps(
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    *,
    exact_match: list[str],
    max_candidates: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, ScopeKey]]:
    to_groups: dict[ScopeKey, list[Any]] = {}
    to_scope_cols = _normalized_scope_cols("to", exact_match)
    from_scope_cols = _normalized_scope_cols("from", exact_match)
    if exact_match:
        for key, group in df_to_work.groupby(to_scope_cols, dropna=False):
            to_groups[_normalize_group_key(key)] = prepare_target_records(group)
    else:
        to_groups[_GLOBAL_SCOPE_KEY] = prepare_target_records(df_to_work)

    candidate_map: dict[str, list[dict[str, Any]]] = {}
    group_map: dict[str, ScopeKey] = {}
    column_index = {name: idx for idx, name in enumerate(df_from_work.columns)}

    for row in df_from_work.itertuples(index=False, name=None):
        group_key: ScopeKey
        if exact_match:
            group_key = tuple(
                _normalize_group_value(row[column_index[col]])
                for col in from_scope_cols
            )
        else:
            group_key = _GLOBAL_SCOPE_KEY

        candidate_map[row[column_index["_from_key"]]] = generate_shortlist_from_records(
            row[column_index["_from_tokens"]],
            row[column_index["_from_char_ngrams"]],
            to_groups.get(group_key, []),
            max_candidates=max_candidates,
        )
        group_map[row[column_index["_from_key"]]] = group_key

    return candidate_map, group_map


def _scope_key_for_row(
    row: tuple[Any, ...],
    *,
    column_index: dict[str, int],
    scope_cols: list[str],
) -> ScopeKey:
    if not scope_cols:
        return _GLOBAL_SCOPE_KEY
    return tuple(_normalize_group_value(row[column_index[col]]) for col in scope_cols)


def _resolve_exact_string_matches(
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    *,
    exact_match: list[str],
    prune_mode: ExactStringPruneMode,
) -> dict[str, Any]:
    """Resolve normalized exact-name matches within the current normalized scope."""

    if prune_mode == "none":
        return {
            "from_to": {},
            "matched_from_keys": set(),
            "matched_to_keys": set(),
        }

    from_scope_cols = _normalized_scope_cols("from", exact_match)
    to_scope_cols = _normalized_scope_cols("to", exact_match)
    to_index = {name: idx for idx, name in enumerate(df_to_work.columns)}
    from_index = {name: idx for idx, name in enumerate(df_from_work.columns)}

    to_key_by_scope_and_name: dict[ScopeKey, dict[str, str]] = {}
    # Exact string hits are cheap and deterministic, so settle them before asking the model.
    for row in df_to_work.itertuples(index=False, name=None):
        scope_key: ScopeKey
        if exact_match:
            scope_key = tuple(
                _normalize_group_value(row[to_index[col]])
                for col in to_scope_cols
            )
        else:
            scope_key = _GLOBAL_SCOPE_KEY
        scope_bucket = to_key_by_scope_and_name.setdefault(scope_key, {})
        scope_bucket[row[to_index["_to_canonical_name"]]] = row[to_index["_to_key"]]

    from_to: dict[str, str] = {}
    for row in df_from_work.itertuples(index=False, name=None):
        scope_key = _scope_key_for_row(
            row,
            column_index=from_index,
            scope_cols=from_scope_cols,
        )
        to_key = to_key_by_scope_and_name.get(scope_key, {}).get(
            row[from_index["_from_canonical_name"]]
        )
        if to_key is None:
            continue
        from_to[row[from_index["_from_key"]]] = to_key

    return {
        "from_to": from_to,
        "matched_from_keys": set(from_to),
        "matched_to_keys": set(from_to.values()),
    }


def _build_exact_match_links(
    exact_matches: dict[str, str],
    *,
    requested_relationship: RequestRelationshipType,
    evidence: bool,
    reason: bool,
) -> dict[str, list[dict[str, Any]]]:
    relationship_value = (
        requested_relationship
        if requested_relationship != "auto"
        else "father_to_father"
    )
    evidence_text = "Resolved by normalized exact string match within the current scope."
    reason_text = "Resolved by normalized exact string match within the current scope."

    exact_links: dict[str, list[dict[str, Any]]] = {}
    for from_key, to_key in exact_matches.items():
        link = {
            "to_key": to_key,
            "link_type": "rename",
            "relationship": relationship_value,
            "score": 1.0,
        }
        if evidence:
            link["evidence"] = evidence_text
        if reason:
            link["reason"] = reason_text
        exact_links[from_key] = [link]
    return exact_links


def _ai_workframes_after_exact_prune(
    df_from_work: pd.DataFrame,
    df_to_work: pd.DataFrame,
    *,
    exact_match_info: dict[str, Any],
    prune_mode: ExactStringPruneMode,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_from_keys = exact_match_info["matched_from_keys"]
    matched_to_keys = exact_match_info["matched_to_keys"]

    if prune_mode == "none":
        return df_from_work, df_to_work
    if prune_mode == "from":
        # Exact-string matches are final outputs; only unmatched from-rows proceed to AI while
        # the full scoped to-side pool remains available for the remaining rows.
        return (
            df_from_work.loc[~df_from_work["_from_key"].isin(matched_from_keys)].reset_index(
                drop=True
            ),
            df_to_work,
        )
    # In prune='to' mode, exact-string matches are final outputs and later AI batches only see
    # the unmatched to-side candidates from the same scope.
    return (
        df_from_work.loc[~df_from_work["_from_key"].isin(matched_from_keys)].reset_index(drop=True),
        df_to_work.loc[~df_to_work["_to_key"].isin(matched_to_keys)].reset_index(drop=True),
    )


def _final_relationship(
    raw_relationship: Any,
    *,
    requested_relationship: RequestRelationshipType,
    link_type: str,
    to_key: str | None,
) -> str:
    if to_key is None or link_type not in _MATCHED_LINK_TYPES:
        return "unknown"
    if requested_relationship != "auto":
        return requested_relationship
    value = str(raw_relationship or "unknown")
    if value in _VALID_RELATIONSHIPS:
        return value
    return "unknown"


def _merge_indicator(*, link_type: str, to_key: str | None) -> str:
    """Return the merge-style indicator used in the final evolution key."""

    if to_key is not None and link_type in _MATCHED_LINK_TYPES:
        return "both"
    return "only_in_from"


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "links_raw": run_dir / "links_raw.jsonl",
        "grounding_notes": run_dir / "grounding_notes.jsonl",
        "second_stage_results": run_dir / "second_stage_results.jsonl",
        "crosswalk_csv": run_dir / "evolution_key.csv",
        "crosswalk_parquet": run_dir / "evolution_key.parquet",
        "review_queue_csv": run_dir / "review_queue.csv",
        "run_metadata_json": run_dir / "run_metadata.json",
    }


def _collect_latest_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    latest_success: dict[str, dict[str, Any]] = {}
    latest_error: dict[str, dict[str, Any]] = {}
    for record in records:
        if "from_key" not in record:
            continue
        if record.get("status") == "ok":
            latest_success[record["from_key"]] = record
        elif record.get("status") == "error":
            latest_error[record["from_key"]] = record
    return latest_success, latest_error


def _second_stage_primary_side(prune_mode: ExactStringPruneMode) -> str | None:
    if prune_mode in {"from", "to"}:
        return prune_mode
    return None


def _unique_search_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        canonical = canonicalize_name(text)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        terms.append(text)
    return terms


def _collect_latest_second_stage_records(
    records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    latest_success: dict[tuple[str, str], dict[str, Any]] = {}
    latest_error: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        primary_side = str(record.get("primary_side", "")).strip()
        primary_key = str(record.get("primary_key", "")).strip()
        if not primary_side or not primary_key:
            continue
        key = (primary_side, primary_key)
        if record.get("status") == "ok":
            latest_success[key] = record
            latest_error.pop(key, None)
        elif record.get("status") == "error":
            latest_success.pop(key, None)
            latest_error[key] = record
    return latest_success, latest_error


def _build_run_metadata(
    *,
    run_id: str,
    request: MappingRequest,
    crosswalk: pd.DataFrame,
    exact_match: list[str],
    warnings: list[str],
    loader_metadata: dict[str, Any],
    start_time: str,
    validation: dict[str, Any],
    error_from_rows: int,
    grounding_attempted_rows: int,
    grounded_rows: int,
    grounding_failed_rows: int,
    second_stage_attempted_rows: int,
    second_stage_rewritten_rows: int,
    second_stage_failed_rows: int,
    replay_key: str | None,
    replay_hit: bool,
    replayed_from_run_id: str | None,
    execution_mode: str,
) -> dict[str, Any]:
    counts = {
        **summarize_counts(crosswalk),
        "from_rows_input": validation["from_rows_input"],
        "from_rows_effective": validation["from_rows_effective"],
        "to_rows_input": validation["to_rows_input"],
        "to_rows_effective": validation["to_rows_effective"],
        "error_from_rows": error_from_rows,
        "grounding_attempted_rows": grounding_attempted_rows,
        "grounded_rows": grounded_rows,
        "grounding_failed_rows": grounding_failed_rows,
        "second_stage_attempted_rows": second_stage_attempted_rows,
        "second_stage_rewritten_rows": second_stage_rewritten_rows,
        "second_stage_failed_rows": second_stage_failed_rows,
    }
    return {
        "run_id": run_id,
        "schema_version": request.schema_version,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "request": request.model_dump(),
        "counts": counts,
        "coverage_by_group": coverage_summary(crosswalk, exact_match),
        "warnings": warnings,
        "loader_metadata": loader_metadata,
        "started_at": start_time,
        "finished_at": now_iso(),
        "replay_key": replay_key,
        "replay_hit": replay_hit,
        "replayed_from_run_id": replayed_from_run_id,
        "execution_mode": execution_mode,
    }


def _finalize_output_artifacts(
    *,
    run_dir: Path,
    crosswalk: pd.DataFrame,
    review_queue: pd.DataFrame,
    metadata: dict[str, Any],
    warnings: list[str],
    output_write_csv: bool,
    output_write_parquet: bool,
    logger: Any,
    run_id: str,
) -> dict[str, Any]:
    paths = _artifact_paths(run_dir)
    artifacts: dict[str, str] = {
        "run_dir": str(run_dir),
        "links_raw_jsonl": str(paths["links_raw"]),
        "review_queue_csv": str(paths["review_queue_csv"]),
        "run_metadata_json": str(paths["run_metadata_json"]),
    }
    if paths["grounding_notes"].exists():
        artifacts["grounding_notes_jsonl"] = str(paths["grounding_notes"])
    if paths["second_stage_results"].exists():
        artifacts["second_stage_results_jsonl"] = str(paths["second_stage_results"])
    if output_write_csv:
        artifacts["evolution_key_csv"] = str(paths["crosswalk_csv"])

    metadata["artifacts"] = artifacts

    if output_write_csv:
        crosswalk.to_csv(paths["crosswalk_csv"], index=False)

    if output_write_parquet:
        try:
            crosswalk.to_parquet(paths["crosswalk_parquet"], index=False)
            metadata["artifacts"]["evolution_key_parquet"] = str(paths["crosswalk_parquet"])
        except Exception as exc:
            warning = f"Parquet write failed ({exc}); wrote CSV fallback instead."
            warnings.append(warning)
            logger.warning("run_id=%s stage=output %s", run_id, warning)
            if "evolution_key_csv" not in metadata["artifacts"]:
                crosswalk.to_csv(paths["crosswalk_csv"], index=False)
                metadata["artifacts"]["evolution_key_csv"] = str(paths["crosswalk_csv"])

    review_queue.to_csv(paths["review_queue_csv"], index=False)
    write_json(paths["run_metadata_json"], metadata)
    return metadata


def preview_pipeline_plan(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    year_from: int | str,
    year_to: int | str,
    map_col_from: str,
    map_col_to: str | None = None,
    exact_match: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    extra_context_cols: list[str] | None = None,
    string_exact_match_prune: ExactStringPruneMode = "none",
    max_candidates: int = 6,
) -> dict[str, Any]:
    """Preview group sizes and candidate budgets without calling LLM."""

    exact_match = exact_match or []
    extra_context_cols = extra_context_cols or []
    validation = validate_inputs_data(
        df_from,
        df_to,
        country=country,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
    )
    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }

    effective_map_col_to = validation["map_col_to"]
    df_from_effective, df_to_effective = _collapse_input_frames(
        df_from,
        df_to,
        map_col_from=map_col_from,
        map_col_to=effective_map_col_to,
        exact_match=exact_match,
    )
    df_from_work, df_to_work = _prepare_workframes(
        df_from_effective,
        df_to_effective,
        map_col_from=map_col_from,
        map_col_to=effective_map_col_to,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        exact_match=exact_match,
        extra_context_cols=extra_context_cols,
    )
    exact_match_info = _resolve_exact_string_matches(
        df_from_work,
        df_to_work,
        exact_match=exact_match,
        prune_mode=string_exact_match_prune,
    )
    ai_from_work, ai_to_work = _ai_workframes_after_exact_prune(
        df_from_work,
        df_to_work,
        exact_match_info=exact_match_info,
        prune_mode=string_exact_match_prune,
    )

    candidate_map, group_map = _build_candidate_maps(
        ai_from_work,
        ai_to_work,
        exact_match=exact_match,
        max_candidates=max_candidates,
    )

    group_counts: dict[str, int] = {}
    for key in group_map.values():
        label = str(key)
        group_counts[label] = group_counts.get(label, 0) + 1

    avg_candidates = 0.0
    if candidate_map:
        avg_candidates = sum(len(candidates) for candidates in candidate_map.values()) / len(
            candidate_map
        )

    return {
        "valid": True,
        "country": country,
        "year_from": year_from,
        "year_to": year_to,
        "from_rows_input": validation["from_rows_input"],
        "from_rows_effective": validation["from_rows_effective"],
        "to_rows_input": validation["to_rows_input"],
        "to_rows_effective": validation["to_rows_effective"],
        "from_rows": int(len(ai_from_work)),
        "to_rows": int(len(ai_to_work)),
        "exact_match": exact_match,
        "string_exact_match_prune": string_exact_match_prune,
        "exact_string_matches": int(len(exact_match_info["matched_from_keys"])),
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
    exact_match: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    extra_context_cols: list[str] | None = None,
    relationship: RequestRelationshipType = "auto",
    string_exact_match_prune: ExactStringPruneMode = "none",
    evidence: bool = False,
    reason: bool = False,
    model: str = "gemini-3.1-flash-lite-preview",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 5,
    max_candidates: int = 6,
    output_dir: str | Path = "outputs",
    seed: int = 42,
    llm_client: BaseLLMClient | None = None,
    temperature: float = 0.75,
    enable_google_search: bool = True,
    cache_enabled: bool = True,
    cache_path: str = "llm_cache.sqlite",
    replay_enabled: bool = False,
    replay_store_dir: str | Path | None = None,
    retry_max_attempts: int = 6,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 20.0,
    retry_jitter: float = 0.2,
    request_timeout_seconds: int | None = 90,
    review_score_threshold: float = 0.6,
    output_write_csv: bool = True,
    output_write_parquet: bool = True,
    loader_metadata: dict[str, Any] | None = None,
    env_search_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete evolution-key pipeline and materialize output artifacts."""

    start_time = now_iso()
    exact_match = exact_match or []
    extra_context_cols = extra_context_cols or []
    loader_metadata = loader_metadata or {}

    validation = validate_inputs_data(
        df_from,
        df_to,
        country=country,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
    )
    if not validation["valid"]:
        raise ValueError("Invalid inputs: " + "; ".join(validation["errors"]))

    map_col_to_effective = validation["map_col_to"]
    request = MappingRequest(
        country=country,
        year_from=year_from,
        year_to=year_to,
        exact_match=exact_match,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        relationship=relationship,
        string_exact_match_prune=string_exact_match_prune,
        evidence=evidence,
        reason=reason,
        model=model,
        batch_size=batch_size,
        max_candidates=max_candidates,
        seed=seed,
        temperature=temperature,
        enable_google_search=enable_google_search,
    )
    llm_backend_name = llm_client.__class__.__name__ if llm_client is not None else "GeminiClient"
    df_from_effective, df_to_effective = _collapse_input_frames(
        df_from,
        df_to,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        exact_match=exact_match,
    )
    replay_identity = build_replay_identity(
        request_payload=request.model_dump(),
        llm_backend=llm_backend_name,
        df_from=df_from_effective,
        df_to=df_to_effective,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        extra_context_cols=extra_context_cols,
    )
    run_identity = {
        "request": request.model_dump(),
        "extra_context_cols": extra_context_cols,
        "llm_backend": llm_backend_name,
        "from_fingerprint": replay_identity["from_fingerprint"],
        "to_fingerprint": replay_identity["to_fingerprint"],
    }

    run_name = _default_run_name(country, year_from, year_to, map_col_from)
    base_dir = ensure_dir(output_dir)
    run_dir = ensure_dir(base_dir / run_name)
    logger = setup_logger(run_dir)
    paths = _artifact_paths(run_dir)

    warnings = list(validation["warnings"])
    run_id = build_run_id(run_identity, seed=seed)
    replay_key = replay_identity["replay_key"] if replay_enabled else None
    replay_dir = (
        resolve_replay_store_dir(replay_store_dir) / replay_key
        if replay_enabled and replay_key is not None
        else None
    )
    logger.info(
        "run_id=%s stage=start country=%s years=%s->%s",
        run_id,
        country,
        year_from,
        year_to,
    )

    if replay_dir is not None:
        try:
            replay_bundle = load_replay_bundle(replay_dir)
        except Exception as exc:
            warning = f"Replay bundle load failed ({exc}); falling back to live run."
            warnings.append(warning)
            logger.warning("run_id=%s stage=replay %s", run_id, warning)
            replay_bundle = None
        if replay_bundle is not None:
            crosswalk = replay_bundle["crosswalk"]
            review_queue = replay_bundle["review_queue"]
            write_jsonl(paths["links_raw"], read_jsonl(replay_bundle["links_raw_path"]))
            grounding_notes_path = replay_bundle.get("grounding_notes_path")
            second_stage_results_path = replay_bundle.get("second_stage_results_path")
            if grounding_notes_path is not None:
                write_jsonl(paths["grounding_notes"], read_jsonl(grounding_notes_path))
            elif paths["grounding_notes"].exists():
                paths["grounding_notes"].unlink()
            if second_stage_results_path is not None:
                write_jsonl(
                    paths["second_stage_results"],
                    read_jsonl(second_stage_results_path),
                )
            elif paths["second_stage_results"].exists():
                paths["second_stage_results"].unlink()
            metadata = _build_run_metadata(
                run_id=run_id,
                request=request,
                crosswalk=crosswalk,
                exact_match=exact_match,
                warnings=warnings,
                loader_metadata=loader_metadata,
                start_time=start_time,
                validation=validation,
                error_from_rows=int(
                    replay_bundle["manifest"].get("counts", {}).get("error_from_rows", 0)
                ),
                grounding_attempted_rows=int(
                    replay_bundle["manifest"]
                    .get("counts", {})
                    .get("grounding_attempted_rows", 0)
                ),
                grounded_rows=int(
                    replay_bundle["manifest"].get("counts", {}).get("grounded_rows", 0)
                ),
                grounding_failed_rows=int(
                    replay_bundle["manifest"]
                    .get("counts", {})
                    .get("grounding_failed_rows", 0)
                ),
                second_stage_attempted_rows=int(
                    replay_bundle["manifest"]
                    .get("counts", {})
                    .get("second_stage_attempted_rows", 0)
                ),
                second_stage_rewritten_rows=int(
                    replay_bundle["manifest"]
                    .get("counts", {})
                    .get("second_stage_rewritten_rows", 0)
                ),
                second_stage_failed_rows=int(
                    replay_bundle["manifest"]
                    .get("counts", {})
                    .get("second_stage_failed_rows", 0)
                ),
                replay_key=replay_key,
                replay_hit=True,
                replayed_from_run_id=replay_bundle["manifest"].get("source_run_id"),
                execution_mode="replay",
            )
            metadata = _finalize_output_artifacts(
                run_dir=run_dir,
                crosswalk=crosswalk,
                review_queue=review_queue,
                metadata=metadata,
                warnings=warnings,
                output_write_csv=output_write_csv,
                output_write_parquet=output_write_parquet,
                logger=logger,
                run_id=run_id,
            )
            logger.info(
                "run_id=%s stage=finish rows=%d execution_mode=replay replay_key=%s",
                run_id,
                len(crosswalk),
                replay_key,
            )
            return crosswalk, metadata
        logger.info("run_id=%s stage=replay miss replay_key=%s", run_id, replay_key)

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
            request_timeout_seconds=request_timeout_seconds,
            env_search_dir=env_search_dir,
        )
    if paths["grounding_notes"].exists():
        paths["grounding_notes"].unlink()
    grounding_enabled = enable_google_search
    if grounding_enabled:
        logger.info(
            "run_id=%s stage=grounding mode=single_pass_structured_json enabled=true",
            run_id,
        )
    effective_batch_size = batch_size
    df_from_work, df_to_work = _prepare_workframes(
        df_from_effective,
        df_to_effective,
        map_col_from=map_col_from,
        map_col_to=map_col_to_effective,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        exact_match=exact_match,
        extra_context_cols=extra_context_cols,
    )
    exact_match_info = _resolve_exact_string_matches(
        df_from_work,
        df_to_work,
        exact_match=exact_match,
        prune_mode=string_exact_match_prune,
    )
    exact_match_links = _build_exact_match_links(
        exact_match_info["from_to"],
        requested_relationship=relationship,
        evidence=evidence,
        reason=reason,
    )
    ai_from_work, ai_to_work = _ai_workframes_after_exact_prune(
        df_from_work,
        df_to_work,
        exact_match_info=exact_match_info,
        prune_mode=string_exact_match_prune,
    )
    from_lookup = df_from_work.set_index("_from_key", drop=False)
    to_lookup = df_to_work.set_index("_to_key", drop=False)

    candidate_map, _group_map = _build_candidate_maps(
        ai_from_work,
        ai_to_work,
        exact_match=exact_match,
        max_candidates=max_candidates,
    )

    links_raw_path = paths["links_raw"]
    second_stage_results_path = paths["second_stage_results"]
    existing_records = _prepare_resume_records(
        records_path=links_raw_path,
        run_id=run_id,
        logger=logger,
        warnings=warnings,
        record_label="raw_link_records",
    )
    completed_from_keys = {
        record["from_key"]
        for record in existing_records
        if record.get("status") == "ok" and "from_key" in record
    }
    exact_match_from_keys = [
        from_key for from_key in exact_match_links if from_key not in completed_from_keys
    ]
    for from_key in exact_match_from_keys:
        append_jsonl(
            links_raw_path,
            {
                "run_id": run_id,
                "timestamp": now_iso(),
                "batch_index": "exact",
                "match_stage": "exact",
                "status": "ok",
                "from_key": from_key,
                "links": exact_match_links[from_key],
            },
        )
    completed_from_keys.update(exact_match_from_keys)
    pending_from_keys = [
        key for key in ai_from_work["_from_key"].tolist() if key not in completed_from_keys
    ]
    logger.info(
        "run_id=%s stage=resume exact_mode=%s exact_matched=%d completed=%d pending=%d",
        run_id,
        string_exact_match_prune,
        len(exact_match_from_keys),
        len(completed_from_keys),
        len(pending_from_keys),
    )

    response_model: BatchResponseModel = get_batch_response_model(
        include_reason=reason,
        include_evidence=evidence,
    )

    run_adjudication_stage_helper(
        pending_from_keys=pending_from_keys,
        batch_size=effective_batch_size,
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
        grounding_notes_path=paths["grounding_notes"],
        warnings=warnings,
        logger=logger,
        from_lookup=from_lookup,
        to_lookup=to_lookup,
        candidate_map=candidate_map,
        extra_context_cols=extra_context_cols,
    )

    final_records = read_jsonl(links_raw_path)
    latest_success, latest_error = _collect_latest_records(final_records)
    grounding_attempted_rows = 0
    grounded_rows = 0
    grounding_failed_rows = 0
    if grounding_enabled and paths["grounding_notes"].exists():
        grounding_records = read_jsonl(paths["grounding_notes"])
        grounding_attempted_rows = sum(
            1 for record in grounding_records if record.get("status") in {"ok", "error"}
        )
        grounded_rows = sum(1 for record in grounding_records if record.get("status") == "ok")
        grounding_failed_rows = sum(
            1 for record in grounding_records if record.get("status") == "error"
        )
    error_from_keys = sorted(set(latest_error) - set(latest_success))
    if error_from_keys:
        failure_warning = (
            f"Adjudication still failed for {len(error_from_keys)} source rows after retries. "
        )
        if evidence:
            failure_warning += (
                "Those rows were kept with error evidence so you can review or rerun them."
            )
        else:
            failure_warning += "Those rows were kept for review or rerun."
        warnings.append(failure_warning)
    if grounding_failed_rows:
        grounding_warning = (
            "Structured grounded adjudication did not complete cleanly for "
            f"{grounding_failed_rows} source rows; unresolved rows were kept "
        )
        if evidence:
            grounding_warning += "with error evidence for review."
        else:
            grounding_warning += "for review."
        warnings.append(grounding_warning)

    crosswalk = pd.DataFrame(
        materialize_rows_helper(
            latest_success,
            latest_error,
            df_from_work=df_from_work,
            df_to_work=df_to_work,
            from_lookup=from_lookup,
            to_lookup=to_lookup,
            exact_match=exact_match,
            candidate_map=candidate_map,
            evidence=evidence,
            reason=reason,
            relationship=relationship,
            country=country,
            year_from=year_from,
            year_to=year_to,
            run_id=run_id,
            final_relationship_fn=_final_relationship,
            merge_indicator_fn=_merge_indicator,
            normalized_scope_col_fn=_normalized_scope_col,
        )
    )
    crosswalk = run_second_stage_helper(
        crosswalk,
        string_exact_match_prune=string_exact_match_prune,
        grounding_enabled=grounding_enabled,
        second_stage_results_path=second_stage_results_path,
        run_id=run_id,
        logger=logger,
        warnings=warnings,
        llm_client=llm_client,
        model=model,
        temperature=temperature,
        seed=seed,
        max_candidates=max_candidates,
        country=country,
        year_from=year_from,
        year_to=year_to,
        relationship=relationship,
        evidence=evidence,
        reason=reason,
        exact_match=exact_match,
        extra_context_cols=extra_context_cols,
        df_from_work=df_from_work,
        df_to_work=df_to_work,
        from_lookup=from_lookup,
        to_lookup=to_lookup,
        prepare_resume_records_fn=_prepare_resume_records,
        collect_latest_second_stage_records_fn=_collect_latest_second_stage_records,
        second_stage_primary_side_fn=_second_stage_primary_side,
        unique_search_terms_fn=_unique_search_terms,
        final_relationship_fn=_final_relationship,
        merge_indicator_fn=_merge_indicator,
        normalized_scope_col_fn=_normalized_scope_col,
    )
    crosswalk, review_queue = finalize_crosswalk_table_helper(
        crosswalk,
        evidence=evidence,
        exact_match=exact_match,
        review_score_threshold=review_score_threshold,
    )

    second_stage_attempted_rows = 0
    second_stage_rewritten_rows = 0
    second_stage_failed_rows = 0
    if second_stage_results_path.exists():
        second_stage_records = read_jsonl(second_stage_results_path)
        latest_second_stage_success, latest_second_stage_error = (
            _collect_latest_second_stage_records(second_stage_records)
        )
        second_stage_attempted_rows = len(latest_second_stage_success) + len(
            latest_second_stage_error
        )
        second_stage_rewritten_rows = sum(
            1
            for record in latest_second_stage_success.values()
            if bool(record.get("rewrite_applied"))
        )
        second_stage_failed_rows = len(latest_second_stage_error)
        if second_stage_failed_rows:
            warnings.append(
                "Second-stage rescue did not complete cleanly for "
                f"{second_stage_failed_rows} primary rows; unresolved rows were kept "
                "for review or rerun."
            )
    if replay_dir is not None and error_from_keys:
        warnings.append(
            "Replay bundle was not updated because the run still has unresolved adjudication "
            "errors."
        )

    metadata = _build_run_metadata(
        run_id=run_id,
        request=request,
        crosswalk=crosswalk,
        exact_match=exact_match,
        warnings=warnings,
        loader_metadata=loader_metadata,
        start_time=start_time,
        validation=validation,
        error_from_rows=len(error_from_keys),
        grounding_attempted_rows=grounding_attempted_rows,
        grounded_rows=grounded_rows,
        grounding_failed_rows=grounding_failed_rows,
        second_stage_attempted_rows=second_stage_attempted_rows,
        second_stage_rewritten_rows=second_stage_rewritten_rows,
        second_stage_failed_rows=second_stage_failed_rows,
        replay_key=replay_key,
        replay_hit=False,
        replayed_from_run_id=None,
        execution_mode="live",
    )
    metadata = _finalize_output_artifacts(
        run_dir=run_dir,
        crosswalk=crosswalk,
        review_queue=review_queue,
        metadata=metadata,
        warnings=warnings,
        output_write_csv=output_write_csv,
        output_write_parquet=output_write_parquet,
        logger=logger,
        run_id=run_id,
    )

    if replay_dir is not None and not error_from_keys:
        try:
            publish_replay_bundle(
                replay_dir=replay_dir,
                replay_key=replay_key or "",
                source_run_id=run_id,
                request_payload=request.model_dump(),
                counts=metadata["counts"],
                llm_backend=llm_backend_name,
                identity=replay_identity["identity"],
                crosswalk=crosswalk,
                review_queue=review_queue,
                links_raw_path=links_raw_path,
                grounding_notes_path=paths["grounding_notes"],
                second_stage_results_path=paths["second_stage_results"],
            )
        except Exception as exc:
            warning = f"Replay bundle publish failed ({exc}); current run outputs are still valid."
            warnings.append(warning)
            logger.warning("run_id=%s stage=replay %s", run_id, warning)
            write_json(paths["run_metadata_json"], metadata)

    logger.info(
        "run_id=%s stage=finish rows=%d execution_mode=live replay_key=%s",
        run_id,
        len(crosswalk),
        replay_key,
    )
    return crosswalk, metadata
