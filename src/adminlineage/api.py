"""Public notebook-friendly API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .export import export_crosswalk_file
from .pipeline import preview_pipeline_plan, run_pipeline
from .validation import validate_inputs_data


def build_evolution_key(
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
    relationship: str = "auto",
    string_exact_match_prune: str = "none",
    evidence: bool = False,
    reason: bool = False,
    model: str = "gemini-3.1-flash-lite-preview",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 25,
    max_candidates: int = 15,
    output_dir: str | Path = "outputs",
    seed: int = 42,
    temperature: float = 0.75,
    enable_google_search: bool = True,
    request_timeout_seconds: int | None = 90,
    env_search_dir: str | Path | None = None,
    replay_enabled: bool = False,
    replay_store_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build an administrative evolution key between two periods."""

    return run_pipeline(
        df_from,
        df_to,
        country=country,
        year_from=year_from,
        year_to=year_to,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        extra_context_cols=extra_context_cols,
        relationship=relationship,
        string_exact_match_prune=string_exact_match_prune,
        evidence=evidence,
        reason=reason,
        model=model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
        max_candidates=max_candidates,
        output_dir=output_dir,
        seed=seed,
        temperature=temperature,
        enable_google_search=enable_google_search,
        request_timeout_seconds=request_timeout_seconds,
        env_search_dir=env_search_dir,
        replay_enabled=replay_enabled,
        replay_store_dir=replay_store_dir,
    )


def preview_plan(
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
    string_exact_match_prune: str = "none",
    max_candidates: int = 15,
) -> dict:
    """Preview grouping and candidate-generation plan without LLM calls."""

    return preview_pipeline_plan(
        df_from,
        df_to,
        country=country,
        year_from=year_from,
        year_to=year_to,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        extra_context_cols=extra_context_cols,
        string_exact_match_prune=string_exact_match_prune,
        max_candidates=max_candidates,
    )


def validate_inputs(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    map_col_from: str,
    map_col_to: str | None = None,
    exact_match: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
) -> dict:
    """Validate inputs and return diagnostics without running the pipeline."""

    return validate_inputs_data(
        df_from,
        df_to,
        country=country,
        map_col_from=map_col_from,
        map_col_to=map_col_to,
        exact_match=exact_match,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
    )


def export_crosswalk(
    *,
    input_path: str | Path,
    output_format: str,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a materialized crosswalk file into another format."""

    return export_crosswalk_file(
        input_path=input_path,
        output_format=output_format,
        output_path=output_path,
    )
