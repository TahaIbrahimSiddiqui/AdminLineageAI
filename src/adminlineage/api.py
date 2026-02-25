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
        anchor_cols=anchor_cols,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        extra_context_cols=extra_context_cols,
        aliases=aliases,
        model=model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
        max_candidates=max_candidates,
        resume_dir=resume_dir,
        run_name=run_name,
        seed=seed,
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
    anchor_cols: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    extra_context_cols: list[str] | None = None,
    aliases: pd.DataFrame | None = None,
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
        anchor_cols=anchor_cols,
        id_col_from=id_col_from,
        id_col_to=id_col_to,
        extra_context_cols=extra_context_cols,
        aliases=aliases,
        max_candidates=max_candidates,
    )


def validate_inputs(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    *,
    country: str,
    map_col_from: str,
    map_col_to: str | None = None,
    anchor_cols: list[str] | None = None,
    id_col_from: str | None = None,
    id_col_to: str | None = None,
    aliases: pd.DataFrame | None = None,
) -> dict:
    """Validate inputs and return diagnostics without running the pipeline."""

    return validate_inputs_data(
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
