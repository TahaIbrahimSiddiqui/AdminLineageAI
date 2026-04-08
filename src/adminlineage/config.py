"""Configuration loading and validation for CLI runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .models import CacheSettings, ReplaySettings, RetrySettings


class RequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str
    year_from: int | str
    year_to: int | str
    map_col_from: str
    map_col_to: str | None = None
    exact_match: list[str] = Field(default_factory=list)
    id_col_from: str | None = None
    id_col_to: str | None = None
    extra_context_cols: list[str] = Field(default_factory=list)
    relationship: Literal[
        "auto",
        "father_to_father",
        "father_to_child",
        "child_to_father",
        "child_to_child",
    ] = "auto"
    string_exact_match_prune: Literal["none", "from", "to"] = "none"
    evidence: bool = False
    reason: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["files", "python_hook"] = "files"
    from_path: str | None = None
    to_path: str | None = None
    callable: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> DataConfig:
        if self.mode == "files":
            if not self.from_path or not self.to_path:
                raise ValueError("data.from_path and data.to_path are required when mode=files")
        if self.mode == "python_hook" and not self.callable:
            raise ValueError("data.callable is required when mode=python_hook")
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["gemini", "mock"] = "gemini"
    model: str = "gemini-3.1-flash-lite-preview"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.75
    seed: int = 42
    enable_google_search: bool = True
    request_timeout_seconds: int | None = 90


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 25
    max_candidates: int = 6
    review_score_threshold: float = 0.6


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_csv: bool = True
    write_parquet: bool = True


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _source_path: Path | None = PrivateAttr(default=None)

    request: RequestConfig
    data: DataConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    replay: ReplaySettings = Field(default_factory=ReplaySettings)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def source_dir(self) -> Path | None:
        if self._source_path is None:
            return None
        return self._source_path.parent


class LoadedFrames(BaseModel):
    """Runtime container for loaded dataframes and loader metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df_from: Any
    df_to: Any
    loader_metadata: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate YAML configuration."""

    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Configuration must be a YAML object: {path_obj}")
    cfg = RunConfig.model_validate(content)
    cfg._source_path = path_obj.resolve()
    return cfg
