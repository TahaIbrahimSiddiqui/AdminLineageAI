"""Pydantic data models used across the package."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import OUTPUT_SCHEMA_VERSION, PROMPT_SCHEMA_VERSION

LinkType = Literal["rename", "split", "merge", "transfer", "no_match", "unknown"]


class MappingRequest(BaseModel):
    """Canonical request object for one evolution-key run."""

    model_config = ConfigDict(extra="forbid")

    country: str
    year_from: int | str
    year_to: int | str
    anchor_cols: list[str] = Field(default_factory=list)
    map_col_from: str
    map_col_to: str
    model: str = "gemini-2.5-pro"
    batch_size: int = 25
    max_candidates: int = 15
    seed: int = 42
    schema_version: str = PROMPT_SCHEMA_VERSION

    @field_validator("country")
    @classmethod
    def _non_empty_country(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("country must be non-empty")
        return value.strip()

    @field_validator("batch_size", "max_candidates")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value


class AdminUnitRecord(BaseModel):
    """Normalized administrative unit row."""

    model_config = ConfigDict(extra="forbid")

    period: str
    name: str
    canonical_name: str
    unit_id: str | int | None = None
    anchors: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)


class CandidateLink(BaseModel):
    """A proposed lineage link for the final crosswalk."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    to_key: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    link_type: LinkType
    evidence: str = ""
    constraints_passed: dict[str, bool] = Field(default_factory=dict)


class LLMChosenLink(BaseModel):
    """Single to-unit decision returned by LLM for one from-unit."""

    model_config = ConfigDict(extra="forbid")

    to_key: str | None = None
    link_type: LinkType
    score: float = Field(ge=0.0, le=1.0)
    evidence: str


class LLMFromDecision(BaseModel):
    """LLM decisions for one from-unit."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    links: list[LLMChosenLink]


class LLMBatchResponse(BaseModel):
    """Strict JSON structure expected from Gemini batch adjudication."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMFromDecision]


class EvolutionKey(BaseModel):
    """Portable representation of one full evolution key artifact."""

    model_config = ConfigDict(extra="forbid")

    request: MappingRequest
    links: list[CandidateLink]
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = OUTPUT_SCHEMA_VERSION


class RetrySettings(BaseModel):
    """Retry settings for transient LLM failures."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    jitter_seconds: float = 0.2


class CacheSettings(BaseModel):
    """LLM cache settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: Literal["sqlite"] = "sqlite"
    path: str = "llm_cache.sqlite"


class RunMetadata(BaseModel):
    """Metadata persisted alongside outputs."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    request: dict[str, Any]
    counts: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = OUTPUT_SCHEMA_VERSION
