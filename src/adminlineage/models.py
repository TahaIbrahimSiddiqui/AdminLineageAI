"""Pydantic data models used across the package."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import PROMPT_SCHEMA_VERSION

LinkType = Literal["rename", "split", "merge", "transfer", "no_match", "unknown"]
RelationshipType = Literal[
    "father_to_father",
    "father_to_child",
    "child_to_father",
    "child_to_child",
    "unknown",
]
RequestRelationshipType = Literal[
    "auto",
    "father_to_father",
    "father_to_child",
    "child_to_father",
    "child_to_child",
]
ExactStringPruneMode = Literal["none", "from", "to"]
SecondStageEventType = Literal["rename", "split", "merge", "transfer", "dissolved", "unknown"]


class MappingRequest(BaseModel):
    """Canonical request object for one evolution-key run."""

    model_config = ConfigDict(extra="forbid")

    country: str
    year_from: int | str
    year_to: int | str
    exact_match: list[str] = Field(default_factory=list)
    map_col_from: str
    map_col_to: str
    relationship: RequestRelationshipType = "auto"
    string_exact_match_prune: ExactStringPruneMode = "none"
    evidence: bool = False
    reason: bool = False
    model: str = "gemini-3.1-flash-lite-preview"
    batch_size: int = 5
    max_candidates: int = 6
    seed: int = 42
    temperature: float = 0.75
    enable_google_search: bool = True
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


class LLMChosenLinkBare(BaseModel):
    """Single to-unit decision returned by LLM for one from-unit."""

    model_config = ConfigDict(extra="forbid")

    to_key: str | None = None
    link_type: LinkType
    relationship: RelationshipType
    score: float = Field(ge=0.0, le=1.0)


class LLMChosenLinkWithEvidence(LLMChosenLinkBare):
    """LLM response item that includes a short evidence summary."""

    evidence: str


class LLMChosenLinkWithReason(LLMChosenLinkBare):
    """Reason-enabled LLM response item."""

    reason: str


class LLMChosenLinkWithEvidenceAndReason(LLMChosenLinkWithEvidence):
    """LLM response item that includes both evidence and reason."""

    reason: str


class LLMFromDecisionBare(BaseModel):
    """LLM decisions for one from-unit."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    links: list[LLMChosenLinkBare]


class LLMFromDecisionWithEvidence(BaseModel):
    """LLM decisions for one from-unit when evidence summaries are enabled."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    links: list[LLMChosenLinkWithEvidence]


class LLMFromDecisionWithReason(BaseModel):
    """LLM decisions for one from-unit when detailed reasons are enabled."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    links: list[LLMChosenLinkWithReason]


class LLMFromDecisionWithEvidenceAndReason(BaseModel):
    """LLM decisions for one from-unit with evidence and reasons enabled."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    links: list[LLMChosenLinkWithEvidenceAndReason]


class LLMBatchResponseBare(BaseModel):
    """Strict JSON structure expected from Gemini without evidence or reason."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMFromDecisionBare]


class LLMBatchResponseWithEvidence(BaseModel):
    """Strict JSON structure expected from Gemini when evidence is enabled."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMFromDecisionWithEvidence]


class LLMBatchResponseWithReason(BaseModel):
    """Strict JSON structure expected from Gemini when detailed reasons are enabled."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMFromDecisionWithReason]


class LLMBatchResponseWithEvidenceAndReason(BaseModel):
    """Strict JSON structure expected from Gemini when evidence and reason are enabled."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMFromDecisionWithEvidenceAndReason]


# Retain the historical alias that older callers and tests still import.
LLMBatchResponseNoReason = LLMBatchResponseWithEvidence

BatchResponse: TypeAlias = (
    LLMBatchResponseBare
    | LLMBatchResponseWithEvidence
    | LLMBatchResponseWithReason
    | LLMBatchResponseWithEvidenceAndReason
)
BatchResponseModel: TypeAlias = (
    type[LLMBatchResponseBare]
    | type[LLMBatchResponseWithEvidence]
    | type[LLMBatchResponseWithReason]
    | type[LLMBatchResponseWithEvidenceAndReason]
)


def get_batch_response_model(
    *, include_reason: bool, include_evidence: bool
) -> BatchResponseModel:
    """Return the strict response model for the requested reason mode."""

    if include_evidence and include_reason:
        return LLMBatchResponseWithEvidenceAndReason
    if include_evidence:
        return LLMBatchResponseWithEvidence
    if include_reason:
        return LLMBatchResponseWithReason
    return LLMBatchResponseBare


class SecondStageResearch(BaseModel):
    """Strict JSON payload for second-stage lineage research."""

    model_config = ConfigDict(extra="forbid")

    event_type: SecondStageEventType
    lineage_hint: str = ""
    notes: str = ""


class SecondStageDecision(BaseModel):
    """Strict JSON payload for second-stage shortlist adjudication."""

    model_config = ConfigDict(extra="forbid")

    primary_key: str
    selected_secondary_keys: list[str] = Field(default_factory=list)
    link_type: LinkType
    relationship: RelationshipType = "unknown"
    score: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    reason: str = ""


class RetrySettings(BaseModel):
    """Retry settings for transient LLM failures."""

    model_config = ConfigDict(extra="forbid")

    # A slightly longer retry window helps Gemini ride out short-lived 503 spikes
    # before we give up on a row and mark it unresolved.
    max_attempts: int = 6
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    jitter_seconds: float = 0.2


class CacheSettings(BaseModel):
    """LLM cache settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: Literal["sqlite"] = "sqlite"
    path: str = "llm_cache.sqlite"


class ReplaySettings(BaseModel):
    """Exact replay settings for fully completed runs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    store_dir: str = ".adminlineage_replay"
