"""Provider-facing LLM interfaces and shared exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Base class for model providers that return structured JSON."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: Any,
        model: str,
        temperature: float,
        seed: int,
        *,
        enable_google_search: bool = False,
    ) -> dict[str, Any]:
        """Generate JSON response validated against schema."""

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float,
        seed: int,
        *,
        enable_google_search: bool = False,
    ) -> str:
        """Generate plain text response for providers that support free-form output."""

        raise LLMServiceError(
            f"{self.__class__.__name__} does not support plain-text generation."
        )


class LLMServiceError(RuntimeError):
    """Raised for provider-side failures or invalid responses."""


class TransientLLMError(LLMServiceError):
    """Raised when a call may succeed on retry."""


class QuotaExceededLLMError(LLMServiceError):
    """Raised when provider quota or billing limits are exhausted."""
