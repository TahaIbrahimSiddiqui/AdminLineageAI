"""Abstract LLM client interface."""

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


class LLMServiceError(RuntimeError):
    """Raised for provider-side failures or invalid responses."""


class TransientLLMError(LLMServiceError):
    """Raised when a call may succeed on retry."""


class QuotaExceededLLMError(LLMServiceError):
    """Raised when the provider rejects calls because billing or quota is exhausted."""
