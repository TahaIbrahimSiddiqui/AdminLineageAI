"""Gemini client implementation with cache, retries, and JSON repair."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

from ..prompts import build_repair_prompt
from ..schema import PROMPT_SCHEMA_VERSION
from ..utils import now_iso
from .base import BaseLLMClient, LLMServiceError, TransientLLMError
from .cache import SQLiteCache
from .retry import retry_call


class GeminiClient(BaseLLMClient):
    """LLM client backed by Gemini models through google-genai SDK."""

    def __init__(
        self,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        cache: SQLiteCache | None = None,
        max_attempts: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 20.0,
        jitter_seconds: float = 0.2,
    ) -> None:
        self.api_key_env = api_key_env
        self.cache = cache
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds

    def _call_model(self, prompt: str, *, model: str, temperature: float, seed: int) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMServiceError(f"Missing Gemini API key in environment variable {self.api_key_env}")

        try:
            from google import genai
        except Exception as exc:
            raise LLMServiceError(
                "google-genai is required for GeminiClient. Install dependency 'google-genai'."
            ) from exc

        client = genai.Client(api_key=api_key)

        def _invoke() -> str:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                        "seed": seed,
                    },
                )
            except Exception as exc:  # Provider/network errors should be retried.
                raise TransientLLMError(str(exc)) from exc
            text = getattr(response, "text", None)
            if not text:
                raise LLMServiceError("Gemini returned an empty response")
            return text

        return retry_call(
            _invoke,
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            jitter_seconds=self.jitter_seconds,
            retry_exceptions=(TransientLLMError,),
        )

    @staticmethod
    def _validate_schema(data: dict[str, Any], schema: Any) -> dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(data).model_dump()
        return data

    def generate_json(
        self,
        prompt: str,
        schema: Any,
        model: str,
        temperature: float,
        seed: int,
    ) -> dict[str, Any]:
        schema_version = PROMPT_SCHEMA_VERSION

        if self.cache is not None:
            cached = self.cache.get(model=model, prompt=prompt, schema_version=schema_version)
            if cached is not None:
                return self._validate_schema(cached, schema)

        raw_text = self._call_model(prompt, model=model, temperature=temperature, seed=seed)
        try:
            parsed = json.loads(raw_text)
            validated = self._validate_schema(parsed, schema)
        except Exception as first_error:
            repair_prompt = build_repair_prompt(
                original_prompt=prompt,
                invalid_output=raw_text,
                error_message=str(first_error),
            )
            repair_text = self._call_model(
                repair_prompt,
                model=model,
                temperature=0.0,
                seed=seed,
            )
            try:
                parsed = json.loads(repair_text)
                validated = self._validate_schema(parsed, schema)
            except Exception as second_error:
                raise LLMServiceError(
                    f"Gemini output invalid after repair: {second_error}"
                ) from second_error

        if self.cache is not None:
            self.cache.set(
                model=model,
                prompt=prompt,
                schema_version=schema_version,
                response_json=validated,
                created_at=now_iso(),
            )

        return validated
