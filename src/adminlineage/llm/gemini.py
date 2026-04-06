"""Gemini client implementation with cache, retries, and JSON repair."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..prompts import build_repair_prompt
from ..schema import PROMPT_SCHEMA_VERSION
from ..utils import load_env_file, now_iso
from .base import (
    BaseLLMClient,
    LLMServiceError,
    QuotaExceededLLMError,
    TransientLLMError,
)
from .cache import SQLiteCache
from .retry import retry_call


class GeminiClient(BaseLLMClient):
    """LLM client backed by Gemini models through google-genai SDK."""

    def __init__(
        self,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        cache: SQLiteCache | None = None,
        max_attempts: int = 6,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 20.0,
        jitter_seconds: float = 0.2,
        min_request_interval_seconds: float = 0.5,
        env_search_dir: str | Path | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.cache = cache
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.env_search_dir = env_search_dir
        self._env_loaded = False
        self._last_request_started_at: float | None = None

    @staticmethod
    def _generation_settings(
        *,
        temperature: float,
        enable_google_search: bool,
    ) -> dict[str, Any]:
        return {
            "temperature": float(temperature),
            "enable_google_search": bool(enable_google_search),
        }

    @staticmethod
    def _build_generate_config(
        *,
        genai_types: Any,
        temperature: float,
        seed: int,
        enable_google_search: bool,
    ) -> Any:
        tools = []
        if enable_google_search:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
        }
        if tools:
            config_kwargs["tools"] = tools
        else:
            config_kwargs["response_mime_type"] = "application/json"
        return genai_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _provider_error_text(exc: Exception) -> str:
        parts: list[str] = []
        for attr in ("message", "status", "code"):
            value = getattr(exc, attr, None)
            if value not in (None, ""):
                parts.append(str(value))

        details = getattr(exc, "details", None)
        if details not in (None, ""):
            if isinstance(details, str):
                parts.append(details)
            else:
                try:
                    parts.append(json.dumps(details, default=str))
                except TypeError:
                    parts.append(str(details))

        parts.append(str(exc))
        return " ".join(parts).lower()

    @classmethod
    def _classify_provider_error(cls, exc: Exception) -> LLMServiceError:
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "").upper()
        text = cls._provider_error_text(exc)

        spending_terms = (
            "spending cap",
            "spending limit",
            "budget exceeded",
            "budget exhausted",
            "insufficient balance",
            "credit balance",
            "payment required",
            "billing account",
            "billing has not been enabled",
        )
        quota_terms = (
            "quota",
            "resource exhausted",
            "resource_exhausted",
            "usage limit",
            "limit exceeded",
        )
        rate_limit_terms = (
            "rate limit",
            "too many requests",
        )
        transient_terms = (
            "temporarily unavailable",
            "try again later",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection closed",
            "connection terminated",
            "remote protocol error",
            "server disconnected without sending a response",
            "stream ended unexpectedly",
            "service unavailable",
        )

        # Billing and quota failures need a hard stop. Transport hiccups should get another shot.
        if code == 402 or any(term in text for term in spending_terms):
            return QuotaExceededLLMError(f"Gemini spending cap reached. {exc}")

        if code == 429:
            if any(term in text for term in rate_limit_terms):
                return TransientLLMError(str(exc))
            if any(term in text for term in quota_terms) or status == "RESOURCE_EXHAUSTED":
                prefix = (
                    "Gemini spending cap reached."
                    if any(term in text for term in spending_terms)
                    else "Gemini quota exhausted."
                )
                return QuotaExceededLLMError(f"{prefix} {exc}")
            return TransientLLMError(str(exc))

        if code == 403 and (
            any(term in text for term in spending_terms)
            or any(term in text for term in quota_terms)
        ):
            prefix = (
                "Gemini spending cap reached."
                if any(term in text for term in spending_terms)
                else "Gemini quota exhausted."
            )
            return QuotaExceededLLMError(f"{prefix} {exc}")

        if code in {408, 409, 425, 500, 502, 503, 504}:
            return TransientLLMError(str(exc))
        if status in {"ABORTED", "CANCELLED", "DEADLINE_EXCEEDED", "INTERNAL", "UNAVAILABLE"}:
            return TransientLLMError(str(exc))
        if "503 unavailable" in text:
            return TransientLLMError(str(exc))
        if any(term in text for term in transient_terms):
            return TransientLLMError(str(exc))

        return LLMServiceError(str(exc))

    def _respect_request_spacing(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return

        now = time.monotonic()
        if self._last_request_started_at is not None:
            wait_seconds = self.min_request_interval_seconds - (
                now - self._last_request_started_at
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        self._last_request_started_at = time.monotonic()

    def _call_model(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
    ) -> str:
        if not self._env_loaded:
            # Notebook runs often start outside the repo root,
            # so look up the caller's .env once here.
            load_env_file(self.env_search_dir)
            self._env_loaded = True
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMServiceError(
                f"Missing Gemini API key in environment variable {self.api_key_env}"
            )

        try:
            from google import genai
        except Exception as exc:
            raise LLMServiceError(
                "google-genai is required for GeminiClient. Install dependency 'google-genai'."
            ) from exc
        genai_types = getattr(genai, "types", None)
        if genai_types is None:
            raise LLMServiceError("google-genai types are unavailable in the installed SDK.")

        client = genai.Client(api_key=api_key)
        config = self._build_generate_config(
            genai_types=genai_types,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
        )

        def _invoke() -> str:
            try:
                self._respect_request_spacing()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                raise self._classify_provider_error(exc) from exc
            text = getattr(response, "text", None)
            if not text:
                raise TransientLLMError("Gemini returned an empty response")
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
        *,
        enable_google_search: bool = False,
    ) -> dict[str, Any]:
        schema_version = PROMPT_SCHEMA_VERSION
        generation_settings = self._generation_settings(
            temperature=temperature,
            enable_google_search=enable_google_search,
        )

        if self.cache is not None:
            cached = self.cache.get(
                model=model,
                prompt=prompt,
                schema_version=schema_version,
                generation_settings=generation_settings,
            )
            if cached is not None:
                return self._validate_schema(cached, schema)

        raw_text = self._call_model(
            prompt,
            model=model,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
        )
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
                enable_google_search=False,
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
                generation_settings=generation_settings,
                response_json=validated,
                created_at=now_iso(),
            )

        return validated
