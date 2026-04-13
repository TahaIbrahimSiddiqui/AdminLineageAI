"""Gemini client implementation with cache, retries, and JSON repair."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .gemini_transport import GeminiTransport
from .llm_cache import SQLiteCache
from .llm_retry import retry_call
from .llm_types import (
    BaseLLMClient,
    LLMServiceError,
    QuotaExceededLLMError,
    TransientLLMError,
)
from .prompts import build_repair_prompt
from .schema import LINK_TYPES, PROMPT_SCHEMA_VERSION, RELATIONSHIP_TYPES
from .utils import load_env_file, now_iso

_VALID_LINK_TYPES = set(LINK_TYPES)
_VALID_RELATIONSHIP_TYPES = set(RELATIONSHIP_TYPES)
_LINK_TYPE_ALIASES = {
    "exact_match": "rename",
}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_MAX_GROUNDED_JSON_ATTEMPTS = 2
_MAX_GROUNDED_TEXT_ATTEMPTS = 2


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
        request_timeout_seconds: int | None = 90,
        env_search_dir: str | Path | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.cache = cache
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.request_timeout_seconds = (
            None
            if request_timeout_seconds is None or int(request_timeout_seconds) <= 0
            else int(request_timeout_seconds)
        )
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
        schema: Any | None = None,
        structured_output: bool = True,
    ) -> Any:
        return GeminiTransport.build_generate_config(
            genai_types=genai_types,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
            schema=schema,
            structured_output=structured_output,
            response_schema_builder=GeminiClient._response_json_schema,
        )

    @classmethod
    def _response_json_schema(cls, schema: Any) -> Any:
        if schema is None:
            return None
        if isinstance(schema, dict):
            return cls._sanitize_json_schema(schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return cls._sanitize_json_schema(schema.model_json_schema())
        return schema

    @classmethod
    def _sanitize_json_schema(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._sanitize_json_schema(item) for item in value]
        if not isinstance(value, dict):
            return value

        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"default", "examples", "description", "title"}:
                continue
            sanitized[key] = cls._sanitize_json_schema(item)

        any_of = sanitized.get("anyOf")
        if isinstance(any_of, list) and any_of:
            simple_types: list[str] = []
            remainder: list[Any] = []
            for option in any_of:
                if (
                    isinstance(option, dict)
                    and isinstance(option.get("type"), str)
                    and set(option.keys()) <= {"type"}
                ):
                    simple_types.append(option["type"])
                else:
                    remainder.append(option)
            if simple_types and not remainder:
                sanitized.pop("anyOf", None)
                sanitized["type"] = simple_types if len(simple_types) > 1 else simple_types[0]

        return sanitized

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
    def _is_unsupported_response_schema_error(cls, exc: Exception) -> bool:
        text = cls._provider_error_text(exc)
        schema_terms = (
            "generation_config.response_schema",
            "response_schema",
            "response_json_schema",
            "json schema",
        )
        failure_terms = (
            "additional_properties",
            "unknown name",
            "unsupported",
            "invalid",
            "not supported",
            "cannot be used",
        )
        return any(term in text for term in schema_terms) and any(
            term in text for term in failure_terms
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        return GeminiTransport.extract_response_text(response)

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
            "getaddrinfo failed",
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided",
            "failed to establish a new connection",
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

    @staticmethod
    def _http_options(timeout_seconds: int | None) -> dict[str, Any] | None:
        return GeminiTransport.http_options(timeout_seconds)

    @classmethod
    def _sdk_http_options(cls, genai_types: Any, timeout_seconds: int | None) -> Any | None:
        return GeminiTransport.sdk_http_options(genai_types, timeout_seconds)

    @classmethod
    def _parse_json_payload(cls, raw_text: str) -> Any:
        stripped = raw_text.strip()
        candidates: list[str] = []

        if stripped:
            candidates.append(stripped)

        fence_match = _JSON_FENCE_RE.search(stripped)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidates.append(stripped[start : end + 1].strip())

        decoder = json.JSONDecoder()
        last_error: Exception | None = None
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
            try:
                parsed, end = decoder.raw_decode(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if candidate[end:].strip() == "":
                return parsed

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("No JSON object found in Gemini response", raw_text, 0)

    def _read_api_key(self) -> str:
        if not self._env_loaded:
            load_env_file(self.env_search_dir)
            self._env_loaded = True
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMServiceError(
                f"Missing Gemini API key in environment variable {self.api_key_env}"
            )
        return api_key

    def _call_generation(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema: Any | None,
        structured_output: bool,
        grounded_attempt_cap: int,
    ) -> str:
        api_key = self._read_api_key()
        transport = GeminiTransport(
            api_key=api_key,
            request_timeout_seconds=self.request_timeout_seconds,
        )

        def _invoke() -> str:
            try:
                self._respect_request_spacing()
                text = transport.generate_text(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    enable_google_search=enable_google_search,
                    schema=schema,
                    structured_output=structured_output,
                    response_schema_builder=self._response_json_schema,
                )
            except LLMServiceError:
                raise
            except Exception as exc:
                raise self._classify_provider_error(exc) from exc
            if not text:
                raise TransientLLMError("Gemini returned an empty response")
            return text

        max_attempts = (
            min(self.max_attempts, grounded_attempt_cap)
            if enable_google_search
            else self.max_attempts
        )

        return retry_call(
            _invoke,
            max_attempts=max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            jitter_seconds=self.jitter_seconds,
            retry_exceptions=(TransientLLMError,),
        )

    def _call_model(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema: Any | None = None,
    ) -> str:
        return self._call_generation(
            prompt,
            model=model,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
            schema=schema,
            structured_output=True,
            grounded_attempt_cap=_MAX_GROUNDED_JSON_ATTEMPTS,
        )

    def _call_text_model(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
    ) -> str:
        return self._call_generation(
            prompt,
            model=model,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
            schema=None,
            structured_output=False,
            grounded_attempt_cap=_MAX_GROUNDED_TEXT_ATTEMPTS,
        )

    @staticmethod
    def _normalize_enum_token(value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _normalize_link_type(cls, value: str) -> str | None:
        normalized = cls._normalize_enum_token(value)
        if normalized in _VALID_LINK_TYPES:
            return normalized
        return _LINK_TYPE_ALIASES.get(normalized)

    @classmethod
    def _normalize_relationship(cls, value: str) -> str | None:
        normalized = cls._normalize_enum_token(value)
        if normalized in _VALID_RELATIONSHIP_TYPES:
            return normalized
        return None

    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._normalize_payload(item) for item in value]
        if not isinstance(value, dict):
            return value

        normalized: dict[str, Any] = {}
        for key, item in value.items():
            item = cls._normalize_payload(item)
            if key == "link_type" and isinstance(item, str):
                item = cls._normalize_link_type(item) or item
            elif key == "relationship" and isinstance(item, str):
                item = cls._normalize_relationship(item) or item
            elif key == "to_key" and isinstance(item, str):
                token = cls._normalize_enum_token(item)
                if token in {"null", "none"}:
                    item = None
            normalized[key] = item
        return normalized

    @classmethod
    def _validate_schema(cls, data: dict[str, Any], schema: Any) -> dict[str, Any]:
        data = cls._normalize_payload(data)
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

        schema_for_call = schema
        try:
            raw_text = self._call_model(
                prompt,
                model=model,
                temperature=temperature,
                seed=seed,
                enable_google_search=enable_google_search,
                schema=schema,
            )
        except LLMServiceError as exc:
            # Grounded structured JSON is the most brittle path in Gemini. If that fails,
            # fall back to prompt-only JSON, and for grounded calls fall back one more time
            # to plain text that we still parse and validate as JSON.
            should_try_prompt_only_json = (
                schema is not None
                and (
                    enable_google_search
                    or self._is_unsupported_response_schema_error(exc)
                )
            )
            if should_try_prompt_only_json:
                try:
                    raw_text = self._call_model(
                        prompt,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                        enable_google_search=enable_google_search,
                        schema=None,
                    )
                    schema_for_call = None
                except LLMServiceError:
                    if not enable_google_search:
                        raise
                    raw_text = self._call_text_model(
                        prompt,
                        model=model,
                        temperature=temperature,
                        seed=seed,
                        enable_google_search=True,
                    )
                    schema_for_call = None
            else:
                raise
        try:
            parsed = self._parse_json_payload(raw_text)
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
                schema=schema_for_call,
            )
            try:
                parsed = self._parse_json_payload(repair_text)
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

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float,
        seed: int,
        *,
        enable_google_search: bool = False,
    ) -> str:
        return self._call_text_model(
            prompt,
            model=model,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
        )
