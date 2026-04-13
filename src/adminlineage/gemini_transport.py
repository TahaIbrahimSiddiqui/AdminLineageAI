"""Direct google-genai transport helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def load_google_genai_sdk() -> tuple[Any, Any]:
    """Load the google-genai SDK and return `(genai, genai.types)`."""

    from google import genai  # type: ignore

    genai_types = getattr(genai, "types", None)
    if genai_types is None:
        raise RuntimeError("google-genai types are unavailable in the installed SDK.")
    return genai, genai_types


@dataclass(frozen=True)
class GeminiTransport:
    """Thin transport around `google-genai` generate_content calls."""

    api_key: str
    request_timeout_seconds: int | None = 90

    @staticmethod
    def http_options(timeout_seconds: int | None) -> dict[str, Any] | None:
        if timeout_seconds is None:
            return None
        return {"timeout": int(timeout_seconds * 1000)}

    @classmethod
    def sdk_http_options(cls, genai_types: Any, timeout_seconds: int | None) -> Any | None:
        http_options = cls.http_options(timeout_seconds)
        if http_options is None:
            return None
        http_options_cls = getattr(genai_types, "HttpOptions", None)
        if http_options_cls is None:
            return http_options
        return http_options_cls(**http_options)

    @staticmethod
    def build_generate_config(
        *,
        genai_types: Any,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema: Any | None = None,
        structured_output: bool = True,
        response_schema_builder: Callable[[Any], Any] | None = None,
    ) -> Any:
        tools = []
        if enable_google_search:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))

        config_kwargs: dict[str, Any] = {
            "temperature": float(temperature),
            "seed": seed,
        }
        if tools:
            config_kwargs["tools"] = tools
        if structured_output:
            config_kwargs["response_mime_type"] = "application/json"
            if schema is not None:
                resolved_schema = (
                    response_schema_builder(schema) if response_schema_builder else schema
                )
                config_kwargs["response_json_schema"] = resolved_schema

        return genai_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def extract_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts = [str(part.text).strip() for part in parts if getattr(part, "text", None)]
            if text_parts:
                return "\n".join(text_parts).strip()
        return ""

    def _build_client(self, genai: Any, genai_types: Any) -> Any:
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        sdk_http_options = self.sdk_http_options(genai_types, self.request_timeout_seconds)
        if sdk_http_options is not None:
            try:
                return genai.Client(http_options=sdk_http_options, **client_kwargs)
            except TypeError as exc:
                if "http_options" not in str(exc):
                    raise
        return genai.Client(**client_kwargs)

    def generate_text(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema: Any | None = None,
        structured_output: bool = True,
        response_schema_builder: Callable[[Any], Any] | None = None,
    ) -> str:
        genai, genai_types = load_google_genai_sdk()
        client = self._build_client(genai, genai_types)
        config = self.build_generate_config(
            genai_types=genai_types,
            temperature=temperature,
            seed=seed,
            enable_google_search=enable_google_search,
            schema=schema,
            structured_output=structured_output,
            response_schema_builder=response_schema_builder,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return self.extract_response_text(response)
