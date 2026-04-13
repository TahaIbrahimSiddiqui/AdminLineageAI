from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

import adminlineage.gemini_client as gemini_module
from adminlineage.gemini_client import GeminiClient
from adminlineage.llm_types import QuotaExceededLLMError
from adminlineage.models import LLMBatchResponseNoReason


class DummyBatchResponse(BaseModel):
    decisions: list[dict[str, object]]


def test_gemini_client_reads_api_key_from_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    captured: dict[str, str] = {}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["config"] = kwargs["config"]
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(api_key_env="TEST_GEMINI_KEY", env_search_dir=tmp_path)
    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert captured["api_key"] == "from_dotenv"
    assert captured["config"].temperature == 0.0
    assert captured["config"].tools == []


def test_gemini_client_builds_search_tool_config(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    captured: dict[str, object] = {}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.response_schema = kwargs.get("response_schema")
            self.response_json_schema = kwargs.get("response_json_schema")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["config"] = kwargs["config"]
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(api_key_env="TEST_GEMINI_KEY", env_search_dir=tmp_path)

    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.75,
        seed=99,
        enable_google_search=True,
        schema=DummyBatchResponse,
    )
    config = captured["config"]
    assert config.temperature == 0.75
    assert config.seed == 99
    assert config.response_mime_type == "application/json"
    assert config.response_schema is None
    assert isinstance(config.response_json_schema, dict)
    assert len(config.tools) == 1
    assert config.tools[0].google_search is not None

    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.75,
        seed=99,
        enable_google_search=True,
        schema=DummyBatchResponse,
    )
    config = captured["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is None
    assert isinstance(config.response_json_schema, dict)
    assert len(config.tools) == 1
    assert config.tools[0].google_search is not None

    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.75,
        seed=99,
        enable_google_search=False,
        schema=DummyBatchResponse,
    )
    config = captured["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is None
    assert isinstance(config.response_json_schema, dict)
    assert config.tools == []


def test_gemini_client_normalizes_link_type_aliases():
    data = {
        "decisions": [
            {
                "from_key": "from_1",
                "links": [
                    {
                        "to_key": "to_1",
                        "link_type": "exact-match",
                        "relationship": "father-to-father",
                        "score": 0.9,
                        "evidence": "Strong lexical match.",
                    }
                ],
            }
        ]
    }

    validated = GeminiClient._validate_schema(data, LLMBatchResponseNoReason)

    assert validated["decisions"][0]["links"][0]["link_type"] == "rename"
    assert validated["decisions"][0]["links"][0]["relationship"] == "father_to_father"


def test_gemini_client_retries_empty_response(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            if calls["count"] == 1:
                return types.SimpleNamespace(text="")
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=2,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
    )

    result = client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert result == '{"decisions":[]}'
    assert calls["count"] == 2


def test_gemini_client_falls_back_when_response_schema_is_rejected(monkeypatch):
    client = GeminiClient()
    calls: list[object] = []

    def fake_call_model(
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema=None,
    ) -> str:
        _ = (prompt, model, temperature, seed, enable_google_search)
        calls.append(schema)
        if schema is not None:
            raise gemini_module.LLMServiceError(
                "400 INVALID_ARGUMENT generation_config.response_schema additional_properties"
            )
        return '{"decisions":[]}'

    monkeypatch.setattr(client, "_call_model", fake_call_model)

    result = client.generate_json(
        prompt="Return JSON only",
        schema=DummyBatchResponse,
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
    )

    assert result == {"decisions": []}
    assert calls == [DummyBatchResponse, None]


def test_gemini_client_parses_fenced_json_without_repair(monkeypatch):
    client = GeminiClient()
    calls: list[str] = []

    def fake_call_model(
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema=None,
    ) -> str:
        _ = (model, temperature, seed, enable_google_search, schema)
        calls.append(prompt)
        return '```json\n{"decisions":[]}\n```'

    monkeypatch.setattr(client, "_call_model", fake_call_model)

    result = client.generate_json(
        prompt="Return JSON only",
        schema=DummyBatchResponse,
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=True,
    )

    assert result == {"decisions": []}
    assert calls == ["Return JSON only"]


def test_gemini_client_converts_timeout_seconds_to_sdk_milliseconds():
    assert GeminiClient._http_options(None) is None
    assert GeminiClient._http_options(90) == {"timeout": 90000}


def test_gemini_client_prefers_sdk_http_options_when_available():
    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

    fake_types = types.SimpleNamespace(HttpOptions=FakeHttpOptions)

    options = GeminiClient._sdk_http_options(fake_types, 90)

    assert isinstance(options, FakeHttpOptions)
    assert options.timeout == 90000


def test_gemini_client_retries_transient_provider_errors(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            if calls["count"] < 4:
                raise RuntimeError("503 UNAVAILABLE")
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=4,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
    )

    result = client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert result == '{"decisions":[]}'
    assert calls["count"] == 4


def test_gemini_client_retries_dns_lookup_errors(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("[Errno 11001] getaddrinfo failed")
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=3,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=0.0,
    )

    result = client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert result == '{"decisions":[]}'
    assert calls["count"] == 3


def test_gemini_client_retries_server_disconnect_errors(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("Server disconnected without sending a response.")
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=2,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=0.0,
    )

    result = client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert result == '{"decisions":[]}'
    assert calls["count"] == 2


def test_gemini_client_retries_rate_limit_errors(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeRateLimitError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("429 RESOURCE_EXHAUSTED. Rate limit exceeded.")
            self.code = 429
            self.status = "RESOURCE_EXHAUSTED"
            self.message = "Rate limit exceeded."
            self.details = {
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            }

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            if calls["count"] < 3:
                raise FakeRateLimitError()
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=3,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=0.0,
    )

    result = client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert result == '{"decisions":[]}'
    assert calls["count"] == 3


def test_gemini_client_raises_spending_cap_error_without_retry(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeBillingError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("429 RESOURCE_EXHAUSTED. Billing budget exhausted.")
            self.code = 429
            self.status = "RESOURCE_EXHAUSTED"
            self.message = "Billing budget exhausted."
            self.details = {
                "error": {
                    "code": 429,
                    "message": "Billing budget exhausted.",
                    "status": "RESOURCE_EXHAUSTED",
                }
            }

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            raise FakeBillingError()

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=4,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=0.0,
    )

    with pytest.raises(QuotaExceededLLMError, match="spending cap reached"):
        client._call_model(
            "{}",
            model="gemini-3.1-flash-lite-preview",
            temperature=0.0,
            seed=42,
            enable_google_search=False,
        )

    assert calls["count"] == 1


def test_gemini_client_spaces_requests(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    slept: list[float] = []
    monotonic_values = iter([0.0, 0.0, 0.4, 1.5])

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(gemini_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(gemini_module.time, "sleep", slept.append)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=1.5,
    )

    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )
    client._call_model(
        "{}",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        seed=42,
        enable_google_search=False,
    )

    assert slept == [pytest.approx(1.1)]


def test_gemini_client_raises_after_repeated_empty_response(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.response_mime_type = kwargs.get("response_mime_type")
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            return types.SimpleNamespace(text="")

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=2,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="empty response"):
        client._call_model(
            "{}",
            model="gemini-3.1-flash-lite-preview",
            temperature=0.0,
            seed=42,
            enable_google_search=False,
        )


def test_gemini_client_repair_call_uses_zero_temperature(monkeypatch):
    client = GeminiClient()
    calls: list[dict[str, object]] = []

    def fake_call_model(
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
        schema=None,
    ) -> str:
        calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "enable_google_search": enable_google_search,
                "schema": schema,
            }
        )
        if len(calls) == 1:
            return "not valid json"
        return '{"decisions":[]}'

    monkeypatch.setattr(client, "_call_model", fake_call_model)

    result = client.generate_json(
        prompt="Return JSON only",
        schema=DummyBatchResponse,
        model="gemini-3.1-flash-lite-preview",
        temperature=0.75,
        seed=42,
        enable_google_search=True,
    )

    assert result == {"decisions": []}
    assert calls[0]["temperature"] == 0.75
    assert calls[0]["enable_google_search"] is True
    assert calls[1]["temperature"] == 0.0
    assert calls[1]["enable_google_search"] is False


def test_gemini_client_generate_text_preserves_semantic_temperature(monkeypatch):
    client = GeminiClient()
    captured: dict[str, object] = {}

    def fake_call_text_model(
        prompt: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        enable_google_search: bool,
    ) -> str:
        captured["prompt"] = prompt
        captured["model"] = model
        captured["temperature"] = temperature
        captured["seed"] = seed
        captured["enable_google_search"] = enable_google_search
        return "grounded notes"

    monkeypatch.setattr(client, "_call_text_model", fake_call_text_model)

    result = client.generate_text(
        prompt="Verify shortlist candidates",
        model="gemini-3.1-flash-lite-preview",
        temperature=0.75,
        seed=7,
        enable_google_search=True,
    )

    assert result == "grounded notes"
    assert captured["temperature"] == 0.75
    assert captured["enable_google_search"] is True


def test_gemini_client_limits_grounded_text_retry_attempts(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    calls = {"count": 0}

    class FakeGoogleSearch:
        pass

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            self.seed = kwargs["seed"]
            self.tools = kwargs.get("tools", [])

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            calls["count"] += 1
            raise RuntimeError("503 UNAVAILABLE")

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.models = FakeModels()

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    fake_google = types.SimpleNamespace(
        genai=types.SimpleNamespace(Client=FakeGenAIClient, types=fake_types)
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)

    client = GeminiClient(
        api_key_env="TEST_GEMINI_KEY",
        env_search_dir=tmp_path,
        max_attempts=6,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        jitter_seconds=0.0,
        min_request_interval_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        client.generate_text(
            prompt="Verify shortlist candidates",
            model="gemini-3.1-flash-lite-preview",
            temperature=0.75,
            seed=42,
            enable_google_search=True,
        )

    assert calls["count"] == 2
