from __future__ import annotations

import types

from adminlineage.gemini_transport import GeminiTransport


def test_transport_http_options_and_sdk_http_options():
    assert GeminiTransport.http_options(None) is None
    assert GeminiTransport.http_options(90) == {"timeout": 90000}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.timeout = kwargs["timeout"]

    fake_types = types.SimpleNamespace(HttpOptions=FakeHttpOptions)
    options = GeminiTransport.sdk_http_options(fake_types, 90)

    assert isinstance(options, FakeHttpOptions)
    assert options.timeout == 90000


def test_transport_build_generate_config_with_search_and_schema_builder():
    captured: dict[str, object] = {}

    class FakeTool:
        def __init__(self, *, google_search=None):
            self.google_search = google_search

    class FakeGoogleSearch:
        pass

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_types = types.SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
    )

    config = GeminiTransport.build_generate_config(
        genai_types=fake_types,
        temperature=0.75,
        seed=42,
        enable_google_search=True,
        schema={"type": "object"},
        structured_output=True,
        response_schema_builder=lambda schema: {"wrapped": schema},
    )

    assert isinstance(config, FakeGenerateContentConfig)
    assert captured["temperature"] == 0.75
    assert captured["seed"] == 42
    assert captured["response_mime_type"] == "application/json"
    assert captured["response_json_schema"] == {"wrapped": {"type": "object"}}
    assert len(captured["tools"]) == 1
