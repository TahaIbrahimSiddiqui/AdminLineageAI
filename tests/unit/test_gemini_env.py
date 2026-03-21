from __future__ import annotations

import types

from adminlineage.llm.gemini import GeminiClient


def test_gemini_client_reads_api_key_from_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_GEMINI_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)

    captured: dict[str, str] = {}

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            _ = kwargs
            return types.SimpleNamespace(text='{"decisions":[]}')

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key
            self.models = FakeModels()

    fake_google = types.SimpleNamespace(genai=types.SimpleNamespace(Client=FakeGenAIClient))
    monkeypatch.setitem(__import__("sys").modules, "google", fake_google)

    client = GeminiClient(api_key_env="TEST_GEMINI_KEY", env_search_dir=tmp_path)
    client._call_model("{}", model="gemini-2.5-pro", temperature=0.0, seed=42)

    assert captured["api_key"] == "from_dotenv"
