from __future__ import annotations

from adminlineage.llm_cache import SQLiteCache


def test_cache_separates_temperature_and_grounding_settings(tmp_path):
    cache = SQLiteCache(tmp_path / "llm_cache.sqlite")
    response_json = {"decisions": []}

    baseline_settings = {
        "temperature": 0.0,
        "enable_google_search": False,
    }
    cache.set(
        model="gemini-3.1-flash-lite-preview",
        prompt="{}",
        schema_version="1.0.0",
        generation_settings=baseline_settings,
        response_json=response_json,
        created_at="2026-04-06T00:00:00Z",
    )

    assert (
        cache.get(
            model="gemini-3.1-flash-lite-preview",
            prompt="{}",
            schema_version="1.0.0",
            generation_settings=baseline_settings,
        )
        == response_json
    )
    assert (
        cache.get(
            model="gemini-3.1-flash-lite-preview",
            prompt="{}",
            schema_version="1.0.0",
            generation_settings={
                "temperature": 0.75,
                "enable_google_search": False,
            },
        )
        is None
    )
    assert (
        cache.get(
            model="gemini-3.1-flash-lite-preview",
            prompt="{}",
            schema_version="1.0.0",
            generation_settings={
                "temperature": 0.0,
                "enable_google_search": True,
            },
        )
        is None
    )
