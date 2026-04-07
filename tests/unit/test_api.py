from __future__ import annotations

import pandas as pd

from adminlineage import api


def test_build_evolution_key_forwards_runtime_options(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_pipeline(df_from, df_to, **kwargs):
        captured["df_from"] = df_from
        captured["df_to"] = df_to
        captured["kwargs"] = kwargs
        return pd.DataFrame(), {"status": "ok"}

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    df_from = pd.DataFrame({"district_name": ["A"]})
    df_to = pd.DataFrame({"District": ["A"]})

    api.build_evolution_key(
        df_from,
        df_to,
        country="India",
        year_from=2011,
        year_to=2025,
        map_col_from="district_name",
        map_col_to="District",
        string_exact_match_prune="from",
        evidence=True,
        output_dir="custom_outputs",
        temperature=0.75,
        enable_google_search=True,
        env_search_dir="custom_env_dir",
        replay_enabled=True,
        replay_store_dir="custom_replay_store",
    )

    kwargs = captured["kwargs"]
    assert captured["df_from"] is df_from
    assert captured["df_to"] is df_to
    assert kwargs["string_exact_match_prune"] == "from"
    assert kwargs["evidence"] is True
    assert kwargs["output_dir"] == "custom_outputs"
    assert kwargs["temperature"] == 0.75
    assert kwargs["enable_google_search"] is True
    assert kwargs["env_search_dir"] == "custom_env_dir"
    assert kwargs["replay_enabled"] is True
    assert kwargs["replay_store_dir"] == "custom_replay_store"
