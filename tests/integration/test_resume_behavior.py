from __future__ import annotations

from pathlib import Path

from adminlineage.llm import MockClient
from adminlineage.pipeline import run_pipeline


def test_resume_skips_completed_from_units(sample_df_from, sample_df_to, tmp_path: Path):
    client = MockClient()

    run_kwargs = dict(
        country="India",
        year_from=1951,
        year_to=2001,
        map_col_from="subdistrict",
        map_col_to="subdistrict",
        anchor_cols=["state", "district"],
        id_col_from="unit_id",
        id_col_to="unit_id",
        model="gemini-2.5-pro",
        batch_size=2,
        max_candidates=3,
        resume_dir=tmp_path,
        run_name="resume_run",
        llm_client=client,
        output_write_parquet=False,
    )

    run_pipeline(sample_df_from, sample_df_to, **run_kwargs)
    first_calls = client.calls
    assert first_calls > 0

    run_pipeline(sample_df_from, sample_df_to, **run_kwargs)
    second_calls = client.calls
    assert second_calls == first_calls

    run_dir = tmp_path / "resume_run"
    assert (run_dir / "links_raw.jsonl").exists()
