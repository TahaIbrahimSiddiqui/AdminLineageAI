from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import adminlineage.cli as cli_module
from adminlineage.cli import main
from adminlineage.llm import MockClient


class TrackingMockClient(MockClient):
    generate_calls = 0

    def generate_json(self, *args, **kwargs):
        type(self).generate_calls += 1
        return super().generate_json(*args, **kwargs)


def _write_example_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_dir = tmp_path / "project"
    data_dir = project_dir / "data"
    config_dir = project_dir / "config"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    df_from = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["f1"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["North Block"],
            "unit_id": ["t1"],
        }
    )
    from_path = data_dir / "from.csv"
    to_path = data_dir / "to.csv"
    df_from.to_csv(from_path, index=False)
    df_to.to_csv(to_path, index=False)

    config_path = config_dir / "config.yml"
    config_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  map_col_to: subdistrict
  exact_match: [state, district]
  string_exact_match_prune: from
  id_col_from: unit_id
  id_col_to: unit_id
  relationship: auto
  reason: false
data:
  mode: files
  from_path: ../data/from.csv
  to_path: ../data/to.csv
llm:
  provider: mock
  model: gemini-3.1-flash-lite-preview
  temperature: 0.75
  enable_google_search: true
pipeline:
  batch_size: 10
  max_candidates: 5
output:
  write_parquet: false
""".strip(),
        encoding="utf-8",
    )
    return from_path, to_path, config_path


def test_cli_run_preview_validate_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, _, config_path = _write_example_files(tmp_path)
    run_cwd = tmp_path / "run_here"
    run_cwd.mkdir()
    monkeypatch.chdir(run_cwd)

    assert main(["preview", "--config", str(config_path)]) == 0
    assert main(["validate", "--config", str(config_path)]) == 0
    assert main(["run", "--config", str(config_path)]) == 0

    run_dir = run_cwd / "outputs" / "india_1951_2001_subdistrict"
    csv_path = run_dir / "evolution_key.csv"
    assert csv_path.exists()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["request"]["temperature"] == 0.75
    assert metadata["request"]["enable_google_search"] is True
    assert metadata["request"]["string_exact_match_prune"] == "from"

    output_jsonl = run_dir / "evolution_key_export.jsonl"
    assert main(["export", "--input", str(csv_path), "--format", "jsonl"]) == 0
    assert output_jsonl.exists()


def test_cli_preview_returns_nonzero_on_invalid_config(tmp_path: Path):
    config_path = tmp_path / "bad.yml"
    config_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: missing
data:
  mode: files
  from_path: from.csv
  to_path: to.csv
llm:
  provider: mock
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame({"subdistrict": ["A"]}).to_csv(tmp_path / "from.csv", index=False)
    pd.DataFrame({"subdistrict": ["B"]}).to_csv(tmp_path / "to.csv", index=False)

    assert main(["preview", "--config", str(config_path)]) == 2


def test_cli_run_uses_replay_store_across_separate_workdirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, config_path = _write_example_files(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "string_exact_match_prune: from",
            "string_exact_match_prune: none",
        )
        + "\nreplay:\n  enabled: true\n  store_dir: ../shared-replay\n",
        encoding="utf-8",
    )

    TrackingMockClient.generate_calls = 0
    monkeypatch.setattr(cli_module, "MockClient", TrackingMockClient)

    first_cwd = tmp_path / "run_one"
    second_cwd = tmp_path / "run_two"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    assert main(["run", "--config", str(config_path)]) == 0
    first_call_count = TrackingMockClient.generate_calls
    assert first_call_count > 0

    first_run_dir = first_cwd / "outputs" / "india_1951_2001_subdistrict"
    first_metadata = json.loads((first_run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert first_metadata["execution_mode"] == "live"
    assert first_metadata["replay_hit"] is False

    monkeypatch.chdir(second_cwd)
    assert main(["run", "--config", str(config_path)]) == 0
    assert TrackingMockClient.generate_calls == first_call_count

    second_run_dir = second_cwd / "outputs" / "india_1951_2001_subdistrict"
    second_metadata = json.loads(
        (second_run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert second_metadata["execution_mode"] == "replay"
    assert second_metadata["replay_hit"] is True
    assert second_metadata["replayed_from_run_id"] == first_metadata["run_id"]
