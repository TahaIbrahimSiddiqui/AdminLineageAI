from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adminlineage.cli import main


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
  model: gemini-2.5-pro
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
