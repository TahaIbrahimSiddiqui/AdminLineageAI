from __future__ import annotations

from pathlib import Path

import pytest

from adminlineage.config import load_config


def test_load_config_files_mode(tmp_path: Path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
  anchor_cols: [state, district]
data:
  mode: files
  from_path: examples/data/from_units.csv
  to_path: examples/data/to_units.csv
llm:
  provider: mock
pipeline:
  batch_size: 5
  max_candidates: 10
output:
  directory: outputs
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.request.country == "India"
    assert cfg.data.mode == "files"


def test_load_config_invalid_files_mode(tmp_path: Path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: subdistrict
data:
  mode: files
  from_path: only_one.csv
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_config(cfg_path)
