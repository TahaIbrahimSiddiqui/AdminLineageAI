from __future__ import annotations

from pathlib import Path

import pandas as pd

from adminlineage.config import load_config
from adminlineage.io import load_frames


def test_load_frames_resolves_paths_relative_to_config_file(tmp_path: Path):
    project_dir = tmp_path / "project"
    data_dir = project_dir / "data"
    config_dir = project_dir / "config"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    from_path = data_dir / "from.csv"
    to_path = data_dir / "to.csv"
    pd.DataFrame({"name": ["A"]}).to_csv(from_path, index=False)
    pd.DataFrame({"name": ["B"]}).to_csv(to_path, index=False)

    config_path = config_dir / "config.yml"
    config_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: name
data:
  mode: files
  from_path: ../data/from.csv
  to_path: ../data/to.csv
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    loaded = load_frames(cfg)

    assert loaded.df_from.shape[0] == 1
    assert Path(loaded.loader_metadata["from_path"]).resolve() == from_path.resolve()
    assert Path(loaded.loader_metadata["to_path"]).resolve() == to_path.resolve()


def test_load_frames_python_hook_imports_from_config_dir(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    module_path = config_dir / "loader_mod.py"
    module_path.write_text(
        """
import pandas as pd

def load_pair(config):
    df_from = pd.DataFrame({"name": ["A"]})
    df_to = pd.DataFrame({"name": ["B"]})
    return df_from, df_to
""".strip(),
        encoding="utf-8",
    )
    config_path = config_dir / "config.yml"
    config_path.write_text(
        """
request:
  country: India
  year_from: 1951
  year_to: 2001
  map_col_from: name
data:
  mode: python_hook
  callable: loader_mod:load_pair
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    loaded = load_frames(cfg)

    assert loaded.df_from.shape[0] == 1
    assert loaded.loader_metadata["mode"] == "python_hook"
