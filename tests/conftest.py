from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_df_from() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["S1", "S1", "S1"],
            "district": ["D1", "D1", "D2"],
            "subdistrict": ["North Block", "East Circle", "River Tehsil"],
            "unit_id": ["f1", "f2", "f3"],
        }
    )


@pytest.fixture
def sample_df_to() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["S1", "S1", "S1", "S1"],
            "district": ["D1", "D1", "D2", "D3"],
            "subdistrict": ["North Block", "East Circle New", "River Tehsil", "Unrelated"],
            "unit_id": ["t1", "t2", "t3", "t4"],
        }
    )


@pytest.fixture
def sample_aliases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "from_alias": ["East Circle"],
            "to_alias": ["East Circle New"],
            "state": ["S1"],
            "district": ["D1"],
        }
    )
