from __future__ import annotations

import pandas as pd

from adminlineage.normalize import add_normalized_columns, canonicalize_name


def test_canonicalize_name_basic():
    value = canonicalize_name("  São-Paulo (North)  ")
    assert value == "são paulo north"


def test_add_normalized_columns():
    df = pd.DataFrame({"name": ["Alpha", "Beta  Circle"]})
    out = add_normalized_columns(df, name_col="name", prefix="from")
    assert "_from_canonical_name" in out.columns
    assert out.loc[1, "_from_canonical_name"] == "beta circle"
