from __future__ import annotations

import pandas as pd

from adminlineage.candidates import build_alias_lookup, generate_shortlist
from adminlineage.normalize import add_normalized_columns


def test_alias_boost_in_shortlist(sample_aliases):
    df_from = pd.DataFrame(
        {
            "state": ["S1"],
            "district": ["D1"],
            "subdistrict": ["East Circle"],
        }
    )
    df_to = pd.DataFrame(
        {
            "state": ["S1", "S1"],
            "district": ["D1", "D1"],
            "subdistrict": ["East Circle New", "Other"],
        }
    )

    df_from = df_from.reset_index(drop=True)
    df_to = df_to.reset_index(drop=True)
    df_from["_from_key"] = ["from_0"]
    df_to["_to_key"] = ["to_0", "to_1"]

    from_norm = add_normalized_columns(df_from, "subdistrict", "from")
    to_norm = add_normalized_columns(df_to, "subdistrict", "to")

    lookup = build_alias_lookup(sample_aliases, ["state", "district"])
    shortlist = generate_shortlist(
        from_norm.iloc[0],
        to_norm,
        max_candidates=5,
        alias_lookup=lookup,
        anchor_cols=["state", "district"],
    )

    assert shortlist[0]["to_key"] == "to_0"
    assert shortlist[0]["alias_hit"] is True
    assert shortlist[0]["score"] >= 0.95
