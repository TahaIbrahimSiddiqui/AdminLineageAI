from __future__ import annotations

import pandas as pd

from adminlineage.candidates import (
    combined_similarity,
    generate_shortlist,
    generate_shortlist_from_records,
    ngram_cosine,
    prepare_target_records,
    token_jaccard,
)
from adminlineage.normalize import add_normalized_columns


def test_generate_shortlist_prefers_best_lexical_match():
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

    shortlist = generate_shortlist(
        from_norm.iloc[0],
        to_norm,
        max_candidates=5,
    )

    assert shortlist[0]["to_key"] == "to_0"
    assert shortlist[0]["score"] >= shortlist[1]["score"]
    assert shortlist[0]["token_jaccard"] >= shortlist[1]["token_jaccard"]


def test_generate_shortlist_from_records_matches_manual_reference():
    df_from = pd.DataFrame(
        {
            "subdistrict": ["East Circle"],
        }
    )
    df_to = pd.DataFrame(
        {
            "subdistrict": ["East Circle New", "East Side", "West Block"],
        }
    )

    df_from["_from_key"] = ["from_0"]
    df_to["_to_key"] = ["to_0", "to_1", "to_2"]

    from_norm = add_normalized_columns(df_from, "subdistrict", "from")
    to_norm = add_normalized_columns(df_to, "subdistrict", "to")
    from_row = from_norm.iloc[0]

    expected = []
    for _, to_row in to_norm.iterrows():
        token_score = token_jaccard(from_row["_from_tokens"], to_row["_to_tokens"])
        ngram_score = ngram_cosine(from_row["_from_char_ngrams"], to_row["_to_char_ngrams"])
        expected.append(
            {
                "to_key": to_row["_to_key"],
                "to_name": to_row["_to_name_raw"],
                "to_canonical_name": to_row["_to_canonical_name"],
                "score": float(round(combined_similarity(token_score, ngram_score), 6)),
                "token_jaccard": float(round(token_score, 6)),
                "ngram_cosine": float(round(ngram_score, 6)),
            }
        )

    expected.sort(key=lambda item: (-item["score"], item["to_canonical_name"], item["to_key"]))
    expected = expected[:2]

    actual = generate_shortlist_from_records(
        from_row["_from_tokens"],
        from_row["_from_char_ngrams"],
        prepare_target_records(to_norm),
        max_candidates=2,
    )

    assert actual == expected
    assert generate_shortlist(from_row, to_norm, max_candidates=2) == expected
