from __future__ import annotations

import pandas as pd

from adminlineage.review import apply_global_flags, build_review_queue


def test_review_queue_thresholds():
    df = pd.DataFrame(
        [
            {
                "from_name": "A",
                "to_name": "B",
                "from_key": "from_0",
                "to_key": "to_0",
                "score": 0.8,
                "link_type": "rename",
            },
            {
                "from_name": "C",
                "to_name": None,
                "from_key": "from_1",
                "to_key": None,
                "score": 0.2,
                "link_type": "no_match",
            },
        ]
    )

    flagged = apply_global_flags(df, low_score_threshold=0.6)
    review = build_review_queue(flagged)

    assert len(review) == 1
    assert review.iloc[0]["from_key"] == "from_1"
