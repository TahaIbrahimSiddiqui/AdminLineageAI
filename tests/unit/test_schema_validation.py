from __future__ import annotations

import pytest

from adminlineage.models import LLMBatchResponse


def test_llm_batch_response_validates():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "rename",
                        "score": 0.9,
                        "evidence": "Name continuity within anchor group.",
                    }
                ],
            }
        ]
    }
    parsed = LLMBatchResponse.model_validate(payload)
    assert parsed.decisions[0].links[0].link_type == "rename"


def test_llm_batch_response_rejects_unknown_type():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "invalid",
                        "score": 0.9,
                        "evidence": "bad",
                    }
                ],
            }
        ]
    }
    with pytest.raises(Exception):
        LLMBatchResponse.model_validate(payload)
