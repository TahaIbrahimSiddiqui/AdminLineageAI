from __future__ import annotations

import pytest
from pydantic import ValidationError

from adminlineage.models import get_batch_response_model


def test_llm_batch_response_validates_without_reason():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "rename",
                        "relationship": "father_to_father",
                        "score": 0.9,
                        "evidence": "Name continuity within exact-match group.",
                    }
                ],
            }
        ]
    }
    parsed = get_batch_response_model(include_reason=False).model_validate(payload)
    assert parsed.decisions[0].links[0].link_type == "rename"


def test_llm_batch_response_validates_with_reason():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "rename",
                        "relationship": "father_to_child",
                        "score": 0.9,
                        "evidence": "Name continuity within exact-match group.",
                        "reason": "The lexical match is strong and no other candidate is close.",
                    }
                ],
            }
        ]
    }
    parsed = get_batch_response_model(include_reason=True).model_validate(payload)
    assert parsed.decisions[0].links[0].reason


def test_llm_batch_response_with_reason_rejects_missing_reason():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "rename",
                        "relationship": "father_to_father",
                        "score": 0.9,
                        "evidence": "bad",
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValidationError):
        get_batch_response_model(include_reason=True).model_validate(payload)


def test_llm_batch_response_rejects_unknown_relationship():
    payload = {
        "decisions": [
            {
                "from_key": "from_0",
                "links": [
                    {
                        "to_key": "to_0",
                        "link_type": "rename",
                        "relationship": "invalid",
                        "score": 0.9,
                        "evidence": "bad",
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValidationError):
        get_batch_response_model(include_reason=False).model_validate(payload)
