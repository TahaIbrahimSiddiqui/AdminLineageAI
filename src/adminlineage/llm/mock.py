"""Mock LLM client for deterministic tests."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .base import BaseLLMClient, LLMServiceError


class MockClient(BaseLLMClient):
    """Deterministic mock that maps each from-unit to top candidate."""

    def __init__(self, default_score: float = 0.9):
        self.default_score = default_score
        self.calls = 0

    def generate_json(
        self,
        prompt: str,
        schema: Any,
        model: str,
        temperature: float,
        seed: int,
    ) -> dict[str, Any]:
        _ = (model, temperature, seed)
        self.calls += 1

        marker = "INPUT_PAYLOAD_JSON:\n"
        if marker not in prompt:
            raise LLMServiceError("MockClient prompt is missing INPUT_PAYLOAD_JSON marker")
        payload_raw = prompt.split(marker, maxsplit=1)[1].strip()
        payload = json.loads(payload_raw)

        decisions: list[dict[str, Any]] = []
        requested_relationship = payload.get("requested_relationship", "auto")
        include_reason = bool(payload.get("include_reason", False))
        for item in payload.get("items", []):
            candidates = item.get("candidates", [])
            if not candidates:
                links = [
                    {
                        "to_key": None,
                        "link_type": "no_match",
                        "relationship": "unknown",
                        "score": 0.0,
                        "evidence": "No candidates available in constrained group.",
                    }
                ]
            else:
                first = candidates[0]
                link_type = "rename"
                if len(candidates) > 1 and first["score"] < 0.5:
                    link_type = "unknown"
                links = [
                    {
                        "to_key": first["to_key"],
                        "link_type": link_type,
                        "relationship": (
                            requested_relationship
                            if requested_relationship != "auto"
                            else "father_to_father"
                        ),
                        "score": max(
                            first["score"],
                            self.default_score if link_type == "rename" else 0.4,
                        ),
                        "evidence": (
                            "Best lexical and contextual candidate within exact-match constraints."
                        ),
                    }
                ]
            if include_reason:
                for link in links:
                    link["reason"] = (
                        (
                            "The lexical score and local context were stronger than the other "
                            "available options."
                        )
                        if link["to_key"] is not None
                        else "No candidate was strong enough to justify a link."
                    )
            decisions.append({"from_key": item["from_key"], "links": links})

        response = {"decisions": decisions}
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(response).model_dump()
        return response
