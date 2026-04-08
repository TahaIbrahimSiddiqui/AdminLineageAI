from __future__ import annotations

import json
from pathlib import Path


def test_packaged_notebook_defaults_max_candidates_to_six():
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "adminlineage_gemini_3_1_flash_lite.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook.get("cells", [])]
    notebook_text = "\n".join(sources)

    assert "max_candidates = 6" in notebook_text
    assert "ADMINLINEAGE_NOTEBOOK_BATCH_SIZE" in notebook_text
    assert "ADMINLINEAGE_NOTEBOOK_STRING_EXACT_MATCH_PRUNE" in notebook_text
