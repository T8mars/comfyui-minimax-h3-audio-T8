from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_audio_refine_blind_review.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_audio_refine_blind_review", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audio_refine_review_is_randomized_and_does_not_reveal_methods(tmp_path):
    tool = _load_tool()
    original = tmp_path / "original.mp4"
    candidate = tmp_path / "candidate.mp4"
    original.write_bytes(b"original")
    candidate.write_bytes(b"candidate")

    result = tool.build_blind_review(
        original=original,
        candidate=candidate,
        output_dir=tmp_path / "review",
        private_dir=tmp_path / "private",
        seed=2608260501,
    )

    assert set(result["mapping"]) == {"A", "B"}
    assert set(result["mapping"].values()) == {"original", "audio_refine"}
    page = (tmp_path / "review" / "blind_review.html").read_text(encoding="utf-8")
    assert "audio_refine" not in page
    assert "original.mp4" not in page
    assert "台词准确" in page
    assert "声音自然" in page
    reveal = json.loads(
        (tmp_path / "private" / "reveal.json").read_text(encoding="utf-8")
    )
    assert reveal["mapping"] == result["mapping"]
    assert not (tmp_path / "review" / "reveal.json").exists()
