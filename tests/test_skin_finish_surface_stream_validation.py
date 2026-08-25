from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_skin_finish_surface_stream as validation  # noqa: E402


def test_surface_stream_validation_is_pinned_single_run_and_low_load():
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert validation.EXPECTED_SOURCE_SHA256 == (
        "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
    )
    assert validation.SURFACE_PARAMETERS == {
        "amount": 0.90,
        "surface_smoothing": 0.25,
        "texture_keep": 0.96,
        "highlight_compression": 0.90,
        "broad_highlight_compression": 0.90,
        "broad_highlight_start": 0.68,
        "broad_highlight_end": 0.94,
        "blemish_balance": 0.10,
        "surface_radius_percent": 2.5,
    }
    assert '"loads_h3": False' in source
    assert '"loads_sam": False' in source
    assert '"stress_or_repeat": False' in source
    assert "chunk_frames=2" in source
    assert "finish_skin_surface" in source
    assert "build_review" in source
    assert "mechanical_diagnostics.json" in source
    assert "stage_candidate_exists" in source
    assert "refusing to overwrite validation evidence" in source
