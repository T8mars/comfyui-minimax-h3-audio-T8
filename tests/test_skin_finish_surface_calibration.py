from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import calibrate_skin_finish_surface as calibration  # noqa: E402


def test_surface_calibration_is_pinned_low_load_and_not_automatic():
    source = Path(calibration.__file__).read_text(encoding="utf-8")
    assert calibration.EXPECTED_SOURCE_SHA256 == (
        "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
    )
    assert list(calibration.SURFACE_ARMS) == ["localized_highlight_oil_v3"]
    assert '"loads_h3": False' in source
    assert '"loads_sam": False' in source
    assert '"runs_full_video": False' in source
    assert '"stress_or_repeat": False' in source
    assert '"automatic_selection": False' in source
    assert "_boundary_diagnostics" in source
    assert '"current_quality_stream"' in source
    assert "finish_skin_surface" in source
