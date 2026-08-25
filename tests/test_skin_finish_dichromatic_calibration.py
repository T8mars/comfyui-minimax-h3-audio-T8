from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import calibrate_skin_finish_dichromatic as calibration  # noqa: E402


def test_dichromatic_calibration_is_one_arm_and_low_load_by_design():
    assert list(calibration.DICHROMATIC_ARM) == [
        "amount",
        "specular_strength",
        "diffuse_radius_percent",
        "specular_threshold_linear",
        "specular_softness_linear",
        "chroma_dilution_threshold",
        "chroma_dilution_softness",
        "minimum_diffuse_chroma",
        "diffuse_chroma_softness",
        "minimum_direction_cosine",
        "maximum_surface_delta",
        "minimum_texture_ratio",
        "maximum_texture_ratio",
    ]
    assert calibration.DICHROMATIC_ARM["amount"] == 0.90
    assert calibration.DICHROMATIC_ARM["specular_strength"] == 0.85
    assert calibration.DEFAULT_OUTPUT.name.endswith("-v4")


def test_calibration_reuses_the_pinned_oily_eight_step_source():
    assert calibration.DEFAULT_SOURCE == calibration.calibration.DEFAULT_SOURCE
    assert calibration.EXPECTED_SOURCE_SHA256 == (
        calibration.calibration.EXPECTED_SOURCE_SHA256
    )
