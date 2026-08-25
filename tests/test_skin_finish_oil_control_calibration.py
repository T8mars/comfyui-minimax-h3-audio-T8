from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import calibrate_skin_finish_oil_control as calibration  # noqa: E402


def test_calibration_arms_are_public_range_and_increase_highlight_reduction():
    calibration._validate_arms()
    fractions = [
        calibration._effective_highlight_residual_fraction(parameters)
        for parameters in calibration.CALIBRATION_ARMS.values()
    ]
    assert fractions == pytest.approx([0.1071875, 0.28875, 0.459375])
    assert fractions == sorted(fractions)
    route_strengths = [
        parameters["highlight_detail_suppression"]
        for parameters in calibration.SPECULAR_ROUTES.values()
    ]
    assert route_strengths == [0.0, 0.35, 0.65]
    for parameters in calibration.SPECULAR_ROUTES.values():
        for key, value in calibration.CALIBRATION_ARMS["balanced"].items():
            assert parameters[key] == value
    upper_strengths = [
        parameters["highlight_detail_suppression"]
        for parameters in calibration.UPPER_BOUND_ROUTES.values()
    ]
    assert upper_strengths == [0.0, 0.65, 1.0]
    for parameters in calibration.UPPER_BOUND_ROUTES.values():
        assert parameters["amount"] == 1.0
        assert parameters["texture_keep"] == 1.0
        assert parameters["shine_control"] == 1.0
    broad_strengths = [
        parameters["highlight_detail_suppression"]
        for parameters in calibration.BROAD_UPPER_BOUND_ROUTES.values()
    ]
    assert broad_strengths == [0.0, 0.65, 1.0]
    for parameters in calibration.BROAD_UPPER_BOUND_ROUTES.values():
        assert parameters["separation_radius_percent"] == 3.0
        assert parameters["positive_detail_threshold"] == 0.004


def test_bin_peak_selection_covers_time_and_is_deterministic():
    scores = [0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.7, 0.2, 0.6, 0.4, 0.5, 0.1]
    assert calibration._select_bin_peaks(scores, bins=3) == [1, 4, 8]
    assert calibration._select_bin_peaks([1.0] * 6, bins=3) == [0, 2, 4]


def test_bin_peak_selection_rejects_empty_or_impossible_contracts():
    with pytest.raises(ValueError, match="must not be empty"):
        calibration._select_bin_peaks([])
    with pytest.raises(ValueError, match="within"):
        calibration._select_bin_peaks([1.0, 2.0], bins=3)
