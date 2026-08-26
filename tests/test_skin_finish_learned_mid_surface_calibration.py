from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "calibrate_skin_finish_learned_mid_surface.py"
PYTHON = Path(r"F:\AI-T8-video-onekey\python\python.exe")


def _plan(tmp_path: Path, mode: str) -> dict:
    result = subprocess.run(
        [
            str(PYTHON),
            str(TOOL),
            "--mode",
            mode,
            "--output",
            str(tmp_path / "must-not-exist"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_mid_surface_calibration_defaults_to_single_frame_plan_only(tmp_path):
    report = _plan(tmp_path, "single")
    assert report["status"] == "PLAN_ONLY"
    assert report["frame_indices"] == [66]
    assert report["fusion_parameters"]["broad_amount"] == 0.65
    assert report["fusion_parameters"]["mid_amount"] == 0.85
    assert report["fusion_parameters"]["fine_split_radius_px"] == 2
    assert report["fusion_parameters"]["broad_radius_px"] == 12
    assert report["minimum_full_frame_masked_mean_change"] == 0.018
    assert report["identity_gate"]["official_sface_cosine_threshold"] == 0.363
    assert report["identity_gate"]["minimum_source_candidate_cosine"] == 0.563
    assert report["identity_gate"]["minimum_improvement_over_raw_gfpgan_proposal"] == 0.02
    assert report["loads_h3"] is False
    assert report["loads_sam"] is False
    assert report["runs_full_video"] is False
    assert report["stress_or_repeat"] is False
    assert report["automatic_accept"] is False
    assert not (tmp_path / "must-not-exist").exists()


def test_six_frame_plan_is_fixed_and_does_not_execute_without_confirmation(tmp_path):
    report = _plan(tmp_path, "six")
    assert report["frame_indices"] == [16, 20, 60, 66, 86, 119]
    assert report["fusion_parameters"]["candidate_rgb_delta_cap"] == 0.10
    assert report["fusion_parameters"]["minimum_source_fine_cosine"] == 0.95
    assert report["fusion_parameters"]["maximum_texture_ratio"] == 1.20
    assert not (tmp_path / "must-not-exist").exists()
