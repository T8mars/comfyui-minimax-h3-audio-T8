from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "calibrate_skin_finish_learned_detail.py"
PYTHON = Path(r"F:\AI-T8-video-onekey\python\python.exe")


def test_learned_detail_calibration_defaults_to_plan_only(tmp_path):
    result = subprocess.run(
        [
            str(PYTHON),
            str(TOOL),
            "--mode",
            "single",
            "--output",
            str(tmp_path / "must-not-exist"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY"
    assert report["frame_indices"] == [66]
    assert report["loads_h3"] is False
    assert report["loads_sam"] is False
    assert report["runs_full_video"] is False
    assert report["stress_or_repeat"] is False
    assert report["automatic_accept"] is False
    assert not (tmp_path / "must-not-exist").exists()


def test_six_frame_plan_is_fixed_and_has_one_predeclared_arm(tmp_path):
    result = subprocess.run(
        [
            str(PYTHON),
            str(TOOL),
            "--mode",
            "six",
            "--output",
            str(tmp_path / "must-not-exist"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["frame_indices"] == [16, 20, 60, 66, 86, 119]
    assert report["fusion_parameters"]["amount"] == 0.70
    assert report["fusion_parameters"]["surface_amount"] == 0.45
    assert report["fusion_parameters"]["maximum_surface_luma_delta"] == 0.035
    assert report["fusion_parameters"]["chroma_amount"] == 0.20
    assert report["fusion_parameters"]["maximum_chroma_component_delta"] == 0.04
    assert report["fusion_parameters"]["candidate_rgb_delta_cap"] == 0.10
    assert report["fusion_parameters"]["maximum_detail_gain"] == 1.80
    assert report["learned_model_sha256"] == (
        "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"
    )
    assert not (tmp_path / "must-not-exist").exists()
