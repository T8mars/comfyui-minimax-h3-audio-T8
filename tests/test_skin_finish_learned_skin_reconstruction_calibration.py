from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "calibrate_skin_finish_learned_skin_reconstruction.py"
PYTHON = Path(r"F:\AI-T8-video-onekey\python\python.exe")


def test_skin_reconstruction_calibration_defaults_to_plan_only(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [str(PYTHON), str(TOOL), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY"
    assert report["frame_indices"] == [66]
    assert report["fusion_parameters"]["amount"] == 0.70
    assert report["fusion_parameters"]["proposal_prefilter_radius_px"] == 1
    assert report["fusion_parameters"]["candidate_rgb_delta_cap"] == 0.12
    assert report["fusion_parameters"]["minimum_masked_mean_abs_change"] == 0.025
    assert report["identity_gate"]["minimum_source_candidate_cosine"] == 0.563
    assert report["minimum_full_frame_masked_mean_change"] == 0.025
    assert report["saves_full_resolution_evidence"] is True
    assert report["loads_h3"] is False
    assert report["loads_sam"] is False
    assert report["runs_full_video"] is False
    assert report["stress_or_repeat"] is False
    assert report["automatic_accept"] is False
    assert not output.exists()
