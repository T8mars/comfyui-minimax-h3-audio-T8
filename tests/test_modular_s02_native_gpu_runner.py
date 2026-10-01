"""Guard all six saved Native Dual GPU probe pairs without starting a Core."""

from copy import deepcopy

import pytest

from tools import run_modular_s02_native_gpu as full
from tools import run_modular_s02_native_resume_gpu as resume


@pytest.mark.parametrize("coarse", [4, 20])
@pytest.mark.parametrize("refine", [3, 4, 5])
def test_saved_full_and_cold_graph_keep_independent_effects_and_high_only(coarse, refine):
    graph, source_sha = full.build_probe_graph(coarse, refine, width=128, height=64, frames=22)
    receipt = {"path": f"NativeDual/LOW{coarse}/LOW-fixed/manifest.json", "sha256": "a" * 64}
    cold, resume_sha = resume.build_resume_graph(coarse, refine, receipt)
    assert len(source_sha) == len(resume_sha) == 64
    assert graph["40"]["inputs"]["length"] == graph["47"]["inputs"]["length"] == 22
    assert cold["47"]["inputs"]["length"] == 22
    assert graph["41"]["inputs"] == graph["42"]["inputs"] == cold["42"]["inputs"]
    assert graph["23"]["inputs"]["av_latent"] == ["45", 0]
    assert cold["23"]["inputs"]["av_latent"] == ["60", 1]
    assert graph["22"]["inputs"]["completed_stage"] == ["13", 2]
    assert cold["22"]["class_type"] == "UNETLoader"
    assert cold["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert cold["60"]["inputs"]["artifact_sha256"] == receipt["sha256"]
    assert "13" not in cold and "50" not in cold and "40" not in cold


def _phase(order, progress):
    return {"terminal": {"type": "execution_success"},
            "events": ([{"type": "executing", "node": node} for node in order] +
                       [{"type": "progress", "node": node} for node in progress])}


def test_full_and_cold_execution_audits_detect_reordered_or_repeated_low():
    complete = _phase(["13", "50", "45", "23", "22", "29", "51"], ["13"] * 4 + ["29"] * 4)
    assert all(full.stage_checks(complete, 4, 4).values())
    cold = _phase(["60", "23", "22", "29", "51"], ["29"] * 4)
    assert all(resume.stage_checks(cold, 4).values())
    parallel = deepcopy(complete)
    parallel["events"][1], parallel["events"][2] = parallel["events"][2], parallel["events"][1]
    assert all(full.stage_checks(parallel, 4, 4).values())
    bad = deepcopy(complete)
    bad["events"][2], bad["events"][3] = bad["events"][3], bad["events"][2]
    assert full.stage_checks(bad, 4, 4)["low_save_eav_lift_independent_high_order"] is False
    cold["events"].insert(3, {"type": "executing", "node": "13"})
    assert resume.stage_checks(cold, 4)["no_low_execution"] is False
