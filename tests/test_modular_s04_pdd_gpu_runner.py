"""Guard the real-asset S04 probe graphs without starting a GPU/Core."""

import pytest

from tools import run_modular_s04_pdd_gpu as full
from tools import run_modular_s04_pdd_resume_gpu as resume


@pytest.mark.parametrize("base,canvas", [("FL2VA", (128, 64)), ("Ref2VA", (128, 128))])
def test_full_and_cold_graph_preserve_one_timeline_and_stage_boundary(base, canvas):
    width, height = canvas
    full_graph, full_sha = full.build_probe_graph(base, width=width, height=height, frames=22)
    receipt = {"path": f"PDD/{base}/LOW-fixed/manifest.json", "sha256": "a" * 64}
    cold_graph, cold_sha = resume.build_resume_graph(base, receipt, [width, height, 22])
    assert len(full_sha) == len(cold_sha) == 64
    assert full_graph["40"]["inputs"]["length"] == 22
    assert full_graph["47"]["inputs"]["length"] == cold_graph["47"]["inputs"]["length"]
    assert full_graph["42"]["inputs"] == cold_graph["42"]["inputs"]
    assert cold_graph["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert cold_graph["60"]["inputs"]["artifact_sha256"] == receipt["sha256"]
    assert cold_graph["22"]["class_type"] == "UNETLoader"
    assert cold_graph["23"]["inputs"]["av_latent"] == ["60", 1]
    assert "13" not in cold_graph and "50" not in cold_graph
    if base == "FL2VA":
        for key in ("97", "98"):
            assert full_graph[key]["inputs"]["width"] == cold_graph[key]["inputs"]["width"]
            assert full_graph[key]["inputs"]["height"] == cold_graph[key]["inputs"]["height"]


def _phase(order, progress):
    return {"terminal": {"type": "execution_success"},
            "events": ([{"type": "executing", "node": node} for node in order] +
                       [{"type": "progress", "node": node} for node in progress])}


def test_stage_audits_allow_independent_loader_lift_order_but_reject_low_in_cold():
    complete = _phase(["13", "50", "23", "22", "29", "51"], ["13"] * 4 + ["29"] * 4)
    assert all(full.stage_execution_checks(complete).values())
    complete["events"][2], complete["events"][3] = complete["events"][3], complete["events"][2]
    assert all(full.stage_execution_checks(complete).values())
    cold = _phase(["60", "23", "22", "29", "51"], ["29"] * 4)
    assert all(resume.stage_checks(cold).values())
    cold["events"].insert(3, {"type": "executing", "node": "13"})
    assert resume.stage_checks(cold)["no_low_execution"] is False
