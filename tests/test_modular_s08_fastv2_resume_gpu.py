"""Guard the private S08 cold HIGH-only probe without running weights."""

from copy import deepcopy
import json

from tools import run_modular_s08_fastv2_resume_gpu as probe


def test_resume_graph_binds_exact_low_without_inserting_low_branch():
    receipt = {"path": "FastH3V2/LOW-example/manifest.json", "sha256": "A" * 64}
    original = json.loads(probe.CANDIDATE.read_text(encoding="utf-8"))
    graph, source_sha = probe.build_resume_graph(receipt)
    expected = deepcopy(original)
    expected["60"]["inputs"].update(artifact_path=receipt["path"], artifact_sha256=receipt["sha256"])
    expected["26"]["inputs"]["min_tokens"] = 0
    expected["47"]["inputs"]["length"] = 22
    expected["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/S08_FastV2_RealSplit/high_only_selected"
    assert graph == expected
    assert source_sha == probe.shared._sha256_file(probe.CANDIDATE)
    assert "13" not in graph and "1" not in graph


def test_resume_stage_checks_require_only_high_and_order():
    phase = {"terminal": {"type": "execution_success"}, "events": [
        *({"type": "executing", "node": node} for node in ("60", "22", "23", "29", "51")),
        *({"type": "progress", "node": "29"} for _ in range(4)),
    ]}
    assert all(probe.stage_checks(phase).values())
    bad = deepcopy(phase)
    bad["events"].append({"type": "executing", "node": "13"})
    assert not probe.stage_checks(bad)["no_low_execution"]
    bad = deepcopy(phase)
    bad["events"][1], bad["events"][2] = bad["events"][2], bad["events"][1]
    assert not probe.stage_checks(bad)["load_independent_model_trained_lift_high_save_in_order"]
