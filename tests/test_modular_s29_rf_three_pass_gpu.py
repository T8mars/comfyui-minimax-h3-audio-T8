"""Guards for the private real-weight RF LOW -> BASE -> RESTART probes."""

from copy import deepcopy

import pytest

from tools import run_modular_s29_rf_three_pass_gpu as full
from tools import run_modular_s29_rf_three_pass_resume_gpu as cold


def test_three_pass_probe_keeps_low_learned_base_and_external_restart():
    graph, source_sha = full.build_probe_graph()
    assert len(source_sha) == 64
    assert [(key, node["class_type"]) for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == [
                ("13", "MiniMaxH3StageSamplerEXPT8"),
                ("29", "MiniMaxH3StageSamplerEXPT8"),
                ("99", "MiniMaxH3StageSamplerEXPT8")]
    assert graph["10"]["inputs"]["stage"] == "dual_low_4"
    assert graph["23"]["inputs"]["av_latent"] == ["45", 0]
    assert graph["88"]["inputs"]["stage"] == "dual_high_3"
    assert graph["26"]["inputs"]["sigmas"] == ["89", 0]
    assert graph["89"]["inputs"]["extra_tail_steps"] == 2
    assert graph["93"]["inputs"]["completed_stage"] == ["29", 2]
    assert graph["90"]["inputs"]["completed_av"] == ["46", 0]
    assert graph["96"]["inputs"]["model"] == ["121", 0]
    assert graph["95"]["inputs"]["prompt_relay_plan"] == ["104", 0]
    assert [graph[key]["inputs"]["length"] for key in ("40", "47", "104")] == [22] * 3
    assert [graph[key]["inputs"]["mode"] for key in ("41", "42", "100")] == ["report_only"] * 3


def test_cold_three_pass_graph_contains_only_restart_from_sealed_base():
    receipt = {"path": "RF/BASE-example/manifest.json", "sha256": "A" * 64}
    graph, source_sha = cold.build_resume_graph(receipt)
    assert len(source_sha) == 64
    assert graph["60"]["inputs"] == {"artifact_path": receipt["path"],
                                     "artifact_sha256": receipt["sha256"],
                                     "expected_stage": "rf_base"}
    assert graph["90"]["inputs"]["completed_av"] == ["60", 0]
    assert graph["93"]["class_type"] == "UNETLoader"
    assert "completed_stage" not in graph["93"]["inputs"]
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["99"]
    assert not {"1", "9", "10", "13", "22", "23", "24", "25", "26", "29",
                "40", "47", "50", "51", "70", "71", "88", "89", "110", "111"} & graph.keys()
    assert graph["104"]["inputs"]["length"] == 22
    assert graph["100"]["inputs"]["mode"] == "report_only"
    with pytest.raises(ValueError, match="22-frame"):
        cold.build_resume_graph(receipt, frames=73)


def test_three_pass_exact_progress_and_stage_order():
    events = ([{"type": "progress", "node": "13"}] * 4 +
              [{"type": "progress", "node": "29"}] * 5 +
              [{"type": "progress", "node": "99"}] * 3 +
              [{"type": "executing", "node": key} for key in
               ("89", "110", "111", "13", "50", "45", "23", "22", "29", "51", "46",
                "90", "93", "120", "121", "99", "103", "102")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(full.stage_checks(phase).values())
    wrong = deepcopy(phase)
    wrong["events"][9]["node"] = "29"
    assert not full.stage_checks(wrong)["exact_low4_rfbase5_restart3_progress"]
    wrong = deepcopy(phase)
    wrong["events"] = [item for item in wrong["events"]
                       if not (item["type"] == "executing" and item["node"] == "23")]
    assert not full.stage_checks(wrong)["three_stage_deferred_models_and_effects_order"]


def test_cold_three_pass_rejects_low_or_base_execution():
    events = ([{"type": "progress", "node": "99"}] * 3 +
              [{"type": "executing", "node": key} for key in
               ("60", "90", "93", "94", "95", "120", "121", "96", "99", "103", "102")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(cold.stage_checks(phase).values())
    wrong = deepcopy(phase)
    wrong["events"].append({"type": "executing", "node": "29"})
    assert not cold.stage_checks(wrong)["no_low_lift_or_rf_base_execution"]
