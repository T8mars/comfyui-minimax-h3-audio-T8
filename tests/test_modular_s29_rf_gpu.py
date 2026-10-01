"""Static guards for the isolated real-weight RF standalone probe."""

from copy import deepcopy

import pytest

from tools import run_modular_s29_rf_gpu as probe
from tools import run_modular_s29_rf_resume_gpu as resume


def test_reduced_rf_probe_preserves_two_explicit_stages_and_effects():
    graph, source_sha = probe.build_probe_graph()
    assert len(source_sha) == 64
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["13", "29"]
    assert graph["22"]["inputs"]["completed_stage"] == ["13", 2]
    assert graph["90"]["inputs"]["completed_av"] == ["45", 0]
    assert graph["26"]["inputs"]["rf_handoff"] == ["90", 0]
    assert graph["14"]["inputs"]["av_latent"] == ["46", 0]
    assert graph["40"]["inputs"]["length"] == ["82", 0]
    assert graph["47"]["inputs"]["length"] == ["82", 0]
    assert [graph[key]["inputs"]["value"] for key in ("80", "81", "82")] == [128, 64, 22]
    assert all(graph[key]["inputs"]["mode"] == "report_only" for key in ("41", "42"))


@pytest.mark.parametrize("width,height,frames", [(0, 64, 22), (96, 65, 22), (128, 64, 0)])
def test_invalid_probe_geometry_rejected(width, height, frames):
    with pytest.raises(ValueError, match="32-aligned"):
        probe.build_probe_graph(width=width, height=height, frames=frames)


def test_exact_progress_and_dependency_order_required():
    events = ([{"type": "progress", "node": "13"}] * 20 +
              [{"type": "progress", "node": "29"}] * 3 +
              [{"type": "executing", "node": key}
               for key in ("13", "50", "45", "22", "90", "29", "51", "46")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(probe.stage_checks(phase).values())
    wrong = deepcopy(phase)
    wrong["events"][20]["node"] = "13"
    assert not probe.stage_checks(wrong)["exact_base_schedule_restart3_progress"]
    missing = deepcopy(phase)
    missing["events"] = [item for item in missing["events"]
                         if not (item["type"] == "executing" and item["node"] == "90")]
    assert not probe.stage_checks(missing)["base_save_eav_handoff_restart_order"]


def test_cold_graph_loads_sealed_base_without_first_stage():
    receipt = {"path": "RF/BASE-example/manifest.json", "sha256": "A" * 64}
    graph, source_sha = resume.build_resume_graph(receipt)
    assert len(source_sha) == 64
    assert graph["60"]["inputs"] == {
        "artifact_path": receipt["path"], "artifact_sha256": receipt["sha256"],
        "expected_stage": "rf_base"}
    assert graph["90"]["inputs"]["completed_av"] == ["60", 0]
    assert graph["22"]["class_type"] == "UNETLoader"
    assert "completed_stage" not in graph["22"]["inputs"]
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["29"]
    assert not {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "88"} & graph.keys()


def test_cold_progress_rejects_first_pass():
    events = ([{"type": "progress", "node": "29"}] * 3 +
              [{"type": "executing", "node": key}
               for key in ("60", "90", "22", "29", "51", "46")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(resume.stage_checks(phase).values())
    wrong = deepcopy(phase)
    wrong["events"].append({"type": "executing", "node": "13"})
    assert not resume.stage_checks(wrong)["no_base_execution"]


def test_detail_mixer_keeps_external_tail_bias_stg_and_cold_boundary():
    graph, full_sha = probe.build_probe_graph("detail_mixer")
    cold, cold_sha = resume.build_resume_graph(
        {"path": "RF/BASE-example/manifest.json", "sha256": "A" * 64}, "detail_mixer")
    assert len(full_sha) == len(cold_sha) == 64
    assert graph["89"]["inputs"]["extra_tail_steps"] == 2
    assert graph["10"]["inputs"]["sigmas"] == ["89", 0]
    assert graph["10"]["inputs"]["model"] == ["111", 0]
    assert graph["26"]["inputs"]["model"] == ["121", 0]
    assert cold["26"]["inputs"]["model"] == ["121", 0]
    assert cold["90"]["inputs"]["completed_av"] == ["60", 0]
    assert "89" not in cold and "110" not in cold and "111" not in cold
    assert "120" in cold and "121" in cold
    assert graph["16"]["inputs"]["filename_prefix"].endswith("detail_mixer_full_selected")
    assert cold["16"]["inputs"]["filename_prefix"].endswith("detail_mixer_cold_selected")
    events = ([{"type": "progress", "node": "13"}] * 22 +
              [{"type": "progress", "node": "29"}] * 3 +
              [{"type": "executing", "node": key}
               for key in ("13", "50", "45", "22", "90", "29", "51", "46")])
    assert all(probe.stage_checks({"terminal": {"type": "execution_success"},
                                   "events": events}, "detail_mixer").values())
