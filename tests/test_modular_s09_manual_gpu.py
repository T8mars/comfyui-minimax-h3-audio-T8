"""Static guards for the private real-weight S09 manual-pass probes."""

from copy import deepcopy

import pytest

from tools import run_modular_s09_manual_gpu as probe
from tools import run_modular_s09_manual_resume_gpu as cold


@pytest.mark.parametrize("noise,mode", [("NativeNoise", "disabled"),
                                        ("FreeNoise", "variance_preserving_blend")])
def test_saved_s09_pair_keeps_independent_model_noise_relay_eav(noise, mode):
    graph, source_sha = probe.build_probe_graph(noise)
    assert len(source_sha) == 64
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["13", "29"]
    assert graph["22"]["inputs"]["completed_stage"] == ["13", 2]
    assert graph["26"]["inputs"]["av_latent"] == ["45", 0]
    assert graph["29"]["inputs"]["latent_image"] == ["45", 0]
    assert graph["40"]["inputs"]["length"] == ["82", 0]
    assert graph["47"]["inputs"]["length"] == ["82", 0]
    assert [graph[key]["inputs"]["mode"] for key in ("83", "84")] == [mode, mode]
    assert [graph[key]["inputs"]["mode"] for key in ("41", "42")] == ["report_only"] * 2
    assert [graph[key]["inputs"]["value"] for key in ("80", "81", "82")] == [128, 64, 22]


def test_s09_progress_and_dependency_order_require_both_passes():
    events = ([{"type": "progress", "node": "13"}] * 20 +
              [{"type": "progress", "node": "29"}] * 3 +
              [{"type": "executing", "node": key} for key in
               ("83", "13", "50", "45", "22", "26", "84", "29", "51", "46")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(probe.stage_checks(phase).values())
    wrong = deepcopy(phase)
    wrong["events"][20]["node"] = "13"
    assert not probe.stage_checks(wrong)["exact_first20_second3_progress"]
    wrong = deepcopy(phase)
    wrong["events"] = [item for item in wrong["events"]
                       if not (item["type"] == "executing" and item["node"] == "22")]
    assert not probe.stage_checks(wrong)["independent_second_model_noise_and_handoff_order"]


@pytest.mark.parametrize("width,height,frames", [(0, 64, 22), (128, 65, 22), (128, 64, 0)])
def test_invalid_s09_probe_geometry_rejected(width, height, frames):
    with pytest.raises(ValueError, match="32-aligned"):
        probe.build_probe_graph("NativeNoise", width=width, height=height, frames=frames)


@pytest.mark.parametrize("noise,mode", [("NativeNoise", "disabled"),
                                        ("FreeNoise", "variance_preserving_blend")])
def test_saved_s09_cold_graph_only_runs_independent_second(noise, mode):
    receipt = {"path": "ManualPass/FIRST-test/manifest.json", "sha256": "a" * 64}
    graph, source_sha = cold.build_resume_graph(noise, receipt)
    assert len(source_sha) == 64
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["29"]
    assert graph["22"]["class_type"] == "UNETLoader"
    assert "completed_stage" not in graph["22"]["inputs"]
    assert graph["26"]["inputs"]["av_latent"] == ["60", 0]
    assert graph["29"]["inputs"]["latent_image"] == ["60", 0]
    assert graph["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert graph["60"]["inputs"]["artifact_sha256"] == receipt["sha256"]
    assert graph["84"]["inputs"]["mode"] == mode
    assert graph["42"]["inputs"]["mode"] == "report_only"
    assert not {"13", "40", "41", "45", "50", "83"} & graph.keys()


def test_s09_cold_events_require_only_second_and_frozen_first():
    events = ([{"type": "progress", "node": "29"}] * 3 +
              [{"type": "executing", "node": key} for key in
               ("60", "22", "24", "26", "84", "29", "51", "46")])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(cold.stage_checks(phase).values())
    extra_first = deepcopy(phase)
    extra_first["events"].append({"type": "executing", "node": "13"})
    assert not cold.stage_checks(extra_first)["no_first_execution"]
    extra_step = deepcopy(phase)
    extra_step["events"].insert(0, {"type": "progress", "node": "13"})
    assert not cold.stage_checks(extra_step)["only_manual_second_three_progress"]
