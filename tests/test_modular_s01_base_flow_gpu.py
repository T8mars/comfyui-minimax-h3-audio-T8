"""Guards for the sealed private S01 real-weight split probes."""

from copy import deepcopy

import pytest

from tools import run_modular_s01_base_flow_gpu as probe


def test_full_keeps_two_native_schedules_models_effects_and_lift():
    graph, source_sha = probe.build_graph()
    assert len(source_sha) == 64
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["13", "29"]
    assert graph["10"]["inputs"]["sigmas"] == ["91", 0]
    assert graph["26"]["inputs"]["sigmas"] == ["93", 1]
    assert graph["22"]["inputs"]["completed_stage"] == ["13", 2]
    assert graph["23"]["inputs"]["av_latent"] == ["45", 0]
    assert graph["9"]["inputs"]["width"] == 128
    assert graph["9"]["inputs"]["height"] == 64
    assert [graph[key]["inputs"]["length"] for key in ("40", "47")] == [22, 22]
    assert [graph[key]["inputs"]["mode"] for key in ("41", "42")] == ["report_only"] * 2


def test_cold_has_only_high_schedule_direct_model_and_exact_stage_load():
    receipt = {"path": "NativeExplicit/base_flow4plus4/LOW-x/manifest.json", "sha256": "a" * 64}
    graph, source_sha = probe.build_graph(cold=True, receipt=receipt)
    assert len(source_sha) == 64
    assert [key for key, node in graph.items()
            if node["class_type"] == "MiniMaxH3StageSamplerEXPT8"] == ["29"]
    assert graph["22"]["class_type"] == "UNETLoader"
    assert "completed_stage" not in graph["22"]["inputs"]
    assert graph["23"]["inputs"]["av_latent"] == ["60", 1]
    assert graph["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert graph["60"]["inputs"]["artifact_sha256"] == receipt["sha256"]
    assert graph["26"]["inputs"]["sigmas"] == ["93", 1]
    assert graph["47"]["inputs"]["length"] == 22
    assert not {"1", "9", "10", "13", "40", "41", "45", "50", "70", "90", "91"} & graph.keys()


def test_progress_checks_require_both_phases_or_high_only():
    full_events = ([{"type": "progress", "node": "13"}] * 4 +
                   [{"type": "progress", "node": "29"}] * 4 +
                   [{"type": "executing", "node": key} for key in
                    ("90", "91", "13", "50", "45", "23", "22", "92", "93", "26", "29", "51", "46")])
    full = {"terminal": {"type": "execution_success"}, "events": full_events}
    assert all(probe.stage_checks(full).values())
    wrong = deepcopy(full)
    wrong["events"][4]["node"] = "13"
    assert not probe.stage_checks(wrong)["exact_low_high_refine_progress"]
    cold_events = ([{"type": "progress", "node": "29"}] * 4 +
                   [{"type": "executing", "node": key} for key in
                    ("60", "23", "22", "92", "93", "26", "29", "51", "46")])
    cold = {"terminal": {"type": "execution_success"}, "events": cold_events}
    assert all(probe.stage_checks(cold, cold=True).values())
    cold["events"].append({"type": "executing", "node": "13"})
    assert not probe.stage_checks(cold, cold=True)["no_low_execution"]


@pytest.mark.parametrize("refine", (3, 4, 5))
def test_published_lbh_graph_keeps_original_parity_table_and_cold_boundary(refine):
    full, _ = probe.build_graph(recipe="lbh", refine=refine)
    receipt = {"path": f"NativeExplicit/lbh4plus{refine}/LOW-x/manifest.json", "sha256": "b" * 64}
    cold, _ = probe.build_graph(recipe="lbh", refine=refine, cold=True, receipt=receipt)
    assert [full[key]["class_type"] for key in ("91", "93")] == [
        "MiniMaxH3LearnedTwoPassParityPlanT8Advanced"] * 2
    for key in ("91", "93"):
        assert {name: full[key]["inputs"][name] for name in
                ("base_steps", "coarse_steps", "refine_steps")} == {
                    "base_steps": 8, "coarse_steps": 4, "refine_steps": refine}
    assert cold["93"]["inputs"]["refine_steps"] == refine
    assert cold["22"]["class_type"] == "UNETLoader"
    assert cold["23"]["inputs"]["av_latent"] == ["60", 1]
    assert not {"13", "40", "41", "45", "50", "90", "91"} & cold.keys()
    phase = {"terminal": {"type": "execution_success"}, "events":
             ([{"type": "progress", "node": "13"}] * 4 +
              [{"type": "progress", "node": "29"}] * refine +
              [{"type": "executing", "node": key} for key in
               ("90", "91", "13", "50", "45", "23", "22", "92", "93", "26", "29", "51", "46")])}
    assert all(probe.stage_checks(phase, recipe="lbh", refine=refine).values())


@pytest.mark.parametrize("coarse,refine", [(coarse, refine) for coarse in (8, 20)
                                            for refine in (3, 4, 5)])
def test_complete_first_graph_keeps_full_native_low_audio_and_cold_high(coarse, refine):
    full, _ = probe.build_graph(recipe="complete", coarse=coarse, refine=refine)
    receipt = {"path": f"NativeExplicit/complete{coarse}plus{refine}/LOW-x/manifest.json",
               "sha256": "c" * 64}
    cold, _ = probe.build_graph(recipe="complete", coarse=coarse, refine=refine,
                                  cold=True, receipt=receipt)
    assert full["90"]["inputs"]["steps"] == coarse
    assert full["10"]["inputs"]["sigmas"] == ["90", 2]
    assert "91" not in full and "91" not in cold
    assert full["93"]["inputs"]["refine_steps"] == refine
    assert full["25"]["inputs"]["second_pass_audio_source"] == "first_pass"
    assert cold["25"]["inputs"]["second_pass_audio_source"] == "first_pass"
    assert full["14"]["inputs"]["av_latent"] == ["94", 0]
    assert cold["14"]["inputs"]["av_latent"] == ["94", 0]
    assert full["94"]["inputs"]["second_pass_input"] == ["25", 0]
    assert cold["94"]["inputs"]["second_pass_output"] == ["46", 0]
    assert ("70" in full) == (coarse == 8)
    assert "70" not in cold
    assert cold["22"]["class_type"] == "UNETLoader"
    assert cold["23"]["inputs"]["av_latent"] == ["60", 1]
    assert not {"1", "9", "10", "13", "40", "41", "45", "50", "90"} & cold.keys()
    full_phase = {"terminal": {"type": "execution_success"}, "events":
                  ([{"type": "progress", "node": "13"}] * coarse +
                   [{"type": "progress", "node": "29"}] * refine +
                   [{"type": "executing", "node": key} for key in
                    ("90", "13", "50", "45", "23", "22", "92", "93", "26", "29", "51", "46", "94")])}
    assert all(probe.stage_checks(full_phase, recipe="complete", coarse=coarse,
                                  refine=refine).values())
    cold_phase = {"terminal": {"type": "execution_success"}, "events":
                  ([{"type": "progress", "node": "29"}] * refine +
                   [{"type": "executing", "node": key} for key in
                    ("60", "23", "22", "92", "93", "26", "29", "51", "46", "94")])}
    assert all(probe.stage_checks(cold_phase, cold=True, recipe="complete",
                                  coarse=coarse, refine=refine).values())


@pytest.mark.parametrize("width,height,frames", [(0, 64, 22), (128, 65, 22), (128, 64, 0)])
def test_invalid_s01_geometry_rejected(width, height, frames):
    with pytest.raises(ValueError, match="32-aligned"):
        probe.build_graph(width=width, height=height, frames=frames)
