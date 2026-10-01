import pytest

from tools.build_modular_native_explicit_workflow import split_graph, RECIPES
from tools.build_modular_native_dual_workflow import VARIANTS
from test_modular_workflows import ancestors


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_original_plans_two_editable_stages_and_real_high_only_restore(recipe, variant):
    graph = split_graph(recipe, variant)
    name, coarse, refine = recipe
    assert not any("Loop" in item["class_type"] or "FastH3V2" in item["class_type"]
                   or "NativeDual" in item["class_type"] for item in graph.values())
    assert graph["26"]["class_type"] == "MiniMaxH3NativeStageBindEXPT8"
    assert graph["26"]["inputs"]["model"] == ["92", 0]
    assert graph["26"]["inputs"]["sampler"] == ["92", 1]
    assert graph["26"]["inputs"]["sigmas"] == ["93", 1]
    assert graph["93"]["inputs"]["model"] == ["92", 0]
    assert graph["93"]["inputs"]["refine_steps"] == refine
    assert graph["25"]["class_type"] == "MiniMaxH3TwoPassLatentReconcileT8Advanced"
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert ("94" in graph) == (name == "complete")
    if name == "complete":
        assert graph["25"]["inputs"]["second_pass_audio_source"] == "first_pass"
        assert graph["14"]["inputs"]["av_latent"] == ["94", 0]
    else:
        assert graph["25"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    samplers = [item for item in graph.values() if item["class_type"] in
                ("SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8")]
    assert len(samplers) == (1 if variant == "resume_effects" else 2)
    if variant == "resume_effects":
        assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50", "70", "90", "91"} & set(graph)
        assert graph["22"]["class_type"] == "UNETLoader"
        assert "completed_stage" not in graph["22"]["inputs"]
        assert graph["60"]["inputs"]["expected_stage"] == "native_low"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 1]
    else:
        assert graph["10"]["inputs"]["stage"] == "native_low"
        assert not {"22", "24", "25", "26", "27", "28", "29", "42", "44", "71", "92", "93"} & ancestors(graph, "13")
        if name == "complete":
            assert graph["90"]["inputs"]["steps"] == coarse
            assert graph["10"]["inputs"]["sigmas"] == ["90", 2]
            assert "91" not in graph
        else:
            assert graph["10"]["inputs"]["sigmas"] == ["91", 0]
        assert "13" in ancestors(graph, "29")
    if variant != "minimal":
        assert graph["92"]["inputs"]["model"] == ["24", 0]
        assert graph["28"]["inputs"]["model"] == ["44", 0]
        assert graph["42"]["inputs"]["mode"] == "report_only"
