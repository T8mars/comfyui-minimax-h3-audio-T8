"""No hidden descents, independent effects, and real removal on RF restore."""
import pytest

from tools.build_modular_rf_workflow import split_graph, ENTRIES, native
from test_modular_workflows import ancestors


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("variant", native.VARIANTS)
def test_each_rf_entry_has_editable_descents_and_anchored_restart_only_restore(entry, variant):
    graph = split_graph(entry, variant)
    third = entry == "two_pass_detail_mixer"
    base_id, base_sample = ("26", "29") if third else ("10", "13")
    restart_id, final_sample, condition_id = ("96", "99", "95") if third else ("26", "29", "24")
    types = [item["class_type"] for item in graph.values()]
    assert not any("Loop" in value or "DetailMixer" in value or "RectifiedFlowRestartSamplerT8Advanced" in value for value in types)
    assert types.count("MiniMaxH3RFHandoffEXPT8") == types.count("MiniMaxH3RFRestartStageSetupEXPT8") == 1
    assert graph[restart_id]["inputs"]["rf_handoff"] == ["90", 0]
    assert graph[condition_id]["inputs"]["width"] == ["90", 3]
    assert graph[condition_id]["inputs"]["height"] == ["90", 4]
    assert "original_template" not in graph["90"]["inputs"]  # Real anchor travels in the base result.
    assert graph[final_sample]["inputs"]["stage_context"] == [restart_id, 4]
    assert graph[final_sample]["inputs"]["latent_image"] == [restart_id, 3]
    samplers = [key for key, node in graph.items() if node["class_type"] in
                ("SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8")]
    if variant == "resume_effects":
        assert samplers == [final_sample]
        restart_loader = "93" if third else "22"
        assert graph[restart_loader]["class_type"] == "UNETLoader"
        assert graph["60"]["inputs"]["expected_stage"] == "rf_base"
        assert graph["90"]["inputs"]["completed_av"] == ["60", 0]
        assert not {"1", "9", "10", "11", "12", "13", "23", "25", "70", "71", "88", "89", "110", "111"} & set(graph)
        assert "MiniMaxH3RFBaseStageSetupEXPT8" not in types
        assert not any("Upscale" in value for value in types)
    else:
        assert len(samplers) == (3 if third else 2)
        assert graph[base_sample]["class_type"] == "MiniMaxH3StageSamplerEXPT8"
        assert graph[base_sample]["inputs"]["stage_context"] == [base_id, 3]
        assert base_sample in ancestors(graph, final_sample)
        assert restart_id not in ancestors(graph, base_sample)
        assert "120" not in ancestors(graph, base_sample)
        if third:
            assert "MiniMaxH3LearnedLatentUpscaleT8Advanced" in types
            assert {"13", "23", "25"} <= ancestors(graph, base_sample)
            if variant == "save_effects":
                assert graph["93"]["inputs"]["completed_stage"] == ["29", 2]
    if entry != "standalone":
        assert graph["120"]["class_type"] == "MiniMaxH3ModelTimeBiasSamplerT8Advanced"
        assert graph["121"]["class_type"] == "MiniMaxH3SpatioTemporalGuidanceT8Advanced"
        if variant != "resume_effects":
            assert graph["89"]["class_type"] == "MiniMaxH3AVTailDetailScheduleT8Advanced"
    if variant != "minimal":
        eav_id = "101" if third else "44"
        assert graph[eav_id]["inputs"]["stage_context"] == [restart_id, 4]
        assert graph[eav_id]["inputs"]["av_latent"] == [restart_id, 3]
        expected_plan = "104" if third else "47"
        assert graph[condition_id]["inputs"]["prompt_relay_plan"] == [expected_plan, 0]
        assert graph[expected_plan]["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
        if third:
            if variant == "resume_effects":
                assert "47" not in graph
            else:
                assert graph["24"]["inputs"]["prompt_relay_plan"] == ["47", 0]
                assert graph["104"]["inputs"] == graph["47"]["inputs"]
                assert graph["104"] is not graph["47"]
                assert "104" not in ancestors(graph, base_sample)
