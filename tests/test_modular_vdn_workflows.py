import pytest

from tools.build_modular_vdn_workflow import split_graph, TRAINING, VARIANTS
from test_modular_workflows import ancestors


@pytest.mark.parametrize("training", TRAINING)
@pytest.mark.parametrize("variant", VARIANTS)
def test_independent_vdn_complete_and_own_tail_actual_output_handoff_and_high_restore(training, variant):
    graph = split_graph(training, variant)
    assert not any("Loop" in item["class_type"] or "FastH3V2" in item["class_type"]
                   or "ParityPlan" in item["class_type"] for item in graph.values())
    assert graph["71"]["inputs"]["model"] == ["22", 0]
    assert graph["71"]["inputs"]["stage"] == training
    assert graph["26"]["inputs"]["stage"] == "vdn_refine"
    assert graph["26"]["inputs"]["model"] == ["71", 0]
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["25"]["inputs"]["second_pass_audio_source"] == "first_pass"
    assert graph["14"]["inputs"]["av_latent"] == ["94", 0]
    assert graph["94"]["inputs"]["fail_on_locked_mismatch"] is True
    completed = ["13", 0] if variant == "minimal" else ["45", 0]
    samplers = [item for item in graph.values() if item["class_type"] in
                ("SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8")]
    assert len(samplers) == (1 if variant == "resume_eav" else 2)
    if variant == "resume_eav":
        assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50", "70"} & set(graph)
        assert graph["60"]["inputs"]["expected_stage"] == "vdn_complete"
        completed = ["60", 0]
    else:
        assert graph["70"]["inputs"]["model"] == ["1", 0]
        assert graph["10"]["inputs"]["stage"] == "vdn_complete"
        assert not {"22", "24", "25", "26", "27", "28", "29", "42", "44", "71"} & ancestors(graph, "13")
        assert "13" in ancestors(graph, "29")
    assert graph["23"]["inputs"]["av_latent"] == completed
    assert graph["26"]["inputs"]["first_pass_latent"] == completed
    if variant != "minimal":
        assert graph["28"]["inputs"]["model"] == ["44", 0]
        assert graph["42"]["inputs"]["mode"] == "report_only"
    if variant in ("save_eav", "resume_eav"):
        assert graph["46"]["inputs"]["av_latent"] == ["51", 0]


@pytest.mark.parametrize("training", TRAINING)
@pytest.mark.parametrize("refine", [3, 4, 5])
@pytest.mark.parametrize("variant", VARIANTS)
def test_vdn_to_clean_native_high_graph_never_uses_lbh_coarse_for_vdn(training, refine, variant):
    from tools.build_modular_vdn_native_workflow import split_graph as mixed_graph
    graph = mixed_graph(training, refine, variant)
    assert graph["71"]["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
    assert graph["71"]["inputs"]["model"] == ["22", 0]
    assert graph["26"]["class_type"] == "MiniMaxH3NativeStageBindEXPT8"
    assert graph["26"]["inputs"]["stage"] == "native_high"
    assert graph["26"]["inputs"]["sigmas"] == ["93", 1]
    assert graph["93"]["inputs"]["refine_steps"] == refine
    assert graph["92"]["inputs"]["model"] == (["71", 0] if variant == "minimal" else ["24", 0])
    assert graph["14"]["inputs"]["av_latent"] == ["94", 0]
    assert graph["23"]["inputs"]["av_latent"][1] == 0
    if variant == "resume_eav":
        assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50", "70"} & set(graph)
        assert graph["60"]["inputs"]["expected_stage"] == "vdn_complete"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 0]
    else:
        assert graph["10"]["inputs"]["stage"] == "vdn_complete"
        assert not {"71", "92", "93", "24", "40"} & ancestors(graph, "13")
    if variant != "minimal":
        assert graph["24"]["inputs"]["model"] == ["71", 0]
        assert graph["24"]["inputs"]["prompt_relay_plan"] == ["40", 0]
        assert graph["25"]["inputs"]["positive"] == ["24", 1]
        assert graph["25"]["inputs"]["highres_template"] == ["24", 2]
