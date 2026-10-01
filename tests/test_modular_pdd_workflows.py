import pytest

from tools.build_modular_pdd_workflow import split_graph, BASES
from tools.build_modular_native_dual_workflow import VARIANTS
from test_modular_workflows import ancestors


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_pdd_independent_models_absolute_stages_joint_audio_and_real_restore(base, variant):
    graph = split_graph(base, variant)
    assert not any("Loop" in item["class_type"] or "FastH3V2" in item["class_type"]
                   or "NativeDual" in item["class_type"] for item in graph.values())
    assert graph["26"]["inputs"] == {"model": ["92", 0], "full_sigmas": ["92", 2],
        "av_latent": ["25", 0], "stage": "pdd_high_4_8"}
    assert graph["92"]["inputs"]["base_variant"] == base
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["25"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    assert not any("AudioAudit" in item["class_type"] for item in graph.values())
    if base == "FL2VA":
        assert graph["24"]["inputs"]["first_frame"] == ["97", 0]
        assert graph["24"]["inputs"]["last_frame"] == ["98", 0]
    else:
        assert graph["24"]["inputs"]["ref_images.ref_image_0"] == ["95", 0]
    samplers = [item for item in graph.values() if item["class_type"] in
                ("SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8")]
    assert len(samplers) == (1 if variant == "resume_effects" else 2)
    if variant == "resume_effects":
        assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50", "90"} & set(graph)
        assert graph["22"]["class_type"] == "UNETLoader"
        assert graph["60"]["inputs"]["expected_stage"] == "pdd_low_0_4"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 1]
    else:
        assert graph["10"]["inputs"]["stage"] == "pdd_low_0_4"
        assert graph["10"]["inputs"]["full_sigmas"] == ["90", 2]
        assert not {"22", "24", "25", "26", "27", "28", "29", "42", "44", "92"} & ancestors(graph, "13")
    if variant != "minimal":
        assert graph["47"]["inputs"]["length"] == (124 if base == "FL2VA" else 22)
        if variant != "resume_effects":
            assert graph["40"]["inputs"]["length"] == graph["47"]["inputs"]["length"]
        assert graph["92"]["inputs"]["model"] == ["24", 0]
        assert graph["24"]["inputs"]["model"] == ["22", 0]
        assert graph["28"]["inputs"]["model"] == ["44", 0]
        assert graph["42"]["inputs"]["mode"] == "report_only"
