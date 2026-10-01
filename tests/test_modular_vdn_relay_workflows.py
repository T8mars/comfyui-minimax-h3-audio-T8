import pytest

from tools.build_modular_vdn_relay_workflow import split_graph, BACKENDS, VARIANTS, vdn
from test_modular_workflows import ancestors


@pytest.mark.parametrize("training", vdn.TRAINING)
@pytest.mark.parametrize("backend,refine", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_per_stage_external_relay_eav_completed_output_and_real_high_only_restore(training, backend, refine, variant):
    graph = split_graph(training, backend, refine, variant)
    effects = variant != "relay"
    resume = variant == "resume_relay_eav"
    assert graph["24"]["inputs"]["prompt_relay_plan"] == ["140", 0]
    assert graph["24"]["inputs"]["model"] == ["71", 0]
    assert graph["25"]["inputs"]["positive"] == ["24", 1]
    assert graph["25"]["inputs"]["highres_template"] == ["24", 2]
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["14"]["inputs"]["av_latent"] == ["94", 0]
    assert graph["25"]["inputs"]["second_pass_audio_source"] == "first_pass"
    if backend == "vdn":
        assert graph["26"]["inputs"]["model"] == ["24", 0]
        assert graph["111"]["inputs"]["model"] == ["26", 0]
        assert graph["44" if effects else "28"]["inputs"]["model"] == ["111", 0]
        assert graph["94"]["inputs"]["second_pass_output"] == ["113", 0]
        assert graph["113"]["inputs"]["runtime"] == ["111", 1]
    else:
        assert graph["71"]["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
        assert graph["92"]["inputs"]["model"] == ["24", 0]
        assert graph["93"]["inputs"]["refine_steps"] == refine
        assert graph["26"]["inputs"]["sigmas"] == ["93", 1]
        assert "111" not in graph and "113" not in graph
    if resume:
        assert not {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "70", "110", "112", "114"} & set(graph)
        assert graph["60"]["inputs"]["expected_stage"] == "vdn_complete"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 0]
        if backend == "vdn":
            assert graph["26"]["inputs"]["first_pass_latent"] == ["60", 0]
    else:
        assert graph["10"]["inputs"]["stage"] == "vdn_complete"
        assert graph["10"]["inputs"]["model"] == ["9", 0]
        assert graph["9"]["inputs"]["model"] == ["70", 0]
        assert graph["9"]["inputs"]["prompt_relay_plan"] == ["40", 0]
        assert graph["110"]["inputs"]["av_latent"] == ["9", 2]
        assert graph["43" if effects else "12"]["inputs"]["model"] == ["110", 0]
        assert graph["23"]["inputs"]["av_latent"] == ["112", 0]
        assert graph["112"]["inputs"]["av_latent"][1] == 0
        assert not {"24", "26", "29", "71", "111", "140"} & ancestors(graph, "13")
    if effects:
        assert graph["42"]["inputs"]["mode"] == "report_only"
        assert graph["28"]["inputs"]["model"] == ["44", 0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_b50_original_own_five_step_tail_is_exposed_without_changing_dmd8(variant):
    graph = split_graph("stage_b_50nfe", "vdn", 5, variant)
    assert graph["26"]["inputs"]["refine_steps"] == 5
    assert graph["26"]["inputs"]["stage"] == "vdn_refine"
    if variant == "resume_relay_eav":
        assert "10" not in graph
    else:
        assert graph["10"]["inputs"]["refine_steps"] == 4
    with pytest.raises(ValueError, match="Unknown VDN"):
        split_graph("stage_dmd_8nfe", "vdn", 5, variant)
