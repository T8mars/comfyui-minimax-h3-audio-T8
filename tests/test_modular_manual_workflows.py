import pytest

from tools.build_modular_manual_pass_workflow import split_graph
from test_modular_workflows import ancestors


@pytest.mark.parametrize("variant", ["minimal", "effects", "save_effects", "resume_effects"])
@pytest.mark.parametrize("free_noise", [False, True])
def test_explicit_manual_output_handoff_and_noise_without_hidden_first(variant, free_noise):
    graph = split_graph(variant, free_noise=free_noise)
    assert not any("Dual" in item["class_type"] or "Upscale" in item["class_type"]
                   or "Loop" in item["class_type"] or "FastH3V2" in item["class_type"] for item in graph.values())
    assert "62" not in graph
    assert graph["26"]["inputs"]["stage"] == "manual_second"
    if variant != "minimal":
        assert graph["24"]["inputs"]["prompt_relay_plan"] == ["47", 0]
        assert graph["47"]["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
        assert graph["47"]["inputs"]["length"] == ["82", 0]
        if variant == "resume_effects":
            assert "40" not in graph
            assert graph["22"]["class_type"] == "UNETLoader"
            assert "13" not in graph
        else:
            assert graph["9"]["inputs"]["prompt_relay_plan"] == ["40", 0]
            assert graph["47"]["inputs"] == graph["40"]["inputs"]
            assert graph["47"] is not graph["40"]
            assert "47" not in ancestors(graph, "13")
    assert graph["29"]["inputs"]["noise"] == ["84", 0]
    assert graph["84"]["inputs"]["mode"] == ("variance_preserving_blend" if free_noise else "disabled")
    assert graph["24"]["inputs"]["width"] == ["80", 0]
    assert graph["24"]["inputs"]["height"] == ["81", 0]
    if variant == "resume_effects":
        assert not {"1", "9", "10", "11", "12", "13", "21", "41", "43", "45", "50", "83"} & set(graph)
        assert graph["60"]["inputs"]["expected_stage"] == "manual_first"
        assert graph["29"]["inputs"]["latent_image"] == ["60", 0]
    else:
        assert not {"22", "24", "26", "27", "28", "29", "42", "44", "84"} & ancestors(graph, "13")
        assert graph["11"]["inputs"]["noise_seed"] == graph["27"]["inputs"]["noise_seed"]
        assert graph["83"]["inputs"]["segment_index"] == graph["84"]["inputs"]["segment_index"]
        if variant == "minimal":
            assert graph["29"]["inputs"]["latent_image"] == ["13", 0]
        else:
            assert graph["45"]["inputs"]["av_latent"] == (["50", 0] if variant == "save_effects" else ["13", 0])
            assert graph["29"]["inputs"]["latent_image"] == ["45", 0]
    assert graph["26"]["inputs"]["av_latent"] == graph["29"]["inputs"]["latent_image"]
