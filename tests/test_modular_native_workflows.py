"""Explicit graph structure and the absence of hidden LOW execution on restore."""
import pytest

from tools.build_modular_native_dual_workflow import split_graph, VARIANTS
from test_modular_workflows import ancestors


@pytest.mark.parametrize("coarse", [4, 20])
@pytest.mark.parametrize("refine", [3, 4, 5])
@pytest.mark.parametrize("variant", VARIANTS)
def test_all_legacy_variants_have_genuinely_editable_stages(coarse, refine, variant):
    graph = split_graph(coarse, refine, variant)
    assert not any("FastH3V2" in item["class_type"] or "Loop" in item["class_type"] for item in graph.values())
    assert "62" not in graph
    setups = [item["inputs"]["stage"] for item in graph.values()
              if item["class_type"] == "MiniMaxH3NativeDualStageSetupEXPT8"]
    assert graph["26"]["inputs"]["stage"] == f"dual_high_{refine}"
    assert graph["25"]["inputs"]["first_pass_steps"] == str(coarse)
    assert graph["25"]["inputs"]["second_audio_source"] == "auto"
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["71"]["inputs"]["model"] == ["22", 0]
    if variant == "resume_effects":
        assert setups == [f"dual_high_{refine}"]
        assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50", "70"} & set(graph)
        assert graph["22"]["class_type"] == "UNETLoader"
        assert graph["60"]["inputs"]["expected_stage"] == f"dual_low_{coarse}"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 1]
    else:
        assert setups == [f"dual_low_{coarse}", f"dual_high_{refine}"]
        assert not {"22", "24", "25", "26", "27", "28", "29", "42", "44", "71"} & ancestors(graph, "13")
        assert ("70" in ancestors(graph, "13")) == (coarse == 4)
        assert "13" in ancestors(graph, "29")
        assert graph["23"]["inputs"]["av_latent"] == (["45", 0] if variant != "minimal" else ["13", 1])
    if variant != "minimal":
        if variant != "resume_effects":
            assert graph["45"]["inputs"]["av_latent"] == ["13", 1]
        assert graph["26"]["inputs"]["model"] == ["24", 0]
        assert graph["25"]["inputs"]["positive"] == ["24", 1]
        assert graph["25"]["inputs"]["highres_template"] == ["24", 2]
        assert graph["28"]["inputs"]["model"] == ["44", 0]
        assert graph["42"]["inputs"]["mode"] == "report_only"
    samplers = [item for item in graph.values() if item["class_type"] in
                ("SamplerCustomAdvanced", "MiniMaxH3StageSamplerEXPT8")]
    assert len(samplers) == (1 if variant == "resume_effects" else 2)
