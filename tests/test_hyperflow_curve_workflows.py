"""Saved public graphs preserve actual stage/effect paths and cold isolation."""
import json

import pytest

from tools.build_hyperflow_curve_workflows import DESTINATION, FILES, graph_for
from tools.audit_modular_new_sampling import _reaches, _workflow_graph
from h3_audio_t8_pkg.modular_sampling.catalogue import CURVE_NODES, ROUTES
from h3_audio_t8_pkg.nodes_hyperflow_curve_exp import NODES


@pytest.mark.parametrize("variant", FILES)
def test_public_three_saved_typed_frontend_graphs_and_cold_no_head(variant):
    graph = graph_for(variant)
    kinds = [node["class_type"] for node in graph.values()]
    workflow = json.loads((DESTINATION / FILES[variant]).read_text(encoding="utf8"))
    assert workflow["extra"]["t8_split_example"]["route"] == "S30"
    actual = [node["type"] for node in workflow["nodes"] if node["type"] != "MarkdownNote"]
    assert actual == kinds
    assert not any(kind.startswith("MiniMaxH3HyperFlow") and "Curve" not in kind for kind in kinds)
    if variant == "full_save":
        assert "MiniMaxH3HyperFlowCurveHeadStageEXPT8" in kinds
        assert graph["29"]["inputs"]["continuous_boundary"] == ["50", 0]
    else:
        assert "MiniMaxH3HyperFlowCurveHeadStageEXPT8" not in kinds
        assert "RandomNoise" not in kinds
        assert "205" not in graph and "201" not in graph
    if variant == "cold_delivery":
        assert "UNETLoader" not in kinds and "CLIPLoader" not in kinds
        assert not any("Effects" in kind or "StageEXPT8" in kind or "FitLoad" in kind for kind in kinds)
    else:
        assert graph["29"]["inputs"]["model"] == ["213", 0]
        assert graph["211"]["inputs"]["model"] == ["24", 0]


def test_s30_actual_registered_types_and_both_external_effect_paths():
    assert CURVE_NODES == tuple(cls.__name__ for cls in NODES)
    assert ROUTES[-1].id == "S30" and set(CURVE_NODES) <= set(ROUTES[-1].public_nodes)
    for variant in ("full_save", "cold_tail"):
        file = DESTINATION / FILES[variant]
        by_id, edges = _workflow_graph(DESTINATION, file.name)
        raw = json.loads(file.read_text(encoding="utf8"))
        api = graph_for(variant)
        mapping = {key: node["id"] for key, node in zip(api, [node for node in raw["nodes"] if node["type"] != "MarkdownNote"], strict=True)}
        assert by_id[mapping["29"]] == "MiniMaxH3HyperFlowCurveTailStageEXPT8"
        for key in ("24", "213", "211", "50"):
            assert _reaches(edges, mapping[key], mapping["29"])
        if variant == "full_save":
            for key in ("9", "203", "201"):
                assert _reaches(edges, mapping[key], mapping["13"])
            assert _reaches(edges, mapping["13"], mapping["29"])
