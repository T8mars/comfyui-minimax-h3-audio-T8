"""Formal S04 PDD graphs retain independent stages and source-only examples."""

import asyncio
from copy import deepcopy
import json

import pytest

from tools.build_formal_pdd_split_workflows import (
    DESTINATION, FILES, FIRST_IMAGE, GLOBAL_PROMPT, LAST_IMAGE,
    REFERENCE_IMAGE, build_suite, graph_for, load_info,
)


@pytest.mark.parametrize("key,filename", FILES.items())
def test_formal_pdd_stages_effects_and_cold_boundary(key, filename):
    base, variant = key
    graph = graph_for(base, variant)
    kinds = [node["class_type"] for node in graph.values()]
    cold = variant == "resume_effects"
    assert kinds.count("MiniMaxH3StageSamplerEXPT8") == (1 if cold else 2)
    assert kinds.count("MiniMaxH3PDDStageSetupEXPT8") == (1 if cold else 2)
    assert kinds.count("MiniMaxH3StageEAVConfigEXPT8") == (1 if cold else 2)
    assert kinds.count("MiniMaxH3PromptRelayPlanT8Advanced") == (1 if cold else 2)
    assert kinds.count("MiniMaxH3StageSaveEXPT8") == (1 if cold else 2)
    assert all(node["inputs"]["global_prompt"] == GLOBAL_PROMPT for node in graph.values()
               if node["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced")
    assert all(node["inputs"]["mode"] == "report_only" for node in graph.values()
               if node["class_type"] == "MiniMaxH3StageEAVConfigEXPT8")
    assert graph["95"]["inputs"]["image"] == (
        FIRST_IMAGE if base == "FL2VA" else REFERENCE_IMAGE)
    if base == "FL2VA":
        assert graph["96"]["inputs"]["image"] == LAST_IMAGE
    assert graph["25"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    if cold:
        assert not {"1", "9", "10", "11", "12", "13", "40", "41", "43", "45", "50", "90"} & set(graph)
        assert graph["22"]["class_type"] == "UNETLoader"
        assert graph["23"]["inputs"]["av_latent"] == ["60", 1]
        assert graph["60"]["inputs"]["expected_stage"] == "pdd_low_0_4"
    else:
        assert graph["10"]["inputs"]["stage"] == "pdd_low_0_4"
        assert graph["26"]["inputs"]["stage"] == "pdd_high_4_8"
        assert graph["23"]["inputs"]["av_latent"] == ["45", 0]
    assert (DESTINATION / filename).is_file()


def test_formal_pdd_rejects_unreviewed_recipes():
    with pytest.raises(ValueError, match="reviewed PDD"):
        graph_for("Other", "save_effects")
    with pytest.raises(ValueError, match="reviewed PDD"):
        graph_for("FL2VA", "minimal")


def test_saved_pdd_suite_matches_live_builder_and_core_static_validation():
    info = load_info()
    import execution

    suite = build_suite(info)
    for key, filename in FILES.items():
        path = DESTINATION / filename
        saved = json.loads(path.read_text(encoding="utf-8"))
        current = suite[key]["workflow"]
        assert {name: value for name, value in saved.items() if name != "id"} == {
            name: value for name, value in current.items() if name != "id"}
        assert suite[key]["audit"]["nodes"] == len(suite[key]["graph"])
        assert saved["extra"]["t8_split_example"] == {
            "schema": "t8.pdd.split-example.v1", "route": "S04", "base": key[0],
            "variant": key[1], "status": "experimental_importable_not_quality_accepted",
        }
        raw = path.read_text(encoding="utf-8")
        for private in ("10A.jpg", "bright concert hall", "artifacts/development", "F:\\"):
            assert private not in raw
        validation_graph = deepcopy(suite[key]["graph"])
        for node in validation_graph.values():
            if node["class_type"] == "LoadImage":
                node["inputs"]["image"] = "10A.jpg"
        valid, error, *_ = asyncio.run(execution.validate_prompt(
            "formal-pdd-regression", validation_graph, None))
        assert valid is True and error is None


@pytest.mark.parametrize("filename", FILES.values())
def test_formal_pdd_frontend_columns_do_not_overlap(filename):
    workflow = json.loads((DESTINATION / filename).read_text(encoding="utf-8"))
    columns = {}
    for node in workflow["nodes"]:
        columns.setdefault(node["pos"][0], []).append(node)
    for column in columns.values():
        ordered = sorted(column, key=lambda node: (node["pos"][1], node["id"]))
        for before, after in zip(ordered, ordered[1:]):
            assert before["pos"][1] + before["size"][1] + 48 <= after["pos"][1], (
                filename, before["id"], after["id"])
