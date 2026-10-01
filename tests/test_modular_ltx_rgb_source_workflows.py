"""Cold graphs must really remove source decoding/encoding/lifting dependencies."""
from copy import deepcopy

import pytest

from tools import build_modular_ltx_rgb_source_workflows as builder


@pytest.mark.parametrize("name,graph", list(builder.generated().items()))
def test_all_six_source_graphs_keep_separate_controls_and_precise_delivery(name, graph):
    by_type = {node["type"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    def edge(kind, field):
        item = next(item for item in by_type[kind]["inputs"] if item["name"] == field)
        return links[item["link"]][1:3]
    if name.endswith("freeze_source"):
        assert builder.SAVE in by_type
        assert not ({"UNETLoader", "CLIPLoader", "CFGGuider", "SamplerCustomAdvanced", builder.BIND, builder.AUDIT} & by_type.keys())
        assert {"VAEEncode", "LTXVLatentUpsampler", "LoadVideo"}.issubset(by_type)
        return
    storage = by_type[builder.LOAD if name.endswith("resume_ltx") else builder.SAVE]
    for slot, (field, _) in enumerate(builder.DATA):
        assert edge(builder.BIND, field) == [storage["id"], slot]
    assert edge(builder.AUDIT, "ltx_latent") == [by_type[builder.BIND]["id"], 4]
    assert edge("SamplerCustomAdvanced", "latent_image") == [by_type[builder.BIND]["id"], 4]
    assert edge("MiniMaxH3OutputTrimT8", "audio") == [by_type[builder.AUDIT]["id"], 1]
    assert edge("MiniMaxH3OutputTrimT8", "duration_seconds") == [storage["id"], 6]
    assert edge("CreateVideo", "bit_depth") == [storage["id"], 7]
    assert edge("CreateVideo", "fps") == edge("MiniMaxH3OutputTrimT8", "fps") == [storage["id"], 5]
    if name.endswith("resume_ltx"):
        assert not ({"LoadVideo", "GetVideoComponents", "VAELoader", "VAEEncode", "LTXVLatentUpsampler",
                     "LatentUpscaleModelLoader", "MiniMaxH3SolEngineDraftToLTXT8Advanced", builder.SAVE} & by_type.keys())
        assert storage["widgets_values"] == ["", ""]


def test_builder_is_deterministic_and_does_not_change_its_source():
    originals = builder.original_graphs()
    before = deepcopy(originals)
    assert builder.generated() == builder.generated()
    for graph in originals.values():
        for mode in ("full_save", "freeze_source", "resume_ltx"):
            builder.build(graph, mode)
    assert originals == before
    with pytest.raises(ValueError, match="Unknown"):
        builder.build(next(iter(originals.values())), "silent_rerun")


@pytest.mark.parametrize("name,original", list(builder.generated().items()))
def test_isolated_media_opt_in_changes_only_explicit_reader_writer(name, original):
    graph = builder.generated(isolated_media=True)[name]
    assert graph["links"] == original["links"]
    assert {n["id"] for n in graph["nodes"]} == {n["id"] for n in original["nodes"]}
    originals = {n["id"]: n for n in original["nodes"]}
    for node in graph["nodes"]:
        before = originals[node["id"]]
        if before["type"] == "GetVideoComponents":
            assert node["type"] == builder.SERIAL_READER and node["outputs"][:5] == before["outputs"]
        elif before["type"] == "SaveVideo":
            assert node["type"] == builder.ISOLATED_WRITER and node["outputs"][:1] == before["outputs"]
            assert node["widgets_values"] == [before["widgets_values"][0], 600, 16., 2.]
        else:
            assert node == before
        assert node["inputs"] == before.get("inputs")
    assert builder.generated()[name] == original
