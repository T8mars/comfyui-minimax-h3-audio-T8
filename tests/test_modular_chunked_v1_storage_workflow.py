"""Fixed three-segment v1 freeze/resume drafts preserve stage dependencies."""
from collections import Counter
from copy import deepcopy

from tools.build_modular_chunked_v1_relay_workflow import build_pair
from tools.build_modular_chunked_v1_storage_workflow import (
    NATIVE_LOAD, NATIVE_SAVE, SEGMENT_LOAD, SEGMENT_SAVE,
    build_storage_pair, freeze_api, freeze_frontend, resume_api, resume_frontend,
)


PASS = "MiniMaxH3ChunkedPass2SegmentEXPT8"


def _frontend_origin(graph, node_id, input_name):
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    item = next(item for item in nodes[node_id]["inputs"] if item["name"] == input_name)
    link = links[item["link"]]
    return link[1], link[2]


def _assert_frontend_links(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    assert len(nodes) == len(graph["nodes"])
    assert len(links) == len(graph["links"])
    for link_id, source_id, source_slot, target_id, target_slot, _dtype in graph["links"]:
        assert link_id in nodes[source_id]["outputs"][source_slot]["links"]
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
    for node in graph["nodes"]:
        for item in node["inputs"]:
            assert item["link"] is None or item["link"] in links


def _assert_api_links(graph):
    for node in graph.values():
        for value in node["inputs"].values():
            if (isinstance(value, list) and len(value) == 2 and
                    isinstance(value[0], str) and value[0].isdigit() and
                    isinstance(value[1], int)):
                assert value[0] in graph
                assert value[1] >= 0


def test_freeze_and_resume_are_additive_pruned_graphs_without_first_segment_replay():
    base_frontend, base_api = build_pair()
    unchanged_frontend, unchanged_api = deepcopy(base_frontend), deepcopy(base_api)
    freeze_frontend_graph = freeze_frontend(base_frontend)
    freeze_api_graph = freeze_api(base_api)
    resume_frontend_graph = resume_frontend(base_frontend)
    resume_api_graph = resume_api(base_api)
    assert base_frontend == unchanged_frontend
    assert base_api == unchanged_api

    freeze_kinds = Counter(node["type"] for node in freeze_frontend_graph["nodes"])
    resume_kinds = Counter(node["type"] for node in resume_frontend_graph["nodes"])
    assert freeze_kinds[PASS] == 1
    assert freeze_kinds[NATIVE_SAVE] == freeze_kinds[SEGMENT_SAVE] == 1
    assert resume_kinds[PASS] == 2
    assert resume_kinds[NATIVE_LOAD] == resume_kinds[SEGMENT_LOAD] == 1
    assert resume_kinds[NATIVE_SAVE] == resume_kinds[SEGMENT_SAVE] == 0
    assert 12 not in {node["id"] for node in resume_frontend_graph["nodes"]}
    assert 23 not in {node["id"] for node in resume_frontend_graph["nodes"]}
    assert "12" not in resume_api_graph and "25" not in resume_api_graph
    assert freeze_api_graph["40"]["inputs"]["confirm_save"] is False
    assert freeze_api_graph["41"]["inputs"]["confirm_save"] is False
    assert resume_api_graph["40"]["inputs"]["checkpoint_path"] == ""
    assert resume_api_graph["40"]["inputs"]["expected_file_sha256"] == ""
    assert resume_api_graph["41"]["inputs"]["artifact_path"] == ""
    assert resume_api_graph["41"]["inputs"]["artifact_sha256"] == ""

    for key in (20, 21, 26, 29):
        assert _frontend_origin(resume_frontend_graph, key, "first_pass_latent") == (40, 0)
        assert resume_api_graph[str(key)]["inputs"]["first_pass_latent"] == ["40", 0]
    for key in (28, 36):
        assert _frontend_origin(resume_frontend_graph, key, "previous_result") == (41, 1)
        assert resume_api_graph[str(key)]["inputs"]["previous_result"] == ["41", 1]
    assert _frontend_origin(resume_frontend_graph, 31, "previous_result") == (28, 1)
    assert resume_api_graph["31"]["inputs"]["previous_result"] == ["28", 1]
    assert _frontend_origin(resume_frontend_graph, 38, "previous_result") == (28, 1)
    assert resume_api_graph["38"]["inputs"]["previous_result"] == ["28", 1]
    for graph in (freeze_frontend_graph, resume_frontend_graph):
        _assert_frontend_links(graph)
    for graph in (freeze_api_graph, resume_api_graph):
        _assert_api_links(graph)


def test_paired_builder_matches_separate_builders_and_preserves_base_route():
    (freeze_frontend_graph, freeze_api_graph), (resume_frontend_graph, resume_api_graph) = (
        build_storage_pair())
    assert len(freeze_frontend_graph["nodes"]) == len(freeze_api_graph) == 25
    assert len(resume_frontend_graph["nodes"]) == len(resume_api_graph) == 29
    assert [node["type"] for node in freeze_frontend_graph["nodes"] if node["id"] == 41] == (
        [SEGMENT_SAVE])
    assert [node["type"] for node in resume_frontend_graph["nodes"] if node["id"] == 41] == (
        [SEGMENT_LOAD])


def test_current_ui_pair_matches_saved_api_image_and_vhs_defaults_without_changing_v1():
    historic = build_storage_pair()
    current = build_storage_pair(current_ui=True)
    for (old_frontend, old_api), (new_frontend, new_api) in zip(historic, current, strict=True):
        assert new_api == old_api
        _assert_frontend_links(new_frontend)
        old_nodes = {node["id"]: node for node in old_frontend["nodes"]}
        new_nodes = {node["id"]: node for node in new_frontend["nodes"]}
        assert new_nodes[5]["widgets_values"] == [new_api["5"]["inputs"]["image"]]
        assert old_nodes[5]["widgets_values"] == ["10A.jpg"]
        if 19 in new_nodes:
            values = new_nodes[19]["widgets_values"]
            settings = new_api["19"]["inputs"]
            assert len(values) == 6
            assert values == [settings[name] for name in (
                "frame_rate", "loop_count", "filename_prefix", "format",
                "pingpong", "save_output")]
            assert len(old_nodes[19]["widgets_values"]) == 10
