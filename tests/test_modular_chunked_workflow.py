"""Opt-in frontend candidate construction never rewires old JSON in place."""
import hashlib
import json
import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from tools.build_modular_chunked_workflow import (
    TEMPLATES, split_workflow, split_api_graph, v1_source_from_v2,
    v3_source_from_v4, append_second_frontend_segment, append_second_api_segment,
    add_external_eav_frontend, add_external_eav_api,
    add_external_relay_frontend, add_external_relay_api,
)


@pytest.mark.parametrize("variant", ["v2", "v3", "v4"])
def test_full_clip_candidate_has_explicit_first_lift_pass2_and_original_template(variant):
    path = TEMPLATES[variant]
    before = path.read_bytes()
    source_sha = hashlib.sha256(before).hexdigest()
    original = json.loads(before)
    if variant == "v3":
        original = v3_source_from_v4(original)
    graph = split_workflow(original)
    assert path.read_bytes() == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_sha
    assert all(node["type"] != "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"
               for node in graph["nodes"])
    if variant == "v3":
        assert not any(node["type"] in ("LoadImageMask", "SetLatentNoiseMask")
                       for node in graph["nodes"])
        assert sum(node["type"] == "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
                   for node in graph["nodes"]) == 1
    kinds = {node["type"]: node for node in graph["nodes"]}
    slice_node = kinds["MiniMaxH3ChunkedSourceSegmentEXPT8"]
    prepare = kinds["MiniMaxH3ChunkedPass2PrepareEXPT8"]
    lift = kinds["MiniMaxH3ChunkedLearnedLiftEXPT8"]
    pass2 = kinds["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert slice_node["widgets_values"] == [0]
    assert pass2["widgets_values"] == [1.0]
    assert pass2["inputs"][10]["link"] is None
    links = {link[0]: link for link in graph["links"]}
    def upstream(node, input_name):
        item = next(item for item in node["inputs"] if item["name"] == input_name)
        return links[item["link"]][1:3]
    assert upstream(lift, "source_segment") == [slice_node["id"], 0]
    assert upstream(lift, "pass2_context") == [prepare["id"], 0]
    assert upstream(pass2, "lifted_segment") == [lift["id"], 0]
    assert upstream(pass2, "source_segment") == [slice_node["id"], 0]
    assert upstream(pass2, "pass2_context") == [prepare["id"], 0]
    assert upstream(pass2, "model") != upstream(pass2, "positive")
    old_executor = next(node for node in original["nodes"]
                        if node["type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced")
    old_links = {link[0]: link for link in original["links"]}
    for name in ("model", "sampler", "sigmas"):
        old_input = next(item for item in old_executor["inputs"] if item["name"] == name)
        assert upstream(pass2, name) != old_links[old_input["link"]][1:3]
    assert upstream(pass2, "positive") == old_links[
        next(item for item in old_executor["inputs"] if item["name"] == "conditioning")["link"]
    ][1:3]
    first_sampler = next(node for node in graph["nodes"] if node["type"] == "SamplerCustomAdvanced")
    high_branch = {upstream(pass2, name)[0] for name in ("model", "sampler", "sigmas")}
    node_by_id = {node["id"]: node for node in graph["nodes"]}

    def ancestors(node_id):
        parents = {links[item["link"]][1] for item in node_by_id[node_id].get("inputs", [])
                   if item.get("link") is not None}
        return parents | set().union(*(ancestors(parent) for parent in parents))

    assert not high_branch & ancestors(first_sampler["id"])
    all_node_ids = {node["id"] for node in graph["nodes"]}
    for link in graph["links"]:
        assert link[1] in all_node_ids and link[3] in all_node_ids
        target = next(node for node in graph["nodes"] if node["id"] == link[3])
        assert target["inputs"][link[4]]["link"] == link[0]


def test_live_core_validates_public_split_stage_ports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes as core_nodes
    from comfy_extras.nodes_preview_any import PreviewAny
    from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
    from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES

    for cls in (*SOURCE_NODES, *STAGE_NODES, PreviewAny):
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)

    class Source:
        RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "NOISE", "SAMPLER", "SIGMAS",
                        "T8_H3_CHUNKED_TWO_PASS_PLAN")
        FUNCTION = "execute"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, "T8ChunkedTestSource", Source)
    graph = {
        "1": {"class_type": "T8ChunkedTestSource", "inputs": {}},
        "2": {"class_type": "MiniMaxH3ChunkedSourceSegmentEXPT8", "inputs": {
            "first_pass_latent": ["1", 2], "plan": ["1", 6], "segment_index": 0}},
        "3": {"class_type": "MiniMaxH3ChunkedPass2PrepareEXPT8", "inputs": {
            "first_pass_latent": ["1", 2], "plan": ["1", 6], "noise": ["1", 3]}},
        "4": {"class_type": "MiniMaxH3ChunkedLearnedLiftEXPT8", "inputs": {
            "source_segment": ["2", 0], "segment_spec": ["2", 1],
            "pass2_context": ["3", 0], "plan": ["1", 6]}},
        "5": {"class_type": "MiniMaxH3ChunkedPass2SegmentEXPT8", "inputs": {
            "model": ["1", 0], "positive": ["1", 1],
            "source_segment": ["2", 0], "lifted_segment": ["4", 0],
            "segment_spec": ["2", 1], "pass2_context": ["3", 0],
            "plan": ["1", 6], "noise": ["1", 3], "sampler": ["1", 4],
            "sigmas": ["1", 5], "cfg": 1.0}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 2]}},
    }
    result = asyncio.run(execution.validate_prompt("chunked-split-ports", deepcopy(graph), None))
    assert result[0] and not result[3] and result[2] == ["6"], result


@pytest.mark.parametrize("variant,mode,builder_name", [
    ("v2", "chunked_two_pass_global_noise", "_chunked_prompt"),
    ("v3", "chunked_two_pass_low_sigma", "_chunked_low_sigma_prompt"),
    ("v4", "chunked_two_pass_masked_low_sigma", "_chunked_masked_low_sigma_prompt"),
])
def test_api_split_mirrors_frontend_stage_topology(variant, mode, builder_name):
    from tools import run_community_update_real_validation as legacy_api

    args = legacy_api._parser().parse_args(["--mode", mode])
    old, _ = getattr(legacy_api, builder_name)(args, "STATIC")
    graph = split_api_graph(old)
    assert old != graph
    assert sum(item["class_type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"
               for item in graph.values()) == 0
    by_type = {item["class_type"]: (key, item) for key, item in graph.items()}
    source_id, _ = by_type["MiniMaxH3ChunkedSourceSegmentEXPT8"]
    context_id, _ = by_type["MiniMaxH3ChunkedPass2PrepareEXPT8"]
    lift_id, lift = by_type["MiniMaxH3ChunkedLearnedLiftEXPT8"]
    pass_id, pass2 = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert lift["inputs"]["source_segment"] == [source_id, 0]
    assert lift["inputs"]["pass2_context"] == [context_id, 0]
    assert pass2["inputs"]["lifted_segment"] == [lift_id, 0]
    assert pass2["inputs"]["source_segment"] == [source_id, 0]
    assert pass2["inputs"]["pass2_context"] == [context_id, 0]
    old_executor = next(item for item in old.values()
                        if item["class_type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced")
    for name in ("model", "sampler", "sigmas"):
        assert pass2["inputs"][name] != old_executor["inputs"][name]
    assert pass2["inputs"]["positive"] == old_executor["inputs"]["conditioning"]
    assert any(item["inputs"].get("source") == [pass_id, 2]
               for item in graph.values() if item["class_type"] == "PreviewAny")


def test_v1_native_aligned_three_segment_frontend_and_api_chain():
    from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
    from h3_audio_t8_pkg import core
    from tools import run_community_update_real_validation as legacy_api

    assert core.align_frame_count(51) == 56
    assert core.video_latent_t(56) == 17
    segments, frames = legacy.compute_temporal_segments(core.video_latent_t(56), 34, 17)
    assert frames == 56 and len(segments) == 3
    source = v1_source_from_v2(json.loads(TEMPLATES["v1"].read_text(encoding="utf-8")))
    source_nodes = {node["id"]: node for node in source["nodes"]}
    assert source_nodes[7]["widgets_values"][3] == 56
    assert source_nodes[13]["widgets_values"][3] == 56
    graph = append_second_frontend_segment(split_workflow(source))
    graph = append_second_frontend_segment(graph, segment_index=2)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    sources = [node for node in graph["nodes"] if node["type"] == "MiniMaxH3ChunkedSourceSegmentEXPT8"]
    passes = [node for node in graph["nodes"] if node["type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert len(sources) == len(passes) == 3
    assert sorted(node["widgets_values"][0] for node in sources) == [0, 1, 2]
    for first, second in zip(passes, passes[1:]):
        prior = next(item for item in second["inputs"] if item["name"] == "previous_result")
        assert links[prior["link"]][1:3] == [first["id"], 1]
    report_link = next(link for link in graph["links"]
                       if nodes[link[3]]["type"] == "PreviewAny" and link[5] == "STRING"
                       and link[1] in {item["id"] for item in passes})
    assert report_link[1:3] == [passes[-1]["id"], 2]
    assert not any(node["type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"
                   for node in graph["nodes"])
    args = legacy_api._parser().parse_args([
        "--mode", "chunked_two_pass", "--frame-count", "56",
        "--temporal-chunk-frames", "34", "--temporal-overlap-frames", "17",
    ])
    old_api, _ = legacy_api._chunked_prompt(args, "STATIC")
    api = append_second_api_segment(split_api_graph(old_api))
    api = append_second_api_segment(api, segment_index=2)
    api_passes = [(key, item) for key, item in api.items()
                  if item["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert len(api_passes) == 3
    assert [item[1]["inputs"]["previous_result"] for item in api_passes[1:]] == [
        [api_passes[0][0], 1], [api_passes[1][0], 1],
    ]
    assert any(item["inputs"].get("source") == [api_passes[2][0], 2]
               for item in api.values() if item["class_type"] == "PreviewAny")


@pytest.mark.parametrize("variant", ["v1", "v2", "v3", "v4"])
def test_external_eav_candidate_has_one_real_apply_audit_per_pass2(variant):
    source = json.loads(TEMPLATES[variant].read_text(encoding="utf-8"))
    if variant == "v1":
        source = v1_source_from_v2(source)
    elif variant == "v3":
        source = v3_source_from_v4(source)
    graph = split_workflow(source)
    if variant == "v1":
        graph = append_second_frontend_segment(graph)
        graph = append_second_frontend_segment(graph, segment_index=2)
    candidate = add_external_eav_frontend(graph)
    links = {link[0]: link for link in candidate["links"]}
    nodes = {node["id"]: node for node in candidate["nodes"]}
    passes = [node for node in candidate["nodes"]
              if node["type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    applies = [node for node in candidate["nodes"]
               if node["type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"]
    audits = [node for node in candidate["nodes"]
              if node["type"] == "MiniMaxH3ChunkedPass2EAVAuditEXPT8"]
    assert len(passes) == len(applies) == len(audits) == (3 if variant == "v1" else 1)
    assert sum(node["type"] == "MiniMaxH3StageEAVConfigEXPT8"
               for node in candidate["nodes"]) == 1
    for pass2 in passes:
        model_link = links[next(item for item in pass2["inputs"]
                                if item["name"] == "model")["link"]]
        apply = nodes[model_link[1]]
        assert apply["type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
        assert any(links[next(item for item in audit["inputs"]
                              if item["name"] == "segment_result")["link"]][1] == pass2["id"]
                   for audit in audits)
    assert candidate["last_node_id"] == max(nodes)
    assert candidate["last_link_id"] == max(links)


def test_external_eav_api_candidate_keeps_each_pass2_independent():
    from tools import run_community_update_real_validation as legacy_api

    args = legacy_api._parser().parse_args([
        "--mode", "chunked_two_pass", "--frame-count", "56",
        "--temporal-chunk-frames", "34", "--temporal-overlap-frames", "17",
    ])
    original, _ = legacy_api._chunked_prompt(args, "STATIC")
    graph = append_second_api_segment(split_api_graph(original))
    graph = add_external_eav_api(append_second_api_segment(graph, segment_index=2))
    passes = [(key, node) for key, node in graph.items()
              if node["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert len(passes) == 3
    assert passes[1][1]["inputs"]["previous_result"] == [passes[0][0], 1]
    assert passes[2][1]["inputs"]["previous_result"] == [passes[1][0], 1]
    for pass_id, pass2 in passes:
        apply_id = pass2["inputs"]["model"][0]
        assert graph[apply_id]["class_type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
        assert any(node["class_type"] == "MiniMaxH3ChunkedPass2EAVAuditEXPT8"
                   and node["inputs"]["segment_result"] == [pass_id, 1]
                   and node["inputs"]["runtime"] == [apply_id, 1]
                   for node in graph.values())


@pytest.mark.parametrize("variant", ["v2", "v3", "v4"])
def test_external_relay_frontend_has_real_paired_stage_and_audit(variant):
    original = json.loads(TEMPLATES[variant].read_text(encoding="utf-8"))
    if variant == "v3":
        original = v3_source_from_v4(original)
    graph = add_external_relay_frontend(split_workflow(original))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    by_type = {node["type"]: node for node in graph["nodes"]}
    plan = by_type["MiniMaxH3PromptRelayPlanT8Advanced"]
    relay_cond = by_type["MiniMaxH3PromptRelayConditioningT8Advanced"]
    bind = by_type["MiniMaxH3ChunkedPass2RelayBindEXPT8"]
    audit = by_type["MiniMaxH3ChunkedPass2RelayAuditEXPT8"]
    pass2 = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    assert relay_cond["widgets_values"][-2] == "apply_exp"
    assert plan["widgets_values"][2] == (22 if variant == "v2" else 124)

    def upstream(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        return links[link_id][1:3]

    assert upstream(relay_cond, "prompt_relay_plan") == [plan["id"], 0]
    assert upstream(bind, "relay_model") == [relay_cond["id"], 0]
    assert upstream(bind, "relay_positive") == [relay_cond["id"], 1]
    assert upstream(bind, "relay_av_latent") == [relay_cond["id"], 2]
    assert upstream(pass2, "model") == [bind["id"], 0]
    assert upstream(pass2, "positive") == [bind["id"], 1]
    assert upstream(audit, "segment_result") == [pass2["id"], 1]
    assert upstream(audit, "runtime") == [bind["id"], 2]
    assert upstream(audit, "segment_spec") == upstream(pass2, "segment_spec")
    for name in ("source_segment", "lifted_segment", "segment_spec",
                 "pass2_context", "plan", "sigmas"):
        assert upstream(bind, name) == upstream(pass2, name)
    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(links)


def test_external_relay_api_candidate_matches_one_pass2_and_rejects_temporal_chunks():
    from tools import run_community_update_real_validation as legacy_api

    args = legacy_api._parser().parse_args(["--mode", "chunked_two_pass_global_noise"])
    original, _ = legacy_api._chunked_prompt(args, "STATIC")
    graph = add_external_relay_api(split_api_graph(original))
    by_type = {node["class_type"]: (key, node) for key, node in graph.items()}
    pass_id, pass2 = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    plan_id, plan = by_type["MiniMaxH3PromptRelayPlanT8Advanced"]
    cond_id, cond = by_type["MiniMaxH3PromptRelayConditioningT8Advanced"]
    bind_id, bind = by_type["MiniMaxH3ChunkedPass2RelayBindEXPT8"]
    _audit_id, audit = by_type["MiniMaxH3ChunkedPass2RelayAuditEXPT8"]
    assert cond["inputs"]["prompt_relay_plan"] == [plan_id, 0]
    assert plan["inputs"]["length"] == 124
    assert cond["inputs"]["execution_mode"] == "apply_exp"
    assert bind["inputs"]["relay_model"] == [cond_id, 0]
    assert bind["inputs"]["relay_positive"] == [cond_id, 1]
    assert pass2["inputs"]["model"] == [bind_id, 0]
    assert pass2["inputs"]["positive"] == [bind_id, 1]
    assert audit["inputs"]["segment_result"] == [pass_id, 1]
    with pytest.raises(ValueError, match="exactly one PASS2"):
        add_external_relay_api(append_second_api_segment(split_api_graph(original)))


@pytest.mark.parametrize("variant", ["v2", "v3", "v4"])
def test_external_relay_plus_eav_candidate_has_single_combined_audit(variant):
    source = json.loads(TEMPLATES[variant].read_text(encoding="utf-8"))
    if variant == "v3":
        source = v3_source_from_v4(source)
    graph = add_external_eav_frontend(
        add_external_relay_frontend(split_workflow(source), with_audit=False),
    )
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    by_type = {node["type"]: node for node in graph["nodes"]}
    pass2 = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    eav = by_type["MiniMaxH3ChunkedPass2EAVApplyEXPT8"]
    bind = by_type["MiniMaxH3ChunkedPass2RelayBindEXPT8"]
    audit = by_type["MiniMaxH3ChunkedPass2EAVAuditEXPT8"]

    def upstream(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        return links[link_id][1:3]

    assert upstream(eav, "model") == [bind["id"], 0]
    assert upstream(pass2, "model") == [eav["id"], 0]
    assert upstream(pass2, "positive") == [bind["id"], 1]
    assert upstream(audit, "runtime") == [eav["id"], 1]
    assert upstream(audit, "segment_result") == [pass2["id"], 1]
    assert "MiniMaxH3ChunkedPass2RelayAuditEXPT8" not in by_type
    assert graph["last_node_id"] == max(nodes)
    assert graph["last_link_id"] == max(links)

    from tools import run_community_update_real_validation as legacy_api
    mode = {"v2": "chunked_two_pass_global_noise",
            "v3": "chunked_two_pass_low_sigma",
            "v4": "chunked_two_pass_masked_low_sigma"}[variant]
    builder = {"v2": legacy_api._chunked_prompt,
               "v3": legacy_api._chunked_low_sigma_prompt,
               "v4": legacy_api._chunked_masked_low_sigma_prompt}[variant]
    old_api, _ = builder(legacy_api._parser().parse_args(["--mode", mode]), "STATIC")
    api = add_external_eav_api(
        add_external_relay_api(split_api_graph(old_api), with_audit=False),
    )
    api_pass = next(item for item in api.values()
                    if item["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8")
    eav_id = api_pass["inputs"]["model"][0]
    assert api[eav_id]["class_type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
    bind_id = api[eav_id]["inputs"]["model"][0]
    assert api[bind_id]["class_type"] == "MiniMaxH3ChunkedPass2RelayBindEXPT8"
    assert not any(item["class_type"] == "MiniMaxH3ChunkedPass2RelayAuditEXPT8"
                   for item in api.values())
