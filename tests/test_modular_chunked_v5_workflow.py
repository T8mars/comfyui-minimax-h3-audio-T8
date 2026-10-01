"""Fixed 192-frame v5 split candidate wiring; frozen source is never rewritten."""
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from build_modular_chunked_v5_workflow import (  # noqa: E402
    TEMPLATE, add_eav_api, add_eav_frontend, add_relay_api, add_relay_frontend,
    split_api, split_frontend,
)
from run_chunked_parity_probe import build_graph  # noqa: E402


def _source():
    before = TEMPLATE.read_bytes()
    return json.loads(before), hashlib.sha256(before).hexdigest()


def test_frontend_v5_is_two_window_graph_with_independent_high_chain():
    original, digest = _source()
    candidate = split_frontend(original)
    nodes = {node["id"]: node for node in candidate["nodes"]}
    links = {link[0]: link for link in candidate["links"]}
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == digest
    assert 15 not in nodes and original["nodes"] != candidate["nodes"]
    assert len(nodes) == len(original["nodes"]) + 5
    assert len(nodes) == len({node["id"] for node in candidate["nodes"]})
    kinds = [node["type"] for node in candidate["nodes"]]
    assert kinds.count("MiniMaxH3ChunkedV5GlobalLiftEXPT8") == 1
    assert kinds.count("MiniMaxH3ChunkedV5PrepareEXPT8") == 1
    assert kinds.count("MiniMaxH3ChunkedV5PASS2WindowEXPT8") == 2
    windows = sorted((node for node in candidate["nodes"]
                      if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda node: node["widgets_values"][0])
    assert [node["widgets_values"] for node in windows] == [[0, 1.0], [1, 1.0]]
    prior = next(item for item in windows[1]["inputs"] if item["name"] == "previous_result")
    assert links[prior["link"]][1:3] == [windows[0]["id"], 1]
    assert next(item for item in windows[0]["inputs"]
                if item["name"] == "previous_result")["link"] is None
    # Both windows consume a clone of DualClock/LoRA, not the LOW branch.
    high_ids = set()
    for window in windows:
        model = next(item for item in window["inputs"] if item["name"] == "model")
        high_ids.add(links[model["link"]][1])
    assert len(high_ids) == 1 and 8 not in high_ids
    high_setup = nodes[high_ids.pop()]
    high_model = next(item for item in high_setup["inputs"] if item["name"] == "model")
    assert links[high_model["link"]][1] != 23
    for link_id, src, src_slot, dst, dst_slot, _type in candidate["links"]:
        assert link_id in nodes[src]["outputs"][src_slot]["links"]
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id


def test_api_v5_replaces_only_executor_and_rewires_outputs():
    original = build_graph()
    del original["26"]
    original["8"]["inputs"]["model"] = ["23", 0]
    frozen = deepcopy(original)
    candidate = split_api(original)
    assert original == frozen and "15" not in candidate
    windows = [(key, item) for key, item in candidate.items()
               if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"]
    assert len(windows) == 2
    first, second = windows
    assert first[1]["inputs"]["window_index"] == 0
    assert second[1]["inputs"]["window_index"] == 1
    assert second[1]["inputs"]["previous_result"] == [first[0], 1]
    assert candidate["18"]["inputs"]["av_latent"] == [second[0], 0]
    assert candidate["17"]["inputs"]["source"] == [second[0], 2]


def test_frontend_v5_rejects_different_window_geometry():
    original, _ = _source()
    next_graph = deepcopy(original)
    next(node for node in next_graph["nodes"] if node["id"] == 14)["widgets_values"][3] = 100
    try:
        split_frontend(next_graph)
    except ValueError as error:
        assert "frozen 192-frame" in str(error)
    else:
        raise AssertionError("A fixed two-window candidate must reject altered geometry")


def test_v5_frontend_external_eav_is_separate_for_both_joint_windows():
    original, digest = _source()
    candidate = add_eav_frontend(split_frontend(original))
    nodes = {node["id"]: node for node in candidate["nodes"]}
    links = {link[0]: link for link in candidate["links"]}
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == digest
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda node: node["widgets_values"][0])
    configs = [node for node in nodes.values() if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
    applies = [node for node in nodes.values() if node["type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"]
    audits = [node for node in nodes.values() if node["type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"]
    assert len(configs) == len(applies) == len(audits) == 2
    assert {node["widgets_values"][0] for node in applies} == {0, 1}
    for window in windows:
        model_id = next(item["link"] for item in window["inputs"] if item["name"] == "model")
        assert nodes[links[model_id][1]]["type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"
    second_apply = next(node for node in applies if node["widgets_values"][0] == 1)
    previous_link = next(item["link"] for item in second_apply["inputs"]
                         if item["name"] == "previous_result")
    assert links[previous_link][1:3] == [windows[0]["id"], 1]
    for link_id, src, src_slot, dst, dst_slot, _kind in candidate["links"]:
        assert link_id in nodes[src]["outputs"][src_slot]["links"]
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id


def test_v5_api_external_eav_has_one_config_and_audit_per_window():
    original = build_graph()
    del original["26"]
    original["8"]["inputs"]["model"] = ["23", 0]
    graph = add_eav_api(split_api(original))
    windows = [item for item in graph.values()
               if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"]
    applies = [item for item in graph.values()
               if item["class_type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"]
    audits = [item for item in graph.values()
              if item["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"]
    assert len(windows) == len(applies) == len(audits) == 2
    assert {item["inputs"]["window_index"] for item in applies} == {0, 1}
    assert len({tuple(item["inputs"]["eav_config"]) for item in applies}) == 2


def test_v5_frontend_relay_keeps_one_global_plan_and_two_local_projections():
    original, digest = _source()
    candidate = add_relay_frontend(split_frontend(original))
    nodes = {node["id"]: node for node in candidate["nodes"]}
    links = {link[0]: link for link in candidate["links"]}
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == digest
    plans = [node for node in nodes.values()
             if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"]
    conds = [node for node in nodes.values()
             if node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"]
    projects = [node for node in nodes.values()
                if node["type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8"]
    audits = [node for node in nodes.values()
              if node["type"] == "MiniMaxH3ChunkedV5RelayAuditEXPT8"]
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda node: node["widgets_values"][0])
    assert len(plans) == len(conds) == 1
    assert len(projects) == len(audits) == len(windows) == 2
    assert plans[0]["widgets_values"][2] == 192
    assert {node["widgets_values"][0] for node in projects} == {0, 1}
    for window in windows:
        for name, slot in (("model", 0), ("positive", 1)):
            link_id = next(item["link"] for item in window["inputs"] if item["name"] == name)
            assert nodes[links[link_id][1]]["type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8"
            assert links[link_id][2] == slot
    second = next(node for node in projects if node["widgets_values"][0] == 1)
    prior = next(item["link"] for item in second["inputs"] if item["name"] == "previous_result")
    assert links[prior][1:3] == [windows[0]["id"], 1]
    for link_id, src, src_slot, dst, dst_slot, _kind in candidate["links"]:
        assert link_id in nodes[src]["outputs"][src_slot]["links"]
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id


def test_v5_api_relay_uses_global_plan_and_separate_window_projectors():
    original = build_graph()
    del original["26"]
    original["8"]["inputs"]["model"] = ["23", 0]
    graph = add_relay_api(split_api(original))
    plans = [item for item in graph.values()
             if item["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"]
    projects = [(key, item) for key, item in graph.items()
                if item["class_type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8"]
    windows = [item for item in graph.values()
               if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"]
    assert len(plans) == 1 and plans[0]["inputs"]["length"] == 192
    assert len(projects) == len(windows) == 2
    assert {item["inputs"]["window_index"] for _, item in projects} == {0, 1}
    assert len({tuple(item["inputs"]["prompt_relay_plan"]) for _, item in projects}) == 1


def test_v5_combined_candidate_uses_only_joint_eav_actual_call_audits():
    original, digest = _source()
    candidate = add_eav_frontend(add_relay_frontend(split_frontend(original), with_audit=False))
    nodes = {node["id"]: node for node in candidate["nodes"]}
    links = {link[0]: link for link in candidate["links"]}
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == digest
    assert sum(node["type"] == "MiniMaxH3ChunkedV5RelayAuditEXPT8"
               for node in nodes.values()) == 0
    assert sum(node["type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"
               for node in nodes.values()) == 2
    applies = [node for node in nodes.values()
               if node["type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"]
    assert len(applies) == 2
    for apply in applies:
        item = next(input_item for input_item in apply["inputs"]
                    if input_item["name"] == "relay_positive")
        assert nodes[links[item["link"]][1]]["type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8"
    for link_id, src, src_slot, dst, dst_slot, _kind in candidate["links"]:
        assert link_id in nodes[src]["outputs"][src_slot]["links"]
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id
    plain = build_graph()
    del plain["26"]
    plain["8"]["inputs"]["model"] = ["23", 0]
    api = add_eav_api(add_relay_api(split_api(plain), with_audit=False))
    assert sum(node["class_type"] == "MiniMaxH3ChunkedV5RelayAuditEXPT8"
               for node in api.values()) == 0
    assert sum(node["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"
               for node in api.values()) == 2
