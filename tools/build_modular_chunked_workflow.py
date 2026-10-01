"""Copy legacy Chunked frontend graphs into opt-in split candidates.

Never writes into examples/workflows. v1 fixes a native-aligned 56-frame/three-segment example;
v2-v4 use one full-clip segment. Changed v1 frame/chunk settings need a newly
computed stage count rather than silently truncating the final output.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "v1": ROOT / "examples/workflows/13-latent-upscale/2026-08-30_H3_Chunked_TwoPass_Global_Noise_v2_Advanced_EXP.json",
    "v2": ROOT / "examples/workflows/13-latent-upscale/2026-08-30_H3_Chunked_TwoPass_Global_Noise_v2_Advanced_EXP.json",
    "v3": ROOT / "examples/workflows/13-latent-upscale/2026-08-30_H3_Mask_Preserving_Low_Sigma_TwoPass_v4_Advanced_EXP.json",
    "v4": ROOT / "examples/workflows/13-latent-upscale/2026-08-30_H3_Mask_Preserving_Low_Sigma_TwoPass_v4_Advanced_EXP.json",
}
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
SPEC = "T8_CHUNKED_SOURCE_SEGMENT"
CONTEXT = "T8_CHUNKED_PASS2_CONTEXT"
RESULT = "T8_CHUNKED_PASS2_RESULT"
EAV_CONFIG = "T8_STAGE_EAV_CONFIG"
EAV_RUNTIME = "T8_STAGE_EAV_RUNTIME"
RELAY_PLAN = "H3_T8_PROMPT_RELAY_PLAN"
RELAY_RUNTIME = "T8_CHUNKED_PASS2_RELAY_RUNTIME"


def v1_source_from_v2(original):
    """Adapt frozen v2 example to native-aligned v1 three-piece video-only sampling."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    plan = nodes[14]
    if (plan["type"] != "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
            or plan["widgets_values"][-1] != "full_clip_safe"):
        raise ValueError("Expected the frozen v2 full-clip template")
    plan["type"] = "MiniMaxH3ChunkedTwoPassPlanT8Advanced"
    plan["title"] = "v1 temporal Chunked Plan — three explicit segments"
    plan["properties"]["Node name for S&R"] = plan["type"]
    plan["widgets_values"].pop()  # v1 has no temporal_strategy widget
    plan["widgets_values"][3] = 34
    plan["widgets_values"][4] = 17
    for node_id in (7, 13):
        nodes[node_id]["widgets_values"][3] = 56
    graph["nodes"] = [node for node in graph["nodes"] if node["type"] != "MarkdownNote"]
    graph["last_node_id"] = max(node["id"] for node in graph["nodes"])
    return graph


def v3_source_from_v4(original):
    """Remove only the v4 mask preparation to expose the older v3 contract."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    plan = nodes[14]
    if (plan["type"] != "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
            or plan["widgets_values"][-1] != "inherit_required"):
        raise ValueError("Expected the frozen v4 masked-plan template")
    plan["type"] = "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
    plan["title"] = "v3 Low-Sigma Plan — complete first pass, no inherited mask"
    plan["properties"]["Node name for S&R"] = plan["type"]
    plan["widgets_values"].pop()
    removed_nodes = {24, 25, 26, 27, 28, 29, 30, 31, 32, 33}
    links = []
    for link in graph["links"]:
        if link[1] == 29 and link[3] in (9, 12):
            link[1:3] = [8, 1]
        if link[1] not in removed_nodes and link[3] not in removed_nodes:
            links.append(link)
    graph["links"] = links
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in removed_nodes]
    used = {link[0]: link for link in links}
    for node in graph["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") is not None and item["link"] not in used:
                raise ValueError("Removing v4 mask chain left a dangling input")
        for slot, item in enumerate(node.get("outputs", [])):
            item["links"] = [link[0] for link in links
                             if link[1] == node["id"] and link[2] == slot]
    graph["last_node_id"] = max(node["id"] for node in graph["nodes"])
    graph["last_link_id"] = max(link[0] for link in links)
    return graph


def split_workflow(original):
    graph = deepcopy(original)
    matches = [node for node in graph["nodes"] if node["type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one old Chunked executor")
    old = matches[0]
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    incoming = {}
    for item in old["inputs"]:
        if item["link"] is not None:
            link = links[item["link"]]
            incoming[item["name"]] = (link[1], link[2])
    if set(incoming) != {"model", "conditioning", "latent", "noise", "sampler", "sigmas", "plan"}:
        raise ValueError("Old graph input contract changed")
    outgoing = {}
    for item in old["outputs"]:
        if len(item["links"]) != 1:
            raise ValueError("Expected one old output consumer per output")
        link = links[item["links"][0]]
        outgoing[item["name"]] = (link[3], link[4])
    if set(outgoing) != {"latent", "report_json"}:
        raise ValueError("Old output contract changed")
    removed = {link[0] for link in graph["links"] if link[1] == old["id"] or link[3] == old["id"]}
    graph["links"] = [link for link in graph["links"] if link[0] not in removed]
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] != old["id"]]
    for node in graph["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") in removed:
                item["link"] = None
        for item in node.get("outputs", []):
            if item.get("links"):
                item["links"] = [link for link in item["links"] if link not in removed]
    next_node = max(nodes) + 1
    slice_id, prep_id, lift_id, pass_id = range(next_node, next_node + 4)
    x, y = old["pos"]

    def make(node_id, kind, title, position, inputs, outputs, widgets=()):
        return {"id": node_id, "type": kind, "title": title, "pos": list(position),
                "size": [390, max(160, 45 + 25 * len(inputs))], "flags": {},
                "order": old["order"], "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {})}
                           for name, dtype, optional in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind, "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}

    additions = [
        make(slice_id, "MiniMaxH3ChunkedSourceSegmentEXPT8", "First-pass segment 0 — no hidden sampling",
             (x, y - 480), [("first_pass_latent", "LATENT", False), ("plan", PLAN, False)],
             [("segment_latent", "LATENT"), ("segment_spec", SPEC), ("report_json", "STRING")], [0]),
        make(prep_id, "MiniMaxH3ChunkedPass2PrepareEXPT8", "Global noise + inherited mask — once",
             (x, y - 180), [("first_pass_latent", "LATENT", False), ("plan", PLAN, False),
                             ("noise", "NOISE", False)],
             [("pass2_context", CONTEXT), ("report_json", "STRING")]),
        make(lift_id, "MiniMaxH3ChunkedLearnedLiftEXPT8", "Learned latent upscale — separate",
             (x + 460, y - 430), [("source_segment", "LATENT", False), ("segment_spec", SPEC, False),
                                   ("pass2_context", CONTEXT, False), ("plan", PLAN, False)],
             [("lifted_segment", "LATENT"), ("report_json", "STRING")]),
        make(pass_id, "MiniMaxH3ChunkedPass2SegmentEXPT8", "PASS2 — external MODEL / conditions / sampler",
             (x + 920, y - 210), [("model", "MODEL", False), ("positive", "CONDITIONING", False),
                                   ("source_segment", "LATENT", False), ("lifted_segment", "LATENT", False),
                                   ("segment_spec", SPEC, False), ("pass2_context", CONTEXT, False),
                                   ("plan", PLAN, False), ("noise", "NOISE", False),
                                   ("sampler", "SAMPLER", False), ("sigmas", "SIGMAS", False),
                                   ("previous_result", RESULT, True), ("negative", "CONDITIONING", True)],
             [("cumulative_av_latent", "LATENT"), ("segment_result", RESULT),
              ("report_json", "STRING")], [1.0]),
    ]
    graph["nodes"].extend(additions)
    nodes = {node["id"]: node for node in graph["nodes"]}
    next_link = max(links) + 1

    def connect(source, target, slot, dtype):
        nonlocal next_link
        source_id, source_slot = source
        link = [next_link, source_id, source_slot, target, slot, dtype]
        graph["links"].append(link)
        nodes[source_id]["outputs"][source_slot].setdefault("links", []).append(next_link)
        nodes[target]["inputs"][slot]["link"] = next_link
        next_link += 1

    connect(incoming["latent"], slice_id, 0, "LATENT")
    connect(incoming["plan"], slice_id, 1, PLAN)
    connect(incoming["latent"], prep_id, 0, "LATENT")
    connect(incoming["plan"], prep_id, 1, PLAN)
    connect(incoming["noise"], prep_id, 2, "NOISE")
    connect((slice_id, 0), lift_id, 0, "LATENT")
    connect((slice_id, 1), lift_id, 1, SPEC)
    connect((prep_id, 0), lift_id, 2, CONTEXT)
    connect(incoming["plan"], lift_id, 3, PLAN)
    # Duplicate the original model/sampler setup with identical defaults. This
    # leaves LOW untouched and gives PASS2 its own explicit model/LoRA branch.
    # The duplicated setup still uses the old source geometry by default;
    # changing LoRA is an intentional new HIGH edit, not a parity claim.
    source_setup = nodes[incoming["model"][0]]
    if source_setup["type"] == "MiniMaxH3PDD8StepSetupT8Advanced":
        branch = (source_setup["id"], incoming["sigmas"][0])
    elif source_setup["type"] == "MiniMaxH3DualClockSamplerT8":
        lora_source = next(item for item in source_setup["inputs"] if item["name"] == "model")
        lora_id = links[lora_source["link"]][1]
        if nodes[lora_id]["type"] != "LoraLoaderBypassModelOnly":
            raise ValueError("Expected the unchanged v4 LOW LoRA branch")
        branch = (lora_id, source_setup["id"], incoming["sigmas"][0])
    else:
        raise ValueError("Unsupported old model setup for this split candidate")
    cloned = {}
    for position, source_id in enumerate(branch):
        original_node = nodes[source_id]
        clone = deepcopy(original_node)
        clone["id"] = max(nodes) + 1
        clone["title"] = "PASS2 separate " + (clone.get("title") or clone["type"])
        clone["pos"] = [x + 430 * position, y + 520]
        clone["order"] = old["order"] + 1 + position
        for item in clone.get("inputs", []):
            item["link"] = None
        for item in clone.get("outputs", []):
            item["links"] = []
        graph["nodes"].append(clone)
        nodes[clone["id"]] = clone
        cloned[source_id] = clone["id"]
        for input_slot, item in enumerate(original_node.get("inputs", [])):
            if item.get("link") is None:
                continue
            old_link = links[item["link"]]
            upstream_id = cloned.get(old_link[1], old_link[1])
            connect((upstream_id, old_link[2]), clone["id"], input_slot, old_link[5])
    pass_model = (cloned[incoming["model"][0]], incoming["model"][1])
    pass_sampler = (cloned[incoming["sampler"][0]], incoming["sampler"][1])
    pass_sigmas = (cloned[incoming["sigmas"][0]], incoming["sigmas"][1])
    for source, slot, dtype in (
        (pass_model, 0, "MODEL"), (incoming["conditioning"], 1, "CONDITIONING"),
        ((slice_id, 0), 2, "LATENT"), ((lift_id, 0), 3, "LATENT"),
        ((slice_id, 1), 4, SPEC), ((prep_id, 0), 5, CONTEXT),
        (incoming["plan"], 6, PLAN), (incoming["noise"], 7, "NOISE"),
        (pass_sampler, 8, "SAMPLER"), (pass_sigmas, 9, "SIGMAS"),
    ):
        connect(source, pass_id, slot, dtype)
    connect((pass_id, 0), *outgoing["latent"], "LATENT")
    connect((pass_id, 2), *outgoing["report_json"], "STRING")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def append_second_frontend_segment(graph, *, segment_index=1):
    """Append the next visible v1 segment and chain its PASS2 result."""
    graph = deepcopy(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    by_type = {node["type"]: node for node in graph["nodes"]}
    first_source = by_type["MiniMaxH3ChunkedSourceSegmentEXPT8"]
    first_lift = by_type["MiniMaxH3ChunkedLearnedLiftEXPT8"]
    first_pass = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    prep = by_type["MiniMaxH3ChunkedPass2PrepareEXPT8"]
    next_link = max(links) + 1

    def origin(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        link = links[item["link"]]
        return link[1], link[2], link[5]

    def clone(node, title, shift):
        item = deepcopy(node)
        item["id"] = max(nodes) + 1
        item["title"] = title
        item["pos"] = [node["pos"][0] + shift, node["pos"][1] + 680]
        item["order"] = node["order"] + 4
        for input_item in item["inputs"]:
            input_item["link"] = None
        for output in item["outputs"]:
            output["links"] = []
        graph["nodes"].append(item)
        nodes[item["id"]] = item
        return item

    def connect(source_id, source_slot, target, input_name, dtype):
        nonlocal next_link
        target_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        graph["links"].append([next_link, source_id, source_slot, target["id"], target_slot, dtype])
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    source = clone(first_source, f"First-pass segment {segment_index}", 0)
    source["widgets_values"] = [segment_index]
    for name in ("first_pass_latent", "plan"):
        source_id, source_slot, dtype = origin(first_source, name)
        connect(source_id, source_slot, source, name, dtype)
    lift = clone(first_lift, f"Learned latent upscale — segment {segment_index}", 0)
    connect(source["id"], 0, lift, "source_segment", "LATENT")
    connect(source["id"], 1, lift, "segment_spec", SPEC)
    connect(prep["id"], 0, lift, "pass2_context", CONTEXT)
    source_id, source_slot, dtype = origin(first_lift, "plan")
    connect(source_id, source_slot, lift, "plan", dtype)
    second = clone(first_pass, f"PASS2 segment {segment_index} — consumes prior result", 0)
    for name in ("model", "positive", "plan", "noise", "sampler", "sigmas"):
        source_id, source_slot, dtype = origin(first_pass, name)
        connect(source_id, source_slot, second, name, dtype)
    for name, source_id, source_slot, dtype in (
        ("source_segment", source["id"], 0, "LATENT"),
        ("lifted_segment", lift["id"], 0, "LATENT"),
        ("segment_spec", source["id"], 1, SPEC),
        ("pass2_context", prep["id"], 0, CONTEXT),
        ("previous_result", first_pass["id"], 1, RESULT),
    ):
        connect(source_id, source_slot, second, name, dtype)
    for slot in (0, 2):
        for link_id in list(first_pass["outputs"][slot]["links"]):
            link = links[link_id]
            link[1] = second["id"]
            first_pass["outputs"][slot]["links"].remove(link_id)
            second["outputs"][slot]["links"].append(link_id)
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def split_api_graph(original):
    """Mirror the frontend replacement on the existing v2/v4 API graph."""
    graph = deepcopy(original)
    matches = [(key, node) for key, node in graph.items()
               if node["class_type"] == "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one old Chunked API executor")
    old_id, old = matches[0]
    inputs = old["inputs"]
    if not all(name in inputs for name in ("model", "conditioning", "latent", "noise",
                                            "sampler", "sigmas", "plan")):
        raise ValueError("Old Chunked API inputs changed")
    next_id = max(map(int, graph)) + 1

    def add(class_type, values):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": class_type, "inputs": values}
        return key

    source_id = add("MiniMaxH3ChunkedSourceSegmentEXPT8", {
        "first_pass_latent": inputs["latent"], "plan": inputs["plan"], "segment_index": 0})
    prep_id = add("MiniMaxH3ChunkedPass2PrepareEXPT8", {
        "first_pass_latent": inputs["latent"], "plan": inputs["plan"], "noise": inputs["noise"]})
    lift_id = add("MiniMaxH3ChunkedLearnedLiftEXPT8", {
        "source_segment": [source_id, 0], "segment_spec": [source_id, 1],
        "pass2_context": [prep_id, 0], "plan": inputs["plan"]})
    setup_id = str(inputs["model"][0])
    setup = graph[setup_id]
    if setup["class_type"] == "MiniMaxH3PDD8StepSetupT8Advanced":
        branch = (setup_id, str(inputs["sigmas"][0]))
    elif setup["class_type"] == "MiniMaxH3DualClockSamplerT8":
        lora_id = str(setup["inputs"]["model"][0])
        if graph[lora_id]["class_type"] != "LoraLoaderBypassModelOnly":
            raise ValueError("Expected the unchanged v4 LOW LoRA branch")
        branch = (lora_id, setup_id, str(inputs["sigmas"][0]))
    else:
        raise ValueError("Unsupported old model setup for API split")
    clones = {}
    for old_branch_id in branch:
        clone = deepcopy(graph[old_branch_id])
        for name, value in clone["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in clones:
                clone["inputs"][name] = [clones[str(value[0])], value[1]]
        clones[old_branch_id] = add(clone["class_type"], clone["inputs"])
    pass_id = add("MiniMaxH3ChunkedPass2SegmentEXPT8", {
        "model": [clones[setup_id], inputs["model"][1]],
        "positive": inputs["conditioning"],
        "source_segment": [source_id, 0], "lifted_segment": [lift_id, 0],
        "segment_spec": [source_id, 1], "pass2_context": [prep_id, 0],
        "plan": inputs["plan"], "noise": inputs["noise"],
        "sampler": [clones[str(inputs["sampler"][0])], inputs["sampler"][1]],
        "sigmas": [clones[str(inputs["sigmas"][0])], inputs["sigmas"][1]],
        "cfg": inputs.get("cfg", 1.0),
    })
    for node in graph.values():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == old_id:
                node["inputs"][name] = [pass_id, {0: 0, 1: 2}[value[1]]]
    del graph[old_id]
    return graph


def append_second_api_segment(graph, *, segment_index=1):
    graph = deepcopy(graph)
    by_type = {node["class_type"]: key for key, node in graph.items()}
    first_source = by_type["MiniMaxH3ChunkedSourceSegmentEXPT8"]
    first_lift = by_type["MiniMaxH3ChunkedLearnedLiftEXPT8"]
    first_pass = by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]
    next_id = max(map(int, graph)) + 1
    second_source, second_lift, second_pass = (str(value) for value in range(next_id, next_id + 3))
    graph[second_source] = deepcopy(graph[first_source])
    graph[second_source]["inputs"]["segment_index"] = segment_index
    graph[second_lift] = deepcopy(graph[first_lift])
    graph[second_lift]["inputs"].update(source_segment=[second_source, 0],
                                        segment_spec=[second_source, 1])
    graph[second_pass] = deepcopy(graph[first_pass])
    graph[second_pass]["inputs"].update(
        source_segment=[second_source, 0], lifted_segment=[second_lift, 0],
        segment_spec=[second_source, 1], previous_result=[first_pass, 1],
    )
    for key, node in graph.items():
        if key == second_pass:
            continue
        for name, value in node["inputs"].items():
            if (isinstance(value, list) and len(value) == 2
                    and value[0] == first_pass and value[1] in (0, 2)):
                node["inputs"][name] = [second_pass, value[1]]
    return graph


def add_external_eav_frontend(graph):
    """Append real per-segment EAV Apply/Audit without touching the old graph."""
    graph = deepcopy(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    passes = [node for node in graph["nodes"]
              if node["type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    if not passes:
        raise ValueError("No explicit Chunked PASS2 segments for external EAV")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": pos,
                "size": [410, max(170, 45 + 25 * len(inputs))], "flags": {},
                "order": max(item["order"] for item in nodes.values()) + 1,
                "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None} for name, dtype in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []} for name, dtype in outputs],
                "properties": {"Node name for S&R": kind, "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        next_node += 1
        graph["nodes"].append(node)
        nodes[node["id"]] = node
        return node

    def connect(source_id, source_slot, target, name, dtype):
        nonlocal next_link
        slot = next(index for index, item in enumerate(target["inputs"]) if item["name"] == name)
        link = [next_link, source_id, source_slot, target["id"], slot, dtype]
        graph["links"].append(link)
        links[next_link] = link
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][slot]["link"] = next_link
        next_link += 1

    def origin(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        link = links[item["link"]]
        return link[1], link[2], link[5]

    x, y = passes[0]["pos"]
    config = make("MiniMaxH3StageEAVConfigEXPT8", "External PASS2 EAV config — report first",
                  [x - 50, y - 700], [], [("eav_config", EAV_CONFIG)],
                  ["report_only", 4.0, 0.15, 0.90, 32, 1.5])
    for index, pass2 in enumerate(passes):
        px, py = pass2["pos"]
        apply = make("MiniMaxH3ChunkedPass2EAVApplyEXPT8",
                     f"External EAV — PASS2 segment {index}", [px - 500, py - 150],
                     [(name, dtype) for name, dtype in (
                         ("model", "MODEL"), ("sigmas", "SIGMAS"),
                         ("source_segment", "LATENT"), ("lifted_segment", "LATENT"),
                         ("segment_spec", SPEC), ("pass2_context", CONTEXT),
                         ("plan", PLAN), ("eav_config", EAV_CONFIG))],
                     [("model", "MODEL"), ("runtime", EAV_RUNTIME),
                      ("report_json", "STRING"), ("stage_context", "T8_STAGE_CONTEXT")])
        audit = make("MiniMaxH3ChunkedPass2EAVAuditEXPT8",
                     f"Actual EAV calls — PASS2 segment {index}", [px + 520, py - 120],
                     [("segment_result", RESULT), ("segment_spec", SPEC),
                      ("runtime", EAV_RUNTIME)],
                     [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")])
        model_item = next(item for item in pass2["inputs"] if item["name"] == "model")
        model_link = links[model_item["link"]]
        model_link[3] = apply["id"]
        model_link[4] = 0
        apply["inputs"][0]["link"] = model_link[0]
        model_item["link"] = None
        for name in ("sigmas", "source_segment", "lifted_segment", "segment_spec",
                     "pass2_context", "plan"):
            source_id, source_slot, dtype = origin(pass2, name)
            connect(source_id, source_slot, apply, name, dtype)
        connect(config["id"], 0, apply, "eav_config", EAV_CONFIG)
        connect(apply["id"], 0, pass2, "model", "MODEL")
        connect(pass2["id"], 1, audit, "segment_result", RESULT)
        source_id, source_slot, dtype = origin(pass2, "segment_spec")
        connect(source_id, source_slot, audit, "segment_spec", dtype)
        connect(apply["id"], 1, audit, "runtime", EAV_RUNTIME)
    graph["last_node_id"] = next_node - 1
    graph["last_link_id"] = next_link - 1
    return graph


def add_external_eav_api(graph):
    graph = deepcopy(graph)
    passes = [(key, node) for key, node in graph.items()
              if node["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    if not passes:
        raise ValueError("No explicit Chunked PASS2 segments for external EAV")
    next_id = max(map(int, graph)) + 1

    def add(kind, inputs):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": inputs}
        return key

    config_id = add("MiniMaxH3StageEAVConfigEXPT8", {
        "mode": "report_only", "tau": 4.0, "start_video_progress": 0.15,
        "end_video_progress": 0.90, "max_workspace_mib": 32, "g_hard_limit": 1.5,
    })
    for pass_id, pass2 in passes:
        values = pass2["inputs"]
        apply_id = add("MiniMaxH3ChunkedPass2EAVApplyEXPT8", {
            "model": values["model"], "sigmas": values["sigmas"],
            "source_segment": values["source_segment"],
            "lifted_segment": values["lifted_segment"],
            "segment_spec": values["segment_spec"],
            "pass2_context": values["pass2_context"], "plan": values["plan"],
            "eav_config": [config_id, 0],
        })
        values["model"] = [apply_id, 0]
        add("MiniMaxH3ChunkedPass2EAVAuditEXPT8", {
            "segment_result": [pass_id, 1],
            "segment_spec": values["segment_spec"], "runtime": [apply_id, 1],
        })
    return graph


def add_external_relay_frontend(graph, *, with_audit=True):
    """Wire stock Relay Plan/Conditioning outside one v2-v4 full-clip PASS2."""
    graph = deepcopy(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    passes = [node for node in graph["nodes"]
              if node["type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    if len(passes) != 1 or any(node["type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
                               for node in graph["nodes"]):
        raise ValueError("External Relay candidate needs exactly one PASS2 and no EAV")
    pass2 = passes[0]
    high = next(node for node in graph["nodes"]
                if node["type"] == "MiniMaxH3AudioConditioningT8"
                and next(item for item in pass2["inputs"] if item["name"] == "positive")["link"]
                in node["outputs"][0]["links"])
    if high["widgets_values"][4] != "FL2VA":
        raise ValueError("This private Relay candidate expects the frozen FL2VA template")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [410, max(180, 48 + 26 * len(inputs))], "flags": {},
                "order": max(item["order"] for item in nodes.values()) + 1, "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None}
                           for name, dtype in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind, "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def connect(source, target, name, dtype):
        nonlocal next_link
        source_id, source_slot = source
        target_slot = next(index for index, item in enumerate(target["inputs"])
                           if item["name"] == name)
        link = [next_link, source_id, source_slot, target["id"], target_slot, dtype]
        graph["links"].append(link)
        links[next_link] = link
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    def origin(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        link = links[link_id]
        return link[1], link[2], link[5]

    x, y = pass2["pos"]
    high_values = high["widgets_values"]
    plan = make("MiniMaxH3PromptRelayPlanT8Advanced", "PASS2 external Relay timeline",
                (x - 920, y - 930), [],
                [("prompt_relay_plan", RELAY_PLAN), ("compiled_prompt", "STRING"),
                 ("length", "INT"), ("timeline_json", "STRING"), ("report_json", "STRING")],
                [high_values[0] + " <Picture 1> is the first frame; <Picture 2> is the last frame.",
                 "Maintain the same subject and scene with subtle natural motion.\n"
                 "Continue the same shot toward the specified last frame without a cut.",
                 high_values[3], "auto_equal", "", "paper_v1", 0.1, False, False])
    media_names = ("drive_audio", "final_audio", "first_frame", "last_frame")
    media_types = {"drive_audio": "AUDIO", "final_audio": "AUDIO",
                   "first_frame": "IMAGE", "last_frame": "IMAGE"}
    relay_cond = make(
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "PASS2 external Relay Conditioning — apply_exp", (x - 460, y - 930),
        [(name, dtype) for name, dtype in
         (("model", "MODEL"), ("clip", "CLIP"), ("video_vae", "VAE"),
          ("audio_vae", "VAE"), ("prompt_relay_plan", RELAY_PLAN))]
        + [(name, media_types[name]) for name in media_names],
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
         ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [high_values[1], high_values[2], high_values[4], high_values[5],
         high_values[6], high_values[7], high_values[8], high_values[9],
         high_values[10], high_values[11], "apply_exp", 256],
    )
    bind = make("MiniMaxH3ChunkedPass2RelayBindEXPT8", "Bind external Relay to PASS2",
                (x + 60, y - 800),
                [("relay_model", "MODEL"), ("relay_positive", "CONDITIONING"),
                 ("relay_av_latent", "LATENT"), ("prompt_relay_plan", RELAY_PLAN),
                 ("source_segment", "LATENT"), ("lifted_segment", "LATENT"),
                 ("segment_spec", SPEC), ("pass2_context", CONTEXT),
                 ("plan", PLAN), ("sigmas", "SIGMAS")],
                [("model", "MODEL"), ("positive", "CONDITIONING"),
                 ("runtime", RELAY_RUNTIME), ("report_json", "STRING")])
    audit = (make("MiniMaxH3ChunkedPass2RelayAuditEXPT8", "Actual PASS2 Relay calls",
                  (x + 540, y - 430),
                  [("segment_result", RESULT), ("segment_spec", SPEC),
                   ("runtime", RELAY_RUNTIME)],
                  [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")])
             if with_audit else None)
    model_item = next(item for item in pass2["inputs"] if item["name"] == "model")
    old_model_link = links[model_item["link"]]
    old_model_link[3:5] = [relay_cond["id"], 0]
    relay_cond["inputs"][0]["link"] = old_model_link[0]
    model_item["link"] = None
    positive_item = next(item for item in pass2["inputs"] if item["name"] == "positive")
    old_positive_id = positive_item["link"]
    old_positive_link = links.pop(old_positive_id)
    nodes[old_positive_link[1]]["outputs"][old_positive_link[2]]["links"].remove(old_positive_id)
    graph["links"] = [link for link in graph["links"] if link[0] != old_positive_id]
    positive_item["link"] = None
    for name in ("clip", "video_vae", "audio_vae", *media_names):
        if next(item for item in high["inputs"] if item["name"] == name)["link"] is not None:
            source_id, source_slot, dtype = origin(high, name)
            connect((source_id, source_slot), relay_cond, name, dtype)
    connect((plan["id"], 0), relay_cond, "prompt_relay_plan", RELAY_PLAN)
    for source, name, dtype in (
        ((relay_cond["id"], 0), "relay_model", "MODEL"),
        ((relay_cond["id"], 1), "relay_positive", "CONDITIONING"),
        ((relay_cond["id"], 2), "relay_av_latent", "LATENT"),
        ((plan["id"], 0), "prompt_relay_plan", RELAY_PLAN),
    ):
        connect(source, bind, name, dtype)
    for name in ("source_segment", "lifted_segment", "segment_spec", "pass2_context",
                 "plan", "sigmas"):
        source_id, source_slot, dtype = origin(pass2, name)
        connect((source_id, source_slot), bind, name, dtype)
    connect((bind["id"], 0), pass2, "model", "MODEL")
    connect((bind["id"], 1), pass2, "positive", "CONDITIONING")
    if audit is not None:
        connect((pass2["id"], 1), audit, "segment_result", RESULT)
        source_id, source_slot, dtype = origin(pass2, "segment_spec")
        connect((source_id, source_slot), audit, "segment_spec", dtype)
        connect((bind["id"], 2), audit, "runtime", RELAY_RUNTIME)
    graph["last_node_id"] = next_node - 1
    graph["last_link_id"] = next_link - 1
    return graph


def add_external_relay_api(graph, *, with_audit=True):
    """Mirror the opt-in v2-v4 frontend topology for real Core validation."""
    graph = deepcopy(graph)
    passes = [(key, node) for key, node in graph.items()
              if node["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8"]
    if len(passes) != 1 or any(node["class_type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
                               for node in graph.values()):
        raise ValueError("External Relay API candidate needs exactly one PASS2 and no EAV")
    pass_id, pass2 = passes[0]
    next_id = max(map(int, graph)) + 1

    def add(kind, values):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": values}
        return key

    positive_ref = pass2["inputs"]["positive"]
    high = graph[str(positive_ref[0])]
    if high["class_type"] != "MiniMaxH3AudioConditioningT8":
        raise ValueError("Expected frozen HIGH conditioning template")
    high_inputs = high["inputs"]
    if high_inputs["task_type"] != "FL2VA":
        raise ValueError("This private Relay candidate expects FL2VA")
    plan_id = add("MiniMaxH3PromptRelayPlanT8Advanced", {
        "global_prompt": high_inputs["prompt"] +
        " <Picture 1> is the first frame; <Picture 2> is the last frame.",
        "local_prompts": "Maintain the same subject and scene with subtle natural motion.\n"
                         "Continue the same shot toward the specified last frame without a cut.",
        "length": high_inputs["length"], "timing_mode": "auto_equal",
        "time_ranges": "", "math_profile": "paper_v1", "epsilon": 0.1,
        "allow_gaps": False, "allow_overlaps": False,
    })
    cond_values = {"model": pass2["inputs"]["model"],
                   "clip": high_inputs["clip"], "video_vae": high_inputs["video_vae"],
                   "audio_vae": high_inputs["audio_vae"],
                   "prompt_relay_plan": [plan_id, 0],
                   "width": high_inputs["width"], "height": high_inputs["height"],
                   "task_type": high_inputs["task_type"],
                   "audio_mode": high_inputs["audio_mode"],
                   "audio_denoise_strength": high_inputs["audio_denoise_strength"],
                   "add_source_as_reference": high_inputs["add_source_as_reference"],
                   "prompt_primary_audio_ordinal": high_inputs["prompt_primary_audio_ordinal"],
                   "strict_prompt_tags": high_inputs["strict_prompt_tags"],
                   "ref_image_size": high_inputs["ref_image_size"],
                   "reference_video_policy": high_inputs["reference_video_policy"],
                   "execution_mode": "apply_exp", "query_chunk_rows": 256}
    for name in ("drive_audio", "final_audio", "first_frame", "last_frame"):
        if name in high_inputs:
            cond_values[name] = high_inputs[name]
    cond_id = add("MiniMaxH3PromptRelayConditioningT8Advanced", cond_values)
    values = pass2["inputs"]
    bind_id = add("MiniMaxH3ChunkedPass2RelayBindEXPT8", {
        "relay_model": [cond_id, 0], "relay_positive": [cond_id, 1],
        "relay_av_latent": [cond_id, 2], "prompt_relay_plan": [plan_id, 0],
        "source_segment": values["source_segment"],
        "lifted_segment": values["lifted_segment"],
        "segment_spec": values["segment_spec"],
        "pass2_context": values["pass2_context"], "plan": values["plan"],
        "sigmas": values["sigmas"],
    })
    values["model"] = [bind_id, 0]
    values["positive"] = [bind_id, 1]
    if with_audit:
        add("MiniMaxH3ChunkedPass2RelayAuditEXPT8", {
            "segment_result": [pass_id, 1],
            "segment_spec": values["segment_spec"], "runtime": [bind_id, 2],
        })
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-eav", action="store_true",
                        help="Append per-segment external EAV Apply/Audit (single-tile only)")
    parser.add_argument("--with-relay", action="store_true",
                        help="Append external Relay Plan/Conditioning/Bind/Audit (v2-v4 full-clip only)")
    args = parser.parse_args()
    if args.with_relay and args.variant == "v1":
        parser.error("This Relay candidate only covers v2-v4 one full-clip PASS2")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Write a new private artifact path; never overwrite an old workflow")
    original = json.loads(TEMPLATES[args.variant].read_text(encoding="utf-8"))
    if args.variant == "v1":
        original = v1_source_from_v2(original)
    elif args.variant == "v3":
        original = v3_source_from_v4(original)
    candidate = split_workflow(original)
    if args.variant == "v1":
        candidate = append_second_frontend_segment(candidate)
        candidate = append_second_frontend_segment(candidate, segment_index=2)
    if args.with_relay:
        candidate = add_external_relay_frontend(candidate, with_audit=not args.with_eav)
    if args.with_eav:
        candidate = add_external_eav_frontend(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "nodes": len(candidate["nodes"]),
                      "links": len(candidate["links"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
