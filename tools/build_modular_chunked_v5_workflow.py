"""Make private fixed-window split copies of the frozen standard Chunked 4+4 graph.

The original workflow is read-only. This fixed 192-frame candidate must be
rebuilt if first-pass length or the temporal window geometry is changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/workflows/13-latent-upscale/2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json"
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
LIFT = "T8_CHUNKED_V5_GLOBAL_LIFT"
PREPARED = "T8_CHUNKED_V5_PREPARED"
RESULT = "T8_CHUNKED_V5_WINDOW_RESULT"


def split_frontend(original, *, window_count=2, temporal_chunk_frames=136,
                   temporal_overlap_frames=34):
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    old = nodes[15]
    if (old["type"] != "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced"
            or nodes[14]["widgets_values"][15] != "standard_joint_4plus4_exp"
            or nodes[7]["widgets_values"][3] != 192
            or nodes[13]["widgets_values"][3] != 192
            or nodes[14]["widgets_values"][3:5] != [136, 34]):
        raise ValueError("Expected the frozen 192-frame/two-window standard4+4 template")
    if (window_count, temporal_chunk_frames, temporal_overlap_frames) not in {
            (2, 136, 34), (3, 102, 34), (4, 85, 34)}:
        raise ValueError("This fixed 192-frame v5 candidate supports only its verified 2/3/4-window plans")
    nodes[14]["widgets_values"][3:5] = [temporal_chunk_frames, temporal_overlap_frames]
    incoming = {}
    for item in old["inputs"]:
        if item["link"] is not None:
            link = links[item["link"]]
            incoming[item["name"]] = (link[1], link[2], link[5])
    outgoing = {}
    for slot, item in enumerate(old["outputs"]):
        for link_id in item.get("links") or []:
            link = links[link_id]
            outgoing.setdefault(slot, []).append((link[3], link[4], link[5]))
    required = {"model", "conditioning", "latent", "noise", "sampler", "sigmas", "plan"}
    if set(incoming) != required or set(outgoing) != {0, 1}:
        raise ValueError("Frozen v5 executor link contract changed")
    removed = {link[0] for link in graph["links"] if link[1] == old["id"] or link[3] == old["id"]}
    graph["links"] = [link for link in graph["links"] if link[0] not in removed]
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] != old["id"]]
    del nodes[old["id"]]
    for node in graph["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") in removed:
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [link for link in item.get("links") or [] if link not in removed]
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [400, max(180, 48 + 25 * len(inputs))], "flags": {},
                "order": old["order"], "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {})}
                           for name, dtype, optional in inputs],
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
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    # Independent HIGH LoRA and AV-clock setup, with the original defaults.
    cloned = {}
    for source_id in (23, 8):
        original_node = nodes[source_id]
        clone = deepcopy(original_node)
        clone["id"] = next_node
        next_node += 1
        clone["title"] = "PASS2 independent " + (clone.get("title") or clone["type"])
        clone["pos"] = [old["pos"][0] + 390 * len(cloned), old["pos"][1] + 650]
        for item in clone.get("inputs", []):
            item["link"] = None
        for item in clone.get("outputs", []):
            item["links"] = []
        graph["nodes"].append(clone)
        nodes[clone["id"]] = clone
        cloned[source_id] = clone["id"]
        for slot, item in enumerate(original_node["inputs"]):
            if item["link"] is None:
                continue
            old_link = links[item["link"]]
            source = (cloned.get(old_link[1], old_link[1]), old_link[2])
            connect(source, clone, clone["inputs"][slot]["name"], old_link[5])
    x, y = old["pos"]
    lift = make("MiniMaxH3ChunkedV5GlobalLiftEXPT8", "Global learned 3D lift — once",
                (x, y - 560), [("partial4_denoised_output", "LATENT", False),
                               ("plan", PLAN, False)],
                [("lifted_full_av", "LATENT"), ("lift_receipt", LIFT),
                 ("report_json", "STRING")])
    prepare = make("MiniMaxH3ChunkedV5PrepareEXPT8", "Global target AV noise — once",
                   (x + 440, y - 560),
                   [("partial4_denoised_output", "LATENT", False),
                    ("lifted_full_av", "LATENT", False), ("lift_receipt", LIFT, False),
                    ("plan", PLAN, False), ("noise", "NOISE", False)],
                   [("prepared", PREPARED), ("report_json", "STRING")])
    window_inputs = [("model", "MODEL", False), ("positive", "CONDITIONING", False),
                     ("partial4_denoised_output", "LATENT", False),
                     ("lifted_full_av", "LATENT", False), ("prepared", PREPARED, False),
                     ("plan", PLAN, False), ("noise", "NOISE", False),
                     ("sampler", "SAMPLER", False), ("sigmas", "SIGMAS", False),
                     ("previous_result", RESULT, True), ("negative", "CONDITIONING", True)]
    outputs = [("cumulative_av_latent", "LATENT"), ("window_result", RESULT),
               ("report_json", "STRING")]
    windows = [make("MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    f"Joint AV PASS2 window {index} — remaining4", (x + 880 + index * 470, y),
                    window_inputs, outputs, [index, 1.0]) for index in range(window_count)]
    for name, source in (("partial4_denoised_output", incoming["latent"]),
                         ("plan", incoming["plan"])):
        connect(source[:2], lift, name, source[2])
    for name, source in (("partial4_denoised_output", incoming["latent"]),
                         ("lifted_full_av", (lift["id"], 0, "LATENT")),
                         ("lift_receipt", (lift["id"], 1, LIFT)),
                         ("plan", incoming["plan"]), ("noise", incoming["noise"])):
        connect(source[:2], prepare, name, source[2])
    for index, window in enumerate(windows):
        wiring = (("model", (cloned[8], 0, "MODEL")),
                  ("positive", incoming["conditioning"]),
                  ("partial4_denoised_output", incoming["latent"]),
                  ("lifted_full_av", (lift["id"], 0, "LATENT")),
                  ("prepared", (prepare["id"], 0, PREPARED)),
                  ("plan", incoming["plan"]), ("noise", incoming["noise"]),
                  ("sampler", (cloned[8], 1, "SAMPLER")),
                  ("sigmas", incoming["sigmas"]))
        for name, source in wiring:
            connect(source[:2], window, name, source[2])
        if index:
            connect((windows[index - 1]["id"], 1), window, "previous_result", RESULT)
    for slot, targets in outgoing.items():
        source_slot = {0: 0, 1: 2}[slot]
        for target_id, target_slot, dtype in targets:
            connect((windows[-1]["id"], source_slot), nodes[target_id],
                    nodes[target_id]["inputs"][target_slot]["name"], dtype)
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def add_eav_frontend(original):
    """Give each fixed v5 PASS2 window its own external EAV Config/Apply/Audit."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda node: node["widgets_values"][0])
    if len(windows) < 2 or any(node["type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"
                                for node in nodes.values()):
        raise ValueError("External v5 EAV expects untouched explicit PASS2 windows")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [410, max(180, 48 + 26 * len(inputs))], "flags": {},
                "order": max(item["order"] for item in nodes.values()) + 1, "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {})}
                           for name, dtype, optional in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind, "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def source(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        return None if item["link"] is None else links[item["link"]][1:3] + [links[item["link"]][5]]

    def connect(parent, child, name, dtype):
        nonlocal next_link
        src_id, src_slot = parent
        target_slot = next(i for i, item in enumerate(child["inputs"]) if item["name"] == name)
        link = [next_link, src_id, src_slot, child["id"], target_slot, dtype]
        graph["links"].append(link)
        nodes[src_id]["outputs"][src_slot]["links"].append(next_link)
        child["inputs"][target_slot]["link"] = next_link
        next_link += 1

    for index, window in enumerate(windows):
        x, y = window["pos"]
        config = make("MiniMaxH3StageEAVConfigEXPT8", f"Window {index} external EAV config",
                      (x, y + 480), [], [("eav_config", "T8_STAGE_EAV_CONFIG")],
                      ["report_only", 4.0, 0.15, 0.90, 32, 1.5])
        apply = make("MiniMaxH3ChunkedV5EAVApplyEXPT8", f"Window {index} external EAV apply",
                     (x + 440, y + 480),
                     [("model", "MODEL", False), ("sigmas", "SIGMAS", False),
                      ("partial4_denoised_output", "LATENT", False),
                      ("lifted_full_av", "LATENT", False),
                      ("prepared", PREPARED, False), ("plan", PLAN, False),
                      ("eav_config", "T8_STAGE_EAV_CONFIG", False),
                      ("previous_result", RESULT, True),
                      ("relay_positive", "CONDITIONING", True)],
                     [("model", "MODEL"), ("runtime", "T8_STAGE_EAV_RUNTIME"),
                      ("report_json", "STRING"), ("stage_context", "T8_STAGE_CONTEXT")],
                     [index])
        audit = make("MiniMaxH3ChunkedV5EAVAuditEXPT8", f"Window {index} actual EAV calls",
                     (x + 880, y + 480),
                     [("window_result", RESULT, False), ("prepared", PREPARED, False),
                      ("runtime", "T8_STAGE_EAV_RUNTIME", False)],
                     [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")])
        old_model = source(window, "model")
        for name in ("sigmas", "partial4_denoised_output", "lifted_full_av", "prepared", "plan"):
            parent = source(window, name)
            connect(parent[:2], apply, name, parent[2])
        connect(old_model[:2], apply, "model", old_model[2])
        connect((config["id"], 0), apply, "eav_config", "T8_STAGE_EAV_CONFIG")
        if index:
            connect((windows[index - 1]["id"], 1), apply, "previous_result", RESULT)
        positive_source = source(window, "positive")
        if nodes[positive_source[0]]["type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8":
            connect(positive_source[:2], apply, "relay_positive", "CONDITIONING")
        old_link_id = next(item for item in window["inputs"] if item["name"] == "model")["link"]
        graph["links"] = [link for link in graph["links"] if link[0] != old_link_id]
        nodes[old_model[0]]["outputs"][old_model[1]]["links"].remove(old_link_id)
        next(item for item in window["inputs"] if item["name"] == "model")["link"] = None
        connect((apply["id"], 0), window, "model", "MODEL")
        connect((window["id"], 1), audit, "window_result", RESULT)
        connect(source(window, "prepared")[:2], audit, "prepared", PREPARED)
        connect((apply["id"], 1), audit, "runtime", "T8_STAGE_EAV_RUNTIME")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def add_relay_frontend(original, *, with_audit=True):
    """Expose one full-clip Relay Plan and exact per-window projections."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda node: node["widgets_values"][0])
    if len(windows) < 2 or any(node["type"] in {
            "MiniMaxH3ChunkedV5EAVApplyEXPT8", "MiniMaxH3ChunkedV5RelayProjectEXPT8"}
            for node in nodes.values()):
        raise ValueError("v5 standalone Relay expects explicit windows and no existing effects")
    old_positive = links[next(item["link"] for item in windows[0]["inputs"]
                              if item["name"] == "positive")]
    high = nodes[old_positive[1]]
    if high["type"] != "MiniMaxH3AudioConditioningT8" or high["widgets_values"][4] != "FL2VA":
        raise ValueError("This private v5 Relay candidate expects frozen FL2VA HIGH conditioning")
    raw_model_link = links[next(item["link"] for item in windows[0]["inputs"]
                                if item["name"] == "model")]
    if any(links[next(item["link"] for item in window["inputs"]
                      if item["name"] == "model")][1:3] != raw_model_link[1:3]
           for window in windows[1:]):
        raise ValueError("PASS2 windows must use the same unchanged raw HIGH branch")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [410, max(180, 48 + 26 * len(inputs))], "flags": {},
                "order": max(item["order"] for item in nodes.values()) + 1, "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {}),
                            **({"widget": {"name": name}} if widget else {})}
                           for name, dtype, optional, widget in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind, "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def origin(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        return None if link_id is None else (links[link_id][1], links[link_id][2], links[link_id][5])

    def connect(source, target, name, dtype):
        nonlocal next_link
        source_id, source_slot = source
        target_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == name)
        link = [next_link, source_id, source_slot, target["id"], target_slot, dtype]
        graph["links"].append(link)
        links[next_link] = link
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    def remove_input_link(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        old_id = item["link"]
        link = links.pop(old_id)
        graph["links"] = [candidate for candidate in graph["links"] if candidate[0] != old_id]
        nodes[link[1]]["outputs"][link[2]]["links"].remove(old_id)
        item["link"] = None

    x, y = windows[0]["pos"]
    high_values = high["widgets_values"]
    plan_node = make(
        "MiniMaxH3PromptRelayPlanT8Advanced", "Global HIGH external Relay timeline",
        (x - 1020, y - 1040), [],
        [("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("compiled_prompt", "STRING"), ("length", "INT"),
         ("timeline_json", "STRING"), ("report_json", "STRING")],
        [high_values[0] + " <Picture 1> is the first frame; <Picture 2> is the last frame.",
         "Maintain the same subject and scene with subtle natural motion.\n"
         "Continue the same shot toward the specified last frame without a cut.",
         high_values[3], "auto_equal", "", "paper_v1", 0.1, False, False],
    )
    relay_inputs = [(name, dtype, optional, widget) for name, dtype, optional, widget in (
        ("model", "MODEL", False, False), ("clip", "CLIP", False, False),
        ("video_vae", "VAE", False, False), ("audio_vae", "VAE", False, False),
        ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN", False, False),
        ("width", "INT", False, True), ("height", "INT", False, True),
        ("drive_audio", "AUDIO", True, False), ("final_audio", "AUDIO", True, False),
        ("first_frame", "IMAGE", True, False), ("last_frame", "IMAGE", True, False),
    )]
    relay_cond = make(
        "MiniMaxH3PromptRelayConditioningT8Advanced", "Full-clip external Relay Conditioning",
        (x - 560, y - 1040), relay_inputs,
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
         ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [high_values[1], high_values[2], high_values[4], high_values[5],
         high_values[6], high_values[7], high_values[8], high_values[9],
         high_values[10], high_values[11], "apply_exp", 256],
    )
    connect(raw_model_link[1:3], relay_cond, "model", "MODEL")
    for name in ("clip", "video_vae", "audio_vae", "width", "height",
                 "drive_audio", "final_audio", "first_frame", "last_frame"):
        source = origin(high, name)
        if source is not None:
            connect(source[:2], relay_cond, name, source[2])
    connect((plan_node["id"], 0), relay_cond, "prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN")
    for index, window in enumerate(windows):
        wx, wy = window["pos"]
        project = make(
            "MiniMaxH3ChunkedV5RelayProjectEXPT8", f"Project global Relay to window {index}",
            (wx + 150, wy - 620),
            [("raw_high_model", "MODEL", False, False),
             ("relay_model", "MODEL", False, False),
             ("relay_positive", "CONDITIONING", False, False),
             ("relay_full_av_latent", "LATENT", False, False),
             ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN", False, False),
             ("partial4_denoised_output", "LATENT", False, False),
             ("lifted_full_av", "LATENT", False, False),
             ("prepared", PREPARED, False, False),
             ("plan", PLAN, False, False), ("sigmas", "SIGMAS", False, False),
             ("previous_result", RESULT, True, False)],
            [("model", "MODEL"), ("positive", "CONDITIONING"),
             ("runtime", "T8_CHUNKED_V5_RELAY_RUNTIME"),
             ("report_json", "STRING")], [index],
        )
        audit = (make(
            "MiniMaxH3ChunkedV5RelayAuditEXPT8", f"Window {index} actual Relay calls",
            (wx + 620, wy - 620),
            [("window_result", RESULT, False, False),
             ("prepared", PREPARED, False, False),
             ("runtime", "T8_CHUNKED_V5_RELAY_RUNTIME", False, False)],
            [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")],
        ) if with_audit else None)
        connect(raw_model_link[1:3], project, "raw_high_model", "MODEL")
        for name, slot, dtype in (("relay_model", 0, "MODEL"),
                                  ("relay_positive", 1, "CONDITIONING"),
                                  ("relay_full_av_latent", 2, "LATENT")):
            connect((relay_cond["id"], slot), project, name, dtype)
        connect((plan_node["id"], 0), project, "prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN")
        for name in ("partial4_denoised_output", "lifted_full_av", "prepared", "plan", "sigmas"):
            source = origin(window, name)
            connect(source[:2], project, name, source[2])
        if index:
            connect((windows[index - 1]["id"], 1), project, "previous_result", RESULT)
        remove_input_link(window, "model")
        remove_input_link(window, "positive")
        connect((project["id"], 0), window, "model", "MODEL")
        connect((project["id"], 1), window, "positive", "CONDITIONING")
        if audit is not None:
            connect((window["id"], 1), audit, "window_result", RESULT)
            prepared_origin = origin(window, "prepared")
            connect(prepared_origin[:2], audit, "prepared", prepared_origin[2])
            connect((project["id"], 2), audit, "runtime", "T8_CHUNKED_V5_RELAY_RUNTIME")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def split_api(original, *, window_count=2, temporal_chunk_frames=136,
              temporal_overlap_frames=34):
    graph = deepcopy(original)
    if graph["15"]["class_type"] != "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced":
        raise ValueError("Expected frozen v5 API executor")
    if (window_count, temporal_chunk_frames, temporal_overlap_frames) not in {
            (2, 136, 34), (3, 102, 34), (4, 85, 34)}:
        raise ValueError("This fixed 192-frame v5 API supports only its verified 2/3/4-window plans")
    graph["14"]["inputs"].update(temporal_chunk_frames=temporal_chunk_frames,
                                   temporal_overlap_frames=temporal_overlap_frames)
    old_inputs = graph["15"]["inputs"]
    next_id = max(map(int, graph)) + 1

    def add(kind, values):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": values}
        return key

    high_lora = add(graph["23"]["class_type"], deepcopy(graph["23"]["inputs"]))
    high_setup_values = deepcopy(graph["8"]["inputs"])
    high_setup_values["model"] = [high_lora, 0]
    high_setup = add(graph["8"]["class_type"], high_setup_values)
    lift = add("MiniMaxH3ChunkedV5GlobalLiftEXPT8", {
        "partial4_denoised_output": old_inputs["latent"], "plan": old_inputs["plan"],
    })
    prepare = add("MiniMaxH3ChunkedV5PrepareEXPT8", {
        "partial4_denoised_output": old_inputs["latent"],
        "lifted_full_av": [lift, 0], "lift_receipt": [lift, 1],
        "plan": old_inputs["plan"], "noise": old_inputs["noise"],
    })
    common = {"model": [high_setup, 0], "positive": old_inputs["conditioning"],
              "partial4_denoised_output": old_inputs["latent"],
              "lifted_full_av": [lift, 0], "prepared": [prepare, 0],
              "plan": old_inputs["plan"], "noise": old_inputs["noise"],
              "sampler": [high_setup, 1], "sigmas": old_inputs["sigmas"],
              "cfg": old_inputs.get("cfg", 1.0)}
    previous = None
    for index in range(window_count):
        inputs = {**common, "window_index": index}
        if previous is not None:
            inputs["previous_result"] = [previous, 1]
        previous = add("MiniMaxH3ChunkedV5PASS2WindowEXPT8", inputs)
    for node in graph.values():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == "15":
                node["inputs"][name] = [previous, {0: 0, 1: 2}[value[1]]]
    del graph["15"]
    return graph


def add_eav_api(original):
    graph = deepcopy(original)
    windows = sorted(((key, item) for key, item in graph.items()
                      if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda pair: pair[1]["inputs"]["window_index"])
    if len(windows) < 2 or any(item["class_type"] == "MiniMaxH3ChunkedV5EAVApplyEXPT8"
                                for item in graph.values()):
        raise ValueError("External v5 EAV expects untouched explicit PASS2 windows")
    next_id = max(map(int, graph)) + 1

    def add(kind, inputs):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": inputs}
        return key

    for window_id, window in windows:
        values = window["inputs"]
        config = add("MiniMaxH3StageEAVConfigEXPT8", {
            "mode": "report_only", "tau": 4.0, "start_video_progress": 0.15,
            "end_video_progress": 0.90, "max_workspace_mib": 32, "g_hard_limit": 1.5,
        })
        inputs = {"model": values["model"], "sigmas": values["sigmas"],
                  "partial4_denoised_output": values["partial4_denoised_output"],
                  "lifted_full_av": values["lifted_full_av"],
                  "prepared": values["prepared"], "plan": values["plan"],
                  "window_index": values["window_index"], "eav_config": [config, 0]}
        if "previous_result" in values:
            inputs["previous_result"] = values["previous_result"]
        if graph[str(values["positive"][0])]["class_type"] == "MiniMaxH3ChunkedV5RelayProjectEXPT8":
            inputs["relay_positive"] = values["positive"]
        apply = add("MiniMaxH3ChunkedV5EAVApplyEXPT8", inputs)
        values["model"] = [apply, 0]
        add("MiniMaxH3ChunkedV5EAVAuditEXPT8", {
            "window_result": [window_id, 1], "prepared": values["prepared"],
            "runtime": [apply, 1],
        })
    return graph


def add_relay_api(original, *, with_audit=True):
    """Use one global external Relay text/conditioning with local layouts."""
    graph = deepcopy(original)
    windows = sorted(((key, item) for key, item in graph.items()
                      if item["class_type"] == "MiniMaxH3ChunkedV5PASS2WindowEXPT8"),
                     key=lambda pair: pair[1]["inputs"]["window_index"])
    if len(windows) < 2 or any(item["class_type"] in {
            "MiniMaxH3ChunkedV5EAVApplyEXPT8", "MiniMaxH3ChunkedV5RelayProjectEXPT8"}
            for item in graph.values()):
        raise ValueError("v5 standalone Relay expects explicit windows and no existing effects")
    high = graph[str(windows[0][1]["inputs"]["positive"][0])]
    if (high["class_type"] != "MiniMaxH3AudioConditioningT8"
            or high["inputs"]["task_type"] != "FL2VA"):
        raise ValueError("This private v5 Relay candidate expects frozen FL2VA HIGH conditioning")
    original_high = high["inputs"]
    raw_model = windows[0][1]["inputs"]["model"]
    if any(window["inputs"]["model"] != raw_model for _, window in windows[1:]):
        raise ValueError("v5 PASS2 windows must share the unchanged raw HIGH branch")
    next_id = max(map(int, graph)) + 1

    def add(kind, inputs):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": inputs}
        return key

    relay_plan = add("MiniMaxH3PromptRelayPlanT8Advanced", {
        "global_prompt": original_high["prompt"]
        + " <Picture 1> is the first frame; <Picture 2> is the last frame.",
        "local_prompts": "Maintain the same subject and scene with subtle natural motion.\n"
                         "Continue the same shot toward the specified last frame without a cut.",
        "length": original_high["length"], "timing_mode": "auto_equal",
        "time_ranges": "", "math_profile": "paper_v1", "epsilon": 0.1,
        "allow_gaps": False, "allow_overlaps": False,
    })
    conditioning = {"model": raw_model, "clip": original_high["clip"],
                    "video_vae": original_high["video_vae"],
                    "audio_vae": original_high["audio_vae"],
                    "prompt_relay_plan": [relay_plan, 0],
                    "width": original_high["width"], "height": original_high["height"],
                    "task_type": original_high["task_type"],
                    "audio_mode": original_high["audio_mode"],
                    "audio_denoise_strength": original_high["audio_denoise_strength"],
                    "add_source_as_reference": original_high["add_source_as_reference"],
                    "prompt_primary_audio_ordinal": original_high["prompt_primary_audio_ordinal"],
                    "strict_prompt_tags": original_high["strict_prompt_tags"],
                    "ref_image_size": original_high["ref_image_size"],
                    "reference_video_policy": original_high["reference_video_policy"],
                    "execution_mode": "apply_exp", "query_chunk_rows": 256}
    for name in ("drive_audio", "final_audio", "first_frame", "last_frame"):
        if name in original_high:
            conditioning[name] = original_high[name]
    relay_cond = add("MiniMaxH3PromptRelayConditioningT8Advanced", conditioning)
    for window_id, window in windows:
        values = window["inputs"]
        project = {"raw_high_model": raw_model, "relay_model": [relay_cond, 0],
                   "relay_positive": [relay_cond, 1],
                   "relay_full_av_latent": [relay_cond, 2],
                   "prompt_relay_plan": [relay_plan, 0],
                   "partial4_denoised_output": values["partial4_denoised_output"],
                   "lifted_full_av": values["lifted_full_av"],
                   "prepared": values["prepared"], "plan": values["plan"],
                   "sigmas": values["sigmas"],
                   "window_index": values["window_index"]}
        if "previous_result" in values:
            project["previous_result"] = values["previous_result"]
        project_id = add("MiniMaxH3ChunkedV5RelayProjectEXPT8", project)
        values["model"] = [project_id, 0]
        values["positive"] = [project_id, 1]
        if with_audit:
            add("MiniMaxH3ChunkedV5RelayAuditEXPT8", {
                "window_result": [window_id, 1], "prepared": values["prepared"],
                "runtime": [project_id, 2],
            })
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-eav", action="store_true")
    parser.add_argument("--with-relay", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Write a new private artifact path; never overwrite an old workflow")
    candidate = split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    if args.with_relay:
        candidate = add_relay_frontend(candidate, with_audit=not args.with_eav)
    if args.with_eav:
        candidate = add_eav_frontend(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "nodes": len(candidate["nodes"]),
                      "links": len(candidate["links"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
