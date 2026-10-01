"""Build a private fixed H16-3 split copy; never edit the old example.

The source example requests 124 aligned frames with guarded 34/17 windows.
That is seven actual native time windows. Changed length or geometry needs a
new graph, not silent truncation of the last frames.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/workflows/13-latent-upscale/2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json"
PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
SPEC = "T8_CHUNKED_SOURCE_SEGMENT"
CONTEXT = "T8_CHUNKED_PASS2_CONTEXT"
H16_RESULT = "T8_H16_PASS2_RESULT"
CORE_RESULT = "T8_CHUNKED_PASS2_RESULT"
WINDOW_COUNT = 7
RELAY_RUNTIME = "T8_H16_RELAY_RUNTIME"


def split_frontend(original):
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    old = nodes[19]
    if (old["type"] != "DeciiaChunkedPass2Sampler"
            or old["widgets_values"] != ["guarded_overlap_exp", 34, 17, 0.999, "refined_exp"]
            or nodes[7]["widgets_values"][3] != 124
            or nodes[14]["widgets_values"][3] != 124
            or nodes[15]["type"] != "MiniMaxH3TwoPassLatentReconcileT8Advanced"
            or nodes[5]["type"] != "LoraLoaderBypassModelOnly"):
        raise ValueError("Expected the frozen 124-frame/7-window H16-3 template")
    expected = {"noise": (18, 0), "guider": (17, 0), "sampler": (16, 1),
                "sigmas": (16, 2), "latent_image": (15, 0)}
    for item in old["inputs"][:5]:
        link = links[item["link"]]
        if tuple(link[1:3]) != expected[item["name"]]:
            raise ValueError("Frozen H16 input contract changed")
    if [(link[3], link[4]) for link in graph["links"] if link[1] == 19] != [(20, 0)]:
        raise ValueError("Frozen H16 output contract changed")

    removed_nodes = {17, 19}
    graph["links"] = [link for link in graph["links"]
                      if link[1] not in removed_nodes and link[3] not in removed_nodes]
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in removed_nodes]
    nodes = {node["id"]: node for node in graph["nodes"]}
    for node in nodes.values():
        for item in node.get("inputs", []):
            if item.get("link") in links and (links[item["link"]][1] in removed_nodes
                                               or links[item["link"]][3] in removed_nodes):
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [link_id for link_id in item.get("links") or []
                             if links[link_id][1] not in removed_nodes
                             and links[link_id][3] not in removed_nodes]
    next_node = max(nodes) + 1
    next_link = max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [405, max(180, 48 + 25 * len(inputs))], "flags": {},
                "order": old["order"], "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {})}
                           for name, dtype, optional in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind,
                               "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def connect(source, target, name, dtype):
        nonlocal next_link
        source_id, source_slot = source
        slot = next(index for index, item in enumerate(target["inputs"])
                    if item["name"] == name)
        if target["inputs"][slot]["link"] is not None:
            raise ValueError(f"Input already connected: {target['id']}:{name}")
        graph["links"].append([next_link, source_id, source_slot, target["id"], slot, dtype])
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][slot]["link"] = next_link
        next_link += 1

    high_lora = deepcopy(nodes[5])
    high_lora["id"] = next_node
    next_node += 1
    high_lora["title"] = "PASS2 independent LoRA (same old default)"
    high_lora["pos"] = [old["pos"][0] - 600, old["pos"][1] + 500]
    for item in high_lora["inputs"]:
        item["link"] = None
    for item in high_lora["outputs"]:
        item["links"] = []
    graph["nodes"].append(high_lora)
    nodes[high_lora["id"]] = high_lora
    connect((4, 0), high_lora, "model", "MODEL")
    mixer_input = next(item for item in nodes[16]["inputs"] if item["name"] == "model")
    old_mixer_link = mixer_input["link"]
    graph["links"] = [link for link in graph["links"] if link[0] != old_mixer_link]
    nodes[5]["outputs"][0]["links"].remove(old_mixer_link)
    mixer_input["link"] = None
    connect((high_lora["id"], 0), nodes[16], "model", "MODEL")

    x, y = old["pos"]
    plan = make("MiniMaxH3H16Pass2PlanEXPT8", "H16 plan — old 34/17 contract",
                (x - 440, y - 580), [("first_pass_latent", "LATENT", False)],
                [("plan", PLAN), ("report_json", "STRING")],
                ["guarded_overlap_exp", 34, 17, 0.999,
                 "minimax_h3_latent_upscaler_3d_fp16.safetensors"])
    prepare = make("MiniMaxH3ChunkedPass2PrepareEXPT8", "H16 once-only global AV noise/mask",
                   (x, y - 580), [("first_pass_latent", "LATENT", False),
                                  ("plan", PLAN, False), ("noise", "NOISE", False)],
                   [("pass2_context", CONTEXT), ("report_json", "STRING")])
    connect((15, 0), plan, "first_pass_latent", "LATENT")
    connect((15, 0), prepare, "first_pass_latent", "LATENT")
    connect((plan["id"], 0), prepare, "plan", PLAN)
    connect((18, 0), prepare, "noise", "NOISE")
    prior = None
    windows = []
    for index in range(WINDOW_COUNT):
        col, row = index % 2, index // 2
        sx, sy = x + col * 1390, y + row * 620
        source = make("MiniMaxH3ChunkedSourceSegmentEXPT8", f"H16 source window {index}",
                      (sx, sy), [("first_pass_latent", "LATENT", False),
                                  ("plan", PLAN, False)],
                      [("segment_latent", "LATENT"), ("segment_spec", SPEC),
                       ("report_json", "STRING")], [index])
        lift = make("MiniMaxH3ChunkedLearnedLiftEXPT8", f"H16 learned same-size lift {index}",
                    (sx + 450, sy), [("source_segment", "LATENT", False),
                                     ("segment_spec", SPEC, False),
                                     ("pass2_context", CONTEXT, False),
                                     ("plan", PLAN, False)],
                    [("lifted_segment", "LATENT"), ("report_json", "STRING")])
        stage = make("MiniMaxH3H16Pass2WindowEXPT8", f"H16 independent PASS2 window {index}",
                     (sx + 900, sy),
                     [("model", "MODEL", False), ("positive", "CONDITIONING", False),
                      ("source_segment", "LATENT", False),
                      ("lifted_segment", "LATENT", False),
                      ("segment_spec", SPEC, False), ("pass2_context", CONTEXT, False),
                      ("plan", PLAN, False), ("noise", "NOISE", False),
                      ("sampler", "SAMPLER", False), ("sigmas", "SIGMAS", False),
                      ("previous_result", H16_RESULT, True),
                      ("negative", "CONDITIONING", True)],
                     [("cumulative_av_latent", "LATENT"),
                      ("window_result", H16_RESULT),
                      ("core_segment_result", CORE_RESULT),
                      ("report_json", "STRING")], ["refined_exp", 1.0])
        for target, name, parent, dtype in (
            (source, "first_pass_latent", (15, 0), "LATENT"),
            (source, "plan", (plan["id"], 0), PLAN),
            (lift, "source_segment", (source["id"], 0), "LATENT"),
            (lift, "segment_spec", (source["id"], 1), SPEC),
            (lift, "pass2_context", (prepare["id"], 0), CONTEXT),
            (lift, "plan", (plan["id"], 0), PLAN),
            (stage, "model", (16, 0), "MODEL"),
            (stage, "positive", (15, 1), "CONDITIONING"),
            (stage, "source_segment", (source["id"], 0), "LATENT"),
            (stage, "lifted_segment", (lift["id"], 0), "LATENT"),
            (stage, "segment_spec", (source["id"], 1), SPEC),
            (stage, "pass2_context", (prepare["id"], 0), CONTEXT),
            (stage, "plan", (plan["id"], 0), PLAN),
            (stage, "noise", (18, 0), "NOISE"),
            (stage, "sampler", (16, 1), "SAMPLER"),
            (stage, "sigmas", (16, 2), "SIGMAS"),
        ):
            connect(parent, target, name, dtype)
        if prior is not None:
            connect((prior["id"], 1), stage, "previous_result", H16_RESULT)
        prior = stage
        windows.append(stage)
    connect((prior["id"], 0), nodes[20], nodes[20]["inputs"][0]["name"], "LATENT")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def add_eav_frontend(original):
    """Give every fixed H16 window its own external EAV config/apply/audit."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8"),
                     key=lambda node: node["id"])
    if len(windows) != WINDOW_COUNT or any(
            node["type"] == "MiniMaxH3H16EAVApplyEXPT8" for node in nodes.values()):
        raise ValueError("H16 EAV needs seven untouched explicit PASS2 windows")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [405, max(180, 48 + 25 * len(inputs))], "flags": {},
                "order": windows[0]["order"], "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {})}
                           for name, dtype, optional in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind,
                               "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def connect(source, target, name, dtype):
        nonlocal next_link
        slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == name)
        if target["inputs"][slot]["link"] is not None:
            raise ValueError(f"Already connected: {target['id']}:{name}")
        graph["links"].append([next_link, source[0], source[1], target["id"], slot, dtype])
        nodes[source[0]]["outputs"][source[1]]["links"].append(next_link)
        target["inputs"][slot]["link"] = next_link
        next_link += 1

    def parent(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        return links[item["link"]][1:3] if item["link"] is not None else None

    audits = []
    for index, window in enumerate(windows):
        x, y = window["pos"]
        positive_source = parent(window, "positive")
        composed_relay = nodes[positive_source[0]]["type"] == "MiniMaxH3H16RelayProjectEXPT8"
        config = make("MiniMaxH3StageEAVConfigEXPT8", f"H16 window {index} EAV config",
                      (x - 450, y + 450), [], [("eav_config", "T8_STAGE_EAV_CONFIG")],
                      ["report_only", 4.0, 0.10 if composed_relay else 0.15, 0.90, 32,
                       3.0 if composed_relay else 1.5])
        apply = make("MiniMaxH3H16EAVApplyEXPT8", f"H16 window {index} EAV apply",
                     (x, y + 450),
                     [("model", "MODEL", False), ("sigmas", "SIGMAS", False),
                      ("source_segment", "LATENT", False),
                      ("lifted_segment", "LATENT", False),
                      ("segment_spec", SPEC, False),
                      ("pass2_context", CONTEXT, False), ("plan", PLAN, False),
                      ("eav_config", "T8_STAGE_EAV_CONFIG", False),
                      ("previous_result", H16_RESULT, True),
                      *([("relay_positive", "CONDITIONING", True)] if composed_relay else [])],
                     [("model", "MODEL"), ("runtime", "T8_STAGE_EAV_RUNTIME"),
                      ("report_json", "STRING"), ("stage_context", "T8_STAGE_CONTEXT")],
                     ["refined_exp"])
        audit = make("MiniMaxH3H16EAVAuditEXPT8", f"H16 window {index} actual EAV calls",
                     (x + 450, y + 450),
                     [("window_result", H16_RESULT, False),
                      ("segment_spec", SPEC, False),
                      ("runtime", "T8_STAGE_EAV_RUNTIME", False)],
                     [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")])
        model_item = next(item for item in window["inputs"] if item["name"] == "model")
        old_link_id = model_item["link"]
        old_model = links[old_link_id]
        graph["links"] = [link for link in graph["links"] if link[0] != old_link_id]
        nodes[old_model[1]]["outputs"][old_model[2]]["links"].remove(old_link_id)
        model_item["link"] = None
        connect((old_model[1], old_model[2]), apply, "model", "MODEL")
        for name, dtype in (("sigmas", "SIGMAS"), ("source_segment", "LATENT"),
                            ("lifted_segment", "LATENT"), ("segment_spec", SPEC),
                            ("pass2_context", CONTEXT), ("plan", PLAN)):
            connect(parent(window, name), apply, name, dtype)
        connect((config["id"], 0), apply, "eav_config", "T8_STAGE_EAV_CONFIG")
        if index:
            connect(parent(window, "previous_result"), apply, "previous_result", H16_RESULT)
        if composed_relay:
            connect(positive_source, apply, "relay_positive", "CONDITIONING")
        connect((apply["id"], 0), window, "model", "MODEL")
        connect((window["id"], 1), audit, "window_result", H16_RESULT)
        connect(parent(window, "segment_spec"), audit, "segment_spec", SPEC)
        connect((apply["id"], 1), audit, "runtime", "T8_STAGE_EAV_RUNTIME")
        audits.append(audit)
    final_item = next(item for item in nodes[20]["inputs"] if item["name"] == "av_latent")
    final_link_id = final_item["link"]
    final_link = links[final_link_id]
    graph["links"] = [link for link in graph["links"] if link[0] != final_link_id]
    nodes[final_link[1]]["outputs"][final_link[2]]["links"].remove(final_link_id)
    final_item["link"] = None
    connect((audits[-1]["id"], 0), nodes[20], "av_latent", "LATENT")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


def add_relay_frontend(original, *, with_audit=True):
    """One external full-clip Relay plan, independently projected to seven windows."""
    graph = deepcopy(original)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8"),
                     key=lambda node: node["id"])
    if len(windows) != WINDOW_COUNT or any(node["type"] in {
            "MiniMaxH3H16EAVApplyEXPT8", "MiniMaxH3H16RelayProjectEXPT8"}
            for node in nodes.values()):
        raise ValueError("H16 Relay expects seven untouched explicit PASS2 windows")
    if (nodes[14]["type"] != "MiniMaxH3AudioConditioningT8"
            or nodes[14]["widgets_values"][4] != "I2VA"
            or nodes[15]["type"] != "MiniMaxH3TwoPassLatentReconcileT8Advanced"):
        raise ValueError("This fixed H16 Relay candidate expects the original I2VA HIGH path")
    next_node, next_link = max(nodes) + 1, max(links) + 1

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [420, max(180, 48 + 25 * len(inputs))], "flags": {},
                "order": windows[0]["order"], "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None,
                            **({"shape": 7} if optional else {}),
                            **({"widget": {"name": name}} if widget else {})}
                           for name, dtype, optional, widget in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind,
                               "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        graph["nodes"].append(node)
        nodes[next_node] = node
        next_node += 1
        return node

    def parent(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        link = links[item["link"]] if item["link"] is not None else None
        return None if link is None else (link[1], link[2], link[5])

    def connect(source, target, name, dtype):
        nonlocal next_link
        slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == name)
        if target["inputs"][slot]["link"] is not None:
            raise ValueError(f"Already connected: {target['id']}:{name}")
        link = [next_link, source[0], source[1], target["id"], slot, dtype]
        graph["links"].append(link)
        links[next_link] = link
        nodes[source[0]]["outputs"][source[1]]["links"].append(next_link)
        target["inputs"][slot]["link"] = next_link
        next_link += 1

    def remove_input(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        old_id = item["link"]
        link = links.pop(old_id)
        graph["links"] = [candidate for candidate in graph["links"] if candidate[0] != old_id]
        nodes[link[1]]["outputs"][link[2]]["links"].remove(old_id)
        item["link"] = None

    original_model = parent(windows[0], "model")
    original_positive = parent(windows[0], "positive")
    if any(parent(window, "model") != original_model
           or parent(window, "positive") != original_positive for window in windows):
        raise ValueError("H16 windows no longer share the frozen HIGH model/conditioning")
    high = nodes[14]
    high_values = high["widgets_values"]
    x, y = windows[0]["pos"]
    plan_node = make(
        "MiniMaxH3PromptRelayPlanT8Advanced", "H16 global HIGH external Relay timeline",
        (x - 1040, y - 1080), [],
        [("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("compiled_prompt", "STRING"), ("length", "INT"),
         ("timeline_json", "STRING"), ("report_json", "STRING")],
        [high_values[0] + " <Picture 1> is the opening frame.",
         "Maintain the same subject and locked-off scene without a cut.\n"
         "Continue the same shot with natural synchronized motion.",
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
        "MiniMaxH3PromptRelayConditioningT8Advanced", "H16 full-clip external Relay Conditioning",
        (x - 560, y - 1080), relay_inputs,
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
         ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [high_values[1], high_values[2], high_values[4], high_values[5],
         high_values[6], high_values[7], high_values[8], high_values[9],
         high_values[10], high_values[11], "apply_exp", 256],
    )
    connect(original_model[:2], relay_cond, "model", "MODEL")
    for name in ("clip", "video_vae", "audio_vae", "width", "height",
                 "drive_audio", "final_audio", "first_frame", "last_frame"):
        origin = parent(high, name)
        if origin is not None:
            connect(origin[:2], relay_cond, name, origin[2])
    connect((plan_node["id"], 0), relay_cond, "prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN")
    audits = []
    for index, window in enumerate(windows):
        wx, wy = window["pos"]
        project = make(
            "MiniMaxH3H16RelayProjectEXPT8", f"Project global Relay to H16 window {index}",
            (wx + 130, wy - 610),
            [("raw_high_model", "MODEL", False, False),
             ("relay_model", "MODEL", False, False),
             ("relay_positive", "CONDITIONING", False, False),
             ("relay_full_av_latent", "LATENT", False, False),
             ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN", False, False),
             ("source_segment", "LATENT", False, False),
             ("lifted_segment", "LATENT", False, False),
             ("segment_spec", SPEC, False, False),
             ("pass2_context", CONTEXT, False, False), ("plan", PLAN, False, False),
             ("sigmas", "SIGMAS", False, False),
             ("previous_result", H16_RESULT, True, False)],
            [("model", "MODEL"), ("positive", "CONDITIONING"),
             ("runtime", RELAY_RUNTIME), ("report_json", "STRING")],
            ["refined_exp"],
        )
        audit = (make(
            "MiniMaxH3H16RelayAuditEXPT8", f"H16 window {index} actual Relay calls",
            (wx + 590, wy - 610),
            [("window_result", H16_RESULT, False, False),
             ("segment_spec", SPEC, False, False),
             ("runtime", RELAY_RUNTIME, False, False)],
            [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")],
        ) if with_audit else None)
        connect(original_model[:2], project, "raw_high_model", "MODEL")
        for name, slot, dtype in (("relay_model", 0, "MODEL"),
                                  ("relay_positive", 1, "CONDITIONING"),
                                  ("relay_full_av_latent", 2, "LATENT")):
            connect((relay_cond["id"], slot), project, name, dtype)
        connect((plan_node["id"], 0), project, "prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN")
        for name in ("source_segment", "lifted_segment", "segment_spec",
                     "pass2_context", "plan", "sigmas"):
            origin = parent(window, name)
            connect(origin[:2], project, name, origin[2])
        if index:
            origin = parent(window, "previous_result")
            connect(origin[:2], project, "previous_result", origin[2])
        remove_input(window, "model")
        remove_input(window, "positive")
        connect((project["id"], 0), window, "model", "MODEL")
        connect((project["id"], 1), window, "positive", "CONDITIONING")
        if audit is not None:
            connect((window["id"], 1), audit, "window_result", H16_RESULT)
            origin = parent(window, "segment_spec")
            connect(origin[:2], audit, "segment_spec", origin[2])
            connect((project["id"], 2), audit, "runtime", RELAY_RUNTIME)
            audits.append(audit)
    if audits:
        remove_input(nodes[20], "av_latent")
        connect((audits[-1]["id"], 0), nodes[20], "av_latent", "LATENT")
    graph["last_node_id"] = max(nodes)
    graph["last_link_id"] = next_link - 1
    return graph


WIDGET_NAMES = {
    1: ["vae_name"], 2: ["vae_name"], 3: ["clip_name", "type", "device"],
    4: ["unet_name", "weight_dtype"], 5: ["lora_name", "strength_model"],
    6: ["image"],
    7: ["prompt", "width", "height", "length", "task_type", "audio_mode",
        "audio_denoise_strength", "add_source_as_reference",
        "prompt_primary_audio_ordinal", "strict_prompt_tags", "ref_image_size",
        "reference_video_policy", "allow_above_reference_area"],
    8: ["steps", "shift_video", "shift_audio", "sampler_name", "scheduler"],
    9: ["base_steps", "coarse_steps", "refine_steps"], 10: [],
    11: ["noise_seed"], 12: [],
    13: ["model_name", "size_mode", "scale_by", "target_megapixels",
         "target_width", "target_height", "aspect_policy", "max_anisotropy",
         "precision", "release_policy"],
    14: ["prompt", "width", "height", "length", "task_type", "audio_mode",
         "audio_denoise_strength", "add_source_as_reference",
         "prompt_primary_audio_ordinal", "strict_prompt_tags", "ref_image_size",
         "reference_video_policy", "allow_above_reference_area"],
    15: ["audio_policy", "second_pass_audio_source", "second_pass_audio_strength"],
    16: ["shift_video", "shift_audio", "enable_tail", "extra_tail_steps",
         "tail_spacing", "enable_model_time_bias", "bias", "bias_start_progress",
         "bias_end_progress", "bias_domain", "enable_stg", "stg_scale",
         "stg_double_blocks", "stg_start_progress", "stg_end_progress",
         "enable_restart", "restart_video_sigma", "restart_steps", "restart_seed"],
    18: ["noise_seed"], 20: [],
}


def split_api(frontend):
    """Mirror the fixed candidate's sampling dependency graph, excluding VHS."""
    nodes = {node["id"]: node for node in frontend["nodes"]}
    graph = {}
    for node_id, node in nodes.items():
        kind = node["type"]
        if kind in {"MarkdownNote", "VHS_VideoCombine"} or node_id == 20:
            continue
        if node_id in WIDGET_NAMES:
            names = WIDGET_NAMES[node_id]
        elif kind == "LoraLoaderBypassModelOnly":
            names = WIDGET_NAMES[5]
        elif kind == "MiniMaxH3H16Pass2PlanEXPT8":
            names = ["temporal_strategy", "temporal_chunk_frames",
                     "temporal_overlap_frames", "anchor_strength", "model_name"]
        elif kind == "MiniMaxH3ChunkedSourceSegmentEXPT8":
            names = ["segment_index"]
        elif kind == "MiniMaxH3H16Pass2WindowEXPT8":
            names = ["audio_output", "cfg"]
        elif kind == "MiniMaxH3StageEAVConfigEXPT8":
            names = ["mode", "tau", "start_video_progress", "end_video_progress",
                     "max_workspace_mib", "g_hard_limit"]
        elif kind == "MiniMaxH3H16EAVApplyEXPT8":
            names = ["audio_output"]
        elif kind == "MiniMaxH3PromptRelayPlanT8Advanced":
            names = ["global_prompt", "local_prompts", "length", "timing_mode",
                     "time_ranges", "math_profile", "epsilon", "allow_gaps", "allow_overlaps"]
        elif kind == "MiniMaxH3PromptRelayConditioningT8Advanced":
            names = ["width", "height", "task_type", "audio_mode",
                     "audio_denoise_strength", "add_source_as_reference",
                     "prompt_primary_audio_ordinal", "strict_prompt_tags",
                     "ref_image_size", "reference_video_policy", "execution_mode",
                     "query_chunk_rows"]
        elif kind == "MiniMaxH3H16RelayProjectEXPT8":
            names = ["audio_output"]
        else:
            names = []
        values = node.get("widgets_values", [])
        if len(names) > len(values):
            raise ValueError(f"Candidate widget contract changed: {node_id} {kind}")
        graph[str(node_id)] = {"class_type": kind,
                               "inputs": dict(zip(names, values[:len(names)], strict=True))}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        if str(target) in graph:
            name = nodes[target]["inputs"][target_slot]["name"]
            graph[str(target)]["inputs"][name] = [str(source), source_slot]
    final = max(node_id for node_id, node in nodes.items()
                if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8")
    audits = [node_id for node_id, node in nodes.items()
              if node["type"] == "MiniMaxH3H16EAVAuditEXPT8"]
    if not audits:
        audits = [node_id for node_id, node in nodes.items()
                  if node["type"] == "MiniMaxH3H16RelayAuditEXPT8"]
    graph["h16_report"] = {"class_type": "PreviewAny",
                           "inputs": {"source": ([str(max(audits)), 1] if audits
                                                 else [str(final), 3])}}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-eav", action="store_true")
    parser.add_argument("--with-relay", action="store_true")
    args = parser.parse_args()
    original = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    frontend = split_frontend(original)
    if args.with_relay:
        frontend = add_relay_frontend(frontend, with_audit=not args.with_eav)
    if args.with_eav:
        frontend = add_eav_frontend(frontend)
    target = ROOT / ("artifacts/development/modular-sampling-m4-h16-relay-20260923/"
                     "candidate-relay-eav-v3" if args.with_relay and args.with_eav else
                     "artifacts/development/modular-sampling-m4-h16-relay-20260923/"
                     "candidate-relay-v1" if args.with_relay else
                     "artifacts/development/modular-sampling-m4-h16-eav-20260923/"
                     "candidate-v1" if args.with_eav else
                     "artifacts/development/modular-sampling-m4-h16-20260923/candidate-v1")
    if target.exists():
        raise FileExistsError("Private candidate already exists; never overwrite evidence")
    target.mkdir(parents=True)
    name = "H16_124F_Seven_Windows_Separate_PASS2"
    if args.with_relay:
        name += "_Relay"
    if args.with_eav:
        name += "_EAV"
    name += "_EXP"
    (target / f"{name}.json").write_text(
        json.dumps(frontend, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / f"{name}.api.json").write_text(
        json.dumps(split_api(frontend), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frontend_nodes": len(frontend["nodes"]),
                      "frontend_links": len(frontend["links"]),
                      "api_nodes": len(split_api(frontend)),
                      "windows": WINDOW_COUNT}, indent=2))


if __name__ == "__main__":
    main()
