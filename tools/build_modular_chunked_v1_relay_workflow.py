"""Build opt-in three-segment Chunked v1 external Prompt Relay candidates.

The fixed 56-frame example is deliberately private. Changing its frame or
chunk settings requires rebuilding the number of segment instances.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools import run_community_update_real_validation as legacy_api
from tools.build_modular_chunked_workflow import (
    ROOT, TEMPLATES, PLAN, SPEC, CONTEXT, RESULT, RELAY_PLAN,
    add_external_eav_api, add_external_eav_frontend,
    append_second_api_segment, append_second_frontend_segment,
    split_api_graph, split_workflow, v1_source_from_v2,
)


RUNTIME = "T8_CHUNKED_V1_LOCAL_RELAY_RUNTIME"
PROJECT = "MiniMaxH3ChunkedV1RelayProjectEXPT8"
AUDIT = "MiniMaxH3ChunkedV1RelayAuditEXPT8"
PASS = "MiniMaxH3ChunkedPass2SegmentEXPT8"


def _frontend_passes(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    passes = []
    for node in graph["nodes"]:
        if node["type"] != PASS:
            continue
        spec_link = links[next(item["link"] for item in node["inputs"]
                               if item["name"] == "segment_spec")]
        index = nodes[spec_link[1]]["widgets_values"][0]
        passes.append((index, node))
    passes.sort(key=lambda item: item[0])
    if [index for index, _ in passes] != [0, 1, 2]:
        raise ValueError("Expected exactly three native-aligned v1 PASS2 segments")
    return [item for _, item in passes]


def add_v1_relay_frontend(graph, *, with_audit=True):
    graph = deepcopy(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    passes = _frontend_passes(graph)
    next_node = max(nodes) + 1
    next_link = max(links) + 1

    def origin(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        link = links[link_id]
        return link[1], link[2], link[5]

    def make(kind, title, pos, inputs, outputs, widgets=()):
        nonlocal next_node
        node = {"id": next_node, "type": kind, "title": title, "pos": list(pos),
                "size": [430, max(180, 48 + 26 * len(inputs))], "flags": {},
                "order": max(item["order"] for item in nodes.values()) + 1,
                "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None}
                           for name, dtype in inputs],
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
        target_slot = next(i for i, item in enumerate(target["inputs"])
                           if item["name"] == name)
        link = [next_link, source_id, source_slot, target["id"], target_slot, dtype]
        graph["links"].append(link)
        links[next_link] = link
        nodes[source_id]["outputs"][source_slot]["links"].append(next_link)
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    def disconnect(target, name):
        item = next(item for item in target["inputs"] if item["name"] == name)
        link_id = item["link"]
        if link_id is None:
            raise ValueError("Expected connected PASS2 " + name)
        link = links.pop(link_id)
        graph["links"].remove(link)
        nodes[link[1]]["outputs"][link[2]]["links"].remove(link_id)
        item["link"] = None

    first = passes[0]
    raw_origin = origin(first, "model")
    positive_origin = origin(first, "positive")
    high = nodes[positive_origin[0]]
    if high["type"] != "MiniMaxH3AudioConditioningT8":
        raise ValueError("Expected frozen HIGH conditioning for the v1 Relay candidate")
    values = high["widgets_values"]
    x, y = first["pos"]
    relay_plan = make(
        "MiniMaxH3PromptRelayPlanT8Advanced", "Full 56-frame PASS2 Relay timeline",
        (x - 1050, y - 1020), [],
        [("prompt_relay_plan", RELAY_PLAN), ("compiled_prompt", "STRING"),
         ("length", "INT"), ("timeline_json", "STRING"),
         ("report_json", "STRING")],
        [values[0] + " <Picture 1> is the first frame; <Picture 2> is the last frame.",
         "Maintain the same scene with continuous natural motion.\n"
         "The subject walks toward the final framing without a cut.",
         values[3], "auto_equal", "", "paper_v1", 0.1, False, False],
    )
    media_types = {"drive_audio": "AUDIO", "final_audio": "AUDIO",
                   "first_frame": "IMAGE", "last_frame": "IMAGE"}
    relay_cond = make(
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "Full-clip external Relay Conditioning — apply_exp", (x - 550, y - 1020),
        [("model", "MODEL"), ("clip", "CLIP"), ("video_vae", "VAE"),
         ("audio_vae", "VAE"), ("prompt_relay_plan", RELAY_PLAN)]
        + list(media_types.items()),
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
         ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [values[1], values[2], values[4], values[5], values[6], values[7],
         values[8], values[9], values[10], values[11], "apply_exp", 256],
    )
    connect(raw_origin[:2], relay_cond, "model", "MODEL")
    for name in ("clip", "video_vae", "audio_vae", *media_types):
        item = next(item for item in high["inputs"] if item["name"] == name)
        if item["link"] is not None:
            source_id, source_slot, dtype = origin(high, name)
            connect((source_id, source_slot), relay_cond, name, dtype)
    connect((relay_plan["id"], 0), relay_cond, "prompt_relay_plan", RELAY_PLAN)

    for index, stage in enumerate(passes):
        stage_inputs = {name: origin(stage, name)
                        for name in ("source_segment", "lifted_segment", "segment_spec",
                                     "pass2_context", "plan", "sigmas")}
        previous_origin = origin(stage, "previous_result") if index else None
        disconnect(stage, "model")
        disconnect(stage, "positive")
        sx, sy = stage["pos"]
        project = make(
            PROJECT, f"Relay projection — v1 segment {index}", (sx - 140, sy - 730),
            [("raw_model", "MODEL"), ("relay_model", "MODEL"),
             ("relay_positive", "CONDITIONING"), ("relay_full_av_latent", "LATENT"),
             ("prompt_relay_plan", RELAY_PLAN), ("source_segment", "LATENT"),
             ("lifted_segment", "LATENT"), ("segment_spec", SPEC),
             ("pass2_context", CONTEXT), ("plan", PLAN), ("sigmas", "SIGMAS"),
             ("previous_result", RESULT)],
            [("model", "MODEL"), ("positive", "CONDITIONING"),
             ("runtime", RUNTIME), ("report_json", "STRING")],
        )
        audit = (make(
            AUDIT, f"Relay calls — v1 segment {index}", (sx + 560, sy - 210),
            [("segment_result", RESULT), ("segment_spec", SPEC), ("runtime", RUNTIME)],
            [("cumulative_av_latent", "LATENT"), ("report_json", "STRING")],
        ) if with_audit else None)
        for source, name, dtype in (
            (raw_origin[:2], "raw_model", "MODEL"),
            ((relay_cond["id"], 0), "relay_model", "MODEL"),
            ((relay_cond["id"], 1), "relay_positive", "CONDITIONING"),
            ((relay_cond["id"], 2), "relay_full_av_latent", "LATENT"),
            ((relay_plan["id"], 0), "prompt_relay_plan", RELAY_PLAN),
        ):
            connect(source, project, name, dtype)
        for name, (source_id, source_slot, dtype) in stage_inputs.items():
            connect((source_id, source_slot), project, name, dtype)
        if previous_origin is not None:
            connect(previous_origin[:2], project, "previous_result", RESULT)
        connect((project["id"], 0), stage, "model", "MODEL")
        connect((project["id"], 1), stage, "positive", "CONDITIONING")
        if audit is not None:
            connect((stage["id"], 1), audit, "segment_result", RESULT)
            connect(stage_inputs["segment_spec"][:2], audit, "segment_spec", SPEC)
            connect((project["id"], 2), audit, "runtime", RUNTIME)
    graph["last_node_id"] = next_node - 1
    graph["last_link_id"] = next_link - 1
    return graph


def add_v1_relay_api(graph, *, with_audit=True):
    graph = deepcopy(graph)
    passes = [(key, node) for key, node in graph.items() if node["class_type"] == PASS]
    if len(passes) != 3:
        raise ValueError("Expected three explicit v1 PASS2 API nodes")
    passes.sort(key=lambda entry: graph[str(entry[1]["inputs"]["segment_spec"][0])]
                ["inputs"]["segment_index"])
    if [graph[str(node["inputs"]["segment_spec"][0])]["inputs"]["segment_index"]
            for _, node in passes] != [0, 1, 2]:
        raise ValueError("Expected v1 segment indices 0, 1, 2")
    first = passes[0][1]["inputs"]
    high_inputs = graph[str(first["positive"][0])]["inputs"]
    if graph[str(first["positive"][0])]["class_type"] != "MiniMaxH3AudioConditioningT8":
        raise ValueError("Expected frozen HIGH conditioning template")
    next_id = max(map(int, graph)) + 1

    def add(kind, inputs):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        graph[key] = {"class_type": kind, "inputs": inputs}
        return key

    plan_id = add("MiniMaxH3PromptRelayPlanT8Advanced", {
        "global_prompt": high_inputs["prompt"] +
        " <Picture 1> is the first frame; <Picture 2> is the last frame.",
        "local_prompts": "Maintain the same scene with continuous natural motion.\n"
                         "The subject walks toward the final framing without a cut.",
        "length": high_inputs["length"], "timing_mode": "auto_equal",
        "time_ranges": "", "math_profile": "paper_v1", "epsilon": 0.1,
        "allow_gaps": False, "allow_overlaps": False,
    })
    cond_inputs = {
        "model": first["model"], "clip": high_inputs["clip"],
        "video_vae": high_inputs["video_vae"], "audio_vae": high_inputs["audio_vae"],
        "prompt_relay_plan": [plan_id, 0], "width": high_inputs["width"],
        "height": high_inputs["height"], "task_type": high_inputs["task_type"],
        "audio_mode": high_inputs["audio_mode"],
        "audio_denoise_strength": high_inputs["audio_denoise_strength"],
        "add_source_as_reference": high_inputs["add_source_as_reference"],
        "prompt_primary_audio_ordinal": high_inputs["prompt_primary_audio_ordinal"],
        "strict_prompt_tags": high_inputs["strict_prompt_tags"],
        "ref_image_size": high_inputs["ref_image_size"],
        "reference_video_policy": high_inputs["reference_video_policy"],
        "execution_mode": "apply_exp", "query_chunk_rows": 256,
    }
    for name in ("drive_audio", "final_audio", "first_frame", "last_frame"):
        if name in high_inputs:
            cond_inputs[name] = high_inputs[name]
    cond_id = add("MiniMaxH3PromptRelayConditioningT8Advanced", cond_inputs)
    for pass_id, stage in passes:
        inputs = stage["inputs"]
        project_inputs = {
            "raw_model": inputs["model"], "relay_model": [cond_id, 0],
            "relay_positive": [cond_id, 1], "relay_full_av_latent": [cond_id, 2],
            "prompt_relay_plan": [plan_id, 0],
            **{name: inputs[name] for name in
               ("source_segment", "lifted_segment", "segment_spec",
                "pass2_context", "plan", "sigmas")},
        }
        if "previous_result" in inputs:
            project_inputs["previous_result"] = inputs["previous_result"]
        project_id = add(PROJECT, project_inputs)
        inputs["model"] = [project_id, 0]
        inputs["positive"] = [project_id, 1]
        if with_audit:
            add(AUDIT, {"segment_result": [pass_id, 1],
                        "segment_spec": inputs["segment_spec"],
                        "runtime": [project_id, 2]})
    return graph


def build_pair(*, with_eav=False):
    original = json.loads(TEMPLATES["v1"].read_text(encoding="utf-8"))
    frontend = split_workflow(v1_source_from_v2(original))
    frontend = append_second_frontend_segment(frontend)
    frontend = append_second_frontend_segment(frontend, segment_index=2)
    frontend = add_v1_relay_frontend(frontend, with_audit=not with_eav)
    if with_eav:
        frontend = add_external_eav_frontend(frontend)
        configs = [node for node in frontend["nodes"]
                   if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
        if len(configs) != 1 or configs[0]["widgets_values"][-1] != 1.5:
            raise ValueError("Expected one unmodified external EAV safety config")
        # An explicit experimental limit for the combined v1 route; the
        # production node default and earlier candidate graphs are unchanged.
        configs[0]["widgets_values"][-1] = 3.0
    args = legacy_api._parser().parse_args([
        "--mode", "chunked_two_pass", "--frame-count", "56",
        "--temporal-chunk-frames", "34", "--temporal-overlap-frames", "17",
    ])
    source_api, _ = legacy_api._chunked_prompt(args, "STATIC")
    api = split_api_graph(source_api)
    api = append_second_api_segment(api)
    api = append_second_api_segment(api, segment_index=2)
    api = add_v1_relay_api(api, with_audit=not with_eav)
    if with_eav:
        api = add_external_eav_api(api)
        configs = [node for node in api.values()
                   if node["class_type"] == "MiniMaxH3StageEAVConfigEXPT8"]
        if len(configs) != 1 or configs[0]["inputs"]["g_hard_limit"] != 1.5:
            raise ValueError("Expected one unmodified external EAV API safety config")
        configs[0]["inputs"]["g_hard_limit"] = 3.0
    return frontend, api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-eav", action="store_true",
                        help="Compose per-segment external EAV with projected Relay")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifact directory; do not overwrite candidates")
    frontend, api = build_pair(with_eav=args.with_eav)
    output.mkdir(parents=True)
    stem = "Chunked_v1_56F_Three_Segments_External_Relay"
    if args.with_eav:
        stem += "_EAV"
    stem += "_EXP"
    (output / (stem + ".json")).write_text(
        json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / (stem + ".api.json")).write_text(
        json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "frontend_nodes": len(frontend["nodes"]),
                      "api_nodes": len(api)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
