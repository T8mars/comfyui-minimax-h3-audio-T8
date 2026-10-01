"""Copy two-/three-person Face examples into separately editable stage drafts.

The old examples reference an unavailable MiniMaxH3SigmaShift. Private copies
use a current native dual-clock MODEL setup per character while retaining each
old er_sde sampler and simple scheduler. Exact old shift parity is unproven.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

TARGET = ROOT / "artifacts/development/modular-sampling-m4-face-multiface-20260923/candidate-v1"


def source_paths(count):
    return (
        ROOT / f"examples/workflows/06-face-refine/2026-08-17_H3_SAM31_{count}Person_Face_Refine_Advanced_EXP.json",
        ROOT / f"tests/fixtures/api/multiface_sam31_{count}person_advanced_api.json",
    )


def _link_source(draft, node, name):
    item = next(value for value in node["inputs"] if value["name"] == name)
    link = draft.links[item["link"]]
    return link[1], link[2]


def split_frontend(source):
    draft = Draft(source)
    nodes = draft.nodes
    old_shift = next(node for node in nodes.values() if node["type"] == "MiniMaxH3SigmaShift")
    model_input = _link_source(draft, old_shift, "model")
    samplers = sorted((node for node in nodes.values() if node["type"] == "SamplerCustomAdvanced"),
                      key=lambda node: node["id"])
    old_shift["type"] = "MiniMaxH3DualClockSamplerT8"
    old_shift["title"] = "Character A native FLOW_AV model setup; er_sde remains external"
    old_shift["properties"]["Node name for S&R"] = old_shift["type"]
    old_shift["inputs"].append({"name": "av_latent", "type": "LATENT", "link": None})
    old_shift["outputs"].extend([
        {"name": "sampler", "type": "SAMPLER", "links": []},
        {"name": "sigmas", "type": "SIGMAS", "links": []}])
    old_shift["widgets_values"] = [8, 12., 3., "er_sde", "simple"]

    for index, stage in enumerate(samplers):
        guider_id, _ = _link_source(draft, stage, "guider")
        sampler_source = _link_source(draft, stage, "sampler")
        sigmas_source = _link_source(draft, stage, "sigmas")
        latent_source = _link_source(draft, stage, "latent_image")
        guider = nodes[guider_id]
        scheduler = nodes[sigmas_source[0]]
        denoise = nodes[latent_source[0]]
        plan_source = _link_source(draft, denoise, "face_plan")
        job = nodes[plan_source[0]]
        decode = next(node for node in nodes.values() if node["type"] == "MiniMaxH3AVDecodeT8"
                      and _link_source(draft, node, "av_latent") == (stage["id"], 0))
        if index == 0:
            setup = old_shift
        else:
            setup = draft.make("MiniMaxH3DualClockSamplerT8",
                f"Character {index + 1} independent FLOW_AV MODEL setup",
                [("model", "MODEL"), ("av_latent", "LATENT")],
                [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS")],
                [8, 12., 3., "er_sde", "simple"], (2450, 1500 + index * 430))
            draft.connect(model_input, setup, "model", "MODEL")
        draft.connect(latent_source, setup, "av_latent", "LATENT")
        for target in (guider, scheduler):
            draft.disconnect(target, "model")
            draft.connect((setup["id"], 0), target, "model", "MODEL")
        bind = draft.make("MiniMaxH3MultiFaceStageBindEXPT8",
            f"Bind character {index + 1} to original parent and repair window",
            [("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"), ("source_frames", "IMAGE"),
             ("parent_frames", "IMAGE"), ("model", "MODEL"), ("sampler", "SAMPLER"),
             ("sigmas", "SIGMAS"), ("av_latent", "LATENT")],
            [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
             ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
            ["require_locked"], (3100, 1450 + index * 450))
        audit = draft.make("MiniMaxH3MultiFaceStageAuditEXPT8",
            f"Audit character {index + 1} before sequential composite",
            [("stage_result", "T8_STAGE_RESULT"), ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
             ("source_frames", "IMAGE"), ("parent_frames", "IMAGE"),
             ("av_latent", "LATENT")],
            [("candidate_av", "LATENT"), ("report_json", "STRING")],
            [], (4020, 1450 + index * 450))
        stage["type"] = "MiniMaxH3StageSamplerEXPT8"
        stage["title"] = f"Character {index + 1} separate er_sde stage; no portable completion"
        stage["properties"]["Node name for S&R"] = stage["type"]
        stage["properties"]["cnr_id"] = "minimax-h3-audio-T8"
        stage["inputs"].append({"name": "stage_context", "type": "T8_STAGE_CONTEXT", "link": None})
        stage["outputs"].extend([
            {"name": "stage_result", "type": "T8_STAGE_RESULT", "links": []},
            {"name": "report_json", "type": "STRING", "links": []}])
        for target, name in ((guider, "model"), (stage, "sampler"),
                             (stage, "sigmas"), (decode, "av_latent")):
            draft.disconnect(target, name)
        for source_slot, name, dtype in [
            (plan_source, "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
            ((job["id"], 1), "source_frames", "IMAGE"),
            ((2, 0), "parent_frames", "IMAGE"),
            ((setup["id"], 0), "model", "MODEL"),
            (sampler_source, "sampler", "SAMPLER"),
            (sigmas_source, "sigmas", "SIGMAS"),
            (latent_source, "av_latent", "LATENT")]:
            draft.connect(source_slot, bind, name, dtype)
        for source_slot, target, name, dtype in [
            ((bind["id"], 0), guider, "model", "MODEL"),
            ((bind["id"], 1), stage, "sampler", "SAMPLER"),
            ((bind["id"], 2), stage, "sigmas", "SIGMAS"),
            ((bind["id"], 3), stage, "stage_context", "T8_STAGE_CONTEXT")]:
            draft.connect(source_slot, target, name, dtype)
        for source_slot, name, dtype in [
            ((stage["id"], 2), "stage_result", "T8_STAGE_RESULT"),
            (plan_source, "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
            ((job["id"], 1), "source_frames", "IMAGE"),
            ((2, 0), "parent_frames", "IMAGE"),
            (latent_source, "av_latent", "LATENT")]:
            draft.connect(source_slot, audit, name, dtype)
        draft.connect((audit["id"], 0), decode, "av_latent", "LATENT")

    graph = draft.graph
    graph["last_node_id"] = max(draft.nodes)
    graph["last_link_id"] = max(draft.links)
    seen = set()
    ordered = []
    while len(ordered) < len(draft.nodes):
        ready = [node for node in graph["nodes"] if node["id"] not in seen and
                 all(item.get("link") is None or draft.links[item["link"]][1] in seen
                     for item in node.get("inputs", []))]
        if not ready:
            raise ValueError("Multi-face candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    return graph


def split_api(frontend, old_api):
    graph = {}
    nodes = {node["id"]: node for node in frontend["nodes"]}
    for node in frontend["nodes"]:
        kind = node["type"]
        if kind == "MarkdownNote":
            continue
        node_id = str(node["id"])
        old = old_api.get(node_id, {"inputs": {}})
        socket_names = {item["name"] for item in node.get("inputs", [])}
        widgets = {name: value for name, value in old["inputs"].items() if name not in socket_names}
        if kind == "MiniMaxH3DualClockSamplerT8":
            widgets = dict(steps=8, shift_video=12., shift_audio=3., sampler_name="er_sde", scheduler="simple")
        elif kind == "MiniMaxH3MultiFaceStageBindEXPT8":
            widgets = {"audio_policy": "require_locked"}
        graph[node_id] = {"class_type": kind, "inputs": widgets}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        if str(target) in graph:
            name = nodes[target]["inputs"][target_slot]["name"]
            graph[str(target)]["inputs"][name] = [str(source), source_slot]
    return graph


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for count in (2, 3):
        source, fixture = source_paths(count)
        old = json.loads(source.read_text(encoding="utf-8"))
        old_api = json.loads(fixture.read_text(encoding="utf-8"))
        frontend = split_frontend(deepcopy(old))
        api = split_api(frontend, old_api)
        stem = f"MultiFace_{count}Person_Separate_Stages_EXP"
        (TARGET / f"{stem}.json").write_text(json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (TARGET / f"{stem}.api.json").write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"people": count, "nodes": len(frontend["nodes"]),
                          "links": len(frontend["links"]), "api_nodes": len(api)}))


if __name__ == "__main__":
    main()
