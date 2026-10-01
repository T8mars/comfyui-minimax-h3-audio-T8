"""Build private Manual/Studio Face Window separate-stage workflow copies.

The Compose-only workflow has no sampler and stays unchanged. Old generation
examples use an unavailable SigmaShift; the private copies explicitly use
the current dual-clock MODEL setup and retain external er_sde/simple selection.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.build_modular_face_parity_workflow import WIDGETS as PARITY_WIDGETS  # noqa: E402

TARGET = ROOT / "artifacts/development/modular-sampling-m4-face-window-20260923/candidate-v1"
VARIANTS = {
    "Manual": (
        "2026-09-05_H3_Face_Refine_Window_Manual_Review_Advanced_EXP.json",
        "face_refine_window_advanced_api.json"),
    "Studio_Serial": (
        "2026-09-05_H3_Face_Refine_Window_Studio_Serial_Advanced_EXP.json",
        "face_refine_window_studio_advanced_api.json"),
}
WIDGETS = {**PARITY_WIDGETS,
    "MiniMaxH3FaceRefineWindowPlanT8Advanced": (
        "fps", "repair_ranges", "range_mode", "context_before_frames", "context_after_frames",
        "min_render_frames", "max_render_frames", "scene_cut_threshold", "overlap_policy",
        "short_shot_policy", "enabled"),
    "MiniMaxH3FaceRefineWindowExtractT8Advanced": ("window_index", "pad_policy"),
    "MiniMaxH3FaceRefineManualReviewT8Advanced": (
        "decision", "accepted_subranges", "confirm_accept", "edge_fade_frames"),
    "MiniMaxH3FaceRefineWindowStudioStartT8Advanced": (
        "studio_id", "execution_mode", "max_retries", "retry_delay_seconds", "release_policy"),
    "MiniMaxH3FaceRefineWindowStudioCommitT8Advanced": (
        "studio_id", "decision", "accepted_subranges", "confirm_accept", "edge_fade_frames"),
    "MiniMaxH3FaceRefineWindowStudioComposeT8Advanced": ("studio_id",),
    "MiniMaxH3FaceWindowStageBindEXPT8": ("audio_policy",),
}


def source_paths(variant):
    example, fixture = VARIANTS[variant]
    return (ROOT / "examples/workflows/06-face-refine" / example,
            ROOT / "tests/fixtures/api" / fixture)


def split_frontend(source):
    draft = Draft(source)
    nodes = draft.nodes
    shift = nodes[10]
    if shift["type"] != "MiniMaxH3SigmaShift" or nodes[18]["type"] != "SamplerCustomAdvanced":
        raise ValueError("Expected the old single-window sampler path")
    shift["type"] = "MiniMaxH3DualClockSamplerT8"
    shift["title"] = "Current FLOW_AV setup; er_sde/simple remain external"
    shift["properties"]["Node name for S&R"] = shift["type"]
    shift["inputs"].append({"name": "av_latent", "type": "LATENT", "link": None})
    shift["outputs"].extend([
        {"name": "sampler", "type": "SAMPLER", "links": []},
        {"name": "sigmas", "type": "SIGMAS", "links": []}])
    shift["widgets_values"] = [8, 12., 3., "er_sde", "simple"]
    # The optional sampler-mask patch depends on this MODEL, so the setup AV
    # must be taken from the preceding per-frame-denoise output, not patch 32.
    draft.connect((13, 0), shift, "av_latent", "LATENT")
    bind = draft.make("MiniMaxH3FaceWindowStageBindEXPT8", "Bind current window, parent and original audio",
        [("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"), ("source_frames", "IMAGE"),
         ("parent_frames", "IMAGE"), ("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN"),
         ("window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING"),
         ("source_audio", "AUDIO"), ("window_audio", "AUDIO"),
         ("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("av_latent", "LATENT")],
        [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
        ["require_locked"], (3130, 500))
    audit = draft.make("MiniMaxH3FaceWindowStageAuditEXPT8", "Audit source before Manual/Studio review",
        [("stage_result", "T8_STAGE_RESULT"), ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
         ("source_frames", "IMAGE"), ("parent_frames", "IMAGE"),
         ("window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN"),
         ("window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING"),
         ("source_audio", "AUDIO"), ("window_audio", "AUDIO"),
         ("av_latent", "LATENT")],
        [("candidate_av", "LATENT"), ("report_json", "STRING")], [], (4050, 500))
    stage = nodes[18]
    stage["type"] = "MiniMaxH3StageSamplerEXPT8"
    stage["title"] = "One source-bound window er_sde stage; completion unverified"
    stage["properties"]["Node name for S&R"] = stage["type"]
    stage["properties"]["cnr_id"] = "minimax-h3-audio-T8"
    stage["inputs"].append({"name": "stage_context", "type": "T8_STAGE_CONTEXT", "link": None})
    stage["outputs"].extend([
        {"name": "stage_result", "type": "T8_STAGE_RESULT", "links": []},
        {"name": "report_json", "type": "STRING", "links": []}])
    for target, name in ((nodes[15], "model"), (stage, "sampler"),
                         (stage, "sigmas"), (nodes[19], "av_latent")):
        draft.disconnect(target, name)
    for source_slot, name, dtype in [
        ((4, 0), "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
        ((27, 0), "source_frames", "IMAGE"), ((2, 0), "parent_frames", "IMAGE"),
        ((26, 0), "window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN"),
        ((27, 2), "window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING"),
        ((2, 1), "source_audio", "AUDIO"), ((27, 1), "window_audio", "AUDIO"),
        ((32, 0), "model", "MODEL"), ((16, 0), "sampler", "SAMPLER"),
        ((17, 0), "sigmas", "SIGMAS"), ((32, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, bind, name, dtype)
    for source_slot, target, name, dtype in [
        ((bind["id"], 0), nodes[15], "model", "MODEL"),
        ((bind["id"], 1), stage, "sampler", "SAMPLER"),
        ((bind["id"], 2), stage, "sigmas", "SIGMAS"),
        ((bind["id"], 3), stage, "stage_context", "T8_STAGE_CONTEXT")]:
        draft.connect(source_slot, target, name, dtype)
    for source_slot, name, dtype in [
        ((18, 2), "stage_result", "T8_STAGE_RESULT"),
        ((4, 0), "face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
        ((27, 0), "source_frames", "IMAGE"), ((2, 0), "parent_frames", "IMAGE"),
        ((26, 0), "window_plan", "H3_T8_FACE_REFINE_WINDOW_PLAN"),
        ((27, 2), "window_mapping", "H3_T8_FACE_REFINE_WINDOW_MAPPING"),
        ((2, 1), "source_audio", "AUDIO"), ((27, 1), "window_audio", "AUDIO"),
        ((32, 1), "av_latent", "LATENT")]:
        draft.connect(source_slot, audit, name, dtype)
    draft.connect((audit["id"], 0), nodes[19], "av_latent", "LATENT")
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
            raise ValueError("Window candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    return graph


def split_api(frontend):
    graph = {}
    nodes = {node["id"]: node for node in frontend["nodes"]}
    for node in frontend["nodes"]:
        kind = node["type"]
        if kind == "MarkdownNote":
            continue
        names = WIDGETS.get(kind, ())
        values = node.get("widgets_values", [])
        if kind == "MiniMaxH3FaceRefineWindowExtractT8Advanced" and len(values) == 1:
            # Studio turns window_index into a linked widget, so LiteGraph
            # serializes only the remaining pad_policy value.
            names = ("pad_policy",)
        if len(values) < len(names):
            raise ValueError(f"Missing widget values for {kind}")
        widgets = dict(zip(names, values[:len(names)], strict=True))
        graph[str(node["id"])] = {"class_type": kind, "inputs": widgets}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        if str(target) in graph:
            name = nodes[target]["inputs"][target_slot]["name"]
            graph[str(target)]["inputs"][name] = [str(source), source_slot]
    return graph


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        source, _fixture = source_paths(variant)
        graph = split_frontend(json.loads(source.read_text(encoding="utf-8")))
        api = split_api(graph)
        stem = f"Face_Window_{variant}_Separate_Stage_EXP"
        (TARGET / f"{stem}.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (TARGET / f"{stem}.api.json").write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "nodes": len(graph["nodes"]),
                          "links": len(graph["links"]), "api_nodes": len(api)}))


if __name__ == "__main__":
    main()
