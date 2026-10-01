"""Create private source-bound Motion Recovery pass-2 copies of both old examples."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.build_modular_face_workflow import WIDGETS as BASE_WIDGETS  # noqa: E402

TARGET = ROOT / "artifacts/development/modular-sampling-m4-motion-20260923/candidate-v2"
SOURCES = {
    "Fullclip": "2026-08-22_H3_Motion_Recovery_Fullclip_Stock20_Advanced_EXP.json",
    "Windowed": "2026-08-22_H3_Motion_Recovery_Windowed_Stock20_Advanced_EXP.json",
}
WIDGETS = {**BASE_WIDGETS,
    "MiniMaxH3DualClockSamplerT8": ("steps", "shift_video", "shift_audio", "sampler_name", "scheduler"),
    "MiniMaxH3MotionOverloadAnalyzeT8Advanced": (
        "mode", "manual_ranges", "overload_threshold", "minimum_profile_contrast",
        "minimum_residual_motion", "max_hold", "bridge_tokens", "minimum_hot_frames", "fps"),
    "MiniMaxH3MotionSegmentPlanT8Advanced": (
        "max_expanded_frames", "window_index", "handle_frames", "coverage"),
    "MiniMaxH3MotionRetimingPrepareT8Advanced": ("audio_seed_mode",),
    "MiniMaxH3MotionRecoveryComposerT8Advanced": (
        "mode", "denoise_fraction", "minimum_second_pass_nfe"),
    "MiniMaxH3MotionRecoverAVT8Advanced": ("audio_mode", "pass1_mix"),
    "MiniMaxH3MotionWindowCollectT8Advanced": (
        "run_name", "store_dir", "write_window", "store_dtype", "feather_frames"),
    "MiniMaxH3MotionAutoGateT8Advanced": ("should_repair",),
}


def source_path(variant):
    return ROOT / "examples/workflows/07-motion-detail" / SOURCES[variant]


def split_frontend(source, variant):
    draft = Draft(source)
    nodes = draft.nodes
    windowed = variant == "Windowed"
    baseline = 13 if windowed else 10
    source_audio_slot = 4 if windowed else 1
    prep = 14 if windowed else 13
    composer = 15 if windowed else 14
    setup = 16 if windowed else 15
    guider = 17 if windowed else 16
    sampler_id = 19 if windowed else 18
    decoder = 20 if windowed else 19
    if nodes[sampler_id]["type"] != "SamplerCustomAdvanced":
        raise ValueError("Expected an independent old Motion Recovery pass-2 sampler")
    bind = draft.make("MiniMaxH3MotionStageBindEXPT8", "Bind retimed pass-2 source and original audio",
        [("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN"), ("baseline_frames", "IMAGE"),
         ("baseline_audio", "AUDIO"), ("smeared_frames", "IMAGE"),
         ("smeared_audio", "AUDIO"), ("prepare_report_json", "STRING"),
         ("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("av_latent", "LATENT"), ("parent_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
         ("parent_frames", "IMAGE"), ("parent_audio", "AUDIO")],
        [("model", "MODEL"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_context", "T8_STAGE_CONTEXT"), ("report_json", "STRING")],
        [], (2740, 600))
    audit = draft.make("MiniMaxH3MotionStageAuditEXPT8", "Audit pass 2 before Recover AV",
        [("stage_result", "T8_STAGE_RESULT"), ("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
         ("baseline_frames", "IMAGE"), ("baseline_audio", "AUDIO"),
         ("smeared_frames", "IMAGE"), ("smeared_audio", "AUDIO"),
         ("prepare_report_json", "STRING"), ("av_latent", "LATENT"),
         ("parent_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
         ("parent_frames", "IMAGE"), ("parent_audio", "AUDIO")],
        [("candidate_av", "LATENT"), ("report_json", "STRING")], [], (3610, 620))
    stage = nodes[sampler_id]
    stage["type"] = "MiniMaxH3StageSamplerEXPT8"
    stage["title"] = "One explicit source-bound motion refinement stage"
    stage["properties"]["Node name for S&R"] = stage["type"]
    stage["properties"]["cnr_id"] = "minimax-h3-audio-T8"
    stage["inputs"].append({"name": "stage_context", "type": "T8_STAGE_CONTEXT", "link": None})
    stage["outputs"].extend([
        {"name": "stage_result", "type": "T8_STAGE_RESULT", "links": []},
        {"name": "report_json", "type": "STRING", "links": []}])
    for target, name in ((nodes[guider], "model"), (stage, "sampler"),
                         (stage, "sigmas"), (nodes[decoder], "av_latent")):
        draft.disconnect(target, name)
    sources = [(composer, 2), (baseline, 0), (baseline, source_audio_slot),
               (prep, 1), (prep, 2), (prep, 4), (setup, 0), (setup, 1),
               (composer, 1), (composer, 0)]
    for (node_id, slot), field in zip(sources, bind["inputs"][:10], strict=True):
        draft.connect((node_id, slot), bind, field["name"], field["type"])
    for slot, target, field, dtype in ((0, nodes[guider], "model", "MODEL"),
                                      (1, stage, "sampler", "SAMPLER"),
                                      (2, stage, "sigmas", "SIGMAS"),
                                      (3, stage, "stage_context", "T8_STAGE_CONTEXT")):
        draft.connect((bind["id"], slot), target, field, dtype)
    draft.connect((sampler_id, 2), audit, "stage_result", "T8_STAGE_RESULT")
    for (node_id, slot), field in zip(sources[:6] + sources[-1:], audit["inputs"][1:8], strict=True):
        draft.connect((node_id, slot), audit, field["name"], field["type"])
    if windowed:
        for target in (bind, audit):
            for source, field, dtype in (((12, 0), "parent_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
                                         ((10, 0), "parent_frames", "IMAGE"),
                                         ((10, 1), "parent_audio", "AUDIO")):
                draft.connect(source, target, field, dtype)
    draft.connect((audit["id"], 0), nodes[decoder], "av_latent", "LATENT")
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
            raise ValueError("Motion candidate has a dependency cycle")
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
        values = node.get("widgets_values", [])
        if isinstance(values, dict):
            widgets = dict(values)
        else:
            names = WIDGETS.get(kind, ())
            if not isinstance(values, list):
                values = [values] if values is not None else []
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
    for variant in SOURCES:
        source = json.loads(source_path(variant).read_text(encoding="utf-8"))
        frontend = split_frontend(source, variant)
        api = split_api(frontend)
        stem = f"Motion_Recovery_{variant}_Source_Bound_Separate_Stage_EXP"
        (TARGET / f"{stem}.json").write_text(
            json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (TARGET / f"{stem}.api.json").write_text(
            json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "nodes": len(frontend["nodes"]),
                          "links": len(frontend["links"]), "api_nodes": len(api)}))


if __name__ == "__main__":
    main()
