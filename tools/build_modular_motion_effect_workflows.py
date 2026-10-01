"""Derive private Motion pass-2 Relay+EAV graphs; preserve prior candidates.

The stock Prompt Relay Plan/Conditioning and Stage EAV Config/Apply/Audit stay
visible and independently editable. Motion's retimed AV remains the sampler
source and original pass-1 audio remains the default delivery branch.
"""
from __future__ import annotations

import hashlib
import json

from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_motion_recovery_workflows import (
    ROOT, SOURCES, TARGET as SOURCE, WIDGETS, split_api,
)

TARGET = ROOT / "artifacts/development/modular-sampling-m4-motion-effects-20260925/candidate-v2"
SHA = {
    "Fullclip": "6e691130730ca2852e1f7191cd49c84470c884e7940955acaba747dc9ec5b1d0",
    "Windowed": "84856333a0d1dbaaca4db6425d76fff465bd50a87c5bb6cd7e31c30be940fede",
}
WIDGETS.update({
    "MiniMaxH3PromptRelayPlanT8Advanced": (
        "global_prompt", "local_prompts", "length", "timing_mode", "time_ranges",
        "math_profile", "epsilon", "allow_gaps", "allow_overlaps"),
    "MiniMaxH3PromptRelayConditioningT8Advanced": (
        "width", "height", "task_type", "audio_mode", "audio_denoise_strength",
        "add_source_as_reference", "prompt_primary_audio_ordinal", "strict_prompt_tags",
        "ref_image_size", "reference_video_policy", "execution_mode", "query_chunk_rows"),
    "MiniMaxH3StageEAVConfigEXPT8": (
        "mode", "tau", "start_video_progress", "end_video_progress",
        "max_workspace_mib", "g_hard_limit"),
})


def _source(variant: str) -> dict:
    if variant not in SOURCES:
        raise ValueError("Unknown Motion Recovery variant")
    path = SOURCE / f"Motion_Recovery_{variant}_Source_Bound_Separate_Stage_EXP.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA[variant]:
        raise ValueError("Pinned Motion Recovery source-bound candidate changed")
    return json.loads(data)


def _linked_source(draft: Draft, target: dict, name: str) -> tuple[int, int]:
    item = next(item for item in target["inputs"] if item["name"] == name)
    return tuple(draft.links[item["link"]][1:3])


def _widget_input(node: dict, name: str):
    item = next(item for item in node["inputs"] if item["name"] == name)
    item["widget"] = {"name": name}


def effect_frontend(variant: str, source_frontend: dict | None = None) -> dict:
    draft = Draft(_source(variant) if source_frontend is None else source_frontend)
    windowed = variant == "Windowed"
    composer = draft.nodes[15 if windowed else 14]
    setup = draft.nodes[16 if windowed else 15]
    bind = draft.nodes[30 if windowed else 28]
    guider = draft.nodes[17 if windowed else 16]
    sampler = draft.nodes[19 if windowed else 18]
    audit = draft.nodes[31 if windowed else 29]
    decoder = draft.nodes[20 if windowed else 19]
    first_conditioning = draft.nodes[5]
    if (_linked_source(draft, bind, "model") != (setup["id"], 0)
            or _linked_source(draft, guider, "conditioning") != (5, 0)
            or _linked_source(draft, decoder, "av_latent") != (audit["id"], 0)):
        raise ValueError("Motion pass-2 candidate effect boundary changed")
    x, y = bind["pos"]
    length = draft.make("MiniMaxH3MotionRelayLengthEXPT8",
        "Signed retimed pass-2 Relay frame count",
        [("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN")],
        [("length", "INT"), ("report_json", "STRING")], [], (x - 1900, y - 1000))
    draft.connect((composer["id"], 2), length, "motion_plan", "H3_T8_MOTION_RECOVERY_PLAN")
    prompt = first_conditioning["widgets_values"][0]
    plan = draft.make("MiniMaxH3PromptRelayPlanT8Advanced",
        "External Relay timeline for the signed retimed clip",
        [("length", "INT")],
        [("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("compiled_prompt", "STRING"), ("length", "INT"),
         ("timeline_json", "STRING"), ("report_json", "STRING")],
        [prompt, "Maintain subject and source action through the expanded motion.\n"
         "Resolve the held action into a continuous natural movement.",
         first_conditioning["widgets_values"][3], "auto_equal", "", "paper_v1",
         .1, False, False], (x - 1430, y - 1000))
    _widget_input(plan, "length")
    draft.connect((length["id"], 0), plan, "length", "INT")
    high_values = first_conditioning["widgets_values"]
    relay_cond = draft.make("MiniMaxH3PromptRelayConditioningT8Advanced",
        "External Relay MODEL + positive for motion pass 2",
        [("model", "MODEL"), ("clip", "CLIP"), ("video_vae", "VAE"),
         ("audio_vae", "VAE"), ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("width", "INT"), ("height", "INT")],
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
         ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
         ("report_json", "STRING")],
        [high_values[1], high_values[2], "T2VA", "native", high_values[6],
         high_values[7], high_values[8], high_values[9], high_values[10],
         high_values[11], "apply_exp", 256], (x - 950, y - 1000))
    for field in ("width", "height"):
        _widget_input(relay_cond, field)
    for field, source, dtype in (
        ("model", (setup["id"], 0), "MODEL"),
        ("clip", (2, 0), "CLIP"), ("video_vae", (3, 0), "VAE"),
        ("audio_vae", (4, 0), "VAE"),
        ("prompt_relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
    ):
        draft.connect(source, relay_cond, field, dtype)
    draft.disconnect(bind, "model")
    draft.connect((relay_cond["id"], 0), bind, "model", "MODEL")
    paired = draft.make("MiniMaxH3MotionRelayBindEXPT8",
        "Authenticate external Relay against signed retimed pass 2",
        [("model", "MODEL"), ("relay_positive", "CONDITIONING"),
         ("relay_av_latent", "LATENT"),
         ("relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
         ("stage_av_latent", "LATENT"),
         ("stage_context", "T8_STAGE_CONTEXT")],
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("report_json", "STRING")], [], (x + 20, y - 1050))
    for field, source, dtype in (
        ("model", (bind["id"], 0), "MODEL"),
        ("relay_positive", (relay_cond["id"], 1), "CONDITIONING"),
        ("relay_av_latent", (relay_cond["id"], 2), "LATENT"),
        ("relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
        ("motion_plan", (composer["id"], 2), "H3_T8_MOTION_RECOVERY_PLAN"),
        ("stage_av_latent", (composer["id"], 0), "LATENT"),
        ("stage_context", (bind["id"], 3), "T8_STAGE_CONTEXT"),
    ):
        draft.connect(source, paired, field, dtype)
    eav_config = draft.make("MiniMaxH3StageEAVConfigEXPT8",
        "Independent motion pass-2 Enhance A Video controls", [],
        [("eav_config", "T8_STAGE_EAV_CONFIG")],
        ["report_only", 4.0, .15, .90, 32, 1.5], (x + 30, y - 470))
    eav_apply = draft.make("MiniMaxH3StageEAVApplyEXPT8",
        "External EAV apply to only motion pass 2",
        [("model", "MODEL"), ("sigmas", "SIGMAS"),
         ("av_latent", "LATENT"), ("stage_context", "T8_STAGE_CONTEXT"),
         ("eav_config", "T8_STAGE_EAV_CONFIG")],
        [("model", "MODEL"), ("runtime", "T8_STAGE_EAV_RUNTIME"),
         ("report_json", "STRING")], [], (x + 500, y - 470))
    for field, source, dtype in (
        ("model", (paired["id"], 0), "MODEL"),
        ("sigmas", (bind["id"], 2), "SIGMAS"),
        ("av_latent", (composer["id"], 0), "LATENT"),
        ("stage_context", (bind["id"], 3), "T8_STAGE_CONTEXT"),
        ("eav_config", (eav_config["id"], 0), "T8_STAGE_EAV_CONFIG"),
    ):
        draft.connect(source, eav_apply, field, dtype)
    draft.disconnect(guider, "model")
    draft.disconnect(guider, "conditioning")
    draft.connect((eav_apply["id"], 0), guider, "model", "MODEL")
    draft.connect((paired["id"], 1), guider, "conditioning", "CONDITIONING")
    eav_audit = draft.make("MiniMaxH3StageEAVAuditEXPT8",
        "Observe actual motion pass-2 Relay + EAV calls",
        [("av_latent", "LATENT"), ("runtime", "T8_STAGE_EAV_RUNTIME")],
        [("av_latent", "LATENT"), ("report_json", "STRING")],
        [], (x + 1480, y - 470))
    draft.connect((audit["id"], 0), eav_audit, "av_latent", "LATENT")
    draft.connect((eav_apply["id"], 1), eav_audit, "runtime", "T8_STAGE_EAV_RUNTIME")
    draft.disconnect(decoder, "av_latent")
    draft.connect((eav_audit["id"], 0), decoder, "av_latent", "LATENT")
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
            raise ValueError("Motion effects candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    if _linked_source(draft, sampler, "stage_context") != (bind["id"], 3):
        raise ValueError("Motion pass-2 StageSampler context changed")
    if windowed:
        segment_plan = draft.nodes[13]
        if (segment_plan["type"] != "MiniMaxH3MotionSegmentPlanT8Advanced"
                or segment_plan["widgets_values"] != [209, 0, 12, "hot_ranges_only"]):
            raise ValueError("Pinned Windowed motion-segment controls changed")
        # Current native UI inserts control_after_generate immediately after
        # window_index. Store that inert UI value explicitly so reopening does
        # not reinterpret handle_frames as the seed control and shift coverage.
        segment_plan["widgets_values"].insert(2, "fixed")
    return graph


def effect_api(frontend: dict, variant: str) -> dict:
    graph = json.loads(json.dumps(frontend))
    if variant == "Windowed":
        segment_plan = next(node for node in graph["nodes"] if node["id"] == 13)
        if segment_plan["widgets_values"] != [209, 0, "fixed", 12, "hot_ranges_only"]:
            raise ValueError("Windowed native seed widget or original values changed")
        segment_plan["widgets_values"].pop(2)
    return split_api(graph)


def main() -> None:
    if TARGET.exists():
        raise FileExistsError("Private Motion effects candidate already exists")
    TARGET.mkdir(parents=True)
    for variant in SOURCES:
        graph = effect_frontend(variant)
        api = effect_api(graph, variant)
        stem = f"Motion_Recovery_{variant}_Relay_EAV_Separate_Pass2_EXP"
        (TARGET / f"{stem}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (TARGET / f"{stem}.api.json").write_text(
            json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, "frontend_nodes": len(graph["nodes"]),
                          "links": len(graph["links"]), "api_nodes": len(api)}))


if __name__ == "__main__":
    main()
