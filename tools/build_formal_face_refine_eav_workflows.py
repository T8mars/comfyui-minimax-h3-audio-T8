"""Build additive S24 Face Refine split-stage and external-effect workflows.

Full-clip standard/anime Face Relay and one local Relay Plan per signed
er_sde repair job are paired. Local Plans do not project a parent-film global
timeline. Exact-source default Core er_sde has a separate completed-stage
adapter; these original no-storage graphs are not cold-delivery examples.
Old examples are read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools import build_modular_face_workflow as standard  # noqa: E402
from tools import build_modular_face_parity_workflow as parity  # noqa: E402
from tools import build_modular_face_multiface_workflows as multiface  # noqa: E402
from tools import build_modular_face_window_workflows as window  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

DESTINATION = ROOT / "examples/workflows/56-face-refine-split"
VARIANTS = ("standard", "anime", "parity", "multiface2", "multiface3",
            "window_manual", "window_studio_serial")
# Current Core er_sde has an exact-source EAV plan and separately verified
# completion identity. Local Ref2VA Relay is one Plan per signed repair job;
# neither certifies face quality.
EFFECTS_BY_VARIANT = {
    variant: ("none", "eav", "relay", "combined")
    for variant in VARIANTS
}
FILES = {(variant, effect): f"S24_Face_{variant}_{effect}_Separate_Stage_EXP.json"
         for variant in VARIANTS for effect in EFFECTS_BY_VARIANT[variant]}
FACE_AUDITS = frozenset({
    "MiniMaxH3FaceStageAuditEXPT8", "MiniMaxH3FaceParityStageAuditEXPT8",
    "MiniMaxH3MultiFaceStageAuditEXPT8", "MiniMaxH3FaceWindowStageAuditEXPT8",
})
CONFIG_WIDGETS = {"mode": "report_only", "tau": 4.0,
                  "start_video_progress": .15, "end_video_progress": .90,
                  "max_workspace_mib": 32, "g_hard_limit": 1.5}
RELAY_PLAN_WIDGETS = ("global_prompt", "local_prompts", "length", "timing_mode",
                      "time_ranges", "math_profile", "epsilon", "allow_gaps", "allow_overlaps")
RELAY_CONDITIONING_WIDGETS = ("width", "height", "task_type", "audio_mode",
    "audio_denoise_strength", "add_source_as_reference", "prompt_primary_audio_ordinal",
    "strict_prompt_tags", "ref_image_size", "reference_video_policy", "execution_mode",
    "query_chunk_rows")
LOCAL_FACE_EVENTS = ("The same face maintains a natural expression and gaze.\n"
                     "The same face turns subtly while identity and mouth timing stay unchanged.")


def _add_widget_inputs(api: dict, node: dict, widgets) -> None:
    """Fill new-node widgets without replacing the converter's connected inputs.

    LiteGraph retains a stored widget value after converting it to a socket.
    The actual source edge, not that inactive value, defines crop geometry and
    timing. All family converters have already resolved those edges here.
    """
    inputs = api[str(node["id"])]["inputs"]
    connected = {item["name"] for item in node.get("inputs", [])
                 if item.get("link") is not None}
    for name, value in dict(widgets).items():
        if name in connected:
            edge = inputs.get(name)
            if not (isinstance(edge, list) and len(edge) == 2
                    and isinstance(edge[0], str) and type(edge[1]) is int):
                raise ValueError(f"Missing converted S24 edge: {node['type']}.{name}")
        else:
            inputs[name] = value


def _source(draft: Draft, node: dict, name: str) -> tuple[int, int]:
    item = next(item for item in node["inputs"] if item["name"] == name)
    if item["link"] is None:
        raise ValueError(f"Unconnected S24 {node['type']}.{name}")
    link = draft.links[item["link"]]
    return link[1], link[2]


def _base(variant: str) -> tuple[dict, object]:
    if variant in ("standard", "anime"):
        source = (standard.SOURCE if variant == "standard" else
                  ROOT / "examples/workflows/06-face-refine/2026-08-16_H3_Face_Refine_Anime_Advanced_EXP.json")
        old = json.loads(source.read_text(encoding="utf-8"))
        return standard.split_frontend(old), standard.split_api
    if variant == "parity":
        old = json.loads(parity.SOURCE.read_text(encoding="utf-8"))
        return parity.split_frontend(old), parity.split_api
    if variant.startswith("multiface"):
        count = int(variant[-1])
        source, fixture = multiface.source_paths(count)
        old = json.loads(source.read_text(encoding="utf-8"))
        old_api = json.loads(fixture.read_text(encoding="utf-8"))
        return multiface.split_frontend(old), lambda graph: multiface.split_api(graph, old_api)
    if variant.startswith("window_"):
        name = "Manual" if variant == "window_manual" else "Studio_Serial"
        source, _ = window.source_paths(name)
        old = json.loads(source.read_text(encoding="utf-8"))
        return window.split_frontend(old), window.split_api
    raise ValueError("Unknown S24 variant")


def _add_relay(frontend: dict) -> dict:
    draft = Draft(frontend)
    by_type = {kind: [node for node in draft.nodes.values() if node["type"] == kind]
               for kind in ("MiniMaxH3AudioConditioningT8",
                            "MiniMaxH3FaceRefineConditioningT8Advanced",
                            "MiniMaxH3FaceRefineSamplerT8Advanced",
                            "MiniMaxH3FaceStageBindEXPT8", "BasicGuider",
                            "CreateVideo", "SaveVideo")}
    if any(len(items) != 1 for items in by_type.values()):
        raise ValueError("Expected one standard full-clip Face refinement branch")
    original, crop, setup, stage_bind, guider, create, save = (
        by_type[kind][0] for kind in by_type)
    values = original["widgets_values"]
    if (len(values) != 12 or values[4] != "T2VA" or values[5] != "lock_source"
            or _source(draft, crop, "positive") != (original["id"], 0)
            or _source(draft, crop, "av_latent") != (original["id"], 1)
            or _source(draft, create, "audio") != (original["id"], 2)):
        raise ValueError("Standard Face conditioner boundary changed")
    plan = draft.make(
        "MiniMaxH3PromptRelayPlanT8Advanced", "Face full-clip external Relay timeline",
        [("length", "INT")],
        [("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"), ("compiled_prompt", "STRING"),
         ("length", "INT"), ("timeline_json", "STRING"), ("report_json", "STRING")],
        [values[0], LOCAL_FACE_EVENTS, values[3], "auto_equal", "", "paper_v1",
         .1, False, False], (1420, -700))
    plan["inputs"][0]["widget"] = {"name": "length"}
    draft.connect(_source(draft, original, "length"), plan, "length", "INT")
    relay_cond = draft.make(
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "External Relay MODEL + CONDITIONING; crop source remains separate",
        [("model", "MODEL"), ("clip", "CLIP"), ("video_vae", "VAE"),
         ("audio_vae", "VAE"), ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("width", "INT"), ("height", "INT"), ("drive_audio", "AUDIO")],
        [("model", "MODEL"), ("positive", "CONDITIONING"), ("av_latent", "LATENT"),
         ("mux_audio", "AUDIO"), ("conditioned_prompt", "STRING"),
         ("media_map_json", "STRING"), ("report_json", "STRING")],
        [values[1], values[2], *values[4:], "apply_exp", 256], (2100, -700))
    for name in ("width", "height"):
        next(item for item in relay_cond["inputs"] if item["name"] == name)["widget"] = {"name": name}
    for name, source, dtype in (
        ("model", _source(draft, setup, "model"), "MODEL"),
        ("clip", _source(draft, original, "clip"), "CLIP"),
        ("video_vae", _source(draft, original, "video_vae"), "VAE"),
        ("audio_vae", _source(draft, original, "audio_vae"), "VAE"),
        ("prompt_relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
        ("width", _source(draft, original, "width"), "INT"),
        ("height", _source(draft, original, "height"), "INT"),
        ("drive_audio", _source(draft, original, "drive_audio"), "AUDIO"),
    ):
        draft.connect(source, relay_cond, name, dtype)
    for target, name, source, dtype in (
        (crop, "positive", (relay_cond["id"], 1), "CONDITIONING"),
        (crop, "av_latent", (relay_cond["id"], 2), "LATENT"),
        (setup, "model", (relay_cond["id"], 0), "MODEL"),
        (create, "audio", (relay_cond["id"], 3), "AUDIO"),
    ):
        draft.disconnect(target, name)
        draft.connect(source, target, name, dtype)
    paired = draft.make(
        "MiniMaxH3FaceRelayBindEXPT8", "Authenticate paired full-clip Face Relay",
        [("model", "MODEL"), ("relay_positive", "CONDITIONING"),
         ("relay_av_latent", "LATENT"), ("relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
         ("face_plan", "H3_T8_FACE_REFINE_PLAN"), ("source_frames", "IMAGE"),
         ("stage_av_latent", "LATENT"), ("stage_context", "T8_STAGE_CONTEXT")],
        [("model", "MODEL"), ("positive", "CONDITIONING"),
         ("report_json", "STRING")], [], (4350, -700))
    for name, source, dtype in (
        ("model", (stage_bind["id"], 0), "MODEL"),
        ("relay_positive", (crop["id"], 0), "CONDITIONING"),
        ("relay_av_latent", (relay_cond["id"], 2), "LATENT"),
        ("relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
        ("face_plan", _source(draft, stage_bind, "face_plan"), "H3_T8_FACE_REFINE_PLAN"),
        ("source_frames", _source(draft, stage_bind, "source_frames"), "IMAGE"),
        ("stage_av_latent", _source(draft, stage_bind, "av_latent"), "LATENT"),
        ("stage_context", (stage_bind["id"], 3), "T8_STAGE_CONTEXT"),
    ):
        draft.connect(source, paired, name, dtype)
    for name, source, dtype in (
        ("model", (paired["id"], 0), "MODEL"),
        ("conditioning", (paired["id"], 1), "CONDITIONING"),
    ):
        draft.disconnect(guider, name)
        draft.connect(source, guider, name, dtype)
    return draft.prune((save["id"],))


def _add_local_relay(frontend: dict) -> dict:
    draft = Draft(frontend)
    binds = sorted((node for node in draft.nodes.values()
                    if node["type"] in ("MiniMaxH3FaceParityStageBindEXPT8",
                                        "MiniMaxH3MultiFaceStageBindEXPT8",
                                        "MiniMaxH3FaceWindowStageBindEXPT8")),
                   key=lambda node: node["id"])
    saves = [node for node in draft.nodes.values() if node["type"] == "SaveVideo"]
    if not binds or len(saves) != 1:
        raise ValueError("Expected separate local Face stages and one delivery")
    for index, stage_bind in enumerate(binds):
        guiders = [node for node in draft.nodes.values()
                   if node["type"] == "BasicGuider" and
                   _source(draft, node, "model") == (stage_bind["id"], 0)]
        if len(guiders) != 1:
            raise ValueError("Local Face stage has no unique BasicGuider")
        guider = guiders[0]
        crop_id, crop_slot = _source(draft, guider, "conditioning")
        crop = draft.nodes[crop_id]
        if crop_slot != 0 or crop["type"] != "MiniMaxH3FaceRefineParityLatentT8Advanced":
            raise ValueError("Local Face CONDITIONING is not the parity crop")
        original_id, original_slot = _source(draft, crop, "positive")
        original = draft.nodes[original_id]
        if (original_slot != 0 or original["type"] != "MiniMaxH3AudioConditioningT8"
                or _source(draft, crop, "av_latent") != (original_id, 1)):
            raise ValueError("Local Face crop has no paired original conditioner")
        values = original["widgets_values"]
        if len(values) != 12 or values[4:6] != ["Ref2VA", "lock_source"]:
            raise ValueError("Local Face must retain Ref2VA and locked source audio")
        model_id, model_slot = _source(draft, stage_bind, "model")
        previous = draft.nodes[model_id]
        setup = (draft.nodes[_source(draft, previous, "model")[0]]
                 if previous["type"] == "MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced"
                 else previous)
        if setup["type"] != "MiniMaxH3DualClockSamplerT8" or model_slot != 0:
            raise ValueError("Local Face MODEL has no independent native setup")
        raw_model = _source(draft, setup, "model")
        y = -900 + index * 520
        plan = draft.make("MiniMaxH3PromptRelayPlanT8Advanced",
            f"Face job {index + 1} · local aligned Relay Plan",
            [("length", "INT")],
            [("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
             ("compiled_prompt", "STRING"), ("length", "INT"),
             ("timeline_json", "STRING"), ("report_json", "STRING")],
            [values[0], LOCAL_FACE_EVENTS, values[3], "auto_equal", "",
             "paper_v1", .1, False, False], (1350, y))
        length_input = next(item for item in original["inputs"] if item["name"] == "length") if any(
            item["name"] == "length" for item in original["inputs"]) else None
        if length_input is not None and length_input["link"] is not None:
            plan["inputs"][0]["widget"] = {"name": "length"}
            draft.connect(_source(draft, original, "length"), plan, "length", "INT")
        base_inputs = [("model", "MODEL"), ("clip", "CLIP"), ("video_vae", "VAE"),
                       ("audio_vae", "VAE"),
                       ("prompt_relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
                       ("width", "INT"), ("height", "INT")]
        optionals = [(item["name"], item["type"]) for item in original["inputs"]
                     if item["name"] not in {"clip", "video_vae", "audio_vae",
                                              "width", "height", "length"}]
        relay_cond = draft.make("MiniMaxH3PromptRelayConditioningT8Advanced",
            f"Face job {index + 1} · external Ref2VA Relay CONDITIONING",
            base_inputs + optionals,
            [("model", "MODEL"), ("positive", "CONDITIONING"),
             ("av_latent", "LATENT"), ("mux_audio", "AUDIO"),
             ("conditioned_prompt", "STRING"), ("media_map_json", "STRING"),
             ("report_json", "STRING")],
            [values[1], values[2], *values[4:], "apply_exp", 256], (2050, y))
        for name, source, dtype in (
            ("model", raw_model, "MODEL"),
            ("prompt_relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
        ):
            draft.connect(source, relay_cond, name, dtype)
        for item in original["inputs"]:
            if item["name"] == "length" or item["link"] is None:
                continue
            if item["name"] in ("width", "height"):
                next(value for value in relay_cond["inputs"]
                     if value["name"] == item["name"])["widget"] = {"name": item["name"]}
            draft.connect(_source(draft, original, item["name"]), relay_cond,
                          item["name"], item["type"])
        draft.disconnect(setup, "model")
        draft.connect((relay_cond["id"], 0), setup, "model", "MODEL")
        consumers = [tuple(link) for link in draft.links.values() if link[1] == original_id]
        for _link_id, _source_id, slot, target_id, target_slot, dtype in consumers:
            target = draft.nodes[target_id]
            name = target["inputs"][target_slot]["name"]
            draft.disconnect(target, name)
            draft.connect((relay_cond["id"], slot + 1), target, name, dtype)
        pair_inputs = [("model", "MODEL"), ("relay_positive", "CONDITIONING"),
                       ("relay_av_latent", "LATENT"),
                       ("relay_plan", "H3_T8_PROMPT_RELAY_PLAN"),
                       ("face_plan", "H3_T8_FACE_REFINE_PARITY_PLAN"),
                       ("source_frames", "IMAGE"), ("stage_av_latent", "LATENT"),
                       ("stage_context", "T8_STAGE_CONTEXT")]
        optional_names = ("parent_frames", "window_plan", "window_mapping",
                          "source_audio", "window_audio")
        pair_inputs.extend((name, {"parent_frames": "IMAGE", "window_plan": "H3_T8_FACE_REFINE_WINDOW_PLAN",
                                   "window_mapping": "H3_T8_FACE_REFINE_WINDOW_MAPPING",
                                   "source_audio": "AUDIO", "window_audio": "AUDIO"}[name])
                           for name in optional_names if any(item["name"] == name
                                                             for item in stage_bind["inputs"]))
        paired = draft.make("MiniMaxH3FaceLocalRelayBindEXPT8",
            f"Face job {index + 1} · authenticate local Relay/source",
            pair_inputs,
            [("model", "MODEL"), ("positive", "CONDITIONING"),
             ("report_json", "STRING")], [], (4400, y))
        for name, source, dtype in (
            ("model", (stage_bind["id"], 0), "MODEL"),
            ("relay_positive", (crop["id"], 0), "CONDITIONING"),
            ("relay_av_latent", (relay_cond["id"], 2), "LATENT"),
            ("relay_plan", (plan["id"], 0), "H3_T8_PROMPT_RELAY_PLAN"),
            ("face_plan", _source(draft, stage_bind, "face_plan"), "H3_T8_FACE_REFINE_PARITY_PLAN"),
            ("source_frames", _source(draft, stage_bind, "source_frames"), "IMAGE"),
            ("stage_av_latent", _source(draft, stage_bind, "av_latent"), "LATENT"),
            ("stage_context", (stage_bind["id"], 3), "T8_STAGE_CONTEXT"),
        ):
            draft.connect(source, paired, name, dtype)
        for name in optional_names:
            if any(item["name"] == name for item in paired["inputs"]):
                source = _source(draft, stage_bind, name)
                dtype = next(item["type"] for item in paired["inputs"] if item["name"] == name)
                draft.connect(source, paired, name, dtype)
        for name, source, dtype in (
            ("model", (paired["id"], 0), "MODEL"),
            ("conditioning", (paired["id"], 1), "CONDITIONING"),
        ):
            draft.disconnect(guider, name)
            draft.connect(source, guider, name, dtype)
    return draft.prune((saves[0]["id"],))


def _add_eav(frontend: dict) -> dict:
    draft = Draft(frontend)
    stages = sorted((node for node in draft.nodes.values()
                     if node["type"] == "MiniMaxH3StageSamplerEXPT8"),
                    key=lambda node: node["id"])
    if not stages:
        raise ValueError("No separate Face stage to decorate")
    for index, stage in enumerate(stages):
        bind_id, bind_slot = _source(draft, stage, "stage_context")
        bind = draft.nodes[bind_id]
        if bind_slot != 3 or "Face" not in bind["type"]:
            raise ValueError("Face stage context is not from its expected Bind")
        guider_id, _ = _source(draft, stage, "guider")
        guider = draft.nodes[guider_id]
        model_source = _source(draft, guider, "model")
        if (guider["type"] != "BasicGuider" or model_source[1] != 0 or
                draft.nodes[model_source[0]]["type"] not in
                {bind["type"], "MiniMaxH3FaceRelayBindEXPT8",
                 "MiniMaxH3FaceLocalRelayBindEXPT8"}):
            raise ValueError("Face guider is not bound to this stage")
        latent_source = _source(draft, stage, "latent_image")
        matching = [node for node in draft.nodes.values()
                    if node["type"] in FACE_AUDITS and
                    _source(draft, node, "stage_result") == (stage["id"], 2)]
        if len(matching) != 1:
            raise ValueError("Missing unique Face source audit")
        face_audit = matching[0]
        decoders = [node for node in draft.nodes.values()
                    if node["type"] == "MiniMaxH3AVDecodeT8" and
                    _source(draft, node, "av_latent") == (face_audit["id"], 0)]
        if len(decoders) != 1:
            raise ValueError("Face source audit must feed one crop decoder")
        decoder = decoders[0]
        y = 600 + 430 * index
        config = draft.make(
            "MiniMaxH3StageEAVConfigEXPT8", f"Face {index + 1} external EAV · report only",
            [], [("eav_config", "T8_STAGE_EAV_CONFIG")],
            list(CONFIG_WIDGETS.values()), (3000, y))
        apply = draft.make(
            "MiniMaxH3StageEAVApplyEXPT8", f"Face {index + 1} EAV before separate sampler",
            [("model", "MODEL"), ("sigmas", "SIGMAS"), ("av_latent", "LATENT"),
             ("stage_context", "T8_STAGE_CONTEXT"), ("eav_config", "T8_STAGE_EAV_CONFIG")],
            [("model", "MODEL"), ("runtime", "T8_STAGE_EAV_RUNTIME"),
             ("report_json", "STRING")], [], (3550, y))
        audit = draft.make(
            "MiniMaxH3StageEAVAuditEXPT8", f"Face {index + 1} actual EAV calls audit",
            [("av_latent", "LATENT"), ("runtime", "T8_STAGE_EAV_RUNTIME")],
            [("av_latent", "LATENT"), ("report_json", "STRING")], [], (4450, y))
        for source, target, name, dtype in (
            (model_source, apply, "model", "MODEL"),
            ((bind_id, 2), apply, "sigmas", "SIGMAS"),
            (latent_source, apply, "av_latent", "LATENT"),
            ((bind_id, 3), apply, "stage_context", "T8_STAGE_CONTEXT"),
            ((config["id"], 0), apply, "eav_config", "T8_STAGE_EAV_CONFIG"),
            ((face_audit["id"], 0), audit, "av_latent", "LATENT"),
            ((apply["id"], 1), audit, "runtime", "T8_STAGE_EAV_RUNTIME"),
        ):
            draft.connect(source, target, name, dtype)
        draft.disconnect(guider, "model")
        draft.connect((apply["id"], 0), guider, "model", "MODEL")
        draft.disconnect(decoder, "av_latent")
        draft.connect((audit["id"], 0), decoder, "av_latent", "LATENT")
    frontend = draft.graph
    frontend["last_node_id"] = max(draft.nodes)
    frontend["last_link_id"] = max(draft.links)
    return frontend


def graph_for(variant: str, effect: str) -> tuple[dict, dict]:
    if variant not in VARIANTS or effect not in EFFECTS_BY_VARIANT[variant]:
        raise ValueError("Unknown S24 Face workflow")
    frontend, convert = _base(variant)
    # The historical SigmaShift socket is named MODEL. The installed
    # DualClock node's socket is named model; keep the current frontend's
    # named edge stable across native Save As without changing its index.
    for node in frontend["nodes"]:
        if node["type"] == "MiniMaxH3DualClockSamplerT8":
            if node["outputs"][0]["name"] not in ("MODEL", "model"):
                raise ValueError("Unexpected DualClock model output")
            node["outputs"][0]["name"] = "model"
        if node["type"] == "GetVideoComponents":
            if [item["name"] for item in node["outputs"]] != [
                    "images", "audio", "fps", "bit_depth"]:
                raise ValueError("Unexpected old video component outputs")
            node["outputs"][3]["type"] = "COMBO"
            node["outputs"].append({"name": "color_space", "type": "COMBO", "links": []})
        if node["type"] == "PreviewImage" and not node["outputs"]:
            node["outputs"] = [{"name": "images", "type": "IMAGE", "links": []}]
        if (variant == "window_studio_serial" and
                node["type"] == "MiniMaxH3FaceRefineWindowExtractT8Advanced"):
            if node["widgets_values"] != ["edge_hold_exp"]:
                raise ValueError("Unexpected Studio linked-window widget source")
            # Current frontend serializes both widgets even when window_index
            # is connected. Preserve the intended pad_policy on Save As.
            node["widgets_values"] = [0, "edge_hold_exp"]
    if effect in ("relay", "combined"):
        frontend = (_add_relay(frontend) if variant in ("standard", "anime")
                    else _add_local_relay(frontend))
    if effect in ("eav", "combined"):
        frontend = _add_eav(frontend)
    api = convert(frontend)
    for node in frontend["nodes"]:
        if node["type"] == "MiniMaxH3StageEAVConfigEXPT8":
            _add_widget_inputs(api, node, CONFIG_WIDGETS)
        elif node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced":
            _add_widget_inputs(api, node,
                zip(RELAY_PLAN_WIDGETS, node["widgets_values"], strict=True))
        elif node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced":
            _add_widget_inputs(api, node,
                zip(RELAY_CONDITIONING_WIDGETS, node["widgets_values"], strict=True))
    frontend.setdefault("extra", {})["t8_split_example"] = {
        "schema": "t8.face-refine.split-example.v1", "route": "S24",
        "variant": variant, "effect": effect,
        "status": "experimental_importable_route_partial",
    }
    spread_frontend_columns(frontend)
    return frontend, api


def build_suite() -> dict:
    return {key: graph_for(*key) for key in FILES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-generated-output-names", action="store_true",
                        help="One-time checked mechanical rewrite of generated S24 files")
    parser.add_argument("--only-new-relay", action="store_true",
                        help="Append only four full-clip Relay graphs; preserve the first nine")
    parser.add_argument("--only-new-ersde-eav", action="store_true",
                        help="Append only five er_sde Face EAV graphs; preserve the first thirteen")
    parser.add_argument("--only-new-local-relay", action="store_true",
                        help="Append only ten local Ref2VA Relay/combined graphs; preserve the first eighteen")
    options = parser.parse_args()
    if sum((options.only_new_relay, options.only_new_ersde_eav,
            options.only_new_local_relay)) > 1:
        parser.error("Select only one append-only S24 slice")
    for key, (frontend, _api) in build_suite().items():
        if options.only_new_relay and key[1] not in ("relay", "combined"):
            continue
        if options.only_new_ersde_eav and (key[0] in ("standard", "anime") or key[1] != "eav"):
            continue
        if options.only_new_local_relay and (key[0] in ("standard", "anime")
                                             or key[1] not in ("relay", "combined")):
            continue
        path = DESTINATION / FILES[key]
        if options.repair_generated_output_names:
            saved = json.loads(path.read_text(encoding="utf-8"))
            old_schema = deepcopy(frontend)
            for node in old_schema["nodes"]:
                if node["type"] == "GetVideoComponents":
                    node["outputs"][3]["type"] = "INT"
                    node["outputs"].pop()
                if (key[0] == "window_studio_serial" and node["type"] == "PreviewImage"
                        and node["id"] == 31):
                    node["outputs"] = []
            historical = deepcopy(frontend)
            if key[0] == "window_studio_serial":
                extract = next(node for node in historical["nodes"]
                               if node["type"] == "MiniMaxH3FaceRefineWindowExtractT8Advanced")
                extract["widgets_values"] = ["edge_hold_exp"]
            normalized_name_only = deepcopy(historical)
            old_shift = min((node for node in historical["nodes"]
                             if node["type"] == "MiniMaxH3DualClockSamplerT8"),
                            key=lambda node: node["id"], default=None)
            if old_shift is not None:
                old_shift["outputs"][0]["name"] = "MODEL"
            if saved not in (historical, normalized_name_only, old_schema, frontend):
                raise ValueError(f"Refuse to overwrite changed Face graph: {path}")
            if saved != frontend:
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    json.dump(frontend, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
        else:
            write_new(path, frontend)
        print(f"{FILES[key]}: {len(frontend['nodes'])} nodes, {len(frontend['links'])} links")


if __name__ == "__main__":
    main()
