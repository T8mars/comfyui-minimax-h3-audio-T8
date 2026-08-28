from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
TEMPLATE = (
    WORKFLOWS
    / "18-audio-refine"
    / "2026-08-26_H3_Audio_Refine_Phase2_Base_Refine4_Advanced_EXP.json"
)


@dataclass(frozen=True)
class SourceSpec:
    source: Path
    output: Path
    profile: str
    model: tuple[int, int]
    positive: tuple[int, int]
    final_latent: tuple[int, int]
    original_audio: tuple[int, int]
    conditioned_prompt: tuple[int, int]
    media_map_json: tuple[int, int]
    conditioning_report: tuple[int, int]
    video_vae: tuple[int, int]
    audio_vae: tuple[int, int]
    frame_count: int = 124
    long_video_context_save_node: int | None = None
    long_video_trim_template_node: int | None = None


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, workflow: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def node_by_id(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def clear_links(node: dict) -> None:
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = None


def update_properties(node: dict, node_type: str) -> None:
    node["type"] = node_type
    properties = dict(node.get("properties") or {})
    properties["cnr_id"] = "minimax-h3-audio-T8"
    properties["Node name for S&R"] = node_type
    node["properties"] = properties


def remove_link(workflow: dict, link_id: int) -> None:
    workflow["links"] = [
        link for link in workflow["links"] if int(link[0]) != int(link_id)
    ]
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") == link_id:
                item["link"] = None
        for item in node.get("outputs", []):
            links = item.get("links")
            if isinstance(links, list) and link_id in links:
                remaining = [value for value in links if value != link_id]
                item["links"] = remaining or None


def connect(
    workflow: dict,
    origin: tuple[int, int],
    target: tuple[int, int],
    value_type: str,
) -> int:
    workflow["last_link_id"] = int(workflow.get("last_link_id", 0)) + 1
    link_id = int(workflow["last_link_id"])
    origin_node = node_by_id(workflow, origin[0])
    target_node = node_by_id(workflow, target[0])
    output = origin_node["outputs"][origin[1]]
    links = output.get("links")
    output["links"] = ([] if links is None else list(links)) + [link_id]
    target_node["inputs"][target[1]]["link"] = link_id
    workflow["links"].append(
        [link_id, origin[0], origin[1], target[0], target[1], value_type]
    )
    return link_id


def clone_template_node(
    workflow: dict,
    template: dict,
    template_id: int,
    *,
    x: float,
    y: float,
) -> dict:
    workflow["last_node_id"] = int(workflow.get("last_node_id", 0)) + 1
    node = copy.deepcopy(node_by_id(template, template_id))
    node["id"] = int(workflow["last_node_id"])
    node["pos"] = [x, y]
    node["order"] = max(int(item.get("order", 0)) for item in workflow["nodes"]) + 1
    clear_links(node)
    workflow["nodes"].append(node)
    return node


def make_route(node: dict, profile: str) -> None:
    update_properties(node, "MiniMaxH3AudioRefineCompatibilityRouteT8Advanced")
    node["title"] = "Audio Refine 4/8-Step Compatibility Route / 音频精修兼容路由"
    node["size"] = [440, 260]
    node["inputs"] = [
        {"name": "audit", "type": "H3_T8_AUDIO_REFINE_AUDIT", "link": None},
        {"name": "refine_model", "type": "MODEL", "link": None},
        {"name": "positive", "type": "CONDITIONING", "link": None},
    ]
    node["outputs"] = [
        {"name": "refine_model", "type": "MODEL", "links": None},
        {"name": "route", "type": "H3_T8_AUDIO_REFINE_COMPAT_ROUTE", "links": None},
        {"name": "decision", "type": "STRING", "links": None},
        {"name": "report_json", "type": "STRING", "links": None},
    ]
    node["widgets_values"] = [profile, 4 if profile == "turbo4" else 8]


def make_plan(node: dict, seed: int) -> None:
    update_properties(node, "MiniMaxH3AudioRefineCompatibilityPlanT8Advanced")
    node["title"] = "Audio Refine Compatibility Plan / 音频精修兼容计划"
    node["inputs"] = [
        {"name": "route", "type": "H3_T8_AUDIO_REFINE_COMPAT_ROUTE", "link": None}
    ]
    node["outputs"] = [
        {"name": "plan", "type": "H3_T8_AUDIO_REFINE_COMPAT_PLAN", "links": None},
        {"name": "decision", "type": "STRING", "links": None},
        {"name": "report_json", "type": "STRING", "links": None},
    ]
    node["widgets_values"] = [4, 0.50, int(seed)]


def make_setup(node: dict) -> None:
    update_properties(node, "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced")
    node["title"] = "Audio Refine Compatibility Setup / 音频精修兼容装配"
    node["inputs"] = [
        {"name": "plan", "type": "H3_T8_AUDIO_REFINE_COMPAT_PLAN", "link": None},
        {"name": "refine_model", "type": "MODEL", "link": None},
        {"name": "positive", "type": "CONDITIONING", "link": None},
        {"name": "av_latent", "type": "LATENT", "link": None},
    ]
    node["widgets_values"] = []


def make_long_video_split(node: dict, segment_index: int = 0) -> None:
    update_properties(node, "MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced")
    node["title"] = "Long Video Original Continuation / Reviewed Delivery Split"
    node["size"] = [460, 230]
    node["inputs"] = [
        {"name": "original_continuation_av_latent", "type": "LATENT", "link": None},
        {"name": "reviewed_delivery_av_latent", "type": "LATENT", "link": None},
        {
            "name": "candidate_selected",
            "type": "BOOLEAN",
            "link": None,
            "widget": {"name": "candidate_selected"},
        },
    ]
    node["outputs"] = [
        {"name": "continuation_av_latent", "type": "LATENT", "links": None},
        {"name": "delivery_av_latent", "type": "LATENT", "links": None},
        {"name": "report_json", "type": "STRING", "links": None},
    ]
    node["widgets_values"] = [False, int(segment_index)]


def make_note(workflow: dict, template: dict, *, x: float, y: float, text: str) -> None:
    note = clone_template_node(workflow, template, 29, x=x, y=y)
    note["title"] = "AUDIO REFINE COMPATIBILITY / 使用边界"
    note["size"] = [650, 410]
    note["widgets_values"] = [text]


def graft(spec: SourceSpec, template: dict) -> None:
    workflow = load(spec.source)
    max_x = max(float(node.get("pos", [0, 0])[0]) for node in workflow["nodes"])
    base_x = max_x + 520
    seed = 2608290000 + len(workflow["nodes"])

    audit = clone_template_node(workflow, template, 16, x=base_x, y=120)
    route = clone_template_node(workflow, template, 17, x=base_x + 450, y=120)
    make_route(route, spec.profile)
    plan = clone_template_node(workflow, template, 18, x=base_x + 930, y=120)
    make_plan(plan, seed)
    setup = clone_template_node(workflow, template, 19, x=base_x + 1380, y=120)
    make_setup(setup)
    sampler = clone_template_node(workflow, template, 20, x=base_x + 1840, y=120)
    audio_audit = clone_template_node(workflow, template, 21, x=base_x + 2250, y=120)
    decode_candidate = clone_template_node(workflow, template, 22, x=base_x + 2670, y=120)
    create_candidate = clone_template_node(workflow, template, 23, x=base_x + 3100, y=120)
    save_candidate = clone_template_node(workflow, template, 24, x=base_x + 3500, y=120)
    save_candidate["widgets_values"] = [
        f"MiniMaxH3/AudioRefineCompat/{spec.profile}_candidate"
    ]
    gate = clone_template_node(workflow, template, 25, x=base_x + 2670, y=560)
    gate["widgets_values"][0] = False
    gate["widgets_values"][1] = int(spec.frame_count)
    decode_selected = clone_template_node(workflow, template, 26, x=base_x + 3530, y=560)
    create_selected = clone_template_node(workflow, template, 27, x=base_x + 3960, y=560)
    save_selected = clone_template_node(workflow, template, 28, x=base_x + 4360, y=560)
    save_selected["widgets_values"] = [
        f"MiniMaxH3/AudioRefineCompat/{spec.profile}_selected"
    ]

    connect(workflow, spec.model, (audit["id"], 0), "MODEL")
    connect(workflow, spec.positive, (audit["id"], 1), "CONDITIONING")
    connect(workflow, spec.final_latent, (audit["id"], 2), "LATENT")
    connect(workflow, spec.conditioned_prompt, (audit["id"], 3), "STRING")
    connect(workflow, spec.media_map_json, (audit["id"], 4), "STRING")
    connect(workflow, spec.conditioning_report, (audit["id"], 5), "STRING")
    connect(workflow, (audit["id"], 0), (route["id"], 0), "H3_T8_AUDIO_REFINE_AUDIT")
    connect(workflow, spec.model, (route["id"], 1), "MODEL")
    connect(workflow, spec.positive, (route["id"], 2), "CONDITIONING")
    connect(workflow, (route["id"], 1), (plan["id"], 0), "H3_T8_AUDIO_REFINE_COMPAT_ROUTE")
    connect(workflow, (plan["id"], 0), (setup["id"], 0), "H3_T8_AUDIO_REFINE_COMPAT_PLAN")
    connect(workflow, (route["id"], 0), (setup["id"], 1), "MODEL")
    connect(workflow, spec.positive, (setup["id"], 2), "CONDITIONING")
    connect(workflow, spec.final_latent, (setup["id"], 3), "LATENT")
    for output_slot, input_slot, value_type in (
        (1, 0, "NOISE"),
        (2, 1, "GUIDER"),
        (3, 2, "SAMPLER"),
        (4, 3, "SIGMAS"),
        (5, 4, "LATENT"),
    ):
        connect(workflow, (setup["id"], output_slot), (sampler["id"], input_slot), value_type)
    connect(workflow, (setup["id"], 5), (audio_audit["id"], 0), "LATENT")
    connect(workflow, (sampler["id"], 0), (audio_audit["id"], 1), "LATENT")
    connect(workflow, (audio_audit["id"], 0), (decode_candidate["id"], 0), "LATENT")
    connect(workflow, spec.video_vae, (decode_candidate["id"], 1), "VAE")
    connect(workflow, spec.audio_vae, (decode_candidate["id"], 2), "VAE")
    connect(workflow, (decode_candidate["id"], 0), (create_candidate["id"], 0), "IMAGE")
    connect(workflow, (decode_candidate["id"], 1), (create_candidate["id"], 1), "AUDIO")
    connect(workflow, (create_candidate["id"], 0), (save_candidate["id"], 0), "VIDEO")
    connect(workflow, spec.final_latent, (gate["id"], 0), "LATENT")
    connect(workflow, (audio_audit["id"], 0), (gate["id"], 1), "LATENT")
    connect(workflow, spec.original_audio, (gate["id"], 2), "AUDIO")
    connect(workflow, (decode_candidate["id"], 1), (gate["id"], 3), "AUDIO")

    selected_origin: tuple[int, int] = (gate["id"], 0)
    if spec.long_video_context_save_node is not None:
        split = clone_template_node(workflow, template, 17, x=base_x + 3100, y=560)
        make_long_video_split(split)
        connect(workflow, spec.final_latent, (split["id"], 0), "LATENT")
        connect(workflow, (gate["id"], 0), (split["id"], 1), "LATENT")
        connect(workflow, (gate["id"], 2), (split["id"], 2), "BOOLEAN")
        context_node = node_by_id(workflow, spec.long_video_context_save_node)
        prior = context_node["inputs"][0].get("link")
        if prior is not None:
            remove_link(workflow, int(prior))
        connect(
            workflow,
            (split["id"], 0),
            (spec.long_video_context_save_node, 0),
            "LATENT",
        )
        selected_origin = (split["id"], 1)

    connect(workflow, selected_origin, (decode_selected["id"], 0), "LATENT")
    connect(workflow, spec.video_vae, (decode_selected["id"], 1), "VAE")
    connect(workflow, spec.audio_vae, (decode_selected["id"], 2), "VAE")

    if spec.long_video_trim_template_node is not None:
        trim = copy.deepcopy(node_by_id(workflow, spec.long_video_trim_template_node))
        workflow["last_node_id"] = int(workflow["last_node_id"]) + 1
        trim["id"] = int(workflow["last_node_id"])
        trim["pos"] = [base_x + 3960, 560]
        trim["order"] = max(int(item.get("order", 0)) for item in workflow["nodes"]) + 1
        clear_links(trim)
        trim["inputs"] = [
            {"name": "frames", "type": "IMAGE", "link": None},
            {"name": "audio", "type": "AUDIO", "link": None, "shape": 7},
        ]
        workflow["nodes"].append(trim)
        connect(workflow, (decode_selected["id"], 0), (trim["id"], 0), "IMAGE")
        connect(workflow, (decode_selected["id"], 1), (trim["id"], 1), "AUDIO")
        connect(workflow, (trim["id"], 0), (create_selected["id"], 0), "IMAGE")
        connect(workflow, (trim["id"], 1), (create_selected["id"], 1), "AUDIO")
    else:
        connect(workflow, (decode_selected["id"], 0), (create_selected["id"], 0), "IMAGE")
        connect(workflow, (decode_selected["id"], 1), (create_selected["id"], 1), "AUDIO")
    connect(workflow, (create_selected["id"], 0), (save_selected["id"], 0), "VIDEO")

    note = (
        "## Audio Refine 只处理最终交付音频\n\n"
        f"本文件保留原始 {spec.profile} 生成链，Audio Refine 在最终 AV latent 后追加4 NFE。"
        "Quality Gate 默认 false：先听原始与 candidate，再决定是否接受；接受后仍强制回填原视频 latent。\n\n"
        "PDD/EAV 不会在精修侧重跑；Prompt Relay 精修会复用同一认证 binding。双采只能接在Pass2最终输出后。"
        "长视频 continuation 必须使用分流节点的 original 输出，delivery 不能喂给下一段。\n\n"
        "模型文件名、大小和SHA不做硬门。每次只串行运行一个工作流；4+4、8+4只是实际NFE成本，不代表训练分布等价。"
    )
    make_note(workflow, template, x=base_x, y=980, text=note)
    save(spec.output, workflow)


def transform_plain_turbo8(template: dict) -> None:
    workflow = copy.deepcopy(template)
    node_by_id(workflow, 9)["widgets_values"][0] = 8
    node_by_id(workflow, 9)["title"] = "Turbo8 first pass · native dual clock 12/3"
    route = node_by_id(workflow, 17)
    make_route(route, "turbo8")
    plan = node_by_id(workflow, 18)
    make_plan(plan, 2608290034)
    setup = node_by_id(workflow, 19)
    make_setup(setup)
    for link_id in (22, 28, 29, 30, 31, 32, 33, 34, 35):
        remove_link(workflow, link_id)
    connect(workflow, (6, 0), (16, 0), "MODEL")
    connect(workflow, (16, 0), (17, 0), "H3_T8_AUDIO_REFINE_AUDIT")
    connect(workflow, (6, 0), (17, 1), "MODEL")
    connect(workflow, (8, 0), (17, 2), "CONDITIONING")
    connect(workflow, (17, 1), (18, 0), "H3_T8_AUDIO_REFINE_COMPAT_ROUTE")
    connect(workflow, (18, 0), (19, 0), "H3_T8_AUDIO_REFINE_COMPAT_PLAN")
    connect(workflow, (17, 0), (19, 1), "MODEL")
    connect(workflow, (8, 0), (19, 2), "CONDITIONING")
    connect(workflow, (12, 0), (19, 3), "LATENT")
    node_by_id(workflow, 24)["widgets_values"] = [
        "MiniMaxH3/AudioRefineCompat/turbo8_candidate"
    ]
    node_by_id(workflow, 28)["widgets_values"] = [
        "MiniMaxH3/AudioRefineCompat/turbo8_selected"
    ]
    node_by_id(workflow, 30)["widgets_values"] = [
        "## Turbo8 + Audio Refine4\n\n首遍真实8 NFE，最终再追加4 NFE；总成本12 NFE。"
        "旧Turbo4 Phase2节点没有改，这个文件使用新兼容路由。"
    ]
    save(
        WORKFLOWS
        / "18-audio-refine"
        / "2026-08-29_H3_Audio_Refine_Turbo8_Plus_Refine4_Advanced_EXP.json",
        workflow,
    )


def main() -> None:
    template = load(TEMPLATE)
    transform_plain_turbo8(template)
    specs = [
        SourceSpec(
            source=WORKFLOWS / "13-latent-upscale" / "2026-08-21_H3_Learned_Latent_TwoPass_I2VA_Native_Speech_Advanced_EXP.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_Learned_TwoPass_Final8_Advanced_EXP.json",
            profile="learned_latent_twopass_final8",
            model=(4, 0), positive=(14, 0), final_latent=(19, 0), original_audio=(20, 1),
            conditioned_prompt=(14, 3), media_map_json=(14, 4), conditioning_report=(14, 5),
            video_vae=(1, 0), audio_vae=(2, 0),
        ),
        SourceSpec(
            source=WORKFLOWS / "19-pdd-acceleration" / "2026-08-27_H3_PDD_Ref2VA_8Step_Advanced_EXP.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_PDD_Ref2VA8_Advanced_EXP.json",
            profile="pdd8",
            model=(1, 0), positive=(6, 0), final_latent=(11, 0), original_audio=(12, 1),
            conditioned_prompt=(6, 3), media_map_json=(6, 4), conditioning_report=(6, 5),
            video_vae=(3, 0), audio_vae=(4, 0),
        ),
        SourceSpec(
            source=WORKFLOWS / "19-pdd-acceleration" / "2026-08-27_H3_PDD_Ref2VA_Learned_Latent_TwoPass_4Plus4_Stable.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_PDD_Ref2VA_4Plus4_Advanced_EXP.json",
            profile="pdd4_plus4",
            model=(4, 0), positive=(14, 0), final_latent=(19, 0), original_audio=(20, 1),
            conditioned_prompt=(14, 3), media_map_json=(14, 4), conditioning_report=(14, 5),
            video_vae=(1, 0), audio_vae=(2, 0),
            frame_count=22,
        ),
        SourceSpec(
            source=WORKFLOWS / "07-motion-detail" / "2026-08-21_H3_Enhance_A_Video_FETA_T2VA_Turbo8_Advanced_EXP.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_EAV_Turbo8_Advanced_EXP.json",
            profile="eav_turbo8",
            model=(1, 0), positive=(6, 0), final_latent=(10, 0), original_audio=(11, 1),
            conditioned_prompt=(6, 3), media_map_json=(6, 4), conditioning_report=(6, 5),
            video_vae=(4, 0), audio_vae=(5, 0),
        ),
        SourceSpec(
            source=WORKFLOWS / "14-prompt-relay" / "2026-08-20_H3_Prompt_Relay_T2VA_Turbo8_Advanced_EXP.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_Prompt_Relay_Turbo8_Advanced_EXP.json",
            profile="prompt_relay_turbo8",
            model=(6, 0), positive=(6, 1), final_latent=(10, 0), original_audio=(11, 1),
            conditioned_prompt=(6, 4), media_map_json=(6, 5), conditioning_report=(6, 6),
            video_vae=(4, 0), audio_vae=(5, 0),
        ),
        SourceSpec(
            source=WORKFLOWS / "04-long-video" / "2026-08-20_H3_Prompt_Relay_Long_Video_Turbo8_Advanced_EXP.json",
            output=WORKFLOWS / "18-audio-refine" / "2026-08-29_H3_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_Advanced_EXP.json",
            profile="long_video_prompt_relay_turbo8",
            model=(9, 0), positive=(9, 1), final_latent=(14, 0), original_audio=(16, 1),
            conditioned_prompt=(9, 4), media_map_json=(9, 5), conditioning_report=(9, 6),
            video_vae=(3, 0), audio_vae=(4, 0),
            long_video_context_save_node=15,
            long_video_trim_template_node=17,
        ),
    ]
    for spec in specs:
        graft(spec, template)


if __name__ == "__main__":
    main()
