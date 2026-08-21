from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows" / "07-motion-detail"
BASE = WORKFLOW_DIR / "2026-08-21_H3_Enhance_A_Video_FETA_Stock20_Advanced_EXP.json"
INSTALLED = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "07-motion-detail"
)

PROMPT = (
    "Night, one continuous cinematic shot on a rain-wet neon street. An adult woman "
    "in a red coat raises one hand, turns, then runs away from camera while the camera "
    "pulls rapidly upward until she becomes a small full-body figure in the street. "
    "Preserve her identity, natural anatomy, stable limbs, crisp clothing and smooth "
    "large-amplitude motion. No dialogue. Synchronized footsteps, coat movement, wind, "
    "traffic and distant city ambience."
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


def _load_image_node(node_id: int, title: str, filename: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "LoadImage",
        "title": title,
        "pos": pos,
        "size": [360, 430],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
            {"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None},
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": None},
        ],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "LoadImage"},
        "widgets_values": [filename, "image"],
    }


def _lora_node(node_id: int) -> dict:
    return {
        "id": node_id,
        "type": "LoraLoaderBypassModelOnly",
        "title": "Corrected Alpha8 Turbo LoRA · 208 modules · strength 1.0",
        "pos": [760, -300],
        "size": [520, 150],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {
                "name": "lora_name",
                "type": "COMBO",
                "widget": {"name": "lora_name"},
                "link": None,
            },
            {
                "name": "strength_model",
                "type": "FLOAT",
                "widget": {"name": "strength_model"},
                "link": None,
            },
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "properties": {"Node name for S&R": "LoraLoaderBypassModelOnly"},
        "widgets_values": [
            "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
            1.0,
        ],
    }


def _append_link(workflow: dict, source: dict, output_slot: int, target: dict, input_slot: int, link_type: str) -> int:
    link_id = int(workflow["last_link_id"]) + 1
    workflow["last_link_id"] = link_id
    workflow["links"].append([link_id, source["id"], output_slot, target["id"], input_slot, link_type])
    source["outputs"][output_slot].setdefault("links", [])
    source["outputs"][output_slot]["links"].append(link_id)
    target["inputs"][input_slot]["link"] = link_id
    return link_id


def _notes(workflow: dict, *, task: str, profile: str) -> None:
    note_nodes = sorted(
        (node for node in workflow["nodes"] if node["type"] == "MarkdownNote"),
        key=lambda node: node["id"],
    )
    schedule = "Stock20 / 20 NFE" if profile == "stock20" else "Turbo8 / 8 NFE"
    if task in {"Ref2VA", "Hybrid"}:
        scope_text = (
            "本模板使用独立 Reference Composer：参考图只参与原生 H3 packed layout，FETA 仍只量测和"
            "缩放目标视频行；参考图、首帧、文本与音频行不会被直接缩放。普通 EAV 节点仍拒绝参考块。"
        )
        conflict_text = (
            "仅支持未打补丁的原生 Stock20；Prompt Relay、普通LoRA、BlockCache、Sage、STG、Long Video、"
            "模型权重 Hybrid artifact 与 denoise mask 仍会主动拒绝。这里的 Hybrid 是任务类型，不是混合权重模型。"
        )
    else:
        scope_text = (
            "EAV 只从目标视频 Q/K 计算跨帧 CFI，并只直接缩放目标视频输出；首帧/尾帧条件段不会被直接缩放。"
        )
        conflict_text = (
            "Ref2VA、Hybrid、denoise mask、Prompt Relay、BlockCache、Sage、STG 和其他 attention/block wrapper "
            "仍会主动拒绝。"
        )
    note_nodes[0]["widgets_values"] = (
        f"## 正确连接与任务范围\n\n当前模板：**{task}、{schedule}、1152×640、124帧**。"
        f"{scope_text}Runtime Audit 必须输出 `apply_exp_verified`。{conflict_text}"
    )
    note_nodes[1]["widgets_values"] = (
        "## 参数与 A/B 方法\n\n`mode=disabled` 才是严格旁路；`tau=0` 不是关闭。"
        "先固定素材、提示词、seed、尺寸和 NFE，只在 `disabled` 与 `apply_exp` 间切换。"
        "`tau=4` 只是上游候选，不是 H3 最优值；`max_workspace_mib=32` 只约束 EAV 的临时分数缓冲，"
        "不代表整套工作流显存。`g_hard_limit=1.5` 超限会报错，不会静默裁剪。"
    )
    if profile == "turbo8_alpha8":
        note_nodes[2]["widgets_values"] = (
            "## Turbo8 严格合同\n\n只使用修正后的 `comfyui_alpha8-T8-convert` LoRA，"
            "`LoraLoaderBypassModelOnly`、208个模块、strength=1.0，并在 EAV 前完成注入。"
            "旧 plain 转换、普通 weight patch、模块数或强度不符都会被拒绝。Turbo8 与 FETA 的画质和音频"
            "已完成一组0.7MP机械/媒体A/B，但没有通过画质或音频非劣结论；能运行不等于比Stock20更好。"
        )
    else:
        note_nodes[2]["widgets_values"] = (
            "## 科学边界\n\n这是 Enhance-A-Video/FETA 对 H3 full-3D packed attention 的实验适配，"
            "不是锐化、修脸或 sigma 重排。目标音频行不会被直接缩放，但联合 Transformer 后续层仍可能"
            "间接改变声音，因此每条成片都要试听。0.7MP 单条通过只证明该任务的机械与媒体合同，不代表"
            "所有素材稳定提质或通用16GB安全。"
        )


def _set_orders(workflow: dict) -> None:
    priority = {
        "UNETLoader": 0,
        "CLIPLoader": 1,
        "VAELoader": 2,
        "LoadImage": 4,
        "LoraLoaderBypassModelOnly": 6,
        "MiniMaxH3AudioConditioningT8": 7,
        "MiniMaxH3DualClockSamplerT8": 8,
        "MiniMaxH3EnhanceAVideoT8Advanced": 9,
        "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced": 9,
        "MiniMaxH3EnhanceAVideoSageComposerT8Advanced": 9,
        "RandomNoise": 10,
        "BasicGuider": 11,
        "SamplerCustomAdvanced": 12,
        "MiniMaxH3EnhanceAVideoAuditT8Advanced": 13,
        "MiniMaxH3AVDecodeT8": 14,
        "VHS_VideoCombine": 15,
        "MarkdownNote": 16,
    }
    for order, node in enumerate(
        sorted(workflow["nodes"], key=lambda item: (priority.get(item["type"], 99), item["id"]))
    ):
        node["order"] = order


def build(task: str, profile: str) -> tuple[str, dict]:
    workflow = json.loads(BASE.read_text(encoding="utf-8"))
    conditioning = _node(workflow, 6)
    dual = _node(workflow, 7)
    eav = _node(workflow, 13)
    save = _node(workflow, 12)

    conditioning["title"] = f"{task} 0.7MP controlled input"
    conditioning["widgets_values"][0] = PROMPT
    conditioning["widgets_values"][1:5] = [1152, 640, 124, task]
    reference_task = task in {"Ref2VA", "Hybrid"}
    if reference_task:
        eav["type"] = "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced"
        eav["properties"]["Node name for S&R"] = eav["type"]
        eav["inputs"] = eav["inputs"][:-1]
        eav["widgets_values"] = eav["widgets_values"][:-1]
    else:
        eav["widgets_values"][-1] = profile
    eav["title"] = f"EAV/FETA tau4 · {task} · {profile}"
    _notes(workflow, task=task, profile=profile)

    if task in {"I2VA", "FL2VA"}:
        first = _load_image_node(
            int(workflow["last_node_id"]) + 1,
            "First-frame anchor · replace with your own image",
            "codex_prompt_relay_fl2va_first.png",
            [300, -520],
        )
        workflow["last_node_id"] = first["id"]
        workflow["nodes"].append(first)
        _append_link(workflow, first, 0, conditioning, 17, "IMAGE")
    if task == "Hybrid":
        first = _load_image_node(
            int(workflow["last_node_id"]) + 1,
            "First-frame anchor · replace with your own image",
            "codex_prompt_relay_fl2va_first.png",
            [300, -520],
        )
        workflow["last_node_id"] = first["id"]
        workflow["nodes"].append(first)
        _append_link(workflow, first, 0, conditioning, 17, "IMAGE")
    if task in {"L2VA", "FL2VA"}:
        last = _load_image_node(
            int(workflow["last_node_id"]) + 1,
            "Last-frame anchor · replace with your own image",
            "codex_prompt_relay_fl2va_last.png",
            [690, -520],
        )
        workflow["last_node_id"] = last["id"]
        workflow["nodes"].append(last)
        _append_link(workflow, last, 0, conditioning, 18, "IMAGE")

    if reference_task:
        reference = _load_image_node(
            int(workflow["last_node_id"]) + 1,
            "Reference image · identity / costume / appearance",
            "replace_with_authorized_reference.png",
            [690 if task == "Hybrid" else 300, -520],
        )
        workflow["last_node_id"] = reference["id"]
        workflow["nodes"].append(reference)
        conditioning["inputs"].append(
            {
                "name": "ref_images.ref_image_0",
                "type": "IMAGE",
                "link": None,
                "shape": 7,
            }
        )
        _append_link(workflow, reference, 0, conditioning, 19, "IMAGE")

    if profile == "turbo8_alpha8":
        lora = _lora_node(int(workflow["last_node_id"]) + 1)
        workflow["last_node_id"] = lora["id"]
        workflow["nodes"].append(lora)
        original = next(link for link in workflow["links"] if link[0] == 1)
        original[3], original[4] = lora["id"], 0
        lora["inputs"][0]["link"] = 1
        _node(workflow, 1)["outputs"][0]["links"] = [1]
        _node(workflow, 7)["inputs"][0]["link"] = None
        _append_link(workflow, lora, 0, dual, 0, "MODEL")
        dual["widgets_values"][0] = 8
        dual["title"] = "Turbo8 native dual clock · shift 12/3"
        save["widgets_values"]["filename_prefix"] = "MiniMaxH3_EAV/eav_t2va_turbo8_tau4_exp"
    else:
        save["widgets_values"]["filename_prefix"] = f"MiniMaxH3_EAV/eav_{task.lower()}_stock20_tau4_exp"

    workflow["extra"]["t8_enhance_a_video"] = {
        "scope": f"{task} {profile} Advanced EXP",
        "paper": "arXiv:2502.07508v3",
        "reference_commit": "16a7899e6f55f85ea19f1d3a415c6dc0c4096176",
        "canvas": "1152x640x124",
        "real_probe": (
            "mechanical native-layout and registration coverage passed; real 0.7MP A/B remains pending"
            if reference_task
            else "one disabled/apply 0.7MP pair passed runtime audit and three strict media decodes"
        ),
        "quality_status": "mechanical_media_pass_quality_audio_noninferiority_unproven",
    }
    _set_orders(workflow)
    suffix = "Turbo8" if profile == "turbo8_alpha8" else "Stock20"
    filename = f"2026-08-21_H3_Enhance_A_Video_FETA_{task}_{suffix}_Advanced_EXP.json"
    return filename, workflow


def build_strict_sage_workflow() -> tuple[str, dict]:
    filename, workflow = build("T2VA", "stock20")
    eav = _node(workflow, 13)
    eav["type"] = "MiniMaxH3EnhanceAVideoSageComposerT8Advanced"
    eav["title"] = "EAV/FETA tau4 + Strict Sage HND · T2VA · Stock20"
    eav["size"] = [560, 440]
    eav["properties"]["Node name for S&R"] = eav["type"]
    eav["inputs"].insert(
        2,
        {
            "name": "task_scope",
            "type": "COMBO",
            "widget": {"name": "task_scope"},
            "link": None,
        },
    )
    eav["widgets_values"].insert(0, "visual")
    save = _node(workflow, 12)
    save["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3_EAV/eav_strict_sage_t2va_stock20_tau4_exp"
    )
    notes = sorted(
        (node for node in workflow["nodes"] if node["type"] == "MarkdownNote"),
        key=lambda node: node["id"],
    )
    notes[0]["title"] = "① Strict Sage 正确接法"
    notes[0]["widgets_values"] = (
        "## 只使用这一个组合节点\n\n不要再串 KJNodes 的 MiniMax H3 Sage Patch、全局"
        "attention override、BlockCache 或 STG。本节点同时拥有 FETA 路由和严格 Sage HND 后端；"
        "Runtime Audit 必须显示20次forward、每次50个FETA测量，并且每次50个成功Sage调用、"
        "failure=0、fallback=0。"
    )
    notes[1]["title"] = "② 模式、范围与 A/B"
    notes[1]["widgets_values"] = (
        "## 参数\n\n`task_scope=visual`支持T2VA/I2VA/FL2VA/L2VA；参考任务请显式改为"
        "`reference`且只用Stock20。`disabled`完全旁路Sage和FETA；`report_only`仍运行Sage、"
        "只关闭FETA增益；`apply_exp`同时运行两者。比较时固定素材、prompt、seed、尺寸和NFE。"
    )
    notes[2]["title"] = "③ 科学边界与显存"
    notes[2]["widgets_values"] = (
        "## 当前结论\n\n该节点不会像ComfyUI原生Sage那样在内核异常时静默回退，失败会直接停止。"
        "本机一条1152×640×124、Stock20实测命中20×50次FETA测量和20×50次Sage调用，"
        "failure=0、fallback=0，并通过三轮严格音视频解码；这不保证比原生attention更快、"
        "更省显存或画质更好。FETA只直接缩放目标视频行，但H3是联合AV Transformer，音频仍可能"
        "间接变化，必须看片并试听。"
    )
    workflow["extra"]["t8_enhance_a_video"] = {
        "scope": "T2VA Stock20 EAV plus Strict Sage Advanced EXP",
        "paper": "Enhance-A-Video arXiv:2502.07508v3",
        "attention_backend": "sageattention.sageattn HND no fallback",
        "canvas": "1152x640x124",
        "validation_status": (
            "one real 1152x640x124 Stock20 run passed 20x50 FETA measurements, "
            "20x50 strict Sage calls, zero failures/fallbacks and three strict AV decodes"
        ),
        "quality_status": "quality_speed_memory_audio_claims_false",
    }
    _set_orders(workflow)
    return (
        "2026-08-21_H3_Enhance_A_Video_FETA_Strict_Sage_T2VA_Stock20_Advanced_EXP.json",
        workflow,
    )


def main() -> None:
    BASE_WORKFLOW = json.loads(BASE.read_text(encoding="utf-8"))
    base_conditioning = _node(BASE_WORKFLOW, 6)
    base_conditioning["widgets_values"][1:3] = [1152, 640]
    _node(BASE_WORKFLOW, 12)["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3_EAV/eav_t2va_stock20_0p7mp_tau4_exp"
    )
    BASE_WORKFLOW["extra"]["t8_enhance_a_video"]["real_probe"] = (
        "1152x640x124, seed 2608217001, tau4, 20x50 measurements"
    )
    _notes(BASE_WORKFLOW, task="T2VA", profile="stock20")
    BASE.write_text(json.dumps(BASE_WORKFLOW, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    specs = [
        ("I2VA", "stock20"),
        ("FL2VA", "stock20"),
        ("L2VA", "stock20"),
        ("T2VA", "turbo8_alpha8"),
        ("Ref2VA", "stock20"),
        ("Hybrid", "stock20"),
    ]
    for task, profile in specs:
        filename, workflow = build(task, profile)
        (WORKFLOW_DIR / filename).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    sage_filename, sage_workflow = build_strict_sage_workflow()
    (WORKFLOW_DIR / sage_filename).write_text(
        json.dumps(sage_workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    INSTALLED.mkdir(parents=True, exist_ok=True)
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        (INSTALLED / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
