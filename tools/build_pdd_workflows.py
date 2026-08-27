#!/usr/bin/env python3
"""Build the frontend-only MiniMax-H3 PDD example workflows."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "examples" / "workflows"
OUTPUT = WORKFLOW_ROOT / "19-pdd-acceleration"
FL_SOURCE = (
    WORKFLOW_ROOT
    / "15-sla-attention"
    / "2026-08-26_H3_Turbo_SLA_Profile_Router_FL2VA_Advanced_EXP.json"
)
REF_SOURCE = (
    WORKFLOW_ROOT
    / "03-image-video-edit"
    / "2026-08-10_H3_Ref2VA_Visual_Reference_Strength_EXP.json"
)
TWO_PASS_SOURCE = (
    WORKFLOW_ROOT
    / "13-latent-upscale"
    / "2026-08-21_H3_Learned_Latent_TwoPass_I2VA_Standard_Advanced_EXP.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def note(node_id: int, title: str, text: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "title": title,
        "pos": pos,
        "size": [930, 430],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {
            "Node name for S&R": "MarkdownNote",
            "cnr_id": "comfy-core",
        },
        "widgets_values": text,
    }


def pdd_node(node_id: int, filename: str, variant: str, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "MiniMaxH3PDD8StepSetupT8Advanced",
        "title": f"PDD 8-Step Setup · {variant}（必须匹配完整基模）",
        "pos": pos,
        "size": [500, 300],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "av_latent", "type": "LATENT", "link": None},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": None},
            {"name": "sampler", "type": "SAMPLER", "links": None},
            {"name": "sigmas", "type": "SIGMAS", "links": None},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "properties": {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": "MiniMaxH3PDD8StepSetupT8Advanced",
        },
        "widgets_values": [filename, variant, 1.0],
    }


def split_sigmas_node(node_id: int, pos: list[int]) -> dict:
    return {
        "id": node_id,
        "type": "SplitSigmas",
        "title": "Split official PDD trajectory · 4 LOW + 4 HIGH",
        "pos": pos,
        "size": [390, 150],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [{"name": "sigmas", "type": "SIGMAS", "link": None}],
        "outputs": [
            {"name": "high_sigmas", "type": "SIGMAS", "links": None},
            {"name": "low_sigmas", "type": "SIGMAS", "links": None},
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "Node name for S&R": "SplitSigmas",
        },
        "widgets_values": [4],
    }


def high_geometry_sampler_node(source: dict) -> dict:
    node = copy.deepcopy(source)
    node["id"] = 16
    node["title"] = "HIGH geometry sampler · reuse PDD MODEL · 12V/3A"
    node["pos"] = [3520, 0]
    node["order"] = 15
    return node


def _set_note(node: dict, title: str, text: str) -> None:
    node["title"] = title
    node["widgets_values"] = [text]


def build_two_pass(variant: str) -> dict:
    """Split the exact PDD eight-step trajectory around one learned upscale.

    PDD's output heads are sigma-indexed.  The first sampler consumes official
    blocks 0..3 and the second consumes blocks 4..7, so the graph still performs
    eight total joint AV model evaluations rather than two independent 8-step
    trajectories.  A fresh high-resolution sampler is required because the
    custom dual-clock sampler closes over packed latent geometry.
    """

    if variant not in {"FL2VA", "Ref2VA"}:
        raise ValueError(f"unsupported PDD two-pass variant: {variant}")
    workflow = read(TWO_PASS_SOURCE)
    workflow["nodes"] = [
        copy.deepcopy(node) for node in workflow["nodes"] if node["id"] != 5
    ]
    by_id = {node["id"]: node for node in workflow["nodes"]}
    base_name = (
        "minimax_h3_fl2va_int8_convrot.safetensors"
        if variant == "FL2VA"
        else "minimax_h3_ref2va_int8_convrot.safetensors"
    )
    adapter_name = (
        "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors"
        if variant == "FL2VA"
        else "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors"
    )
    task_type = variant
    ref2va_stable = variant == "Ref2VA"
    low_width, low_height = (864, 480) if ref2va_stable else (512, 288)
    high_width, high_height = (1312, 736) if ref2va_stable else (1024, 576)
    frame_count = 22 if ref2va_stable else 124
    scale_by = 1.5 if ref2va_stable else 2.0
    prompt = (
        "One continuous cinematic shot of the same adult woman at a stable portrait scale. "
        "Natural blinking, subtle head motion and realistic cloth movement. She says clearly "
        "in Mandarin exactly once: <d>你在干嘛呢，我在这里呀，看看效果如何。</d> "
        "Synchronized speech, natural ambience, no added words, no repeated mumbling, no "
        "subtitles, no cuts or sudden scale changes."
        if variant == "FL2VA"
        else "Use <Picture 1> as the visual identity and appearance reference. In one continuous "
        "cinematic portrait, keep the same adult woman at a stable scale while she turns gently "
        "and blinks once. She says clearly in Mandarin exactly once: <d>你在干嘛呢，我在这里呀，"
        "看看效果如何。</d> Synchronized speech, natural ambience, no added words, no repeated "
        "mumbling, no subtitles, cuts or additional people."
    )

    by_id[4]["title"] = f"Full non-pruned H3 {variant} base"
    by_id[4]["widgets_values"][0] = base_name
    by_id[6]["title"] = (
        "First frame · replace with your image"
        if variant == "FL2VA"
        else "Reference image · Picture 1"
    )
    by_id[6]["widgets_values"] = ["10A.jpg"]

    source_sampler = copy.deepcopy(by_id[8])
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in {8, 9, 16}]
    workflow["nodes"].extend(
        [
            pdd_node(8, adapter_name, variant, [880, 0]),
            split_sigmas_node(9, [1320, 0]),
            high_geometry_sampler_node(source_sampler),
        ]
    )
    by_id = {node["id"]: node for node in workflow["nodes"]}

    for node_id, width, height, title in (
        (7, low_width, low_height, f"LOW {variant} Conditioning · {low_width}x{low_height}"),
        (14, high_width, high_height, f"HIGH {variant} Conditioning · auto size from upscaler"),
    ):
        conditioning = by_id[node_id]
        conditioning["title"] = title
        conditioning["widgets_values"][0] = prompt
        conditioning["widgets_values"][1] = width
        conditioning["widgets_values"][2] = height
        conditioning["widgets_values"][3] = frame_count
        conditioning["widgets_values"][4] = task_type
        if node_id == 14:
            conditioning["widgets_values"][-1] = True
        if variant == "Ref2VA" and not any(
            item["name"] == "ref_images.ref_image_0"
            for item in conditioning["inputs"]
        ):
            conditioning["inputs"].append(
                {
                    "name": "ref_images.ref_image_0",
                    "type": "IMAGE",
                    "link": None,
                }
            )

    by_id[12]["title"] = "PASS 1 · PDD blocks 0–3 · use denoised_output"
    by_id[13]["title"] = f"Learned video-latent upscale · default {scale_by:g}x"
    by_id[13]["widgets_values"][0] = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    by_id[13]["widgets_values"][1] = "scale_by"
    by_id[13]["widgets_values"][2] = scale_by
    by_id[13]["widgets_values"][4] = high_width
    by_id[13]["widgets_values"][5] = high_height
    by_id[13]["widgets_values"][8] = "fp16"
    by_id[13]["widgets_values"][9] = "offload_after"
    by_id[15]["title"] = "Reconcile HIGH latent · native joint AV continuation"
    by_id[15]["widgets_values"] = ["auto", "legacy_policy", 0.0]
    by_id[17]["title"] = "HIGH guider · PDD model"
    by_id[18]["title"] = "PASS 2 fresh noise"
    by_id[19]["title"] = "PASS 2 · PDD blocks 4–7 · decode output"
    by_id[20]["title"] = "Decode completed HIGH AV"
    by_id[21]["title"] = "Save PDD 4+4 synchronized MP4"
    by_id[21]["widgets_values"][2] = (
        f"MiniMaxH3_PDD/{variant.lower()}_learned_twopass_4plus4_"
        f"{low_width}x{low_height}_scale_{str(scale_by).replace('.', 'p')}"
    )
    by_id[21]["widgets_values"][3] = "video/h265-mp4"

    if variant == "FL2VA":
        last_frame = copy.deepcopy(by_id[6])
        last_frame["id"] = 27
        last_frame["title"] = "Last frame · replace with your image"
        last_frame["pos"] = [0, 1440]
        last_frame["order"] = 26
        workflow["nodes"].append(last_frame)

    notes = {
        22: (
            "1 · Exact PDD 4+4 / 精确PDD双采",
            "## 1 · One official PDD trajectory, not 8+8\nThe PDD Setup emits the official nine sigma values. `SplitSigmas step=4` sends blocks 0–3 to LOW and blocks 4–7 to HIGH. Total Transformer work remains exactly 8 joint AV forwards. Do not run two complete PDD schedules and do not change the split unless a separately trained PDD contract is available.",
        ),
        23: (
            "2 · Geometry handoff / 几何交接",
            "## 2 · Why HIGH has another sampler setup\nPass 1 must send `denoised_output` into Learned Latent Upscale. The upscaler changes packed video geometry, so HIGH creates a new dual-clock sampler from the already-patched PDD MODEL and the reconciled HIGH latent. Its generated full sigmas are intentionally unused; PASS 2 receives the lower half from the original PDD schedule.",
        ),
        24: (
            "3 · Audio continuation / 音频继续采样",
            "## 3 · Keep native joint AV continuation\nReconcile stays at `second_pass_audio_source=legacy_policy`. PASS 2 continues both video and audio through PDD blocks 4–7; do not freeze pass-1 audio or insert Two-Pass Audio Audit in this native route. Decode the PASS 2 `output`, not `denoised_output`.",
        ),
        25: (
            "4 · Safe starting size / 建议起始尺寸",
            (
                "## 4 · Tested Ref2VA default\nThe formal Ref2VA workflow starts at `864x480x22` (about 0.4MP) with `scale_by=1.5`; 32-pixel alignment produces `1312x736x22`. One serial 16GB run passed with only 634MiB minimum free VRAM, so keep one job at a time. The upscaler's actual width/height already feed HIGH Conditioning."
                if ref2va_stable
                else "## 4 · Conservative FL2VA experiment\nThe FL2VA example starts at `512x288x124` and learned-upscales to `1024x576x124`. Its actual width/height already feed HIGH Conditioning. FL2VA has static contract coverage but no equivalent real two-pass render yet, so keep one job at a time."
            ),
        ),
        26: (
            "5 · Models and limits / 模型与边界",
            f"## 5 · Required {variant} assets\nUse the full non-pruned `{base_name}` base, `{adapter_name}` through the dedicated PDD Setup, and `minimax_h3_latent_upscaler_3d_fp16.safetensors`. Do not add Turbo/SLA/ordinary LoRA. "
            + (
                "This Ref2VA 4+4 preset is the supported tested route; its one accepted render does not prove every model build, duration or 16GB environment safe."
                if ref2va_stable
                else "This FL2VA 4+4 route remains experimental until an equivalent real render is completed."
            ),
        ),
    }
    for node_id, (title, text) in notes.items():
        _set_note(by_id[node_id], title, text)

    endpoints = [
        (3, 0, 7, 0, "CLIP"),
        (1, 0, 7, 1, "VAE"),
        (2, 0, 7, 2, "VAE"),
        (4, 0, 8, 0, "MODEL"),
        (7, 1, 8, 1, "LATENT"),
        (8, 2, 9, 0, "SIGMAS"),
        (8, 0, 10, 0, "MODEL"),
        (7, 0, 10, 1, "CONDITIONING"),
        (11, 0, 12, 0, "NOISE"),
        (10, 0, 12, 1, "GUIDER"),
        (8, 1, 12, 2, "SAMPLER"),
        (9, 0, 12, 3, "SIGMAS"),
        (7, 1, 12, 4, "LATENT"),
        (12, 1, 13, 0, "LATENT"),
        (13, 1, 14, 3, "INT"),
        (13, 2, 14, 4, "INT"),
        (3, 0, 14, 0, "CLIP"),
        (1, 0, 14, 1, "VAE"),
        (2, 0, 14, 2, "VAE"),
        (13, 0, 15, 0, "LATENT"),
        (14, 1, 15, 1, "LATENT"),
        (14, 0, 15, 2, "CONDITIONING"),
        (8, 0, 16, 0, "MODEL"),
        (15, 0, 16, 1, "LATENT"),
        (16, 0, 17, 0, "MODEL"),
        (15, 1, 17, 1, "CONDITIONING"),
        (18, 0, 19, 0, "NOISE"),
        (17, 0, 19, 1, "GUIDER"),
        (16, 1, 19, 2, "SAMPLER"),
        (9, 1, 19, 3, "SIGMAS"),
        (15, 0, 19, 4, "LATENT"),
        (19, 0, 20, 0, "LATENT"),
        (1, 0, 20, 1, "VAE"),
        (2, 0, 20, 2, "VAE"),
        (20, 0, 21, 0, "IMAGE"),
        (20, 1, 21, 1, "AUDIO"),
    ]
    if variant == "FL2VA":
        endpoints.extend(
            [
                (6, 0, 7, 5, "IMAGE"),
                (27, 0, 7, 6, "IMAGE"),
                (6, 0, 14, 7, "IMAGE"),
                (27, 0, 14, 8, "IMAGE"),
            ]
        )
    else:
        endpoints.extend(
            [
                (6, 0, 7, 7, "IMAGE"),
                (6, 0, 14, 9, "IMAGE"),
            ]
        )
    relink(workflow, endpoints)
    title = (
        "MiniMax H3 PDD Ref2VA learned latent two-pass 4+4 Stable"
        if ref2va_stable
        else "MiniMax H3 PDD FL2VA learned latent two-pass 4+4 Advanced EXP"
    )
    workflow.setdefault("extra", {})["workflow_title"] = title
    workflow["extra"]["workflow_name"] = title
    return workflow


def relink(workflow: dict, endpoints: list[tuple[int, int, int, int, str]]) -> None:
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for node in workflow["nodes"]:
        for input_ in node.get("inputs", []):
            input_["link"] = None
        for output in node.get("outputs", []):
            output["links"] = None
    links = []
    for link_id, (source, source_slot, target, target_slot, type_) in enumerate(
        endpoints, 1
    ):
        links.append([link_id, source, source_slot, target, target_slot, type_])
        by_id[target]["inputs"][target_slot]["link"] = link_id
        output = by_id[source]["outputs"][source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
    workflow["links"] = links
    workflow["last_link_id"] = len(links)
    workflow["last_node_id"] = max(by_id)


def build_fl2va() -> dict:
    workflow = read(FL_SOURCE)
    workflow["nodes"] = [
        copy.deepcopy(node)
        for node in workflow["nodes"]
        if node["id"] not in {8, 9, 13, 16, 17, 18}
    ]
    conditioning = next(node for node in workflow["nodes"] if node["id"] == 7)
    conditioning["title"] = "FL2VA Conditioning · 首帧与尾帧"
    conditioning["widgets_values"][-1] = False
    workflow["nodes"].extend(
        [
            pdd_node(
                8,
                "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors",
                "FL2VA",
                [880, 0],
            ),
            note(
                16,
                "① PDD 正确连接（不要加普通 LoRA）",
                "## 正确连接\n\n`完整 FL2VA 基模 → PDD 8-Step Setup → BasicGuider`。同时把 Conditioning 的 `av_latent` 接入 PDD 节点；PDD 会统一输出 MODEL、SAMPLER、SIGMAS。不要再接普通 Load LoRA、Turbo LoRA、SLA LoRA 或另一套 PDD。普通 LoRA 节点会丢失 PDD 的动态视频/音频输出头。",
                [0, 1150],
            ),
            note(
                17,
                "② 官方固定参数与模型匹配",
                "## 固定 8 NFE 契约\n\n此节点固定使用 `Euler + simple + 8 NFE + video shift 12 + audio shift 3 + CFG 1`；工作流中无需再放 Dual-Clock 节点。LoRA strength 官方值为 `1.0`。必须搭配 `minimax_h3_fl2va_int8_convrot.safetensors` 这类完整、非 pruned FL2VA 基模；Ref2VA PDD 与 FL2VA PDD 不能互换。原生 MODEL 不保留文件名，节点能验证 PDD 文件变体，但基模选择仍由工作流和用户负责。",
                [970, 1150],
            ),
            note(
                18,
                "③ 用途、显存与验证边界",
                "## 用途与边界\n\nPDD 是 32 个训练时间间隔按每 4 个融合为一次前向，因此实际为 8 NFE；它不是简单的 8 步普通 LoRA。动态主干采用 bypass residual，不把 BF16 LoRA 合并再量化进 INT8 基模。当前已通过两变体静态、调度和 CPU/meta 装配验证；真实画质、速度与峰值显存仍以本机最终渲染为准。首次建议先用 736×416、124 帧串行测试；不要同时排队多个大任务。",
                [1940, 1150],
            ),
        ]
    )
    next(node for node in workflow["nodes"] if node["id"] == 15)[
        "widgets_values"
    ][2] = "MiniMaxH3_PDD/fl2va_8step_736x416_124f"
    relink(
        workflow,
        [
            (3, 0, 7, 0, "CLIP"),
            (1, 0, 7, 1, "VAE"),
            (2, 0, 7, 2, "VAE"),
            (5, 0, 7, 5, "IMAGE"),
            (6, 0, 7, 6, "IMAGE"),
            (4, 0, 8, 0, "MODEL"),
            (7, 1, 8, 1, "LATENT"),
            (8, 0, 11, 0, "MODEL"),
            (7, 0, 11, 1, "CONDITIONING"),
            (10, 0, 12, 0, "NOISE"),
            (11, 0, 12, 1, "GUIDER"),
            (8, 1, 12, 2, "SAMPLER"),
            (8, 2, 12, 3, "SIGMAS"),
            (7, 1, 12, 4, "LATENT"),
            (12, 0, 14, 0, "LATENT"),
            (1, 0, 14, 1, "VAE"),
            (2, 0, 14, 2, "VAE"),
            (14, 0, 15, 0, "IMAGE"),
            (14, 1, 15, 1, "AUDIO"),
        ],
    )
    workflow["extra"]["workflow_title"] = "MiniMax H3 PDD FL2VA 8-Step Advanced EXP"
    workflow["extra"]["workflow_name"] = workflow["extra"]["workflow_title"]
    return workflow


def build_ref2va() -> dict:
    workflow = read(REF_SOURCE)
    workflow["nodes"] = [
        copy.deepcopy(node)
        for node in workflow["nodes"]
        if node["id"] not in {7, 8}
    ]
    image = next(node for node in workflow["nodes"] if node["id"] == 5)
    image["widgets_values"] = ["codex_prompt_relay_fl2va_first.png", "image"]
    conditioning = next(node for node in workflow["nodes"] if node["id"] == 6)
    conditioning["title"] = "Ref2VA Conditioning · 参考图"
    conditioning["widgets_values"].append(False)
    workflow["nodes"].extend(
        [
            pdd_node(
                8,
                "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors",
                "Ref2VA",
                [1320, 0],
            ),
            note(
                14,
                "① Ref2VA PDD 的输入方式",
                "## 参考生视频连接\n\n参考图接入 Conditioning 的 `ref_image_0`，提示词使用 `<Picture 1>`；完整 Ref2VA 基模和 Conditioning 输出的 `av_latent` 一起接入 PDD 节点。PDD 的 MODEL、SAMPLER、SIGMAS 分别直连 BasicGuider / SamplerCustomAdvanced。不要再串普通 LoRA、Turbo/SLA LoRA、Dual-Clock 或第二个 PDD 节点。",
                [0, 1050],
            ),
            note(
                15,
                "② 固定参数与不可互换项",
                "## 官方 PDD 参数\n\n固定 `8 NFE / Euler / simple / 12V / 3A / CFG 1`，strength 使用 `1.0`。必须搭配完整、非 pruned 的 `minimax_h3_ref2va_int8_convrot.safetensors`；FL2VA PDD 不能用于 Ref2VA 基模。节点会检查 PDD metadata、四个动态输出头、258 个主干 adapter 和 AdaLN 宽度 2688，但无法从 MODEL 对象反推出用户加载的具体文件名。",
                [970, 1050],
            ),
            note(
                16,
                "③ 创作建议与验证状态",
                "## 使用建议\n\n先用单张清晰、主体明确的参考图验证，提示词明确人物、动作、镜头和声音。PDD 的 8 步是蒸馏路径，不等于把普通 20 步工作流直接改成 8。当前节点已通过结构、schedule、两变体动态装配验证；真实渲染仍需观察参考遵循、音频稳定、速度和显存峰值。16GB 机器请串行执行，不要并发排队。",
                [1940, 1050],
            ),
        ]
    )
    save = next(node for node in workflow["nodes"] if node["id"] == 13)
    values = save["widgets_values"]
    if isinstance(values, dict):
        values["filename_prefix"] = "MiniMaxH3_PDD/ref2va_8step_736x416_124f"
    relink(
        workflow,
        [
            (2, 0, 6, 0, "CLIP"),
            (3, 0, 6, 1, "VAE"),
            (4, 0, 6, 2, "VAE"),
            (5, 0, 6, 7, "IMAGE"),
            (1, 0, 8, 0, "MODEL"),
            (6, 1, 8, 1, "LATENT"),
            (8, 0, 10, 0, "MODEL"),
            (6, 0, 10, 1, "CONDITIONING"),
            (9, 0, 11, 0, "NOISE"),
            (10, 0, 11, 1, "GUIDER"),
            (8, 1, 11, 2, "SAMPLER"),
            (8, 2, 11, 3, "SIGMAS"),
            (6, 1, 11, 4, "LATENT"),
            (11, 0, 12, 0, "LATENT"),
            (3, 0, 12, 1, "VAE"),
            (4, 0, 12, 2, "VAE"),
            (12, 0, 13, 0, "IMAGE"),
            (12, 1, 13, 1, "AUDIO"),
        ],
    )
    workflow.setdefault("extra", {})["workflow_title"] = (
        "MiniMax H3 PDD Ref2VA 8-Step Advanced EXP"
    )
    workflow["extra"]["workflow_name"] = workflow["extra"]["workflow_title"]
    return workflow


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "2026-08-27_H3_PDD_FL2VA_8Step_Advanced_EXP.json": build_fl2va(),
        "2026-08-27_H3_PDD_Ref2VA_8Step_Advanced_EXP.json": build_ref2va(),
        "2026-08-27_H3_PDD_FL2VA_Learned_Latent_TwoPass_4Plus4_Advanced_EXP.json": build_two_pass("FL2VA"),
        "2026-08-27_H3_PDD_Ref2VA_Learned_Latent_TwoPass_4Plus4_Stable.json": build_two_pass("Ref2VA"),
    }
    for name, workflow in outputs.items():
        path = OUTPUT / name
        path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
