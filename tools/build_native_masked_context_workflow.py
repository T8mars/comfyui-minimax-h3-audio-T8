from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-08-09_H3_Long_Video_22F_EXP.json"
)
OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Advanced_EXP.json"
)
STARTER_OUTPUT = (
    ROOT
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Segment0_Starter_Advanced_EXP.json"
)
NODE_TYPE = "MiniMaxH3NativeMaskedVideoContextT8Advanced"
COLOR_MATCH_NODE_TYPE = "MiniMaxH3LongVideoColorMatchT8Advanced"
SOURCE_LEGACY_EMA_LORA = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
NEW_EMA_B_LORA = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"


SETUP_NOTE = """# Native Masked Video Context · Plan B（独立 EXP）

这是一份单独工作流，不替换现有 Long Video 默认路线，也不修改旧节点。

1. 第 0 段必须先用配套的 `Plan_B_Segment0_Starter` 工作流生成并保存上下文；不要使用旧双时钟默认工作流。本节点只接受第 1 段及以后。
2. Planner、Context Load、Long Video Conditioning 必须属于同一 chain / segment。
3. Conditioning 已固定为 `context_audio=video_only`；节点直接复制上一段原生 video latent 尾部，并只锁住当前段开头的画面区域。
4. Output Trim 仍按 Planner 精确移除 5 / 22 / 39 帧重建头，随后 Color Match 默认开启：读取上一段实际输出尾帧，用全局 Lab 色彩/对比度匹配和 8x5 局部分区补偿处理续段开头，并在 24 帧内渐隐；可在节点中关闭。
"""


STARTER_NOTE = """# Native Masked Video Context · 第 0 段启动器

这是独立 Plan B 的配套起始工作流，不替换现有 Long Video 默认路线。

1. 先在这里设置唯一 `chain_id`，保持 `segment_index=0`、`is_final_segment=false`并运行第 0 段。
2. 本工作流和续段 Plan B 都固定 ComfyUI 原生 AV `euler + native_flow`，避免当前核心下旧 `dual_clock_euler` 的异常音频。
3. 第 0 段完成并保存 context 后，打开配套 Plan B 工作流，填写完全相同的 `chain_id`，从 `segment_index=1`继续。
4. 本工作流按用户指定固定新版 step600 `EMA_B` LoRA；旧通用EMA在人审中声音非常轻且不正常，不再用于本Plan B。
5. 不要把其他旧路线保存的 context 混入这条链；模型、LoRA、画布、帧数、shift或采样合同变化时应使用新 `chain_id`。
6. 原生音乐与对白的正常听感仍需人工试听；机械解码和ASR不能代替听感。
7. Color Match 默认开启；第0段不改画面，只保存实际输出尾帧的 Lab 统计和 8x5 局部颜色参考。关闭时画面逐像素旁路，但仍记录尾帧，方便后续重新开启。
"""


BOUNDARY_NOTE = """# 声音所有权与验收边界

- 上一段音频不会注入当前段；当前段 audio tensor 和已有 Vocal Lock audio mask 原样旁路。
- 本工作流固定使用 ComfyUI 原生 AV `euler + native_flow`。当前核心下旧 `dual_clock_euler` 的同 Seed 对照产生异常 DC 偏置和噪音，不能用于本路线的原生音频。
- LoRA固定为用户指定的`minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`；旧通用EMA的人审声音非常轻且不正常，禁止回退。
- 不做视频 VAE 解码再编码，不安装 ComfyUI 运行时 monkey patch。
- Color Match 位于 Output Trim 之后，只改显示域 RGB；默认开启、可关闭，不改audio或原生AV latent。每段只参考上一段实际输出的最后5帧，先做 ComfyUI 内置 ColorTransfer 同类的 pooled Reinhard Lab 色彩/对比度匹配，再做 8x5 局部分区 RGB 补偿；总像素通道改变量上限0.02并在24帧内渐隐，疑似切镜会放弃校正。
- 如果目标画面前缀已有其他锁定 mask、报告不匹配或画布不一致，会直接报错，不猜测覆盖。
- 首轮与第二轮旧双时钟样例的人审结论为画面平局且均无明显问题，但两路都有杂音，音频未通过。原生 AV Euler 修复路线仍需完整续段试听，所以继续标记 Advanced EXP，不宣称优于旧路线、音频非劣或通用 16GB 安全。
"""


def _by_type(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {node_type}, found {len(matches)}")
    return matches[0]


def _remove_link(workflow: dict, link_id: int) -> None:
    links = [link for link in workflow["links"] if link[0] == link_id]
    if len(links) != 1:
        raise ValueError(f"Expected exactly one link {link_id}, found {len(links)}")
    link = links[0]
    workflow["links"].remove(link)
    source = next(node for node in workflow["nodes"] if node["id"] == link[1])
    target = next(node for node in workflow["nodes"] if node["id"] == link[3])
    source_links = source["outputs"][link[2]].get("links") or []
    source["outputs"][link[2]]["links"] = [item for item in source_links if item != link_id]
    if not source["outputs"][link[2]]["links"]:
        source["outputs"][link[2]]["links"] = None
    target["inputs"][link[4]]["link"] = None


def _add_link(
    workflow: dict,
    link_id: int,
    source_id: int,
    output_slot: int,
    target_id: int,
    input_slot: int,
    link_type: str,
) -> None:
    source = next(node for node in workflow["nodes"] if node["id"] == source_id)
    target = next(node for node in workflow["nodes"] if node["id"] == target_id)
    if target["inputs"][input_slot].get("link") is not None:
        raise ValueError(f"Target {target_id}:{input_slot} is already connected")
    output_links = source["outputs"][output_slot].get("links") or []
    source["outputs"][output_slot]["links"] = [*output_links, link_id]
    target["inputs"][input_slot]["link"] = link_id
    workflow["links"].append(
        [link_id, source_id, output_slot, target_id, input_slot, link_type]
    )


def _note(node_id: int, title: str, pos: list[int], text: str, order: int) -> dict:
    return {
        "id": node_id,
        "type": "MarkdownNote",
        "pos": pos,
        "size": [1080, 300],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {"Node name for S&R": "MarkdownNote"},
        "widgets_values": [text],
        "color": "#432",
        "bgcolor": "#653",
    }


def _pin_native_av_euler(dual_clock: dict, context_save: dict) -> None:
    if dual_clock["widgets_values"][3:] != ["dual_clock_euler", "native_flow"]:
        raise ValueError("Source Long Video sampler contract changed")
    dual_clock["widgets_values"][3] = "euler"
    context_save["widgets_values"][1] = (
        "4-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
    )


def _pin_new_ema_b_lora(workflow: dict) -> None:
    lora = _by_type(workflow, "LoraLoaderBypassModelOnly")
    if lora["widgets_values"] != [SOURCE_LEGACY_EMA_LORA, 1.0]:
        raise ValueError("Source Long Video LoRA contract changed")
    lora["widgets_values"][0] = NEW_EMA_B_LORA
    lora["title"] = "User-selected step600 EMA_B LoRA (old generic EMA rejected)"


def _color_match_node(node_id: int, pos: list[int], order: int) -> dict:
    return {
        "id": node_id,
        "type": COLOR_MATCH_NODE_TYPE,
        "pos": pos,
        "size": [500, 330],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {"name": "frames", "type": "IMAGE", "link": None},
            {"name": "context", "type": "H3_T8_CONTEXT", "link": None},
            {"name": "chain_id", "type": "STRING", "link": None},
            {"name": "segment_index", "type": "INT", "link": None},
        ],
        "outputs": [
            {"name": "frames", "type": "IMAGE", "links": None},
            {"name": "status", "type": "STRING", "links": None},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "title": "Default-on Lab + local seam Color Match (optional)",
        "properties": {"Node name for S&R": COLOR_MATCH_NODE_TYPE},
        "widgets_values": [True, 5, 24, 1.0, 0.0005, 0.02, 0.18],
        "color": "#243",
        "bgcolor": "#365",
    }


def _insert_color_match(
    workflow: dict,
    *,
    node_id: int,
    order: int,
) -> None:
    planner = _by_type(workflow, "MiniMaxH3LongVideoPlannerT8")
    context_load = _by_type(workflow, "MiniMaxH3LongVideoContextLoadT8")
    output_trim = _by_type(workflow, "MiniMaxH3OutputTrimT8")
    create_video = _by_type(workflow, "CreateVideo")
    image_link = create_video["inputs"][0]["link"]
    if image_link is None:
        raise ValueError("CreateVideo images input is not connected")
    link = next(item for item in workflow["links"] if item[0] == image_link)
    if (link[1], link[2]) != (output_trim["id"], 0):
        raise ValueError("Expected Output Trim frames to feed CreateVideo directly")
    _remove_link(workflow, image_link)

    for node in workflow["nodes"]:
        if node["id"] in {create_video["id"], _by_type(workflow, "SaveVideo")["id"]}:
            node["pos"][0] += 520
    color_node = _color_match_node(
        node_id,
        [output_trim["pos"][0] + 470, output_trim["pos"][1]],
        order,
    )
    workflow["nodes"].append(color_node)
    next_link = max(item[0] for item in workflow["links"]) + 1
    additions = [
        (output_trim["id"], 0, color_node["id"], 0, "IMAGE"),
        (context_load["id"], 0, color_node["id"], 1, "H3_T8_CONTEXT"),
        (planner["id"], 0, color_node["id"], 2, "STRING"),
        (planner["id"], 1, color_node["id"], 3, "INT"),
        (color_node["id"], 0, create_video["id"], 0, "IMAGE"),
    ]
    for offset, (source, output, target, input_slot, link_type) in enumerate(additions):
        _add_link(
            workflow,
            next_link + offset,
            source,
            output,
            target,
            input_slot,
            link_type,
        )


def build_starter(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    if workflow.get("version") != 0.4:
        raise ValueError("Source must be a ComfyUI frontend workflow version 0.4")

    planner = _by_type(workflow, "MiniMaxH3LongVideoPlannerT8")
    conditioning = _by_type(workflow, "MiniMaxH3LongVideoConditioningT8")
    dual_clock = _by_type(workflow, "MiniMaxH3DualClockSamplerT8")
    context_save = _by_type(workflow, "MiniMaxH3LongVideoContextSaveT8")
    save_video = _by_type(workflow, "SaveVideo")
    _pin_new_ema_b_lora(workflow)

    if planner["widgets_values"][1] != 0 or planner["widgets_values"][6] is not False:
        raise ValueError("Source Long Video starter Planner contract changed")
    planner["title"] = "1. Plan B starter: segment_index=0, is_final_segment=false"
    if conditioning["widgets_values"][0] != "video_and_audio":
        raise ValueError("Source Long Video Conditioning audio mode changed")
    conditioning["widgets_values"][0] = "video_only"
    conditioning["title"] = "3. Plan B starter conditioning (no previous audio context)"
    _pin_native_av_euler(dual_clock, context_save)
    dual_clock["title"] = "4-step current-core native AV Euler"
    context_save["title"] = "4. Save Plan B segment-0 context for continuation"
    save_video["title"] = "7. Save Plan B segment 0"
    save_video["widgets_values"][0] = "MiniMaxH3/long_video_masked_plan_b_segment0"

    note_id = max(node["id"] for node in workflow["nodes"]) + 1
    workflow["nodes"].append(
        _note(note_id, "Plan B 第0段使用顺序", [360, 580], STARTER_NOTE, note_id - 1)
    )
    _insert_color_match(
        workflow,
        node_id=note_id + 1,
        order=note_id,
    )
    workflow["id"] = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "t8-h3-native-masked-video-context-plan-b-segment0-native-av-euler",
        )
    )
    workflow["revision"] = 0
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow["links"].sort(key=lambda link: link[0])
    return workflow


def build(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    if workflow.get("version") != 0.4:
        raise ValueError("Source must be a ComfyUI frontend workflow version 0.4")

    planner = _by_type(workflow, "MiniMaxH3LongVideoPlannerT8")
    context_load = _by_type(workflow, "MiniMaxH3LongVideoContextLoadT8")
    conditioning = _by_type(workflow, "MiniMaxH3LongVideoConditioningT8")
    dual_clock = _by_type(workflow, "MiniMaxH3DualClockSamplerT8")
    sampler = _by_type(workflow, "SamplerCustomAdvanced")
    context_save = _by_type(workflow, "MiniMaxH3LongVideoContextSaveT8")
    decode = _by_type(workflow, "MiniMaxH3AVDecodeT8")
    output_trim = _by_type(workflow, "MiniMaxH3OutputTrimT8")
    create_video = _by_type(workflow, "CreateVideo")
    save_video = _by_type(workflow, "SaveVideo")
    _pin_new_ema_b_lora(workflow)

    planner["widgets_values"][1] = 1
    planner["title"] = "1. Plan B continuation segment_index: 1, 2, 3..."
    if conditioning["widgets_values"][0] != "video_and_audio":
        raise ValueError("Source Long Video Conditioning audio mode changed")
    conditioning["widgets_values"][0] = "video_only"
    conditioning["title"] = "3. Long-video conditioning (video-only context for Plan B)"
    context_save["title"] = "5. Save the sampled AV tail for the next segment"
    decode["title"] = "6. Decode the current segment only"
    output_trim["title"] = "7. Remove the reconstructed AV head exactly"
    create_video["title"] = "9. Create one exact-duration synchronized segment"
    save_video["title"] = "10. Save the independent Plan B segment"
    save_video["widgets_values"][0] = "MiniMaxH3/long_video_masked_plan_b_segment"
    _pin_native_av_euler(dual_clock, context_save)

    latent_to_dual = dual_clock["inputs"][1]["link"]
    latent_to_sampler = sampler["inputs"][4]["link"]
    if latent_to_dual is None or latent_to_sampler is None:
        raise ValueError("Source conditioning latent route is incomplete")
    _remove_link(workflow, latent_to_dual)
    _remove_link(workflow, latent_to_sampler)

    for node in workflow["nodes"]:
        if node["id"] >= dual_clock["id"]:
            node["pos"][0] += 520

    masked_node = {
        "id": 18,
        "type": NODE_TYPE,
        "pos": [1770, 300],
        "size": [500, 250],
        "flags": {},
        "order": 17,
        "mode": 0,
        "inputs": [
            {"name": "av_latent", "type": "LATENT", "link": None},
            {"name": "context", "type": "H3_T8_CONTEXT", "link": None},
            {"name": "planner_report_json", "type": "STRING", "link": None},
            {"name": "conditioning_report_json", "type": "STRING", "link": None},
        ],
        "outputs": [
            {"name": "av_latent", "type": "LATENT", "links": None},
            {"name": "trim_context_frames", "type": "INT", "links": None},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "title": "4. Plan B: hard-lock previous native video latent only",
        "properties": {"Node name for S&R": NODE_TYPE},
        "widgets_values": [],
        "color": "#432",
        "bgcolor": "#653",
    }
    workflow["nodes"].extend(
        [
            masked_node,
            _note(19, "Plan B 使用顺序", [360, 580], SETUP_NOTE, 18),
            _note(20, "Plan B 边界", [1510, 650], BOUNDARY_NOTE, 19),
        ]
    )
    _insert_color_match(workflow, node_id=21, order=20)

    next_link = max(link[0] for link in workflow["links"]) + 1
    additions = [
        (conditioning["id"], 2, masked_node["id"], 0, "LATENT"),
        (context_load["id"], 0, masked_node["id"], 1, "H3_T8_CONTEXT"),
        (planner["id"], 9, masked_node["id"], 2, "STRING"),
        (conditioning["id"], 6, masked_node["id"], 3, "STRING"),
        (masked_node["id"], 0, dual_clock["id"], 1, "LATENT"),
        (masked_node["id"], 0, sampler["id"], 4, "LATENT"),
    ]
    for offset, (source, output, target, input_slot, link_type) in enumerate(additions):
        _add_link(
            workflow,
            next_link + offset,
            source,
            output,
            target,
            input_slot,
            link_type,
        )

    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "t8-h3-native-masked-video-context-plan-b")
    )
    workflow["revision"] = 0
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow["links"].sort(key=lambda link: link[0])
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--starter-output", type=Path, default=STARTER_OUTPUT)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--starter-mirror", type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    rendered = json.dumps(build(source), ensure_ascii=False, indent=2) + "\n"
    starter_rendered = json.dumps(build_starter(source), ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    args.starter_output.parent.mkdir(parents=True, exist_ok=True)
    args.starter_output.write_text(starter_rendered, encoding="utf-8")
    print(args.starter_output)
    if args.mirror is not None:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
        args.mirror.write_text(rendered, encoding="utf-8")
        print(args.mirror)
    if args.starter_mirror is not None:
        args.starter_mirror.parent.mkdir(parents=True, exist_ok=True)
        args.starter_mirror.write_text(starter_rendered, encoding="utf-8")
        print(args.starter_mirror)


if __name__ == "__main__":
    main()
