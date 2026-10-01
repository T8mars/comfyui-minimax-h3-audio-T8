"""Build S17 multimodal SPEED 2-stage split research workflows."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

from tools import build_formal_speed_split_workflows as base
from tools import build_h3_speed_multimodal_validation as old_media
from tools import build_h3_speed_reference_validation as old_refs
from tools.audit_modular_sampling_compat import write_new
from tools.modular_frontend_layout import spread_frontend_columns


DESTINATION = base.speed.ROOT / "examples/workflows/51-speed-multimodal-split"
CASES = {item["name"]: item for item in (*old_media.CASES, *old_refs.CASES)}
MODES = tuple(CASES)
EFFECTS = base.EFFECTS[2]
VARIANTS = base.VARIANTS[2]
FILES = {
    (mode, kind, scope, variant):
        f"SPEED_S17_MM_{mode}_{kind}_{scope}_{variant}_EXP.json"
    for mode in MODES for kind, scope in EFFECTS for variant in VARIANTS
}
SOURCE_VIDEO = "replace_with_exact_24fps_source.mp4"
REFERENCE_VIDEO = "replace_with_2_to_15s_reference_video.mp4"
REFERENCE_IMAGE = "replace_with_reference_image.png"
RELAY_EVENTS = (
    "Maintain one coherent shot and the same subjects.\n"
    "Continue the motion naturally without a scene change.\n"
    "Finish the same shot with stable detail and synchronized ambience."
)


def _media_nodes(mode: str) -> tuple[dict, dict]:
    inputs = {}
    nodes = {}
    if mode in ("i2va_lock_source", "fl2va_remix_source", "l2va_native"):
        nodes["90"] = base.speed.node("LoadVideo", file=SOURCE_VIDEO)
        nodes["91"] = base.speed.node("Video Slice", video=["90", 0], start_time=0.0,
                                      duration=124 / 24, strict_duration=True)
        nodes["92"] = base.speed.node("GetVideoComponents", video=["91", 0])
        if mode != "l2va_native":
            nodes["93"] = base.speed.node("ImageFromBatch", image=["92", 0], batch_index=0, length=1)
            inputs["first_frame"] = ["93", 0]
        if mode != "i2va_lock_source":
            nodes["94"] = base.speed.node("ImageFromBatch", image=["92", 0], batch_index=-1, length=1)
            inputs["last_frame"] = ["94", 0]
        if mode != "l2va_native":
            inputs["drive_audio"] = ["92", 1]
    elif mode == "ref_image_native":
        nodes["95"] = base.speed.node("LoadImage", image=REFERENCE_IMAGE)
        inputs["ref_images.ref_image_0"] = ["95", 0]
    elif mode in ("hybrid_first_image_audio", "ref_video_audio_native"):
        nodes["90"] = base.speed.node("LoadVideo", file=REFERENCE_VIDEO)
        nodes["91"] = base.speed.node("Video Slice", video=["90", 0], start_time=0.0,
                                      duration=2.0, strict_duration=True)
        nodes["92"] = base.speed.node("GetVideoComponents", video=["91", 0])
        if mode == "hybrid_first_image_audio":
            nodes["93"] = base.speed.node("ImageFromBatch", image=["92", 0], batch_index=0, length=1)
            nodes["95"] = base.speed.node("LoadImage", image=REFERENCE_IMAGE)
            inputs.update(first_frame=["93", 0],
                          **{"ref_images.ref_image_0": ["95", 0],
                             "ref_audios.ref_audio_0": ["92", 1]})
        else:
            inputs["ref_videos.ref_video_0"] = ["92", 0]
            inputs["ref_video_audios.ref_video_audio_0"] = ["92", 1]
    else:
        raise ValueError("Unreviewed SPEED multimodal source")
    return nodes, inputs


def graph_for(mode: str, kind: str, scope: str, variant: str) -> dict:
    if mode not in CASES or (kind, scope) not in EFFECTS or variant not in VARIANTS:
        raise ValueError("Expected a reviewed S17 multimodal route/effect/storage variant")
    case = CASES[mode]
    graph = base.graph_for(2, kind, scope, variant)
    media_nodes, media_inputs = _media_nodes(mode)
    graph.update(media_nodes)
    for index in range(2):
        key = str(11 + 12 * index)
        if key not in graph:
            continue
        source = graph[key]["inputs"]
        source.update(task_type=case["task_type"],
                      audio_mode=case.get("audio_mode", "native"),
                      audio_denoise_strength=case.get("audio_denoise_strength", 1.0),
                      **media_inputs)
        source["prompt"] = case["prompt"]
        setup = graph[str(12 + 12 * index)]["inputs"]
        setup.update(execution_scope="multimodal_research_exp", reuse_t2va_text=False)
        relay_key = str(16 + 12 * index)
        if relay_key in graph:
            settings = graph[relay_key]["inputs"]
            settings.update(global_prompt=case["prompt"], local_prompts=RELAY_EVENTS)
            package = ("h3_audio_t8_pkg" if "h3_audio_t8_pkg" in sys.modules
                       else "_t8_modular_compat_capture")
            _, source["prompt"], *_ = import_module(
                package + ".prompt_relay_advanced").build_prompt_relay_plan(**settings)
    audio_link = ["92", 1] if mode == "i2va_lock_source" else ["80", 1]
    graph["83"] = base.speed.node("MiniMaxH3OutputTrimT8", frames=["80", 0], audio=audio_link,
                                   start_seconds=0.0, duration_seconds=124 / 24, fps=24.0)
    graph["81"]["inputs"].update(images=["83", 0], audio=["83", 1])
    graph["82"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_SPEED_{mode}_EXP"
    return graph


def load_live_info() -> dict:
    info = base.speed.load_live_info()
    import nodes
    from comfy_extras.nodes_images import ImageFromBatch
    from comfy_extras.nodes_video import GetVideoComponents, LoadVideo, VideoSlice
    from tools.build_modular_fast_h3_v2_workflow import native_info

    extra = {"LoadImage": nodes.NODE_CLASS_MAPPINGS["LoadImage"],
             "LoadVideo": LoadVideo, "Video Slice": VideoSlice,
             "GetVideoComponents": GetVideoComponents, "ImageFromBatch": ImageFromBatch}
    nodes.NODE_CLASS_MAPPINGS.update(extra)
    info.update({name: native_info(name, cls) for name, cls in extra.items()})
    return info


def _note(mode: str, kind: str, scope: str, variant: str) -> str:
    return (
        f"S17 SPEED {mode}／{kind}:{scope}／{variant} 两阶段分离研究图。沿用旧 SPEED "
        "多模态示例的原始输入模式：视频素材先按目标124帧或参考2秒显式切片，"
        "再取首帧／末帧／原音频／参考；每个活动阶段用同一真实素材在该阶段画布重新编码。"
        "I2VA lock_source 交付用切片源音频，其余用生成音频，末端精确裁到124帧。"
        "两阶段 MODEL/可插 LoRA、Source、条件、Sampler、噪声与 DCT 转段独立；"
        "Relay Plan 和 EAV Config 按文件名阶段独立外置，EAV 默认 report_only。"
        "冷图仅凭真实阶段 manifest 路径及 SHA 选择冻结前段，前段不重采；"
        "效果标签记录完整图来源，不代表冷图会重新执行冻结效果。"
        "请替换素材占位并核对精确24fps、原音频与参考2至15秒范围；"
        "无效文件和空回执不可直接排队。手工 sigma 已被旧用户盲评否决，"
        "仅机械示范，不作为质量／速度推荐或旧图逐帧替代。原尺寸预训练GPU、"
        "实际 Relay/EAV 画质、其它后端／LoRA、显存和人工验收均未由本图证明。")


def build_suite(info: dict) -> dict:
    result = {}
    for key in FILES:
        mode, kind, scope, variant = key
        graph, selected = base.speed.selected_frontend_schema(graph_for(*key), info)
        title = f"S17 SPEED {mode} {kind}:{scope} {variant} EXP"
        workflow = base.speed.convert(graph, selected, title)
        by_id = {item["id"]: item for item in workflow["nodes"]}
        for source_key, node_id in zip(graph, by_id):
            if graph[source_key]["class_type"] == "UNETLoader":
                by_id[node_id]["title"] = f"Stage MODEL {source_key} — independent LoRA input"
        note_id = workflow["last_node_id"] + 1
        workflow["nodes"].append({
            "id": note_id, "type": "MarkdownNote", "title": "先读 / S17 multimodal SPEED EXP",
            "pos": [0, -800], "size": [1180, 580], "flags": {}, "order": len(graph),
            "mode": 0, "inputs": [], "outputs": [], "properties": {},
            "widgets_values": [_note(*key)],
        })
        workflow["last_node_id"] = note_id
        workflow.setdefault("extra", {})["t8_split_example"] = {
            "schema": "t8.speed.multimodal-split-example.v1", "route": "S17",
            "input_mode": mode, "stages": 2, "effect_kind": kind,
            "effect_scope": scope, "variant": variant,
            "status": "experimental_importable_not_quality_accepted",
        }
        workflow["extra"]["workflow_title"] = title
        spread_frontend_columns(workflow)
        audit = base.speed.audit_candidate(graph, workflow, selected)
        result[key] = {"graph": graph, "workflow": workflow, "audit": audit}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    options = parser.parse_args()
    destination = options.output_dir.resolve()
    if not destination.is_relative_to(base.speed.ROOT.resolve()):
        parser.error("Output must remain inside this project")
    suite = build_suite(load_live_info())
    for key, filename in FILES.items():
        write_new(destination / filename, suite[key]["workflow"])
        print(f"{filename}: {suite[key]['audit']['nodes']} nodes, "
              f"{suite[key]['audit']['edges']} edges")


if __name__ == "__main__":
    main()
