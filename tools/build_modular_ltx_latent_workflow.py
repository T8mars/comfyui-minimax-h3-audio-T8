"""Build a private learned H3 AV latent -> externally sampled LTX candidate."""

import argparse
import json
from pathlib import Path
import uuid

from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/development/modular-sampling-m4-ltx-latent-20260924/candidate-v1"
CERTIFIED_OUTPUT = ROOT / "artifacts/development/modular-sampling-m4-ltx-latent-20260924/candidate-v3"
STEM = "H3_Learned_Latent_to_LTX_Separate_Refiner_EXP"


def build_prompt(checkpoint_path="", expected_file_sha256="", source_directory="",
                 model_directory="", certified_sample=False):
    graph = {
        "1": {"class_type": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced", "inputs": {
            "checkpoint_path": checkpoint_path, "expected_manifest_json": "",
            "expected_file_sha256": expected_file_sha256, "hash_chunk_megabytes": 8}},
        "2": {"class_type": "MiniMaxH3LTXLatentAdapterEXPT8", "inputs": {
            "h3_latent": ["1", 0], "source_frames": 73, "source_fps": 24.0,
            "frame_policy": "exact", "source_directory": source_directory,
            "model_directory": model_directory, "device": "cpu", "precision": "float32",
            "reference_prefix_latents": 0, "normalization": "comfy_normalized"}},
        "3": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
            "weight_dtype": "default"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["3", 0], "lora_name": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
            "strength_model": 0.8}},
        "5": {"class_type": "MiniMaxH3SolEngineLTXRefinerSetupT8Advanced", "inputs": {
            "model": ["4", 0], "enabled": True, "attention_backend": "auto_sol_attn",
            "min_tokens": 4096, "kernel_precision": "bf16_official", "verbose": False}},
        "6": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "type": "ltxv", "device": "default"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["6", 0],
            "text": "Replace with the exact prompt matching the H3 checkpoint and adapter input."}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["6", 0], "text": ""}},
        "9": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["7", 0], "negative": ["8", 0], "frame_rate": 24.0}},
        "10": {"class_type": "CFGGuider", "inputs": {
            "model": ["5", 0], "positive": ["9", 0], "negative": ["9", 1], "cfg": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "12": {"class_type": "MiniMaxH3LTXLearnedStageBindEXPT8", "inputs": {
            "h3_latent": ["1", 0], "original_h3_av": ["2", 1],
            "ltx_video_latent": ["2", 0], "adapter_report_json": ["2", 4],
            "model": ["5", 0], "noise": ["11", 0], "guider": ["10", 0],
            "sampler": ["5", 1], "sigmas": ["5", 2], "setup_report_json": ["5", 4]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["12", 1], "sampler": ["12", 2],
            "sigmas": ["12", 3], "latent_image": ["12", 4]}},
        "14": {"class_type": "MiniMaxH3LTXLearnedStageAuditEXPT8", "inputs": {
            "stage_boundary": ["12", 5], "h3_latent": ["1", 0],
            "original_h3_av": ["2", 1], "ltx_video_latent": ["2", 0],
            "adapter_report_json": ["2", 4], "model": ["5", 0],
            "noise": ["12", 0], "guider": ["12", 1], "sampler": ["12", 2],
            "sigmas": ["12", 3], "setup_report_json": ["5", 4],
            "candidate_latent": ["13", 0]}},
        "15": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "16": {"class_type": "MiniMaxH3LTXOriginalAudioDecodeEXPT8", "inputs": {
            "original_h3_av": ["14", 1], "adapter_report_json": ["2", 4],
            "audio_vae": ["15", 0]}},
        "17": {"class_type": "MiniMaxH3SolEngineTAEHVLoaderT8Advanced", "inputs": {
            "model_name": "taeltx2_3_wide.pth"}},
        "18": {"class_type": "MiniMaxH3SolEngineTAEHVDecodeT8Advanced", "inputs": {
            "latent": ["14", 0], "taehv": ["17", 0],
            "execution_mode": "auto_official", "precision": "bf16_official"}},
        "19": {"class_type": "MiniMaxH3OutputTrimT8", "inputs": {
            "frames": ["18", 0], "audio": ["16", 0], "start_seconds": 0.0,
            "duration_seconds": ["14", 2], "fps": ["2", 3]}},
        "20": {"class_type": "CreateVideo", "inputs": {
            "images": ["19", 0], "audio": ["19", 1], "fps": ["2", 3],
            "color_space": "sRGB", "codec": "none", "bit_depth": 8}},
        "21": {"class_type": "SaveVideo", "inputs": {
            "video": ["20", 0], "filename_prefix": "MiniMaxH3/learned_latent_ltx_candidate",
            "format": "auto", "codec": "auto"}},
    }
    if certified_sample:
        graph["13"]["class_type"] = "MiniMaxH3LTXLearnedStageSampleEXPT8"
        graph["13"]["inputs"]["stage_boundary"] = ["12", 5]
        graph["14"]["class_type"] = "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8"
        graph["14"]["inputs"]["sample_proof"] = ["13", 2]
    return graph


def build_workflow(info, certified_sample=False, **paths):
    graph = build_prompt(certified_sample=certified_sample, **paths)
    workflow = convert(graph, info, "H3 learned LATENT to separate LTX refiner / EXP")
    normalize_native_widget_inputs(workflow)
    for index, node in enumerate(workflow["nodes"]):
        node["pos"] = [(index % 7) * 670, (index // 7) * 820]
        node["size"] = [620, 620]
    note = (
        "## Learned H3→LTX 分离精修候选\n\n"
        "先用原生 H3 AV Checkpoint Save 保存含音频的联合 AV，再填本图 Load 的相对路径和 SHA。"
        "普通 LoadLatent 视频模板不能被当成带音轨的联合 AV。此图仅接受 73 帧/24fps/exact，"
        "124 帧的 pad/crop 需另做明确音频时长对齐。"
        "填写已核验 Sana adapter 源码目录与官方模型目录；第一个 LATENT 由 learned adapter 直接输出，"
        "没有 RGB 绕路、全 LTX VAE encode 或第二个 x2 放大器。H3 原音只经过 H3 Audio VAE，"
        "不转为 LTX 音频。LTX 的模型、LoRA、提示词、噪声、采样器和 sigma 仍在外部节点上。\n\n"
        "本图需替换所有本地素材/权重占位再运行；CPU Core schema 通过不等于采样/GPU音画或人审通过。"
        + ("v3 的独立 Stage Sample 实调 Core 采样并输出仅当前进程有效的三步回执；"
           "Stage Audit 必须接此回执才报告采样执行。旧 v1 图不接时仍只审来源/形状，"
           "两者均不授权跨进程缓存、预训练音画或人审。"
           if certified_sample else
           "Stage Audit 只核对来源/形状/连接，不证明采样器确实执行，也不授权跨进程缓存。")
    )
    nid = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": nid, "type": "MarkdownNote", "title": "必读：联合AV与分离边界",
        "pos": [0, 2460], "size": [1250, 600], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = nid
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL,
        "t8:modular-ltx-learned:20260924:joint-av-exact73"
        + (":certified-sample-v3" if certified_sample else "")))
    workflow["extra"]["route"] = "S27-learned-latent"
    workflow["extra"]["qualification"] = (
        "candidate_sampler_proof_unexecuted" if certified_sample else "candidate_static_only")
    return workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--certified-sample", action="store_true")
    options = parser.parse_args()
    output = (options.output or (CERTIFIED_OUTPUT if options.certified_sample
                                 else DEFAULT_OUTPUT)).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Choose a new private artifacts candidate directory")
    import asyncio
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT.parents[1]))
    package_name = "_t8_ltx_learned_builder"
    spec = importlib.util.spec_from_file_location(package_name, ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    own = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}
    import nodes as core_nodes
    for name in ("nodes_custom_sampler.py", "nodes_video.py", "nodes_lt.py"):
        if not asyncio.run(core_nodes.load_custom_node(
                str(ROOT.parents[1] / "comfy_extras" / name), module_parent="comfy_extras")):
            raise RuntimeError("Could not load required Core node module: " + name)
    info = {}
    for item in build_prompt(certified_sample=options.certified_sample).values():
        name = item["class_type"]
        cls = next((cls for cls in classes if cls.define_schema().node_id == name), None)
        if cls is None:
            cls = core_nodes.NODE_CLASS_MAPPINGS[name]
        value = (own[name] if name in own else cls.GET_NODE_INFO_V1()
                 if hasattr(cls, "GET_NODE_INFO_V1") else {
                     "input": cls.INPUT_TYPES(), "output": cls.RETURN_TYPES,
                     "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES),
                     "display_name": name, "python_module": cls.__module__,
                 })
        info[name] = value
        if isinstance(value, dict):
            value["cnr_id"] = "minimax-h3-audio-T8" if name in own else "comfy-core"
    output.mkdir(parents=True)
    frontend = build_workflow(info, certified_sample=options.certified_sample)
    api = build_prompt(certified_sample=options.certified_sample)
    (output / (STEM + ".json")).write_text(json.dumps(frontend, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (output / (STEM + ".api.json")).write_text(json.dumps(api, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(str(output))


if __name__ == "__main__":
    raise SystemExit(main())
