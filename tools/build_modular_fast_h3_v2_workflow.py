"""Build an editable split candidate using live Core schemas, without sampling."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_modular_sampling_compat import capture, write_new  # noqa: E402
from api_to_frontend_workflow import convert  # noqa: E402
from audit_progressive_workflows import audit_candidate  # noqa: E402
from build_fast_h3_v2_workflows import selected_frontend_schema  # noqa: E402
from run_fast_h3_v2_probe import build_graph  # noqa: E402


def split_graph(profile="dense_compat_exp"):
    graph, _ = build_graph(profile=profile, width=448, height=256, frames=73,
                           min_tokens=12288, instrument=False)
    graph["10"]["class_type"] = "MiniMaxH3FastH3V2StageSetupEXPT8"
    graph["10"]["inputs"]["stage"] = "low_0_4"
    graph["21"]["inputs"]["source"] = ["10", 4]
    # Independent input sockets: user may insert separate LoRA/patch chains here.
    graph["22"] = deepcopy(graph["1"])
    graph["23"] = {"class_type": "MiniMaxH3LearnedLatentUpscaleT8Advanced", "inputs": {
        "av_latent": ["13", 1], "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "size_mode": "scale_by", "scale_by": 2.0, "target_megapixels": .46,
        "target_width": 896, "target_height": 512, "aspect_policy": "preserve_source",
        "max_anisotropy": 1.05, "precision": "fp16", "release_policy": "offload_after"}}
    graph["24"] = deepcopy(graph["9"])
    graph["24"]["inputs"].update(width=["23", 1], height=["23", 2])
    graph["25"] = {"class_type": "MiniMaxH3TwoPassLatentReconcileT8Advanced", "inputs": {
        "learned_latent": ["23", 0], "highres_template": ["24", 1], "positive": ["24", 0],
        "audio_policy": "auto", "second_pass_audio_source": "legacy_policy", "second_pass_audio_strength": 0.0}}
    graph["26"] = {"class_type": "MiniMaxH3FastH3V2StageSetupEXPT8", "inputs": {
        "model": ["22", 0], "av_latent": ["25", 0], "stage": "high_4_8", "profile": profile,
        "min_tokens": 12288}}
    graph["27"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 2609032102}}
    graph["28"] = {"class_type": "BasicGuider", "inputs": {"model": ["26", 0], "conditioning": ["25", 1]}}
    graph["29"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["27", 0], "guider": ["28", 0], "sampler": ["26", 1],
        "sigmas": ["26", 2], "latent_image": ["25", 0]}}
    graph["30"] = {"class_type": "MiniMaxH3FastH3V2RuntimeAuditEXPT8", "inputs": {
        "model": ["26", 0], "sampled_av_latent": ["29", 0]}}
    graph["31"] = {"class_type": "PreviewAny", "inputs": {"source": ["30", 1]}}
    graph["32"] = {"class_type": "PreviewAny", "inputs": {"source": ["26", 4]}}
    graph["14"]["inputs"]["av_latent"] = ["30", 0]
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/Modular_FastH3_V2_EXP"
    return graph


def native_info(name, cls):
    if hasattr(cls, "GET_NODE_INFO_V1"):
        return cls.GET_NODE_INFO_V1()
    inputs = cls.INPUT_TYPES()
    return {"input": inputs, "input_order": {key: list(group) for key, group in inputs.items()},
            "output": cls.RETURN_TYPES, "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES),
            "output_is_list": getattr(cls, "OUTPUT_IS_LIST", [False] * len(cls.RETURN_TYPES)),
            "name": name, "display_name": name, "category": getattr(cls, "CATEGORY", ""),
            "output_node": getattr(cls, "OUTPUT_NODE", False),
            "python_module": cls.__module__}


def split_graph_with_effects():
    graph = split_graph("dense_compat_exp")
    # Explicit Relay MODEL/COND/latent outputs are bound independently at each
    # real resolution. A shared immutable Plan is not a shared runtime binding.
    for key, model_id in (("9", "1"), ("24", "22")):
        item = graph[key]
        item["class_type"] = "MiniMaxH3PromptRelayConditioningT8Advanced"
        item["inputs"].pop("prompt")
        item["inputs"].pop("length")
        item["inputs"].update(model=[model_id, 0], prompt_relay_plan=["40", 0],
                              execution_mode="apply_exp", query_chunk_rows=256)
        for target in graph.values():
            for name, value in list(target["inputs"].items()):
                if isinstance(value, list) and len(value) == 2 and value[0] == key:
                    target["inputs"][name] = [key, {0: 1, 1: 2}[value[1]]]
    graph["10"]["inputs"]["model"] = ["9", 0]
    graph["26"]["inputs"]["model"] = ["24", 0]
    graph["40"] = {"class_type": "MiniMaxH3PromptRelayPlanT8Advanced", "inputs": {
        "global_prompt": "A continuous cinematic shot of a woman in a bright concert hall, natural motion and ambience.",
        "local_prompts": "She turns toward the camera and smiles.\nShe raises her hand and waves gently.",
        "length": 73, "timing_mode": "auto_equal", "time_ranges": "", "math_profile": "paper_v1",
        "epsilon": .1, "allow_gaps": False, "allow_overlaps": False}}
    graph["47"] = deepcopy(graph["40"])
    graph["24"]["inputs"]["prompt_relay_plan"] = ["47", 0]
    for key in ("41", "42"):
        graph[key] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
            "mode": "report_only", "tau": 4., "start_video_progress": .15,
            "end_video_progress": .9, "max_workspace_mib": 32, "g_hard_limit": 1.5}}
    for key, setup, config, latent_input in (("43", "10", "41", ["9", 2]), ("44", "26", "42", ["25", 0])):
        graph[key] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
            "model": [setup, 0], "sigmas": [setup, 2], "av_latent": latent_input,
            "stage_context": [setup, 3], "eav_config": [config, 0]}}
    graph["12"]["inputs"]["model"] = ["43", 0]
    graph["28"]["inputs"]["model"] = ["44", 0]
    graph["45"] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
        "av_latent": ["13", 1], "runtime": ["43", 1]}}
    graph["46"] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
        "av_latent": ["29", 0], "runtime": ["44", 1]}}
    graph["23"]["inputs"]["av_latent"] = ["45", 0]
    graph["30"]["inputs"]["sampled_av_latent"] = ["46", 0]
    return graph


def split_graph_with_results(graph=None):
    graph = split_graph() if graph is None else deepcopy(graph)
    for sampler, setup in (("13", "10"), ("29", "26")):
        graph[sampler]["class_type"] = "MiniMaxH3StageSamplerEXPT8"
        graph[sampler]["inputs"]["stage_context"] = [setup, 3]
    # A second identical UNETLoader is not independent in Core's cache. Load
    # HIGH only after LOW completed; this also avoids sharing a live
    # model_sampling owner between the two stage/effect branches.
    graph["22"]["class_type"] = "MiniMaxH3StageUNETLoaderAfterEXPT8"
    graph["22"]["inputs"]["completed_stage"] = ["13", 2]
    graph["62"] = {"class_type": "MiniMaxH3FastH3V2CompletedLowX0EXPT8",
                   "inputs": {"low_stage_result": ["13", 2]}}
    for key, sampler, prefix in (("50", "13", "FastH3V2/LOW"), ("51", "29", "FastH3V2/HIGH")):
        graph[key] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
            "stage_result": [sampler, 2], "prefix": prefix}}
    if "45" in graph:
        graph["45"]["inputs"]["av_latent"] = ["62", 0]
        graph["46"]["inputs"]["av_latent"] = ["51", 0]
    else:
        graph["23"]["inputs"]["av_latent"] = ["62", 0]
        graph["30"]["inputs"]["sampled_av_latent"] = ["51", 0]
    return graph


def resume_high_graph(artifact_path="REPLACE_WITH_SAVED_LOW_PATH/manifest.json", artifact_sha256="0" * 64,
                      with_effects=False):
    graph = split_graph_with_results(split_graph_with_effects() if with_effects else None)
    graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
        "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "low_0_4"}}
    graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
    graph["62"]["inputs"]["low_stage_result"] = ["60", 3]
    graph["23"]["inputs"]["av_latent"] = ["62", 0]
    graph["22"]["inputs"]["completed_stage"] = ["60", 3]
    # Actually remove the LOW body. A hidden muted/disconnected LOW output node
    # can still execute in Core; a recovery graph must contain no LOW sampler.
    keep = set()

    def include(key):
        if key in keep:
            return
        keep.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                include(str(value[0]))

    for key in ("16", "31", "32", "51", "61"):
        include(key)
    return {key: item for key, item in graph.items() if key in keep}


def load_live_info():
    snapshot = capture()
    core_root = ROOT.parents[1] if len(ROOT.parents) > 1 else None
    if core_root is not None and (core_root / "comfy").is_dir():
        sys.path.insert(0, str(core_root))
    import nodes as core_nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_video import CreateVideo, SaveVideo
    from comfy_extras.nodes_preview_any import PreviewAny
    package = sys.modules["_t8_modular_compat_capture"]
    project_classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    selected = {cls.define_schema().node_id: cls for cls in project_classes}
    selected.update({cls.__name__: cls for cls in (BasicGuider, RandomNoise, SamplerCustomAdvanced,
                                                  CreateVideo, SaveVideo, PreviewAny)})
    for name in ("UNETLoader", "CLIPLoader", "VAELoader"):
        selected[name] = core_nodes.NODE_CLASS_MAPPINGS[name]
    core_nodes.NODE_CLASS_MAPPINGS.update(selected)
    info = {item["id"]: item["info"] for item in snapshot["nodes"]}
    info.update({name: native_info(name, cls) for name, cls in selected.items() if name not in info})
    return json.loads(json.dumps(info))


def build_candidate(graph, info):
    graph, selected_info = selected_frontend_schema(graph, info)
    workflow = convert(graph, selected_info, "FastH3 V2 — editable LOW / learned upscale / HIGH (EXP)")
    mapping = {key: index + 1 for index, key in enumerate(graph)}
    titles = {"1": "LOW MODEL — insert your LOW LoRA chain", "22": "HIGH independent MODEL load — insert HIGH LoRA chain",
              "9": "LOW conditioning", "24": "HIGH conditioning — actual upscale dimensions",
              "10": "LOW 0:4 — exact V2 stage", "26": "HIGH 4:8 — exact V2 stage",
              "13": "PASS 1 — use denoised_output for learned handoff", "29": "PASS 2 — final joint AV",
              "23": "Existing learned 3D upscale — video only", "25": "Reconcile HIGH template / joint audio"}
    # Two aligned lanes, explicit middle handoff. All ordinary ports stay visible.
    positions = {"1": (0, 0), "6": (0, 380), "7": (0, 670), "8": (0, 890),
                 "9": (400, 0), "10": (800, 0), "11": (800, 340), "12": (1160, 0), "13": (1520, 0),
                 "19": (1900, 0), "20": (2250, 0), "21": (1160, 320), "22": (0, 1300),
                 "23": (400, 1300), "24": (800, 1300), "25": (1200, 1300), "26": (1560, 1300),
                 "27": (1560, 1650), "28": (1920, 1300), "29": (2280, 1300), "30": (2640, 1300),
                 "31": (3000, 1300), "32": (1920, 1630), "14": (3000, 1800), "15": (3360, 1800), "16": (3720, 1800)}
    effects = any(item["class_type"] == "MiniMaxH3StageEAVApplyEXPT8" for item in graph.values())
    if effects:
        titles.update({"9": "LOW Relay — paired MODEL / COND / latent", "24": "HIGH Relay — rebuilt real layout",
                       "40": "External LOW Relay Plan — editable without changing HIGH",
                       "47": "External HIGH Relay Plan — editable without changing LOW",
                       "41": "External LOW EAV config", "42": "External HIGH EAV config",
                       "43": "LOW EAV + Relay owner", "44": "HIGH EAV + Relay owner",
                       "45": "LOW actual effect audit — handoff x0", "46": "HIGH actual effect audit"})
        positions.update({"40": (-450, -250), "47": (350, 1050), "41": (800, 650), "42": (1560, 2000),
                          "43": (1160, 0), "12": (1520, 0), "13": (1880, 0), "45": (2250, 0),
                          "19": (2620, 0), "20": (2990, 0), "44": (1920, 1300), "28": (2280, 1300),
                          "29": (2640, 1300), "46": (3000, 1300), "30": (3370, 1300), "31": (3740, 1300),
                          "14": (3370, 1800), "15": (3740, 1800), "16": (4110, 1800)})
    results = "51" in graph
    recovery = "60" in graph
    if results:
        titles.update({"50": "Save completed LOW — copy path + SHA for later HIGH-only reuse",
                       "51": "Save completed HIGH — exact output and denoised state",
                       "62": "Verified completed LOW x0 — never unfinished x_sigma",
                       "60": "Load frozen LOW — enter BOTH saved path and SHA; LOW does not execute",
                       "61": "Frozen LOW provenance / implementation report"})
        positions.update({"50": (1850, 350), "51": (2600, 900), "60": (0, 700), "61": (450, 700),
                          "62": (2070, 550), "30": (3000, 1300), "31": (3380, 1300)})
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, node_id in mapping.items():
        if key in titles:
            by_id[node_id]["title"] = titles[key]
        if key in positions:
            by_id[node_id]["pos"] = list(positions[key])
    note_id = workflow["last_node_id"] + 1
    note = (
        "分离式 FastH3 V2 4 + learned upscale + 4 / EXP 候选，未获得 GPU 或人审资格。\n\n"
        "上排 LOW，下排 HIGH：各自 MODEL → 可插独立 LoRA → Stage Setup → BasicGuider → 标准采样器。"
        "两阶段各自提示词、噪声节点；不是隐藏 Loop。需要 full V2 student，不要把旧 Turbo/EMA 加速 LoRA 当作 V2 模型。\n\n"
        "LOW 必须接 denoised_output 到原有 learned 3D upscaler；HIGH 重建实际尺寸条件后 Reconcile。"
        "auto + legacy_policy 保留联合音频继续采样；不是把未完成 LOW 音频锁死。最终解码 HIGH output。\n\n"
        "默认 dense_compat_exp 是显式兼容分支；可分别改 trained_vsa_exp，但需查看真实 VSA eligibility/dispatch。"
        "阶段 Context 只是描述，不是采样完成回执；当前图没有跨重启恢复或 GPU/人审资格。\n\n"
        "模型/CLIP/VAE/放大器文件须在本机存在。可修改 LOW/HIGH LoRA；高采专属修改不应触发 LOW 重算，"
        "仍受 ComfyUI 真实依赖及缓存寿命约束。旧工作流和旧节点没有被替换。"
    )
    if effects:
        note += ("\n\n本图外置两份Relay Plan，LOW/HIGH提示事件可分别编辑；各阶段在实际尺寸分别绑定MODEL/条件。"
                 "两份EAV Config可分别关闭/report_only/apply_exp；默认只审计，不增强。"
                 "注意：绝对1-video_sigma窗口不重置，默认.15-.90可能让LOW4全部不激活。"
                 "原生VSA的FETA桥接已有CPU数学/producer检查，真实CUDA与稀疏Relay时间偏置仍未认证。"
                 "本图明确选择dense_compat_exp，不冒充稀疏组合已验收。")
    if results:
        note = note.replace("标准采样器", "单阶段原生采样器 + 结果回执")
        note = note.replace("当前图没有跨重启恢复或 GPU/人审资格。",
                            "本图提供显式阶段保存/读取；没有完整预训练 GPU/人审资格。")
        note += ("\n\nStage Sampler仍只调用一次原生采样；不是双采封装。Save会在output/MiniMaxH3/stage_artifacts"
                 "写新目录，输出artifact_path及artifact_sha256，两个都必须保存。LOW必须用denoised_output交接。"
                 "Completed LOW x0节点先核实V2 0:4真实完成回执，才把x0送往独立learned放大器；不把unfinished x_sigma误用。"
                 "读取是显式冻结选中结果，不会自动判断你当前改过的LOW模型/提示词仍相同；要改变LOW请重新采样并保存。"
                 "坏SHA、错阶段、未完成文件拒绝；原生阶段EAV、普通Dense Relay及其组合已绑定实际效果身份。"
                 "稀疏Relay/第三方后端等未适配组合仍可新采样和归档，但不授予认证跨运行复用。")
        note += ("\n\nHIGH MODEL使用完成LOW后独立加载节点；两个同参数原生UNETLoader可能被Core缓存为同一底层对象，"
                 "MODEL clone也不隔离共享网络的live model_sampling。此节点不隐藏采样，但可增加CPU RAM/VRAM占用；"
                 "请按设备余量安排模型释放与分阶段执行。HIGH模型文件和LoRA仍可独立选择。")
    if recovery:
        note += ("\n\n这是只跑HIGH的恢复图：LOW模型、条件、采样器、输出节点已实际移除，不是隐藏或静音。"
                 "先从上一张保存图复制LOW的artifact_path和artifact_sha256到Load；占位符不能运行。"
                 "原learned3D放大和HIGH实际尺寸条件仍在图中可编辑。共享旧LOW设置没有接入本图，不会静默重采LOW。")
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / Qualification boundaries",
        "pos": [0, -650], "size": [1250, 580], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = note_id
    audit = audit_candidate(graph, workflow, selected_info)
    return graph, workflow, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-effects", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--with-results", action="store_true")
    variants.add_argument("--resume-high", action="store_true")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifact directory; reviewed examples are promoted separately")
    info = load_live_info()
    selected = (resume_high_graph(with_effects=args.with_effects) if args.resume_high else
                split_graph_with_results(split_graph_with_effects() if args.with_effects else None) if args.with_results
                else split_graph_with_effects() if args.with_effects else split_graph())
    graph, workflow, audit = build_candidate(selected, info)
    import execution
    validation = asyncio.run(execution.validate_prompt("t8-modular-v2-cpu-validation", deepcopy(graph), None))
    report = {"serialization": audit, "core_validation": validation,
              "qualification": "CPU live schema / graph validation; no browser, weights loaded, GPU or human review."}
    destination.mkdir(parents=True)
    name = ("FastH3_V2_Frozen_LOW_Resume_HIGH_EXP" if args.resume_high else
            "FastH3_V2_Separate_LOW_HIGH_Save_Stages_EXP" if args.with_results else
            "FastH3_V2_Separate_LOW_HIGH_EAV_Relay_EXP" if args.with_effects else "FastH3_V2_Separate_LOW_HIGH_EXP")
    if args.with_effects and (args.with_results or args.resume_high):
        name = name.removesuffix("_EXP") + "_EAV_Relay_EXP"
    write_new(destination / f"{name}.api.json", graph)
    write_new(destination / f"{name}.json", workflow)
    write_new(destination / "audit.json", report)
    used = {item["class_type"] for item in graph.values()}
    write_new(destination / "object-info.json", {name: info[name] for name in sorted(used)})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not validation[0]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
