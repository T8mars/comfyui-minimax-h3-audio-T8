"""Build editable P7 segment candidates with explicit delivery and acceptance.

This is a CPU graph/serialization builder. It never queues a prompt or claims
pretrained GPU, browser round-trip, or human audio/video qualification.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_hyperflow_workflow as base  # noqa: E402


ROOT, shared = base.ROOT, base.shared
SEGMENTS = ("segment0", "segment1")
EFFECTS = (("none", "both"), *((kind, scope) for kind in ("eav", "relay", "combined")
                                for scope in ("low", "high", "both")))
RESUME_EFFECTS = EFFECTS
CASES = tuple((segment, kind, scope, variant) for segment in SEGMENTS
              for kind, scope in EFFECTS for variant in ("minimal", "save")) + tuple(
                  (segment, kind, scope, "resume_high") for segment in SEGMENTS
                  for kind, scope in RESUME_EFFECTS)
CHAIN_ID = "modular_p7_8s_new_chain_exp"
BASE_MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
LOW_PATH_PLACEHOLDER = "REPLACE_WITH_SAVED_P7_LOW_MANIFEST_PATH"
PARENT_ID_PLACEHOLDER = "REPLACE_WITH_SEGMENT0_ACCEPTED_CANDIDATE_ID"
PARENT_SHA_PLACEHOLDER = "0" * 64


def _node(name, **inputs):
    return {"class_type": name, "inputs": inputs}


def split_graph(segment="segment0", kind="combined", scope="both", variant="minimal", *,
                artifact_path=LOW_PATH_PLACEHOLDER, artifact_sha256=PARENT_SHA_PLACEHOLDER):
    """Return an API graph. A resume graph contains no LOW MODEL/noise/sampler."""
    if (segment, kind, scope, variant) not in CASES:
        raise ValueError("Unknown P7 segment/effects/storage candidate")
    old = shared.split_graph()
    graph = {key: deepcopy(old[key]) for key in ("1", "6", "7", "8")}
    graph["50"] = deepcopy(old["22"])
    for key in ("1", "50"):
        graph[key]["inputs"]["unet_name"] = BASE_MODEL

    is_continuation = segment == "segment1"
    segment_seed = 123456789 + int(is_continuation)
    if is_continuation:
        graph["2"] = _node("MiniMaxH3HyperFlowP7AcceptedParentEXPT8", chain_id=CHAIN_ID,
            segment_index=1, parent_candidate_id=PARENT_ID_PLACEHOLDER,
            parent_revision=1, previous_job_sha256=PARENT_SHA_PLACEHOLDER,
            context_frames=22, width=896, height=448, low_width=448, low_height=224)
        graph["3"] = _node("MiniMaxH3HyperFlowP7PrepareContextsEXPT8",
                           accepted_parent=["2", 0], video_vae=["7", 0])
        contexts = ["3", 0]
    else:
        graph["2"] = _node("MiniMaxH3HyperFlowP7InitialSegmentEXPT8", chain_id=CHAIN_ID,
            context_frames=22, width=896, height=448, low_width=448, low_height=224)
        contexts = ["2", 0]

    graph["10"] = _node("MiniMaxH3HyperFlowFreshLoaderEXPT8", model=["1", 0],
                        hyperflow_file=base.WEIGHT)
    graph["11"] = _node("MiniMaxH3HyperFlowFreshLoaderEXPT8", model=["50", 0],
                        hyperflow_file=base.WEIGHT)
    prompt = ("A single continuous bright daytime camera move through a living room. "
              "Stable furniture, crisp details, natural light and quiet ambience.")
    for phase, condition, loader, setup, noise, guider, sampler, result, plan, project, relay, config, eav in (
        ("low", "12", "10", "14", "15", "16", "17", "18", "30", "31", "32", "33", "34"),
        ("high", "20", "11", "22", "23", "24", "25", "26", "40", "41", "42", "43", "44"),
    ):
        selected = scope in (phase, "both")
        active = selected and (variant != "resume_high" or phase == "high")
        phase_prompt = prompt + (" LOW framing and natural motion." if phase == "low"
                                  else " HIGH crisp texture and stable detail.")
        graph[condition] = _node("MiniMaxH3HyperFlowP7ConditioningEXPT8",
            contexts=contexts, phase=phase, clip=["6", 0], video_vae=["7", 0],
            audio_vae=["8", 0], prompt=phase_prompt, length=124)
        selected_model = [loader, 0]
        positive, negative = [condition, 1], [condition, 2]
        if selected and kind in ("relay", "combined"):
            graph[plan] = _node("MiniMaxH3PromptRelayPlanT8Advanced",
                global_prompt=phase_prompt,
                local_prompts=("The camera glides slowly past the sofa.\n"
                               "The camera continues toward the window.\n"
                               "Sunlight shifts across the living room.\n"
                               "The camera settles on the far wall."),
                length=193, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
                epsilon=.1, allow_gaps=False, allow_overlaps=False)
            graph[project] = _node("MiniMaxH3HyperFlowP7RelayProjectEXPT8",
                contexts=contexts, global_plan=[plan, 0], length=124,
                accepted_end_frame=192 if is_continuation else 124)
            graph[condition]["inputs"]["prompt"] = [project, 1]
        if phase == "low" and variant == "resume_high":
            graph["19"] = _node("MiniMaxH3HyperFlowP7LowLoadEXPT8",
                low_phase=[condition, 0], artifact_path=artifact_path,
                artifact_sha256=artifact_sha256)
            continue
        if active and kind in ("relay", "combined"):
            graph[relay] = _node("MiniMaxH3HyperFlowP7RelayApplyEXPT8",
                model=selected_model, prepared_phase=[condition, 0], clip=["6", 0],
                projected_relay=[project, 0], query_chunk_rows=256, mode="apply_exp")
            selected_model, positive, negative = [relay, 0], [relay, 1], [relay, 2]
        if phase == "low":
            graph[setup] = _node("MiniMaxH3HyperFlowP7LowSetupEXPT8",
                prepared_phase=[condition, 0], model=selected_model,
                positive=positive, negative=negative)
        else:
            graph[setup] = _node("MiniMaxH3HyperFlowP7HighSetupEXPT8",
                                 high_handoff=["21", 0], model=selected_model)
            positive = [setup, 5]
        sampled_model = [setup, 0]
        if active and kind in ("eav", "combined"):
            graph[config] = _node("MiniMaxH3StageEAVConfigEXPT8", mode="report_only",
                tau=4., start_video_progress=.15, end_video_progress=.9,
                max_workspace_mib=32, g_hard_limit=1.5)
            graph[eav] = _node("MiniMaxH3HyperFlowP7StageEAVApplyEXPT8", model=sampled_model,
                sigmas=[setup, 2], av_latent=[setup, 3], stage_context=[setup, 4],
                eav_config=[config, 0], prepared_phase=[condition, 0])
            sampled_model = [eav, 0]
        graph[noise] = _node("RandomNoise", noise_seed=segment_seed if phase == "low"
                             else segment_seed + 1)
        graph[guider] = _node("BasicGuider", model=sampled_model, conditioning=positive)
        graph[sampler] = _node("MiniMaxH3StageSamplerEXPT8", noise=[noise, 0],
            guider=[guider, 0], sampler=[setup, 1], sigmas=[setup, 2],
            latent_image=[setup, 3], stage_context=[setup, 4])
        if phase == "low":
            graph[result] = _node("MiniMaxH3HyperFlowP7LowResultEXPT8",
                                  prepared_phase=[condition, 0], stage_result=[sampler, 2])
        else:
            graph[result] = _node("MiniMaxH3HyperFlowP7HighResultEXPT8",
                                  high_handoff=["21", 0], stage_result=[sampler, 2])

    low_result = ["19", 0] if variant == "resume_high" else ["18", 0]
    if variant == "save":
        graph["19"] = _node("MiniMaxH3HyperFlowP7LowSaveEXPT8", low_result=low_result)
        low_result = ["19", 0]
    graph["21"] = _node("MiniMaxH3HyperFlowP7HighHandoffEXPT8",
        learned_lift=["35", 0], high_phase=["20", 0], positive=["42", 1]
        if "42" in graph else ["20", 1], negative=["42", 2]
        if "42" in graph else ["20", 2])
    graph["35"] = _node("MiniMaxH3HyperFlowP7LearnedLiftEXPT8",
                        low_result=low_result, upscaler_model=UPSCALER)
    high_result = ["26", 0]
    if variant in ("save", "resume_high"):
        graph["27"] = _node("MiniMaxH3HyperFlowP7HighSaveEXPT8", high_result=high_result)
        high_result = ["27", 0]
    graph["28"] = _node("MiniMaxH3HyperFlowP7CandidateSaveEXPT8",
        high_result=high_result, video_vae=["7", 0], audio_vae=["8", 0], candidate_id="",
        is_final_segment=is_continuation, final_frame_count=68 if is_continuation else 0,
        model_id="h3_p7_modular_exp", seed=segment_seed,
        color_match=True, bit_depth=8, crf=18)
    graph["29"] = _node("MiniMaxH3HyperFlowP7CandidateAcceptEXPT8",
                        candidate_json_path=["28", 0], job_sha256=["28", 2],
                        accept_candidate=False)
    return base.common.prune(graph, ("28", "29"))


def load_live_info():
    return base.load_live_info()


def build_candidate(segment, kind, scope, variant, info, *, graph_override=None):
    graph, selected = shared.selected_frontend_schema(
        split_graph(segment, kind, scope, variant) if graph_override is None
        else deepcopy(graph_override), info)
    workflow = shared.convert(graph, selected,
        f"HyperFlow P7 {segment} — separate LOW/HIGH — {kind}/{scope}/{variant} EXP")
    positions = {"1": (-1050, 200), "50": (-1050, 1950), "6": (-1450, 800),
        "7": (-1450, 1150), "8": (-1450, 1500), "2": (-1050, -300), "3": (-650, -300),
        "10": (-630, 200), "11": (-630, 1950), "12": (-170, 200), "20": (-170, 1950),
        "14": (300, 200), "15": (300, 880), "16": (750, 200), "17": (1200, 200),
        "18": (1650, 200), "19": (2050, 200), "35": (2500, 200),
        "21": (2950, 1600), "22": (3350, 1950), "23": (3350, 2600),
        "24": (3800, 1950), "25": (4250, 1950), "26": (4700, 1950),
        "27": (5100, 1950), "28": (5550, 1950), "29": (6000, 1950),
        "30": (-620, -850), "31": (-170, -850), "32": (300, -850),
        "33": (300, 1300), "34": (750, 1300), "40": (-620, 1150),
        "41": (-170, 1150), "42": (300, 1150), "43": (3350, 3100),
        "44": (3800, 3100)}
    for key, node in zip(graph, workflow["nodes"]):
        node["pos"] = list(positions.get(key, (0, 0)))
        node["title"] = selected[graph[key]["class_type"]].get("display_name", graph[key]["class_type"])
    note = (
        "P7 分离式候选：LOW 0:4 真正取 denoised_output → 原 learned3D → 原音频 reconcile → "
        "HIGH fresh 4:8；不是旧一体 Loop，亦不是 continuous/full8 fresh 路线。两路完整底模/内容 "
        "LoRA 插槽、提示、噪声、Sampler 和外置效果可独立编辑；HyperFlow 原权重必须用专用 Loader。\n\n"
        "保存候选节点内部按旧 P7 交付顺序解码 HIGH、裁掉续段前22帧、可选调色，非末段保存 "
        "LOW 视频＋完成 HIGH 音频上下文和审计。段0输出124帧，续段渲染124帧并只交付68帧，"
        "总计192帧/8秒。Accept 默认 false：先预览，再显式改 true；旧链、旧图不自动改写。\n\n"
        "续段图的 parent_candidate_id / parent_revision / previous_job_sha256 必须从前一张图 "
        "P7 Candidate Accept 的输出填入。HIGH-only 图的 LOW Load path/SHA 也必须换成实际冻结 "
        "LOW Save 的回执；这是显式选定，不会自动匹配当前 LOW 编辑。占位符不可运行。\n\n"
        "EAV Config 默认为 report_only，改为 apply_exp 才真正启用；Relay 是各阶段独立 Plan→投影→"
        "MODEL/CONDITIONING 成对绑定。默认四个连续局部事件让段0及续段各有至少两个活动事件；"
        "改动 Plan 后请核对各段窗口覆盖。LOW-only/ HIGH-only/两边均有候选；resume_high 不含 LOW "
        "模型/噪声/采样/EAV/Relay Apply；若冻结LOW用了Relay，则保留LOW Plan→投影→Conditioning "
        "仅重建来源身份并核验已存回执。LOW编辑不表示重采样或自动匹配缓存。原音画接缝、人审画质未获本图资格。\n\n"
        "本生成器只校验 CPU Core API/front-end 序列化，不排队、不运行 GPU，不证明浏览器保存重载或整片音画。"
    )
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / P7 staged delivery",
        "pos": [-1450, -1500], "size": [1400, 1060], "flags": {}, "order": len(graph),
        "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = note_id
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    destination = options.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = load_live_info()
    import execution
    rows = []
    for case in CASES:
        graph, workflow, audit = build_candidate(*case, info)
        validation = asyncio.run(execution.validate_prompt("t8-hyperflow-p7-modular", deepcopy(graph), None))
        expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
        name = "HyperFlow_P7_" + "_".join(case) + "_EXP"
        rows.append({"name": name, "serialization": audit, "core_validation": validation,
            "all_outputs_valid": bool(validation[0] and not validation[3]
                                      and set(validation[2]) == expected)})
        shared.write_new(destination / (name + ".api.json"), graph)
        shared.write_new(destination / (name + ".json"), workflow)
    shared.write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "CPU live schema/serialization only; not browser, GPU, media or human qualification."})
    shared.write_new(destination / "object-info.json", info)
    failed = [row["name"] for row in rows if not row["all_outputs_valid"]]
    print(json.dumps({"candidates": len(rows), "failed": failed, "output": str(destination)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
