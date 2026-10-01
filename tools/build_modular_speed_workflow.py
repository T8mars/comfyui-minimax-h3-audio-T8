"""Build private editable SPEED 2/3-stage candidates; never queues sampling."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

try:
    from .api_to_frontend_workflow import convert
    from .audit_progressive_workflows import audit_candidate
    from .build_fast_h3_v2_workflows import selected_frontend_schema
    from .build_modular_fast_h3_v2_workflow import load_live_info
except ImportError:  # Direct script invocation from tools/.
    from api_to_frontend_workflow import convert
    from audit_progressive_workflows import audit_candidate
    from build_fast_h3_v2_workflows import selected_frontend_schema
    from build_modular_fast_h3_v2_workflow import load_live_info


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = "minimax_h3_fl2va_int8_convrot.safetensors"
CLIP_FILE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SEED = 2609231701


def node(name, **inputs):
    return {"class_type": name, "inputs": inputs}


def relay_params():
    return dict(global_prompt="One cinematic night street, the same woman and natural city ambience.",
                local_prompts="She looks toward the camera.\nShe walks past the lights.\nThe camera follows steadily.",
                length=124, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
                epsilon=0.1, allow_gaps=False, allow_overlaps=False)


def split_graph(stages=2, *, with_effects=False, resume_stage=None,
                effects_by_stage=None, save_stages=True):
    if stages not in (2, 3):
        raise ValueError("This candidate builder supports two or three explicit SPEED stages")
    if resume_stage is not None and not 1 <= resume_stage < stages:
        raise ValueError("Resume must begin at a later stage inside this SPEED plan")
    if effects_by_stage is None:
        effects_by_stage = {index: frozenset(("relay", "eav")) if with_effects else frozenset()
                            for index in range(stages)}
    else:
        if any(index not in range(stages) for index in effects_by_stage):
            raise ValueError("Effect stage must be inside the SPEED plan")
        effects_by_stage = {index: frozenset(effects_by_stage.get(index, ()))
                            for index in range(stages)}
        if any(not effects <= {"relay", "eav"} for effects in effects_by_stage.values()):
            raise ValueError("Only external Relay and EAV stage effects are supported")
    relay_settings = relay_params()
    relay_prompt = None
    if any("relay" in effects for effects in effects_by_stage.values()):
        package = ("h3_audio_t8_pkg" if "h3_audio_t8_pkg" in sys.modules
                   else "_t8_modular_compat_capture")
        build_prompt_relay_plan = importlib.import_module(
            package + ".prompt_relay_advanced",
        ).build_prompt_relay_plan
        _, relay_prompt, *_ = build_prompt_relay_plan(**relay_settings)
    plain_prompt = ("A cinematic night city in rain, camera gliding forward through reflections and mist. "
                    "Natural synchronized rain and distant traffic, no speech.")
    graph = {
        "1": node("CLIPLoader", clip_name=CLIP_FILE, type="minimax", device="default"),
        "2": node("VAELoader", vae_name=VIDEO_VAE),
        "3": node("VAELoader", vae_name=AUDIO_VAE),
        "4": node("MiniMaxH3SPEEDPlanT8Advanced", width=736, height=416, steps=20,
                  scales="0.5,1.0" if stages == 2 else "0.4,0.7,1.0",
                  transition_mode="manual_sigmas",
                  manual_transition_sigmas="0.85" if stages == 2 else "0.94,0.78",
                  delta=0.01, shift_video=12.0, transform="dct",
                  profile_policy="require_validated_profile", fallback_policy="error"),
    }
    previous_spec_link = previous_result_link = None
    if resume_stage is not None:
        graph["5"] = node("MiniMaxH3SPEEDStageLoadEXPT8", speed_plan=["4", 0], seed=SEED,
                          shift_audio=3.0, next_stage_index=resume_stage,
                          artifact_path="", artifact_sha256="")
        previous_spec_link, previous_result_link = ["5", 1], ["5", 0]
    for index in range(resume_stage or 0, stages):
        stage_effects = effects_by_stage[index]
        prefix = str(10 + index * 12)
        model_id, source_id, setup_id = (str(int(prefix) + offset) for offset in (0, 1, 2))
        noise_id, sample_id, transition_id = (str(int(prefix) + offset) for offset in (3, 4, 5))
        relay_plan_id, relay_id, config_id = (str(int(prefix) + offset) for offset in (6, 7, 8))
        effect_id, audit_id = (str(int(prefix) + offset) for offset in (9, 10))
        save_id = str(int(prefix) + 11)
        graph[model_id] = node("UNETLoader", unet_name=MODEL_FILE, weight_dtype="default")
        graph[source_id] = node("MiniMaxH3SPEEDSourceT8Advanced",
            clip=["1", 0], video_vae=["2", 0], audio_vae=["3", 0],
            prompt=relay_prompt if "relay" in stage_effects else plain_prompt, length=124,
            task_type="T2VA", audio_mode="native", audio_denoise_strength=1.0,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True, ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            checkpoint_fingerprint="unrecorded", vae_fingerprint="unrecorded")
        setup_inputs = dict(model=[model_id, 0], speed_plan=["4", 0], speed_source=[source_id, 0],
                            stage_index=index, shift_audio=3.0, seed=SEED,
                            execution_scope="strict_t2va_stock20", reuse_t2va_text=True)
        if previous_spec_link is not None:
            setup_inputs["previous_spec"] = previous_spec_link
        graph[setup_id] = node("MiniMaxH3SPEEDStageSetupEXPT8", **setup_inputs)
        selected_model, selected_positive, selected_latent = ([setup_id, item] for item in (0, 1, 2))
        if "relay" in stage_effects:
            # Distinct Plan/Config nodes make every stage independently editable.
            graph[relay_plan_id] = node("MiniMaxH3PromptRelayPlanT8Advanced", **deepcopy(relay_settings))
            graph[relay_id] = node("MiniMaxH3SPEEDRelayApplyEXPT8",
                model=[setup_id, 0], stage_spec=[setup_id, 5], speed_source=[source_id, 0],
                prompt_relay_plan=[relay_plan_id, 0], execution_mode="apply_exp", query_chunk_rows=256)
            selected_model, selected_positive, selected_latent = ([relay_id, item] for item in (0, 1, 2))
        if "eav" in stage_effects:
            graph[config_id] = node("MiniMaxH3StageEAVConfigEXPT8", mode="report_only", tau=0.1,
                start_video_progress=0.0, end_video_progress=1.0,
                max_workspace_mib=32, g_hard_limit=3.0)
            graph[effect_id] = node("MiniMaxH3StageEAVApplyEXPT8", model=selected_model,
                sigmas=[setup_id, 4], av_latent=selected_latent,
                stage_context=[setup_id, 10], eav_config=[config_id, 0])
            selected_model = [effect_id, 0]
        if index == 0:
            graph[noise_id] = node("MiniMaxH3SPEEDModalityStableNoiseT8Advanced", seed=SEED)
            noise_output = [noise_id, 0]
        else:
            graph[transition_id] = node("MiniMaxH3SPEEDDCTTransitionEXPT8",
                completed_stage=previous_result_link, next_stage=[setup_id, 5], dct_chunk_size=64)
            noise_output = [transition_id, 0]
        graph[sample_id] = node("MiniMaxH3SPEEDStageSampleEXPT8",
            model=selected_model, positive=selected_positive, av_latent=selected_latent,
            sampler=[setup_id, 3], sigmas=[setup_id, 4], noise=noise_output,
            stage_spec=[setup_id, 5])
        if "eav" in stage_effects:
            graph[audit_id] = node("MiniMaxH3StageEAVAuditEXPT8",
                av_latent=[sample_id, 0], runtime=[effect_id, 1])
        previous_spec_link, previous_result_link = [setup_id, 5], [sample_id, 1]
        if index < stages - 1 and save_stages:
            graph[save_id] = node("MiniMaxH3SPEEDStageSaveEXPT8", stage_result=[sample_id, 1])
            previous_result_link = [save_id, 0]
    final_latent = [audit_id, 0] if "eav" in stage_effects else [sample_id, 0]
    graph["80"] = node("MiniMaxH3AVDecodeT8", av_latent=final_latent,
                       video_vae=["2", 0], audio_vae=["3", 0])
    graph["81"] = node("CreateVideo", images=["80", 0], audio=["80", 1], fps=24.0, bit_depth=8)
    graph["82"] = node("SaveVideo", video=["81", 0],
                       filename_prefix=f"MiniMaxH3/Modular_SPEED_{stages}stage_EXP", format="mp4")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", type=int, choices=(2, 3), default=2)
    parser.add_argument("--with-effects", action="store_true")
    parser.add_argument("--resume-stage", type=int, choices=(1, 2))
    args = parser.parse_args()
    target = args.output_dir.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "artifacts"):
        parser.error("Choose a new private artifact directory")
    info = load_live_info()
    graph = split_graph(args.stages, with_effects=args.with_effects, resume_stage=args.resume_stage)
    graph, selected = selected_frontend_schema(graph, info)
    title = f"SPEED {args.stages}-stage editable DCT transition (research EXP)"
    if args.resume_stage is not None:
        title += f" — resume stage {args.resume_stage}"
    workflow = convert(graph, selected, title)
    by_id = {item["id"]: item for item in workflow["nodes"]}
    for key, node_id in zip(graph, by_id):
        if graph[key]["class_type"] == "UNETLoader":
            by_id[node_id]["title"] = f"Stage MODEL {key} — insert independent LoRA here"
    note = ("SPEED 分离式研究候选，不是质量／速度推荐。每个 StageSample 只采一个分辨率阶段；"
            "DCTTransition 单独处理视频频域和联合音频时间重索引。各阶段 MODEL/LoRA、Source、"
            "SAMPLER、NOISE 可独立接线；改后段不会在图结构上重跑前段。"
            "手工 sigma 只为机械示范，旧用户盲评已否决固定计划；不把本图当正式默认。"
            "外置 Relay Plan 可独立改词；重建 AV 目标必须与本阶段 Source 目标完全一致。"
            "EAV 默认 report_only，实际调用看 Audit。"
            "显式 Load 需填已保存阶段的相对manifest路径及精确SHA；它冻结所选旧阶段，不会按当前前段编辑自动命中。"
            "效果阶段只有真实覆盖回执合格才可显式保存；后段默认复用 Source 文本，若要沿用/改写外置 Relay，"
            "请在后段另接独立 Relay Plan。无GPU音画/显存或真人质量资格；旧整链节点与工作流未修改。")
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "先读 / SPEED EXP",
        "pos": [0, -800], "size": [1180, 460], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = note_id
    audit = audit_candidate(graph, workflow, selected)
    import execution
    validation = asyncio.run(execution.validate_prompt("t8-speed-split-cpu", deepcopy(graph), None))
    target.mkdir(parents=True)
    stem = f"SPEED_{args.stages}Stage_{'EAV_Relay' if args.with_effects else 'Minimal'}"
    stem += f"_Resume{args.resume_stage}" if args.resume_stage is not None else ""
    stem += "_DRAFT"
    (target / f"{stem}.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / f"{stem}_api.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "audit.json").write_text(json.dumps({"serialization": audit, "core_validation": validation,
        "qualification": "Private CPU/Core graph only; no browser, weights, GPU, complete AV, or human acceptance."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"nodes": len(graph), "status": audit.get("status"), "core_validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
