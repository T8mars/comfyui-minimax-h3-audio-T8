"""Build the explicit accepted-parent FastH3 V2 second-window candidate.

The source fields are deliberate placeholders. The default graph renders a raw
90-frame window; --with-delivery trims it to 68 fresh frames. Neither variant
accepts or assembles segments into the final 8s movie.
"""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_fast_h3_v2_workflow as base  # noqa: E402
from audit_modular_sampling_compat import write_new  # noqa: E402
from run_dual_model_pilot import BUND_KOREAN_MV_GLOBAL_PROMPT, BUND_KOREAN_MV_LOCAL_PROMPTS  # noqa: E402

ROOT = base.ROOT
SEED = 2609152202  # Old loop's increment policy for segment 1.


def _prune(graph, outputs):
    keep = set()

    def include(key):
        if key in keep:
            return
        keep.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                include(str(value[0]))

    for key in outputs:
        include(key)
    return {key: item for key, item in graph.items() if key in keep}


def continuation_graph(*, resume_high=False, with_delivery=False,
                       with_current_recipe=False, with_stage_attestation=False,
                       with_condition_provenance=False, with_handoff_provenance=False,
                       with_job_binding=False, with_media_provenance=False,
                       with_candidate_save=False, with_color_match=False,
                       with_frozen_low_bundle=False,
                       first_frame=""):
    if with_stage_attestation and not with_current_recipe:
        raise ValueError("Current stage attestation requires the editable recipe")
    if with_condition_provenance and not with_stage_attestation:
        raise ValueError("Condition provenance requires current stage attestation")
    if with_handoff_provenance and not with_condition_provenance:
        raise ValueError("Handoff provenance requires condition provenance")
    if with_job_binding and not with_handoff_provenance:
        raise ValueError("Current job binding requires handoff provenance")
    if with_media_provenance and not with_job_binding:
        raise ValueError("Current media provenance requires current job binding")
    if with_media_provenance and not with_delivery:
        raise ValueError("Current media provenance requires exact final-68 delivery trim")
    if with_candidate_save and not with_media_provenance:
        raise ValueError("Candidate save requires authenticated current media")
    if with_color_match and not with_media_provenance:
        raise ValueError("External Color Match requires authenticated current media")
    if with_frozen_low_bundle and not (with_current_recipe and with_condition_provenance):
        raise ValueError("Frozen LOW bundle needs current recipe and condition provenance")
    if resume_high and with_current_recipe and not with_frozen_low_bundle:
        raise ValueError("Frozen LOW needs a separately authenticated saved current recipe")
    if with_current_recipe:
        image = Path(first_frame)
        if (not first_frame or image.is_absolute() or image.drive
                or any(part in ("", ".", "..") for part in image.parts)
                or ":" in first_frame or "\\" in first_frame):
            raise ValueError("Recipe first frame must be a safe Core input-relative image filename")
    graph = base.split_graph_with_results(base.split_graph_with_effects())
    graph["70"] = {"class_type": "MiniMaxH3ContinuationSourceEXPT8", "inputs": {
        "chain_id": "REPLACE_WITH_ACCEPTED_CHAIN", "segment_index": 1,
        "parent_candidate_id": "REPLACE_WITH_ACCEPTED_CANDIDATE", "parent_revision": 1,
        "previous_job_sha256": "0" * 64, "context_frames": 22,
        "width": 512, "height": 768, "low_width": 256, "low_height": 384}}
    graph["71"] = {"class_type": "MiniMaxH3ContinuationContextsEXPT8", "inputs": {
        "accepted_source": ["70", 0], "video_vae": ["7", 0]}}
    graph["77"] = {"class_type": "MiniMaxH3FastH3V2AcceptedWindowEXPT8", "inputs": {
        "contexts": ["71", 0], "total_accepted_frames": 192}}
    for key, phase in (("72", "low"), ("73", "high")):
        graph[key] = {"class_type": "MiniMaxH3FastH3V2AcceptedContextEXPT8",
                      "inputs": {"contexts": ["71", 0], "phase": phase}}
    for key in ("40", "47"):
        graph[key]["inputs"].update(global_prompt=BUND_KOREAN_MV_GLOBAL_PROMPT,
            local_prompts=BUND_KOREAN_MV_LOCAL_PROMPTS, length=193,
            timing_mode="percent", time_ranges="0-43.75\n43.75-87.5\n87.5-100")
    for key, plan in (("74", "40"), ("75", "47")):
        graph[key] = {"class_type": "MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8", "inputs": {
            "contexts": ["71", 0], "global_plan": [plan, 0], "length": ["77", 0]}}
    for key, port, projected, model in (("9", "72", "74", "1"), ("24", "73", "75", "22")):
        graph[key] = {"class_type": "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced", "inputs": {
            "model": [model, 0], "clip": ["6", 0], "video_vae": ["7", 0], "audio_vae": ["8", 0],
            "context": [port, 0], "segment_index": [port, 1], "context_frames": [port, 2],
            "context_audio": "video_and_audio", "prompt_relay_plan": [projected, 0],
            "width": [port, 3], "height": [port, 4], "length": ["77", 0], "task_type": "I2VA",
            "audio_mode": "native", "audio_denoise_strength": 1.0,
            "add_source_as_reference": False, "prompt_primary_audio_ordinal": 0,
            "strict_prompt_tags": True, "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s", "execution_mode": "apply_exp",
            "query_chunk_rows": 256, "first_frame_reuse": "segment0_only"}}
    graph["23"]["inputs"].update(size_mode="target_dimensions", scale_by=2.0,
        target_width=512, target_height=768, aspect_policy="honor_dimensions_exp")
    graph["25"] = {"class_type": "MiniMaxH3FastH3V2AcceptedReconcileEXPT8", "inputs": {
        "contexts": ["71", 0], "learned_latent": ["23", 0],
        "highres_template": ["24", 2], "positive": ["24", 1]}}
    graph["76"] = {"class_type": "MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8", "inputs": {
        "contexts": ["71", 0], "reconciled_av": ["25", 0],
        "mode": "high_native_mask_ramp_exp"}}
    for key, input_name in (("26", "av_latent"), ("29", "latent_image"), ("44", "av_latent")):
        graph[key]["inputs"][input_name] = ["76", 0]
    for key in ("41", "42"):
        graph[key]["inputs"]["mode"] = "disabled"  # Old V2 loop default; external switches remain editable.
    graph["11"]["inputs"]["noise_seed"] = SEED
    graph["27"]["inputs"]["noise_seed"] = SEED
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/FastH3V2_Accepted_Second_Raw90_EXP"
    outputs = ["16", "31", "32", "51", "72", "73", "74", "75", "76", "77"]
    if with_delivery:
        graph["78"] = {"class_type": "MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8", "inputs": {
            "contexts": ["71", 0], "total_accepted_frames": 192}}
        graph["79"] = {"class_type": "MiniMaxH3OutputTrimT8", "inputs": {
            "frames": ["14", 0], "audio": ["14", 1],
            "start_seconds": ["78", 5], "duration_seconds": ["78", 6], "fps": 24.0}}
        graph["80"] = {"class_type": "PreviewAny", "inputs": {"source": ["79", 2]}}
        graph["15"]["inputs"].update(images=["79", 0], audio=["79", 1])
        graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/FastH3V2_Accepted_Second_New68_Unaccepted_EXP"
        outputs.extend(("78", "80"))
    if with_current_recipe:
        if "78" not in graph:
            graph["78"] = {"class_type": "MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8", "inputs": {
                "contexts": ["71", 0], "total_accepted_frames": 192}}
        graph["82"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        graph["83"] = {"class_type": "MiniMaxH3FastH3V2CurrentRecipeEXPT8", "inputs": {
            "model_pass1": ["1", 0], "model_pass2": ["22", 0],
            "clip": ["6", 0], "video_vae": ["7", 0], "audio_vae": ["8", 0],
            "first_frame": ["82", 0], "low_global_plan": ["40", 0],
            "high_global_plan": ["47", 0], "low_eav_config": ["41", 0],
            "high_eav_config": ["42", 0], "upscale_report_json": ["23", 3],
            "chain_id": ["78", 0], "low_width": ["72", 3], "low_height": ["72", 4],
            "width": ["73", 3], "height": ["73", 4],
            "total_accepted_frames": 192, "first_seed": SEED - 1}}
        graph["84"] = {"class_type": "PreviewAny", "inputs": {"source": ["83", 3]}}
        outputs.append("84")
    if with_condition_provenance:
        replacements = {"9": "89", "24": "90"}
        for old, new in replacements.items():
            graph[new] = graph.pop(old)
            graph[new]["class_type"] = "MiniMaxH3FastH3V2ConditionProvenanceEXPT8"
        for item in graph.values():
            for name, value in item["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and value[0] in replacements:
                    item["inputs"][name] = [replacements[value[0]], value[1]]
    if with_handoff_provenance:
        graph["91"] = graph.pop("23")
        graph["91"]["class_type"] = "MiniMaxH3FastH3V2UpscaleProvenanceEXPT8"
        for item in graph.values():
            for name, value in item["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and value[0] == "23":
                    item["inputs"][name] = ["91", value[1]]
    if with_stage_attestation:
        for key, preview, sampler, raw, model, global_plan, projected, eav, stage in (
                ("85", "86", "13", "1", "43", "40", "74", "41", "low_0_4"),
                ("87", "88", "29", "22", "44", "47", "75", "42", "high_4_8")):
            sample = graph[sampler]["inputs"]
            setup = "10" if stage == "low_0_4" else "26"
            audit_node = ("MiniMaxH3FastH3V2OriginStageAttestEXPT8"
                          if with_condition_provenance else
                          "MiniMaxH3FastH3V2CurrentStageAttestEXPT8")
            graph[key] = {"class_type": audit_node, "inputs": {
                "current_recipe": ["83", 0], "stage_result": [sampler, 2],
                "raw_model": [raw, 0], "stage_model": [model, 0],
                "guider": sample["guider"], "noise": sample["noise"],
                "sigmas": sample["sigmas"], "source_latent": sample["latent_image"],
                "stage_context": [setup, 3], "global_plan": [global_plan, 0],
                "projected_plan": [projected, 0], "eav_config": [eav, 0],
                "segment_index": ["78", 1], "phase": stage}}
            if with_condition_provenance:
                conditioner = "89" if stage == "low_0_4" else "90"
                graph[key]["inputs"].update(condition_receipt=[conditioner, 7],
                                             conditioned_latent=[conditioner, 2])
            graph[preview] = {"class_type": "PreviewAny", "inputs": {"source": [key, 2]}}
            outputs.append(preview)
    if with_handoff_provenance:
        graph["92"] = {"class_type": "MiniMaxH3FastH3V2CurrentHandoffAttestEXPT8", "inputs": {
            "current_recipe": ["83", 0], "low_result": ["13", 2], "high_result": ["29", 2],
            "low_attestation": ["85", 0], "high_attestation": ["87", 0],
            "low_condition_receipt": ["89", 7], "high_condition_receipt": ["90", 7],
            "upscale_receipt": ["91", 4], "upscaled_latent": ["91", 0],
            "upscale_report_json": ["91", 3], "high_template": ["90", 2],
            "high_positive": ["90", 1], "high_source_latent": graph["29"]["inputs"]["latent_image"],
            "segment_index": ["78", 1], "contexts": ["71", 0]}}
        graph["93"] = {"class_type": "PreviewAny", "inputs": {"source": ["92", 2]}}
        outputs.append("93")
    if with_job_binding:
        graph["94"] = {"class_type": "MiniMaxH3FastH3V2CurrentJobBindEXPT8", "inputs": {
            "current_recipe": ["83", 0], "handoff_attestation": ["92", 0],
            "segment_index": ["78", 1], "contexts": ["71", 0]}}
        graph["95"] = {"class_type": "PreviewAny", "inputs": {"source": ["94", 3]}}
        outputs.append("95")
    if with_media_provenance:
        graph["96"] = {"class_type": "MiniMaxH3FastH3V2CurrentMediaPrepareEXPT8", "inputs": {
            "job_binding": ["94", 0], "high_result": ["29", 2],
            "video_vae": ["7", 0], "audio_vae": ["8", 0],
            "start_seconds": ["78", 5], "duration_seconds": ["78", 6], "fps": 24.}}
        graph["97"] = {"class_type": "PreviewAny", "inputs": {"source": ["96", 3]}}
        graph["15"]["inputs"].update(images=["96", 0], audio=["96", 1])
        graph["80"]["inputs"]["source"] = ["96", 3]
        outputs.append("97")
    if with_color_match:
        graph["105"] = {"class_type": "MiniMaxH3FastH3V2ExternalColorMatchEXPT8", "inputs": {
            "media_receipt": ["96", 2], "enabled": True, "mode": "bounded_motion_color_exp"}}
        graph["106"] = {"class_type": "PreviewAny", "inputs": {"source": ["105", 3]}}
        graph["15"]["inputs"].update(images=["105", 0], audio=["105", 1])
        outputs.append("106")
    if with_candidate_save:
        graph["98"] = {"class_type": ("MiniMaxH3FastH3V2ColoredCandidateSaveEXPT8"
                                      if with_color_match else "MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8"),
                       "inputs": {("color_receipt" if with_color_match else "media_receipt"):
                                  (["105", 2] if with_color_match else ["96", 2]), "candidate_id": ""}}
        graph["99"] = {"class_type": "PreviewAny", "inputs": {"source": ["98", 3]}}
        outputs.append("99")
    if with_frozen_low_bundle and not resume_high:
        graph["101"] = {"class_type": "MiniMaxH3FastH3V2FrozenLOWBundleSaveEXPT8", "inputs": {
            "current_recipe": ["83", 0], "low_stage_result": ["13", 2],
            "low_attestation": ["85", 0], "low_condition_receipt": ["89", 7],
            "artifact_path": ["50", 2], "artifact_sha256": ["50", 3]}}
        graph["103"] = {"class_type": "PreviewAny", "inputs": {"source": ["101", 2]}}
        outputs.append("103")
    if resume_high:
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": "REPLACE_WITH_SAVED_LOW_PATH/manifest.json",
            "artifact_sha256": "0" * 64, "expected_stage": "low_0_4"}}
        graph["62"]["inputs"]["low_stage_result"] = ["60", 3]
        graph["22"]["inputs"]["completed_stage"] = ["60", 3]
        graph["91" if with_handoff_provenance else "23"]["inputs"]["av_latent"] = ["62", 0]
        if with_frozen_low_bundle:
            graph["102"] = {"class_type": "MiniMaxH3FastH3V2FrozenLOWBundleLoadEXPT8", "inputs": {
                "artifact_path": graph["60"]["inputs"]["artifact_path"],
                "artifact_sha256": graph["60"]["inputs"]["artifact_sha256"],
                "low_stage_result": ["60", 3]}}
            for item in graph.values():
                for name, value in item["inputs"].items():
                    if isinstance(value, list) and len(value) == 2 and value[0] == "83":
                        item["inputs"][name] = ["102", value[1]]
            if with_handoff_provenance:
                graph["92"]["inputs"].update(low_result=["60", 3],
                    low_attestation=["102", 4], low_condition_receipt=["102", 5])
            graph["104"] = {"class_type": "PreviewAny", "inputs": {"source": ["102", 3]}}
            outputs.append("104")
        outputs = [key for key in outputs if key not in ("72", "74", "86")]
        outputs.append("60")
    else:
        outputs.append("50")
    # Explicit outputs above include the source reports for canvas inspection;
    # private graph pruning must not leave disconnected LOW samplers in resume.
    return _prune(graph, outputs)


def build_candidate(graph, info, *, resume_high=False, with_delivery=False,
                    with_current_recipe=False, with_stage_attestation=False,
                    with_condition_provenance=False, with_handoff_provenance=False,
                    with_job_binding=False, with_media_provenance=False,
                    with_candidate_save=False, with_color_match=False,
                    with_frozen_low_bundle=False):
    if with_current_recipe and "LoadImage" not in info:
        import nodes as core_nodes
        info = {**info, "LoadImage": base.native_info(
            "LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])}
    graph, selected = base.selected_frontend_schema(graph, info)
    title = "FastH3 V2 · Accepted second window · frozen LOW" if resume_high else \
        "FastH3 V2 · Accepted second window · independent LOW/HIGH"
    if with_delivery:
        title += " · verified AV trim preview"
    if with_current_recipe:
        title += " · current editable recipe fingerprint"
    if with_stage_attestation:
        title += " · two current stage input attestations"
    if with_condition_provenance:
        title += " · actual CLIP/VAE condition origin receipts"
    if with_handoff_provenance:
        title += " · accepted LOW/upscale/HIGH handoff proof"
    if with_job_binding:
        title += " · current job/parent SHA match"
    if with_media_provenance:
        title += " · bound HIGH AV decode/68-frame trim"
    if with_color_match:
        title += " · external accepted-picture Color Match"
    if with_candidate_save:
        title += " · unaccepted candidate save"
    if with_frozen_low_bundle:
        title += " · attested frozen LOW load" if resume_high else " · save attested frozen LOW"
    workflow = base.convert(graph, selected, title)
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "第二窗口边界 / Read first",
        "pos": [-1800, -800], "size": [1250, 700], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [
            "仅私有 EXP 候选；必须填写真实已接受直接前段的 chain/candidate/revision/job SHA。"
            "Source 会重验真实 MP4 和 AV context；占位符不能执行。LOW 从已接受 RGB24 尾39帧重编码，"
            "HIGH 用完成 AV；二者分别走原生长视频 Relay 条件、V2 DMD4步、外置 EAV。"
            "旧循环默认 EAV disabled，本图保留独立开关。训练 VSA＋Relay 尚无兼容偏置内核，"
            "因此当前只用 dense_compat_exp；不是训练VSA同质替代。\n\n"
            "本窗口 90 帧包含22帧上下文和68帧新画面，第一窗口应为124帧；两段接受后是192帧/8秒。"
            + ("此图用独立 OutputTrim 裁去22帧重叠并只预览/保存新68帧；父段身份和时间坐标从已接受来源输出，"
             + ("当前作业与父片身份逐项核验；仅保存未接受候选，不自动接受或拼接。"
                if with_candidate_save else
                "但当前编辑后的模型/LoRA/提示词尚无跨段 job 合同认证，不保存候选、不接受/拼接。")
             if with_delivery else "此图只保存原始90帧，不执行父片接受、上下文去重、颜色/音频接缝、拼接或最终8秒交付。")
            + ("当前配方指纹单独绑定两路原始模型/LoRA、组件、首帧、全局Relay、EAV及实际放大回执；"
               "尚非Stage执行或候选接受证明。" if with_current_recipe else "")
            + ("LOW/HIGH另各自核对真实Stage回执与当前原始模型、采样输入、Relay投影和EAV；"
               + ("仍不等于候选接受。" if with_condition_provenance else
                  "仍不证明CLIP/VAE/首帧来源或候选接受。")
               if with_stage_attestation else "")
            + ("条件调用回执绑定实际CLIP、双VAE和Guider；第二窗口不复用首帧，父片来源和HIGH交接另需认证。"
               if with_condition_provenance else "")
            + ("已接受父片上下文及LOW→放大→HIGH输入只读核验；仍未接受候选。"
               if with_handoff_provenance and with_candidate_save else
               "已接受父片上下文及LOW→放大→HIGH输入只读核验；仍不保存或接受候选。"
               if with_handoff_provenance else "")
            + ("当前作业SHA与已接受直接父片的作业SHA必须完全相等；不匹配即拒绝交付。"
               if with_job_binding else "")
            + ("已完成HIGH经一次原生AV解码并精确裁去22帧上下文，画音回执可重验。"
               if with_media_provenance and with_candidate_save else
               "已完成HIGH经一次原生AV解码并精确裁去22帧上下文，画音回执可重验；不保存候选。"
               if with_media_provenance else "")
            + ("来源绑定画音只保存为未接受续段候选；需另外预览并显式接受。"
               if with_candidate_save else "")
            + ("外置 Color Match 读取并核验已接受直接父片视频，仅修新68帧RGB；音频/latent不改。"
               if with_color_match else "")
            + ("只读选定LOW阶段manifest和配方/条件旁证后运行HIGH；不加载或重采LOW。"
               if with_frozen_low_bundle and resume_high else
               "额外保存与已完成LOW阶段manifest绑定的当前配方和条件旁证。"
               if with_frozen_low_bundle else "")
            + "Stage Save/Load 是阶段恢复，不是已接受链交付。旧一体工作流仍保留。"
            "缺完整双段同条件GPU、最终解码、人审和自动接收/拼接资格。"]})
    workflow["last_node_id"] = note_id
    return graph, workflow, base.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-delivery", action="store_true",
                        help="Also export verified final-window AV trim, without candidate acceptance")
    parser.add_argument("--with-current-recipe", action="store_true",
                        help="Also fingerprint the current editable recipe in the full graph only")
    parser.add_argument("--with-stage-attestation", action="store_true",
                        help="Audit both completed StageResults against the current recipe and live inputs")
    parser.add_argument("--with-condition-provenance", action="store_true",
                        help="Use an opt-in conditioning wrapper and bind its actual call to both stage audits")
    parser.add_argument("--with-handoff-provenance", action="store_true",
                        help="Bind accepted parent, completed LOW x0, learned upscale and actual HIGH source")
    parser.add_argument("--with-job-binding", action="store_true",
                        help="Require current completed recipe SHA to equal the accepted direct parent's job SHA")
    parser.add_argument("--with-media-provenance", action="store_true",
                        help="Use one native HIGH AV decode/68-frame trim call and capture exact media provenance")
    parser.add_argument("--with-candidate-save", action="store_true",
                        help="Write an unaccepted source-bound candidate; never auto-accept")
    parser.add_argument("--with-color-match", action="store_true",
                        help="Apply the unchanged accepted-picture RGB Color Match as an external node")
    parser.add_argument("--with-frozen-low-bundle", action="store_true",
                        help="Save an attested LOW recipe sidecar and build a HIGH-only loader variant")
    parser.add_argument("--first-frame", default="",
                        help="Existing Core input-relative first-frame image when fingerprinting the recipe")
    args = parser.parse_args()
    if args.with_stage_attestation and not args.with_current_recipe:
        parser.error("--with-stage-attestation requires --with-current-recipe")
    if args.with_condition_provenance and not args.with_stage_attestation:
        parser.error("--with-condition-provenance requires --with-stage-attestation")
    if args.with_handoff_provenance and not args.with_condition_provenance:
        parser.error("--with-handoff-provenance requires --with-condition-provenance")
    if args.with_job_binding and not args.with_handoff_provenance:
        parser.error("--with-job-binding requires --with-handoff-provenance")
    if args.with_media_provenance and not args.with_job_binding:
        parser.error("--with-media-provenance requires --with-job-binding")
    if args.with_media_provenance and not args.with_delivery:
        parser.error("--with-media-provenance requires --with-delivery")
    if args.with_candidate_save and not args.with_media_provenance:
        parser.error("--with-candidate-save requires --with-media-provenance")
    if args.with_color_match and not args.with_media_provenance:
        parser.error("--with-color-match requires --with-media-provenance")
    if args.with_frozen_low_bundle and not (args.with_current_recipe and args.with_condition_provenance):
        parser.error("--with-frozen-low-bundle requires --with-current-recipe and --with-condition-provenance")
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    if args.with_current_recipe:
        image = (ROOT.parents[1] / "input" / args.first_frame).resolve()
        if not image.is_file() or not image.is_relative_to((ROOT.parents[1] / "input").resolve()):
            parser.error("Recipe first frame must exist inside this Core's input directory")
    info = base.load_live_info()
    import execution
    rows = []
    for resume in ((False, True) if args.with_frozen_low_bundle or not args.with_current_recipe else (False,)):
        graph, workflow, audit = build_candidate(
            continuation_graph(resume_high=resume, with_delivery=args.with_delivery,
                               with_current_recipe=args.with_current_recipe,
                               with_stage_attestation=args.with_stage_attestation,
                               with_condition_provenance=args.with_condition_provenance,
                               with_handoff_provenance=args.with_handoff_provenance,
                               with_job_binding=args.with_job_binding,
                               with_media_provenance=args.with_media_provenance,
                               with_candidate_save=args.with_candidate_save,
                               with_color_match=args.with_color_match,
                               with_frozen_low_bundle=args.with_frozen_low_bundle,
                               first_frame=args.first_frame), info,
            resume_high=resume, with_delivery=args.with_delivery,
            with_current_recipe=args.with_current_recipe,
            with_stage_attestation=args.with_stage_attestation,
            with_condition_provenance=args.with_condition_provenance,
            with_handoff_provenance=args.with_handoff_provenance,
            with_job_binding=args.with_job_binding,
            with_media_provenance=args.with_media_provenance,
            with_candidate_save=args.with_candidate_save,
            with_color_match=args.with_color_match,
            with_frozen_low_bundle=args.with_frozen_low_bundle)
        validation = asyncio.run(execution.validate_prompt("fast-v2-accepted-continuation", deepcopy(graph), None))
        name = "FastH3V2_Accepted_Second_Frozen_LOW" if resume else "FastH3V2_Accepted_Second_Full"
        if args.with_delivery:
            name += "_Final68_Unaccepted"
        if args.with_current_recipe:
            name += "_CurrentRecipe"
        if args.with_stage_attestation:
            name += "_StageAttest"
        if args.with_condition_provenance:
            name += "_ConditionProvenance"
        if args.with_handoff_provenance:
            name += "_HandoffProvenance"
        if args.with_job_binding:
            name += "_CurrentJobBound"
        if args.with_media_provenance:
            name += "_MediaProvenance"
        if args.with_candidate_save:
            name += "_CandidateSave"
        if args.with_color_match:
            name += "_ExternalColorMatch"
        if args.with_frozen_low_bundle:
            name += "_LOWBundleLoad" if resume else "_LOWBundleSave"
        write_new(destination / (name + ".api.json"), graph)
        write_new(destination / (name + ".json"), workflow)
        rows.append({"name": name, "serialization": audit, "core_validation": validation,
                     "all_outputs_valid": bool(validation[0] and not validation[3])})
    static_boundary = ("Frozen LOW recipe bundle and accepted-parent provenance wired but not executed by static validation. "
                       if args.with_frozen_low_bundle else
                       "Candidate save and media/current job/parent provenance wired but not executed by static validation. "
                       if args.with_candidate_save else
                       "Media, current job/parent, handoff, condition and stage provenance wired but not executed by static validation. "
                       if args.with_media_provenance else
                       "Current job/parent, handoff, condition and stage provenance wired but not executed by static validation. "
                       if args.with_job_binding else
                       "Handoff, condition and stage provenance wired but not executed by static validation. "
                       if args.with_handoff_provenance else
                       "Condition provenance and stage-attestation nodes wired but not executed by static validation. "
                       if args.with_condition_provenance else
                       "Stage-attestation nodes wired but not executed by static validation. "
                       if args.with_stage_attestation else
                       "Optional current recipe is not stage-attested. ")
    write_new(destination / "audit.json", {"candidates": rows,
        "qualification": "Static live-Core and frontend serialization only. " + static_boundary
                         + ("Parent equality not executed by static validation. No acceptance, "
                            if args.with_candidate_save else
                            "Parent equality not executed by static validation. No candidate save, "
                            "acceptance, GPU, full two-window assembly, or quality pass."
                            if args.with_job_binding else
                            "No accepted source bound, GPU, first window, candidate acceptance, "
                            "assembly, or quality pass.")})
    if not all(row["all_outputs_valid"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
