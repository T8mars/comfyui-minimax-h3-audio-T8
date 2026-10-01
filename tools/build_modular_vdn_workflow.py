"""Independent VDN full/tail candidates; no GPU queue or legacy JSON overwrite."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_modular_fast_h3_v2_workflow as shared  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ("stage_dmd_8nfe", "stage_b_50nfe")
VARIANTS = ("minimal", "eav", "save_eav", "resume_eav")


def split_graph(training=TRAINING[0], variant="minimal", *, refine_steps=4,
                artifact_path="REPLACE_WITH_LOW/manifest.json", artifact_sha256="0"*64):
    if (training not in TRAINING or variant not in VARIANTS
            or type(refine_steps) is not int or refine_steps not in (4, 5)
            or (training == "stage_dmd_8nfe" and refine_steps != 4)):
        raise ValueError("Unknown VDN training/workflow variant")
    graph = shared.split_graph()
    for key in ("1", "22"):
        graph[key]["inputs"]["unet_name"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    for key, base in (("70", "1"), ("71", "22")):
        graph[key] = {"class_type": "MiniMaxH3VDNModelComposerT8Advanced", "inputs": {
            "model": [base, 0], "vdn_root": "OpenVDN/vdn-minimax-h3", "stage": training,
            "verify_hashes": True, "allow_structural_base": True}}
    for key, composer, latent, stage in (("10", "70", ["9", 1], "vdn_complete"),
                                        ("26", "71", ["25", 0], "vdn_refine")):
        graph[key] = {"class_type": "MiniMaxH3VDNStageSetupEXPT8", "inputs": {
            "model": [composer, 0], "av_latent": latent, "stage": stage,
            "refine_steps": refine_steps if stage == "vdn_refine" else 4}}
    for key in ("19", "20", "30", "31"):
        graph.pop(key)
    graph["25"]["inputs"].update(audio_policy="first_pass", second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    if variant != "minimal":
        for key in ("41", "42"):
            graph[key] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
                "mode": "report_only", "tau": 4., "start_video_progress": .15,
                "end_video_progress": .9, "max_workspace_mib": 32, "g_hard_limit": 1.5}}
        for key, setup, config, latent in (("43", "10", "41", ["9", 1]), ("44", "26", "42", ["25", 0])):
            graph[key] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
                "model": [setup, 0], "sigmas": [setup, 2], "av_latent": latent,
                "stage_context": [setup, 3], "eav_config": [config, 0]}}
        graph["12"]["inputs"]["model"] = ["43", 0]
        graph["28"]["inputs"]["model"] = ["44", 0]
        for key, sampled, effect in (("45", "13", "43"), ("46", "29", "44")):
            graph[key] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
                "av_latent": [sampled, 0], "runtime": [effect, 1]}}
    if variant in ("save_eav", "resume_eav"):
        # shared helper expects old V2 audit30 solely to route the saved HIGH;
        # retain our explicit VDN graph and build these four nodes directly.
        for sampler, setup, key, prefix in (("13", "10", "50", "LOW"), ("29", "26", "51", "HIGH")):
            graph[sampler]["class_type"] = "MiniMaxH3StageSamplerEXPT8"
            graph[sampler]["inputs"]["stage_context"] = [setup, 3]
            graph[key] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
                "stage_result": [sampler, 2], "prefix": f"VDN/{training}/{prefix}"}}
        graph["45"]["inputs"]["av_latent"] = ["50", 0]
        graph["46"]["inputs"]["av_latent"] = ["51", 0]
    completed = ["13", 0] if variant == "minimal" else ["45", 0]
    graph["23"]["inputs"]["av_latent"] = completed
    graph["26"]["inputs"]["first_pass_latent"] = completed
    graph["94"] = {"class_type": "MiniMaxH3TwoPassAudioAuditT8Advanced", "inputs": {
        "second_pass_input": ["25", 0], "second_pass_output": ["29", 0] if variant == "minimal" else ["46", 0],
        "expected_audio_strength": 0., "fail_on_locked_mismatch": True, "locked_atol": 1e-5}}
    graph["14"]["inputs"]["av_latent"] = ["94", 0]
    graph["72"] = {"class_type": "PreviewAny", "inputs": {"source": ["94", 1]}}
    graph["16"]["inputs"]["filename_prefix"] = f"MiniMaxH3/Modular_VDN_{training}_EXP"
    if variant == "resume_eav":
        graph["60"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
            "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "expected_stage": "vdn_complete"}}
        graph["61"] = {"class_type": "PreviewAny", "inputs": {"source": ["60", 4]}}
        graph["23"]["inputs"]["av_latent"] = ["60", 0]
        graph["26"]["inputs"]["first_pass_latent"] = ["60", 0]
        keep = set()
        def include(key):
            if key in keep:
                return
            keep.add(key)
            for value in graph[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    include(str(value[0]))
        for key in ("16", "32", "51", "61", "72"):
            include(key)
        graph = {key: value for key, value in graph.items() if key in keep}
    return graph


def build_candidate(training, variant, info):
    graph, workflow, _ = shared.build_candidate(split_graph(training, variant), info)
    mapping = {key: index + 1 for index, key in enumerate(graph)}
    titles = {"9": "LOW VDN conditioning", "24": "HIGH actual-size conditioning",
        "10": "VDN full original training trajectory", "26": "VDN fresh tail on its own grid",
        "13": "PASS 1 — completed OUTPUT", "29": "PASS 2 — own-stage tail",
        "25": "Original reconcile — completed first-pass audio",
        "43": "LOW external VDN EAV", "44": "HIGH external VDN EAV",
        "70": "LOW VDN Composer — independent MODEL / content LoRA",
        "71": "HIGH VDN Composer — independent MODEL / content LoRA",
        "94": "Original completed-audio audit / relock", "45": "Actual LOW VDN effect audit"}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for key, title in titles.items():
        if key in mapping:
            by_id[mapping[key]]["title"] = title
    for key, position in (("70", (-400, 0)), ("71", (-400, 1300)), ("94", (3370, 1500)), ("72", (3750, 1500))):
        if key in mapping:
            by_id[mapping[key]]["pos"] = list(position)
    note = (f"VDN {training}完整首采→原learned3D→原Reconcile→独立VDN尾段，非4+4/LBH。\n\n"
        "两路MODEL/Composer/内容LoRA、条件、NOISE独立。首采接完成OUTPUT；HIGH尺寸来自放大器，"
        "默认保留完成音频，并在解码前原AudioAudit校验重锁。EAV两份配置默认report_only，实际保留VDN窗口和linear。\n\n"
        "本图只配置EAV，不含VDN Relay；外置Relay图使用build_modular_vdn_relay_workflow.py的专属Apply/Audit。"
        "已识别branch普通加载路径有保存/恢复，未知patch和low-VRAM side-model身份仍未认证。"
        "恢复图真实移除LOW链，只读完成LOW槽0，填写path和SHA后只跑HIGH。"
        "随机tiny CPU/Core/序列化候选，不代表完整权重/learned/GPU/浏览器/人审；旧图不动。")
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["widgets_values"] = [note]
    workflow["extra"]["workflow_title"] = f"VDN {training} full + own tail — {variant} EXP"
    _, selected = shared.selected_frontend_schema(graph, info)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new private artifacts directory")
    info = shared.load_live_info()
    import execution
    records = []
    for training in TRAINING:
        for variant in VARIANTS:
            graph, workflow, audit = build_candidate(training, variant, info)
            validation = asyncio.run(execution.validate_prompt("vdn-stage-candidate", deepcopy(graph), None))
            name = f"VDN_{training}_{variant}_EXP"
            shared.write_new(destination / f"{name}.api.json", graph)
            shared.write_new(destination / f"{name}.json", workflow)
            records.append({"name": name, "serialization": audit, "core_validation": validation})
    shared.write_new(destination / "audit.json", {"candidates": records,
        "qualification": "Core/serialization only; not browser, GPU or human. This EAV-only graph omits the separate VDN Relay adapter."})
    shared.write_new(destination / "object-info.json", info)
    failed = [item["name"] for item in records if not item["core_validation"][0]]
    print(json.dumps({"candidates": len(records), "failed": failed, "output": str(destination)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
