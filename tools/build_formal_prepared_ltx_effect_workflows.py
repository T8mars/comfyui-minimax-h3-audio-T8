"""Add six independent Prepared LTX EAV/Relay full and cold-decode EXP graphs.

Existing S28 graphs are neither read as templates nor overwritten. Paths and
hashes are intentionally empty local prerequisites, not shipped private assets.
"""
from __future__ import annotations

import argparse
import json
import uuid

from tools.build_formal_prepared_ltx_split_workflows import DESTINATION, _node_info
from tools.build_modular_prepared_ltx_workflows import build_prompt as base_prompt
from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs

EFFECTS = ("EAV", "Relay", "EAV_Relay")


def build_prompt(*, effect, resume=False, bundle_path="", lease_path="",
                 expected_sha256="", local_prompts="", text_encoder_name=None):
    if effect not in EFFECTS:
        raise ValueError("Choose an explicit Prepared LTX effect graph")
    graph = base_prompt(resume=resume, bundle_path=bundle_path,
                        lease_path=lease_path, expected_sha256=expected_sha256)
    bind = {"prepared_bundle": ["1", 0], "relay_cache_manifest_path": "",
            "relay_mode": "report_only", "relay_max_workspace_mib": 32}
    if "EAV" in effect:
        graph["6"] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
            "mode": "report_only", "tau": 4., "start_video_progress": .15,
            "end_video_progress": .90, "max_workspace_mib": 32, "g_hard_limit": 1.5}}
        bind["eav_config"] = ["6", 0]
    if "Relay" in effect:
        if not isinstance(text_encoder_name, str) or not text_encoder_name:
            raise ValueError("Choose the actual text_encoders menu filename")
        graph["8"] = {"class_type": "MiniMaxH3PreparedLTXRelayPlanEXPT8", "inputs": {
            "prepared_bundle": ["1", 0], "local_prompts": local_prompts,
            "timing_mode": "auto_equal", "time_ranges": "", "epsilon": .1,
            "allow_gaps": False, "allow_overlaps": False}}
        graph["9"] = {"class_type": "MiniMaxH3PreparedLTXRelayEncodeEXPT8", "inputs": {
            "prepared_bundle": ["1", 0], "ltx_relay_plan": ["8", 0],
            "text_encoder_name": text_encoder_name, "cache_id": "prepared_ltx_relay_01",
            "resume_existing": True, "serial_lease_path": lease_path}}
        bind.update(ltx_relay_plan=["8", 0], relay_cache_manifest_path=["9", 0])
    graph["7"] = {"class_type": "MiniMaxH3PreparedLTXEffectsBindEXPT8", "inputs": bind}
    for node_id in ("2", "3"):
        graph[node_id]["inputs"]["prepared_bundle"] = ["7", 0]
    return graph


def _info():
    info = _node_info()
    from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
    from h3_audio_t8_pkg.modular_sampling.prepared_ltx_effects_nodes import NODES as effects
    from h3_audio_t8_pkg.modular_sampling.prepared_ltx_relay_cache_nodes import NODES as encoder
    for cls in (MiniMaxH3StageEAVConfigEXPT8, *effects, *encoder):
        value = cls.GET_NODE_INFO_V1()
        value["cnr_id"] = "minimax-h3-audio-T8"
        info[cls.define_schema().node_id] = value
    return info


def build_workflow(info, *, effect, resume=False, **paths):
    options = info["MiniMaxH3PreparedLTXRelayEncodeEXPT8"]["input"]["required"]["text_encoder_name"][1]["options"]
    native = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    selected = native if native in options else (options[0] if options else None)
    graph = build_prompt(effect=effect, resume=resume,
                         text_encoder_name=paths.pop("text_encoder_name", selected), **paths)
    title = f"Prepared LTX / {effect} External / {'Cold Decode' if resume else 'Generate and Decode'} / EXP"
    workflow = convert(graph, info, title)
    normalize_native_widget_inputs(workflow)
    positions = {"MiniMaxH3PreparedGenerationBundleEXPT8": [0, 0],
        "MiniMaxH3PreparedLTXRelayPlanEXPT8": [0, 420],
        "MiniMaxH3PreparedLTXRelayEncodeEXPT8": [660, 420],
        "MiniMaxH3StageEAVConfigEXPT8": [660, 0],
        "MiniMaxH3PreparedLTXEffectsBindEXPT8": [1320, 180],
        "MiniMaxH3PreparedLTXGenerateEXPT8": [1980, 0],
        "MiniMaxH3PreparedLTXLoadGenerationEXPT8": [1980, 0],
        "MiniMaxH3PreparedLTXDecodeEXPT8": [2640, 0]}
    for node in workflow["nodes"]:
        node["pos"] = positions[node["type"]]
        node["size"] = [580, 350]
    note = (
        "## Prepared LTX 外置效果与独立阶段\n\n"
        "只接受匹配的 Prepared LTX bundle，不自动转换任意 H3 latent。填写本机 bundle、"
        "绝对 serial lease 路径，改变模型/提示词/效果/实现后使用新的 chain_id 与 cache_id。"
        "旧图与原 bundle 不修改；Plan/Encode 接原 bundle，Generate/Load/Decode 接 Effects Bind 输出。\n\n"
        "EAV 和 Relay 默认 report_only，不偷偷施加效果。Relay local_prompts 留空为显式 global-only"
        "旁路，不编码/不加局部效果。填真实局部事件后，独立 Encode 使用匹配的原生 INT8 Gemma/"
        "connector，需64GiB空闲主存和2GiB GPU运行预留；正常 text_encoders 下拉不是任意CLIP兼容认证。"
        "缓存命中不重编码，坏SHA不会自动重建。compiled_prompt不是编码特征。\n\n"
        + ("冷图无Generate：复制原生成的准确 SHA；同配置的Encode命中缓存，Load只读已有latent，"
           "Decode只解码保留原声。不要改旧state身份。" if resume else
           "Generate只采样保存完成latent；Decode只解码，原声不替换成新联合音频。")
        + "\n\n单个真实组合输入已完成原生生成/冷解码/第三缓存及完整画音机械验收；"
        "不代表本图所有控件值、任意素材/后端、人审画质或一般加速。"
    )
    nid = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": nid, "type": "MarkdownNote", "title": "必读：外置效果和缓存边界",
        "pos": [1980, 430], "size": [1190, 650], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = nid
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL,
        f"t8:prepared-ltx-effects:20260930:{effect}:{resume}"))
    workflow["extra"].update(prepared_route="ltx_refine", external_effect=effect,
        split_stage="decode_only" if resume else "generation_then_decode")
    return workflow


def generated():
    info = _info()
    return {DESTINATION / f"S28_Prepared_LTX_{effect}_{'DecodeOnly' if resume else 'Generate_Decode'}_External_EXP.json":
            build_workflow(info, effect=effect, resume=resume)
            for effect in EFFECTS for resume in (False, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args()
    pending = {}
    for path, graph in generated().items():
        raw = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite modified S28 effects graph: {path}")
        if not path.exists():
            pending[path] = raw
    if pending and not options.write:
        print(f"S28 effects: {len(pending)} missing; no files changed")
        return 1
    for path, raw in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    print(f"S28 effects: six verified, {len(pending)} created; no sampling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
