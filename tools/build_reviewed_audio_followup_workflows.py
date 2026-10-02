"""Append reusable B8/C1/C2 followup templates without rewriting legacy graphs.

These are generic editable templates, NOT the private, asset-bound canvas
captures used for human review. Cold templates deliberately retain placeholders.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples/workflows"
DESTINATION = WORKFLOWS / "64-reviewed-audio-followup"
OLD_RF = "MiniMaxH3RFRestartStageSetupEXPT8"
JOINT_RF = "MiniMaxH3RFRestartJointClockSetupEXPT8"
AUDIO_PLAN = "MiniMaxH3AudioRefineCompatibilityPlanT8Advanced"
LONG_STEM = "S26_2026-08-29_H3_Audio_Refine_Long_Video_Prompt_Relay_Turbo8_Advanced_EXP"

# Pin the inputs, not a whole dirty checkout. This builder cannot silently
# regenerate a different recipe or overwrite any modified old/new workflow.
SOURCES = {
    "C1_RF_JointClock_Full_Save_EXP.json": (
        "62-rf-restart-split/S29_RF_standalone_save_effects_Separate_EXP.json",
        "a06f9bdac46bd36ffc849505566f90e261ac6aa4c600a784272a6aac455f01e7"),
    "C1_RF_JointClock_Resume_EXP.json": (
        "62-rf-restart-split/S29_RF_standalone_resume_effects_Separate_EXP.json",
        "343da78afe72f9aee011ca539b64d44d97b2301f7475b636aea635d6d58c8081"),
    "C2_RF_DetailMixer_JointClock_Full_Save_EXP.json": (
        "62-rf-restart-split/S29_RF_detail_mixer_save_effects_Separate_EXP.json",
        "5709527c930680889dcd6f1ab542df727020c987d183e61a6a2af172da4c21c9"),
    "C2_RF_DetailMixer_JointClock_Resume_EXP.json": (
        "62-rf-restart-split/S29_RF_detail_mixer_resume_effects_Separate_EXP.json",
        "5d11fd68d5e98c8685ee29b1ed4555e9e07e17deac6659bc673109110726b51d"),
    "B8_LongRelay_035_Freeze_Video_EXP.json": (
        f"59-audio-refine-split/{LONG_STEM}_freeze_video_Separate_EXP.json",
        "4a99fc23361e169d772d49a830bfd851044c4e337d0931ec6dc27fee6565620b"),
    "B8_LongRelay_035_Resume_Audio_EAV_EXP.json": (
        f"59-audio-refine-split/effects/{LONG_STEM}_resume_audio_Separate_EXP_eav_TailEffects.json",
        "06a5b69efa5330c90dd7c17233a2cc53e726297a717072ca9cfb97a17e9aa443"),
}

COMMON_NOTE = (
    "可复用分离式 EXP 模板，不是人审样片的逐值原生画布副本。"
    "2026-10-02 用户通过的是绑定固定素材/权重/同源检查点的 B8/C1/C2 修正版，"
    "不能泛化到本模板默认画布或任意模型。旧图未覆盖。"
    "模型、输入文件、提示词、帧数及时间布局须按自己的任务核实。"
    "冷恢复必须填写对应完整图真实 Save 的路径/manifest/SHA；占位值故意不能执行。"
    "保留原音轨和旧失败记录，不自动 QualityGate 接受或 confirm_save。"
)
RF_NOTE = (
    "显式 JointClock RF：BASE output＋原模板 → Handoff → 单独 RESTART。"
    "RF 已按各自时钟重噪，此新节点避免第二次 audio rebase；"
    "Core anchor/NOISE/masks、原联合 AV、3步/seed/shift/sigma 与效果连接保留。"
    "两阶段 MODEL/LoRA/条件/Relay/EAV 独立；冷图只加载真实 RF BASE 后跑 RESTART。"
    "不要把旧策略 StageContext 当作新策略重启结果使用。"
)
B8_NOTE = (
    "B8 同源4步音频尾采候选 audio_denoise=0.35；不是 RF 节点或通用降噪修复。"
    "第一张只生成/审核视频并显式 Freeze，第二张只读检查点后音频精修。"
    "独立 Long Relay Plan/原时间投影与外置 EAV report_only 均保留。"
    "恢复时三份 Save 值、frame guard、segment/context/chain 与检查点必须一致；"
    "新素材改链名，不能沿用审核用私有检查点。"
    "QualityGate 默认为不接受，需自己试听后再选择候选。"
)


def single(graph, kind):
    matches = [node for node in graph["nodes"] if node["type"] == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {kind}, found {len(matches)}")
    return matches[0]


def transform(source, filename):
    graph = deepcopy(source)
    rf = filename.startswith(("C1_", "C2_"))
    if rf:
        restart = single(graph, OLD_RF)
        restart["type"] = JOINT_RF
        restart["properties"]["Node name for S&R"] = JOINT_RF
        restart["title"] = "RF RESTART · Joint Clock / no double audio rebase"
    elif "Resume_Audio" in filename:
        plan = single(graph, AUDIO_PLAN)
        # Actual schema order: refine_steps, audio_denoise, refine_seed.
        if plan["widgets_values"] != [4, 0.50, 2608290022]:
            raise ValueError("Pinned B8 four-step/denoise/seed contract changed")
        plan["widgets_values"][1] = 0.35
        plan["title"] = "B8 AUDIO TAIL · original 4 steps / denoise 0.35"
    for note in (node for node in graph["nodes"] if node["type"] == "MarkdownNote"):
        # Replace the legacy RF instruction to keep a double rebase. Other
        # recipes retain their exact checkpoint instructions and append scope.
        text = RF_NOTE if rf else note["widgets_values"][0] + "\n\n" + B8_NOTE
        note["widgets_values"] = [text + "\n\n" + COMMON_NOTE]
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "t8:reviewed-audio-template:" + filename))
    graph["extra"]["workflow_title"] = filename.removesuffix(".json")
    return graph


def generated():
    result = {}
    for filename, (relative, expected) in SOURCES.items():
        path = WORKFLOWS / relative
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Pinned legacy input changed: {relative}")
        result[DESTINATION / filename] = transform(json.loads(content), filename)
    return result


def write_missing(expected, *, write=False):
    pending = {}
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified template: {path}")
        else:
            pending[path] = content
    if pending and not write:
        return len(pending)
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Append missing files only")
    args = parser.parse_args()
    expected = generated()
    missing = write_missing(expected, write=args.write)
    print(json.dumps({"templates": len(expected), "missing": missing,
                      "legacy_overwritten": False, "human_review_captures": False}))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
