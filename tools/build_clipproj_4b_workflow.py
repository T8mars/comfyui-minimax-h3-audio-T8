from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples"
    / "workflows"
    / "12-system-memory"
    / "2026-08-22_H3_ClipProj_8B_T2VA_Bridge_Advanced_EXP.json"
)
TARGET = (
    ROOT
    / "examples"
    / "workflows"
    / "12-system-memory"
    / "2026-08-23_H3_ClipProj_4B_T2VA_Bridge_Advanced_EXP.json"
)
SOURCE_SHA256 = "67BBF9CD42583BB1805F66B832743585AB835EEC783AF648C8289F12F18610E7"
ENCODER_REVISION = "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96"
ENCODER_SHA256 = "54BD5144DF0BBC25DD6CCADFCB826B521445A1B06AE5A42570BDD2974CA87094"
PROJECTION_REVISION = "2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808"
PROJECTION_SHA256 = "0184E5C8D666A131962506D21949C2D8A8C6F33445B7B5E347E9A7E0A5BAA819"
RUNTIME_RUN_ID = "20260823-213048-8066329d"
RUNTIME_MEDIA_SHA256 = "839442EB88BC05C5433A579BC55C0E34803AEC45DCF7979B20EB3CA80B035E5A"
RUNTIME_VIDEO_SHA256 = "5078716547E0CB863BCD58387524A1C3E11D975096EE0CAD8FC59D3066BF18B1"
RUNTIME_AUDIO_SHA256 = "DC7D75BF49B34405F1159DBC9E1208555404F51C1D413A63D884FDBA8D3BE2C7"


def _canonical_workflow_sha256(path: Path) -> str:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        workflow,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _single_node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node.get("type") == node_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {node_type!r} node, found {len(matches)}")
    return matches[0]


def build() -> dict:
    observed_sha = _canonical_workflow_sha256(SOURCE)
    if observed_sha != SOURCE_SHA256:
        raise ValueError(
            "the reviewed 8B source workflow changed; audit it and update the locked SHA "
            f"before regenerating the 4B derivative (expected {SOURCE_SHA256}, got {observed_sha})"
        )

    workflow = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    loader = _single_node(workflow, "CLIPLoader")
    apply = _single_node(workflow, "ClipProjApply")
    audit = _single_node(workflow, "MiniMaxH3ClipProjCompatibilityAuditT8Advanced")
    conditioning = _single_node(workflow, "MiniMaxH3AudioConditioningT8")
    combine = _single_node(workflow, "VHS_VideoCombine")

    if loader["widgets_values"] != [
        "qwen3vl_8b_fp8_scaled.safetensors",
        "boogu",
        "default",
    ]:
        raise ValueError("the reviewed 8B CLIP loader contract changed")
    if apply["widgets_values"] != ["mmh3-8b-ClipProj-v3.1.safetensors"]:
        raise ValueError("the reviewed 8B projection contract changed")
    if audit["widgets_values"] != [
        "8B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-8b-ClipProj-v3.1.safetensors",
        False,
        False,
        "block_hard_conflicts",
    ]:
        raise ValueError("the reviewed 8B audit contract changed")

    loader["widgets_values"] = [
        "qwen3vl_4b_fp8_scaled.safetensors",
        "krea2",
        "default",
    ]
    apply["title"] = "External ClipProj 0.1.13+ - 4B projection"
    apply["widgets_values"] = ["mmh3-4b-ClipProj-v3.1.safetensors"]
    audit["title"] = "Fail-closed 4B to 5120 H3 conditioning contract"
    audit["widgets_values"] = [
        "4B",
        "qwen3_vl",
        "fp8",
        "stock_pageable",
        "mmh3-4b-ClipProj-v3.1.safetensors",
        False,
        False,
        "block_hard_conflicts",
    ]

    # This is deliberately a bounded smoke graph. It does not replace the reviewed
    # 736x416x124 8B example or any native 32B workflow.
    conditioning["widgets_values"][1:4] = [256, 256, 22]
    combine["widgets_values"]["filename_prefix"] = "MiniMaxH3/clipproj_4b_t2va_smoke"

    notes = {
        node.get("title"): node
        for node in workflow["nodes"]
        if node.get("type") == "MarkdownNote"
    }
    notes["1. Actual assets and installation"]["widgets_values"] = [
        "## 4B ClipProj独立桥接\r\n\r\n"
        "需要单独安装 nicolab28/ComfyUI-ClipProj >= 0.1.13 并重启ComfyUI。"
        "编码器必须是含视觉塔的 qwen3vl_4b_fp8_scaled.safetensors，不能使用同宽度的纯文本"
        "qwen_3_4b.safetensors。编码器固定HF revision "
        f"{ENCODER_REVISION}，SHA-256 {ENCODER_SHA256}。矩阵固定HF revision "
        f"{PROJECTION_REVISION}，SHA-256 {PROJECTION_SHA256}，结构为2560到5120。"
    ]
    notes["2. Scientific boundary and A/B"]["widgets_values"] = [
        "## 单条真实运行已通过，仍需质量对照\r\n\r\n"
        "这里使用官方Load CLIP的pageable路径，再由外部ClipProj Apply投影2560到5120。"
        "审计节点设为block_hard_conflicts，维度、版本、架构或纯文本编码器不合就停止。"
        "2026-08-23已在8188未运行且空闲显存13,649MiB时完成一条固定seed的"
        "256x256x22、4 NFE T2VA：43.812秒，峰值15,015MiB、最低余量1,095MiB，结束后"
        "显存只比基线高14MiB；22帧H.264与32kHz双声道AAC严格解码通过，音频无NaN/Inf。"
        f"成片SHA-256为{RUNTIME_MEDIA_SHA256}。该PASS只证明这套资产和短链能运行，仍应"
        "与8B和原生32B固定参数对照并完整试听。不能外推中文、专名、对白、参考媒体、画质、"
        "速度、显存收益、重复稳定性或普遍16GB安全。"
    ]

    workflow.setdefault("extra", {})["t8_clipproj_4b"] = {
        "status": "ASSET_AND_SINGLE_T2VA_RUNTIME_PASS",
        "source_workflow_sha256": SOURCE_SHA256,
        "encoder_revision": ENCODER_REVISION,
        "encoder_sha256": ENCODER_SHA256,
        "projection_revision": PROJECTION_REVISION,
        "projection_sha256": PROJECTION_SHA256,
        "runtime_reason": "ONE_FIXED_256X256X22_4NFE_T2VA_PASS",
        "runtime_evidence": {
            "run_id": RUNTIME_RUN_ID,
            "seed": 123456789,
            "elapsed_seconds": 43.812,
            "peak_used_mib": 15015,
            "minimum_free_mib": 1095,
            "final_minus_baseline_mib": 14,
            "media_sha256": RUNTIME_MEDIA_SHA256,
            "decoded_video_sha256": RUNTIME_VIDEO_SHA256,
            "decoded_audio_sha256": RUNTIME_AUDIO_SHA256,
            "strict_media_contract": "22x256x256 H264 plus 32kHz stereo AAC",
        },
    }
    return workflow


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write(*, install: bool = False) -> list[Path]:
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    targets = [TARGET]
    if install:
        comfy_root = ROOT.parents[1]
        targets.append(
            comfy_root
            / "user"
            / "default"
            / "workflows"
            / "MiniMax H3 T8"
            / "12-system-memory"
            / TARGET.name
        )
    for target in targets:
        _atomic_write(target, payload)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reviewed ClipProj 4B T2VA frontend workflow."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="also copy the generated frontend workflow into the local ComfyUI user menu",
    )
    args = parser.parse_args()
    for path in write(install=args.install):
        print(path)


if __name__ == "__main__":
    main()
