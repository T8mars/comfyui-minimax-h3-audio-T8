"""Build the editable local SFX comparison; no models, generation or downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import build_modular_fast_h3_v2_workflow as shared
from tools.build_nov05_core_examples import build_candidate


ROOT = Path(__file__).resolve().parents[1]
NAME = "Local_SFX_AB_EXP"
MEDIA = {
    "source_video": "TaoMateHD_2LOW_1HIGH_5s_00001_.mp4",
    "source_audio": "source_original.wav",
    "external_audio": "external_mmaudio.wav",
    "h3_audio": "h3_candidate.wav",
}
NOTE = """局部音效三选对照 / 未发布 EXP

先上传四个真实文件：原片视频、source_original.wav、external_mmaudio.wav、h3_candidate.wav。后三项是已生成的 PCM，图中不加载 MMAudio、不生成 H3，也不需要 VHS。未带素材的示例需要重新选择文件，不能用占位声音代替人审。

本例固定为 24 fps、5 秒原片。源 PCM 先精确裁至 [0,5) 秒，去掉 AAC 解码尾部 padding，再把同一个结果同时接 source_audio 与 protected_voice_audio：这里保护的是原片整条音轨（对白和环境声），不是分离出来的人声。在原声之上只于 1.2–3.5 秒添加候选，两个候选都从 1.2 秒取声，边界淡化 0.01 秒。使用 target_rms_ceiling 只衰减不放大，候选 RMS 上限 -56 dBFS；这是本段轻微衣袖声音的低电平示例，换素材应调整。

selection 三选：original=原片原声；external=原声+MMAudio；h3=原声+H3。默认 original。右侧三条对照始终可播放，另一个“所选成片”输出跟随 selection。画面全部来自同一个 GetVideoComponents，绝不换成 H3 音效候选视频。Safe AV Save 固定 24 fps；换其它帧率素材需改接与其匹配的保存节点。

窗口外及保护音轨的精确保留指编码前 PCM；重新编码 AAC 不是逐字节无损。RMS 上限不等于感知响度。旧例强制归一到 -30 dBFS，曾把该 H3 候选放大约66.8倍；本例保留安静候选的原生电平。电平修正不能修复模型本身的音色或内容，仍需听审；本图不作人声分离或自动优胜判断。详细连接与素材来源见 docs/NOV05_LOCAL_SFX.md。"""


def recipe():
    def node(kind, title, **inputs):
        return {"class_type": kind, "inputs": inputs, "_meta": {"title": title}}

    graph = {
        "1": node("LoadVideo", "01 原片画面 · 5 秒 / 24 fps", file=MEDIA["source_video"]),
        "2": node("GetVideoComponents", "原片解码 · 三路共享同一画面", video=["1", 0]),
        "3": node("LoadAudio", "02 原片整条音轨 · 不是分离人声", audio=MEDIA["source_audio"]),
        "4": node("TrimAudioDuration", "源 PCM 精确裁至 5 秒 · 去除尾部 padding",
                  audio=["3", 0], start_index=0., duration=5.),
        "5": node("LoadAudio", "03 MMAudio 真实 PCM 候选", audio=MEDIA["external_audio"]),
        "6": node("LoadAudio", "04 H3 真实 PCM 候选 · 不替换画面", audio=MEDIA["h3_audio"]),
        "7": node("MiniMaxH3LocalSFXCandidatesT8", "选择原声 / MMAudio / H3 · 原声整轨保护",
                  source_audio=["4", 0], external_sfx_audio=["5", 0],
                  start_seconds=1.2, end_seconds=3.5, selection="original",
                  level_policy="target_rms_ceiling", target_rms_dbfs=-56.,
                  candidate_start_seconds=1.2, fade_seconds=.01, headroom_policy="none",
                  protected_voice_audio=["4", 0], h3_sfx_audio=["6", 0]),
    }
    for key, label, audio, suffix in (
        ("8", "A · 原片原声", ["4", 0], "original"),
        ("9", "B · 原声 + MMAudio 局部音效", ["7", 1], "external"),
        ("10", "C · 原声 + H3 局部音效", ["7", 2], "h3"),
        ("11", "所选成片 · 跟随 selection", ["7", 0], "selected"),
    ):
        graph[key] = node("MiniMaxH3SafeAVSaveT8Advanced", label, images=["2", 0],
                          audio=audio, filename_prefix="Nov05/SFX_"+suffix+"_5s",
                          crf=18, preserve_audio_length=False)
    return graph


def load_live_info():
    info = shared.load_live_info()
    import nodes
    from comfy_extras.nodes_audio import LoadAudio, TrimAudioDuration
    from comfy_extras.nodes_video import GetVideoComponents, LoadVideo
    for cls in (LoadAudio, TrimAudioDuration, GetVideoComponents, LoadVideo):
        name = cls.define_schema().node_id
        nodes.NODE_CLASS_MAPPINGS[name] = cls
        info[name] = shared.native_info(name, cls)
    return json.loads(json.dumps(info))


def build(info):
    source = recipe()
    used = {node["class_type"] for node in source.values()}
    graph, workflow, _, selected = build_candidate(source, {name: info[name] for name in used}, NAME, NOTE)
    layout = {
        1: (0, 0, 360, 300), 2: (400, 0, 360, 180),
        3: (0, 380, 360, 200), 4: (400, 380, 360, 200),
        5: (0, 670, 360, 200), 6: (0, 960, 360, 200),
        7: (830, 380, 430, 440),
        8: (1330, 0, 400, 330), 9: (1770, 0, 400, 330),
        10: (2210, 0, 400, 330), 11: (1330, 430, 400, 330),
    }
    for node in workflow["nodes"]:
        if node["id"] in layout:
            x, y, width, height = layout[node["id"]]
            node.update(pos=[x, y], size=[width, height])
        else:
            node.update(pos=[830, 900], size=[1240, 460])
    workflow["extra"]["ds"] = {"scale": .65, "offset": [80, 100]}
    return graph, workflow, shared.audit_candidate(graph, workflow, selected), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    if evidence.exists() or not evidence.is_relative_to((ROOT / "artifacts").resolve()):
        parser.error("Use a new evidence directory under this checkout artifacts/")
    path = args.output / (NAME + ".json")
    if path.exists():
        raise FileExistsError(path)
    graph, workflow, audit, selected = build(load_live_info())
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True)
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for suffix, value in (("api", graph), ("schema", selected), ("receipt", {
        "serialization": audit, "media_execution": False, "human_quality_reviewed": False,
        "note": "Core prompt validation with actual files and native UI roundtrip are separate checks.",
    })):
        (evidence / (NAME + "." + suffix + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
