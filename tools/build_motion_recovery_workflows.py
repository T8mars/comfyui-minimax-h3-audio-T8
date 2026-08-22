from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows" / "07-motion-detail"
BASE = WORKFLOW_DIR / "2026-08-21_H3_Enhance_A_Video_FETA_Stock20_Advanced_EXP.json"
INSTALLED_DIR = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "07-motion-detail"
)

PROMPT = (
    "Night, one continuous high-speed cinematic shot. An adult woman in flowing red Hanfu "
    "launches across a moonlit rooftop and spins rapidly through the air while the camera "
    "whip-pans with her. Between 1.5 and 3.5 seconds she completes the fastest turn and sword "
    "swing. Preserve coherent anatomy, stable limbs, crisp fabric folds and full motion amplitude. "
    "She says clearly in Mandarin exactly once: <d>你在干嘛呢，我在这里呀，看看效果如何。</d> "
    "No repeated words, no subtitles. Synchronized voice, rushing wind, cloth movement, sword "
    "whooshes, sparks and distant night ambience."
)


def _clone(source: dict, node_id: int, title: str, pos: tuple[int, int]) -> dict:
    node = deepcopy(source)
    node["id"] = node_id
    node["title"] = title
    node["pos"] = list(pos)
    node["order"] = node_id - 1
    node["mode"] = 0
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = [] if item.get("links") is not None else None
    return node


def _linked(name: str, value_type: str) -> dict:
    return {"name": name, "type": value_type, "link": None}


def _widget(name: str, value_type: str) -> dict:
    return {
        "name": name,
        "type": value_type,
        "widget": {"name": name},
        "link": None,
    }


def _output(name: str, value_type: str) -> dict:
    return {"name": name, "type": value_type, "links": []}


def _custom(
    node_id: int,
    node_type: str,
    title: str,
    pos: tuple[int, int],
    size: tuple[int, int],
    inputs: list[dict],
    outputs: list[dict],
    widgets: list,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets,
    }


def _note(node_id: int, title: str, text: str, pos: tuple[int, int]) -> dict:
    return _custom(
        node_id,
        "MarkdownNote",
        title,
        pos,
        (620, 330),
        [],
        [],
        [text],
    )


def _connect(workflow: dict, source: dict, output_slot: int, target: dict, input_slot: int) -> None:
    source_output = source["outputs"][output_slot]
    target_input = target["inputs"][input_slot]
    if source_output["type"] != target_input["type"]:
        raise ValueError(
            f"link type mismatch {source['type']}:{source_output['type']} -> "
            f"{target['type']}:{target_input['type']}"
        )
    link_id = workflow["last_link_id"] + 1
    workflow["last_link_id"] = link_id
    workflow["links"].append(
        [
            link_id,
            source["id"],
            output_slot,
            target["id"],
            input_slot,
            source_output["type"],
        ]
    )
    source_output.setdefault("links", [])
    source_output["links"].append(link_id)
    target_input["link"] = link_id


def _motion_nodes(windowed: bool) -> dict[str, dict]:
    plan_type = "H3_T8_MOTION_RECOVERY_PLAN"
    analyzer = _custom(
        12,
        "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
        "P0 · Analyze pass-1 motion / 分析动作过载",
        (3300, 100),
        (540, 510),
        [
            _linked("av_latent", "LATENT"),
            _linked("frames", "IMAGE"),
            _widget("mode", "COMBO"),
            _widget("manual_ranges", "STRING"),
            _widget("overload_threshold", "FLOAT"),
            _widget("minimum_profile_contrast", "FLOAT"),
            _widget("minimum_residual_motion", "FLOAT"),
            _widget("max_hold", "INT"),
            _widget("bridge_tokens", "INT"),
            _widget("minimum_hot_frames", "INT"),
            _widget("fps", "FLOAT"),
        ],
        [
            _output("motion_plan", plan_type),
            _output("heat_preview", "IMAGE"),
            _output("should_repair", "BOOLEAN"),
            _output("expanded_frames", "INT"),
            _output("report_json", "STRING"),
        ],
        ["auto_conservative_exp", "", 3.5, 2.0, 0.01, 3, 2, 5, 24.0],
    )
    prepare = _custom(
        14 if windowed else 13,
        "MiniMaxH3MotionRetimingPrepareT8Advanced",
        "P1 · Expand selected time / 扩时并编码二采初值",
        (4700 if windowed else 3980, 80),
        (540, 330),
        [
            _linked("frames", "IMAGE"),
            _linked("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
            _linked("video_vae", "VAE"),
            _widget("audio_seed_mode", "COMBO"),
            _linked("audio_vae", "VAE"),
            _linked("source_audio", "AUDIO"),
        ],
        [
            _output("av_latent", "LATENT"),
            _output("smeared_frames", "IMAGE"),
            _output("smeared_audio_seed", "AUDIO"),
            _output("motion_plan", plan_type),
            _output("report_json", "STRING"),
        ],
        ["follow_original_0p5"],
    )
    composer = _custom(
        15 if windowed else 14,
        "MiniMaxH3MotionRecoveryComposerT8Advanced",
        "P1 · Partial second-pass schedule / 二采后段Sigma",
        (5340 if windowed else 4620, 100),
        (500, 300),
        [
            _linked("av_latent", "LATENT"),
            _linked("sigmas", "SIGMAS"),
            _linked("motion_plan", plan_type),
            _widget("mode", "COMBO"),
            _widget("denoise_fraction", "FLOAT"),
            _widget("minimum_second_pass_nfe", "INT"),
        ],
        [
            _output("av_latent", "LATENT"),
            _output("sigmas", "SIGMAS"),
            _output("motion_plan", plan_type),
            _output("actual_second_pass_nfe", "INT"),
            _output("report_json", "STRING"),
        ],
        ["apply_exp", 0.48, 2],
    )
    recover = _custom(
        21 if windowed else 20,
        "MiniMaxH3MotionRecoverAVT8Advanced",
        "P1 · Recover original clock / 回到原始音画时钟",
        (7350 if windowed else 6600, 80),
        (520, 330),
        [
            _linked("generated_frames", "IMAGE"),
            _linked("generated_audio", "AUDIO"),
            _linked("pass1_audio", "AUDIO"),
            _linked("motion_plan", plan_type),
            _widget("audio_mode", "COMBO"),
            _widget("pass1_mix", "FLOAT"),
        ],
        [
            _output("frames", "IMAGE"),
            _output("audio", "AUDIO"),
            _output("report_json", "STRING"),
        ],
        ["pass1_original", 0.8],
    )
    output = {"analyzer": analyzer, "prepare": prepare, "composer": composer, "recover": recover}
    if windowed:
        output["segment"] = _custom(
            13,
            "MiniMaxH3MotionSegmentPlanT8Advanced",
            "P2 · Select one budgeted window / 选择一个预算窗口",
            (4000, 80),
            (560, 430),
            [
                _linked("frames", "IMAGE"),
                _linked("baseline_audio", "AUDIO"),
                _linked("motion_plan", plan_type),
                _widget("max_expanded_frames", "INT"),
                _widget("window_index", "INT"),
                _widget("handle_frames", "INT"),
                _widget("coverage", "COMBO"),
            ],
            [
                _output("segment_frames", "IMAGE"),
                _output("window_plan", plan_type),
                _output("first_frame", "IMAGE"),
                _output("last_frame", "IMAGE"),
                _output("segment_audio", "AUDIO"),
                _output("window_count", "INT"),
                _output("window_index", "INT"),
                _output("report_json", "STRING"),
            ],
            [209, 0, 12, "hot_ranges_only"],
        )
        output["collect"] = _custom(
            22,
            "MiniMaxH3MotionWindowCollectT8Advanced",
            "P2 · Save/collect signed windows / 保存与断点收集",
            (8000, 80),
            (590, 430),
            [
                _linked("baseline_frames", "IMAGE"),
                _linked("baseline_audio", "AUDIO"),
                _linked("recovered_segment", "IMAGE"),
                _linked("recovered_audio", "AUDIO"),
                _linked("window_plan", plan_type),
                _widget("run_name", "STRING"),
                _widget("store_dir", "STRING"),
                _widget("write_window", "BOOLEAN"),
                _widget("store_dtype", "COMBO"),
                _widget("feather_frames", "INT"),
            ],
            [
                _output("frames", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("complete", "BOOLEAN"),
                _output("windows_on_disk", "INT"),
                _output("report_json", "STRING"),
            ],
            ["motion_recovery_stock20_demo", "", True, "float32_exact", 6],
        )
    return output


def build(windowed: bool) -> dict:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    source_by_type = {node["type"]: node for node in base["nodes"]}
    nodes: dict[str, dict] = {
        "unet": _clone(source_by_type["UNETLoader"], 1, "H3 FL2VA Stock20 model", (-980, 100)),
        "clip": _clone(source_by_type["CLIPLoader"], 2, "H3 CLIP", (-980, 430)),
        "video_vae": _clone(source_by_type["VAELoader"], 3, "H3 video VAE", (-980, 720)),
    }
    vae_nodes = [node for node in base["nodes"] if node["type"] == "VAELoader"]
    nodes["video_vae"] = _clone(vae_nodes[0], 3, "H3 video VAE", (-980, 720))
    nodes["audio_vae"] = _clone(vae_nodes[1], 4, "H3 audio VAE", (-980, 940))
    nodes["conditioning"] = _clone(
        source_by_type["MiniMaxH3AudioConditioningT8"],
        5,
        "Pass 1 T2VA conditioning / 原始基线",
        (-300, 120),
    )
    nodes["conditioning"]["widgets_values"][0] = PROMPT
    nodes["conditioning"]["widgets_values"][1:5] = [
        1152 if windowed else 736,
        640 if windowed else 416,
        124,
        "T2VA",
    ]
    nodes["dual"] = _clone(
        source_by_type["MiniMaxH3DualClockSamplerT8"],
        6,
        "Native Stock20 dual clock · 20 calls",
        (500, 120),
    )
    nodes["guider"] = _clone(source_by_type["BasicGuider"], 7, "Shared guider", (1040, 80))
    nodes["noise1"] = _clone(source_by_type["RandomNoise"], 8, "Pass 1 noise", (1040, 300))
    nodes["noise1"]["widgets_values"] = [2608222201, "fixed"]
    nodes["sample1"] = _clone(
        source_by_type["SamplerCustomAdvanced"], 9, "PASS 1 · baseline generation", (1460, 100)
    )
    nodes["decode1"] = _clone(
        source_by_type["MiniMaxH3AVDecodeT8"], 10, "Decode pass 1 for analysis", (2050, 100)
    )
    nodes["save1"] = _clone(
        source_by_type["VHS_VideoCombine"], 11, "Save pass-1 baseline", (2680, -180)
    )
    nodes["save1"]["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3_MotionRecovery/pass1_baseline"
    )
    motion = _motion_nodes(windowed)
    nodes.update(motion)

    pass2_ids = (16, 17, 18, 19, 20) if windowed else (15, 16, 17, 18, 19)
    nodes["dual2"] = _clone(
        source_by_type["MiniMaxH3DualClockSamplerT8"],
        pass2_ids[0],
        "Pass 2 DualClock · rebuild for expanded latent",
        (5360 if windowed else 4640, 420),
    )
    nodes["guider2"] = _clone(
        source_by_type["BasicGuider"],
        pass2_ids[1],
        "Pass 2 guider · bound to expanded clock",
        (5900 if windowed else 5180, 400),
    )
    nodes["noise2"] = _clone(
        source_by_type["RandomNoise"], pass2_ids[2], "Pass 2 fresh noise", (6300 if windowed else 5580, 420)
    )
    nodes["noise2"]["widgets_values"] = [2608222201, "fixed"]
    nodes["sample2"] = _clone(
        source_by_type["SamplerCustomAdvanced"],
        pass2_ids[3],
        "PASS 2 · partial V2V motion recovery",
        (6000 if windowed else 5300, 80),
    )
    nodes["decode2"] = _clone(
        source_by_type["MiniMaxH3AVDecodeT8"],
        pass2_ids[4],
        "Decode expanded pass 2",
        (6700 if windowed else 6000, 80),
    )
    gate_id = 23 if windowed else 21
    nodes["gate"] = _custom(
        gate_id,
        "MiniMaxH3MotionAutoGateT8Advanced",
        "AUTO · Skip pass 2 on ABSTAIN / 平静片自动旁路",
        (8720 if windowed else 7260, 80),
        (560, 350),
        [
            _widget("should_repair", "BOOLEAN"),
            _linked("baseline_frames", "IMAGE"),
            _linked("baseline_audio", "AUDIO"),
            _linked("motion_plan", "H3_T8_MOTION_RECOVERY_PLAN"),
            _linked("repaired_frames", "IMAGE"),
            _linked("repaired_audio", "AUDIO"),
        ],
        [
            _output("frames", "IMAGE"),
            _output("audio", "AUDIO"),
            _output("did_repair", "BOOLEAN"),
            _output("report_json", "STRING"),
        ],
        [False],
    )
    save_id = gate_id + 1
    nodes["save_final"] = _clone(
        source_by_type["VHS_VideoCombine"],
        save_id,
        "Save recovered synchronized MP4",
        (9380 if windowed else 7920, 80),
    )
    nodes["save_final"]["widgets_values"]["filename_prefix"] = (
        "MiniMaxH3_MotionRecovery/windowed_recovered"
        if windowed
        else "MiniMaxH3_MotionRecovery/fullclip_recovered"
    )

    notes = [
        _note(
            save_id + 1,
            "① What it does / 它解决什么",
            "## Motion Recovery is a second V2V pass / 这是二次V2V时间超采样\n"
            "它不是 EAV、STG、锐化或修脸。Pass 1 先生成普通 H3 视频；Analyzer 只测量动作过载，"
            "选中的帧在 Pass 2 中获得更多生成位置，再按 hold 组取回一帧恢复原始 24fps/124帧。"
            "新增节点不修改原有采样器、Conditioning 或 MODEL wrapper。",
            (-980, -520),
        ),
        _note(
            save_id + 2,
            "② Analyzer first / 先看分析结果",
            "## Analyze before spending compute / 先分析再分配算力\n"
            "本工作流已使用 `auto_conservative_exp`。只有绝对分数、轮廓对比和解码残差三道门"
            "同时通过才会请求第二次采样；平静片会自动 ABSTAIN。Auto Gate 使用 ComfyUI 原生"
            "lazy input，ABSTAIN 时二采分支不会执行，并逐对象原样传递 pass-1 画面与音频。"
            "`manual_ranges` 只保留给明确的诊断实验。",
            (-320, -520),
        ),
        _note(
            save_id + 3,
            "③ Audio policy / 声音策略",
            "## Safe audio default / 默认保留一采原音\n"
            "Prepare 的 `follow_original_0p5` 只把相位声码器拉伸后的 pass-1 声音作为联合 AV Transformer 引导。"
            "Recover 保持 `pass1_original`，交付音轨不经过 phase-vocoder 或音频VAE往返；"
            "完整试听已发现 `pass2_recovered_exp` 中段会突然变成远处声音再恢复，只保留为诊断模式；"
            "`blend_exp` 在单条I2VA、`pass1_mix=0.8`时听感正常，但仍是单素材EXP结论。",
            (340, -520),
        ),
        _note(
            save_id + 4,
            "④ Parameters / 建议参数",
            "## Starting values / 起始参数\n"
            "Stock20、shift 12/3、Euler；二采 `denoise_fraction=0.48` 表示保留原 Sigma 序列最后约48%的调用，"
            "不是 sigma=0.48。Pass 2 必须用扩时latent重新建立第二个DualClock和Guider，不能复用pass-1 sampler。"
            "先用 hold 2～3；hold 越大并不保证更好，只会增加有效帧与显存。"
            "首轮不要叠加 EAV/STG/RF Restart/BlockCache；这些组合尚未验收。",
            (1000, -520),
        ),
    ]
    if windowed:
        notes.append(
            _note(
                save_id + 5,
                "⑤ Window resume / 分窗续跑",
                "## Queue one window at a time / 每次排一个窗口\n"
                "Segment Plan 默认 `max_expanded_frames=209`、12帧handle、只覆盖热点。先看 `window_count`，"
                "把 `window_index` 从0依次排到最后；Collect 按 parent plan hash 写入独立目录。"
                "中断后保留相同 `run_name` 与计划，缺失窗口才需重跑。默认 `float32_exact`；"
                "`float16_half_disk` 更省盘但不是像素精确。未收齐时输出仍是 pass-1 基线。",
                (1660, -520),
            )
        )
    else:
        notes.append(
            _note(
                save_id + 5,
                "⑤ Memory boundary / 显存边界",
                "## Full-clip template is not a 16GB guarantee / 整段模板不是16GB保证\n"
                "本模板用 736×416 降低首次尝试风险；有效帧仍可能因范围和 hold 大幅增加。"
                "0.7MP 或更长热点请使用 Windowed 模板。项目没有设置全局2MP/帧数禁令，"
                "但也不会把单机成功宣传为通用显存安全。",
                (1660, -520),
            )
        )
    nodes["notes"] = notes

    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8-motion-recovery-{windowed}")),
        "revision": 0,
        "last_node_id": max(
            [node["id"] for node in nodes.values() if isinstance(node, dict)]
            + [node["id"] for node in notes]
        ),
        "last_link_id": 0,
        "nodes": [node for node in nodes.values() if isinstance(node, dict)] + notes,
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.55 if windowed else 0.62, "offset": [90, 500]},
            "ue_links": [],
            "t8_motion_recovery": {
                "scope": "windowed Stock20 T2VA Advanced EXP" if windowed else "full-clip Stock20 T2VA Advanced EXP",
                "reference": "matlowai/ComfyUI-MAINodes@aadd17764007122700738095d71a3a4e8108c866",
                "quality_status": "mechanical_contract_only_real_quality_pending",
                "audio_default": "pass1_original",
                "audio_validation": "pass2_failed_midwindow_distance_jump; blend_0p8_single_clip_pass",
            },
        },
        "version": 0.4,
    }

    # Remove the temporary list holder from the real node sequence.
    workflow["nodes"] = [node for node in workflow["nodes"] if node.get("type")]

    _connect(workflow, nodes["unet"], 0, nodes["dual"], 0)
    _connect(workflow, nodes["clip"], 0, nodes["conditioning"], 0)
    _connect(workflow, nodes["video_vae"], 0, nodes["conditioning"], 1)
    _connect(workflow, nodes["audio_vae"], 0, nodes["conditioning"], 2)
    _connect(workflow, nodes["conditioning"], 1, nodes["dual"], 1)
    _connect(workflow, nodes["dual"], 0, nodes["guider"], 0)
    _connect(workflow, nodes["conditioning"], 0, nodes["guider"], 1)
    _connect(workflow, nodes["noise1"], 0, nodes["sample1"], 0)
    _connect(workflow, nodes["guider"], 0, nodes["sample1"], 1)
    _connect(workflow, nodes["dual"], 1, nodes["sample1"], 2)
    _connect(workflow, nodes["dual"], 2, nodes["sample1"], 3)
    _connect(workflow, nodes["conditioning"], 1, nodes["sample1"], 4)
    _connect(workflow, nodes["sample1"], 0, nodes["decode1"], 0)
    _connect(workflow, nodes["video_vae"], 0, nodes["decode1"], 1)
    _connect(workflow, nodes["audio_vae"], 0, nodes["decode1"], 2)
    _connect(workflow, nodes["decode1"], 0, nodes["save1"], 0)
    _connect(workflow, nodes["decode1"], 1, nodes["save1"], 1)
    _connect(workflow, nodes["sample1"], 0, motion["analyzer"], 0)
    _connect(workflow, nodes["decode1"], 0, motion["analyzer"], 1)

    active_frames = nodes["decode1"]
    active_frames_slot = 0
    active_audio = nodes["decode1"]
    active_audio_slot = 1
    active_plan = motion["analyzer"]
    active_plan_slot = 0
    if windowed:
        segment = motion["segment"]
        _connect(workflow, nodes["decode1"], 0, segment, 0)
        _connect(workflow, nodes["decode1"], 1, segment, 1)
        _connect(workflow, motion["analyzer"], 0, segment, 2)
        active_frames, active_frames_slot = segment, 0
        active_audio, active_audio_slot = segment, 4
        active_plan, active_plan_slot = segment, 1

    _connect(workflow, active_frames, active_frames_slot, motion["prepare"], 0)
    _connect(workflow, active_plan, active_plan_slot, motion["prepare"], 1)
    _connect(workflow, nodes["video_vae"], 0, motion["prepare"], 2)
    _connect(workflow, nodes["audio_vae"], 0, motion["prepare"], 4)
    _connect(workflow, active_audio, active_audio_slot, motion["prepare"], 5)
    _connect(workflow, motion["prepare"], 0, nodes["dual2"], 1)
    _connect(workflow, nodes["unet"], 0, nodes["dual2"], 0)
    _connect(workflow, nodes["dual2"], 0, nodes["guider2"], 0)
    _connect(workflow, nodes["conditioning"], 0, nodes["guider2"], 1)
    _connect(workflow, motion["prepare"], 0, motion["composer"], 0)
    _connect(workflow, nodes["dual2"], 2, motion["composer"], 1)
    _connect(workflow, motion["prepare"], 3, motion["composer"], 2)
    _connect(workflow, nodes["noise2"], 0, nodes["sample2"], 0)
    _connect(workflow, nodes["guider2"], 0, nodes["sample2"], 1)
    _connect(workflow, nodes["dual2"], 1, nodes["sample2"], 2)
    _connect(workflow, motion["composer"], 1, nodes["sample2"], 3)
    _connect(workflow, motion["composer"], 0, nodes["sample2"], 4)
    _connect(workflow, nodes["sample2"], 0, nodes["decode2"], 0)
    _connect(workflow, nodes["video_vae"], 0, nodes["decode2"], 1)
    _connect(workflow, nodes["audio_vae"], 0, nodes["decode2"], 2)
    _connect(workflow, nodes["decode2"], 0, motion["recover"], 0)
    _connect(workflow, nodes["decode2"], 1, motion["recover"], 1)
    _connect(workflow, active_audio, active_audio_slot, motion["recover"], 2)
    _connect(workflow, motion["composer"], 2, motion["recover"], 3)

    final_frames, final_frames_slot = motion["recover"], 0
    final_audio, final_audio_slot = motion["recover"], 1
    if windowed:
        collect = motion["collect"]
        _connect(workflow, nodes["decode1"], 0, collect, 0)
        _connect(workflow, nodes["decode1"], 1, collect, 1)
        _connect(workflow, motion["recover"], 0, collect, 2)
        _connect(workflow, motion["recover"], 1, collect, 3)
        _connect(workflow, motion["composer"], 2, collect, 4)
        final_frames, final_frames_slot = collect, 0
        final_audio, final_audio_slot = collect, 1
    _connect(workflow, motion["analyzer"], 2, nodes["gate"], 0)
    _connect(workflow, nodes["decode1"], 0, nodes["gate"], 1)
    _connect(workflow, nodes["decode1"], 1, nodes["gate"], 2)
    _connect(workflow, motion["analyzer"], 0, nodes["gate"], 3)
    _connect(workflow, final_frames, final_frames_slot, nodes["gate"], 4)
    _connect(workflow, final_audio, final_audio_slot, nodes["gate"], 5)
    _connect(workflow, nodes["gate"], 0, nodes["save_final"], 0)
    _connect(workflow, nodes["gate"], 1, nodes["save_final"], 1)

    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["nodes"].sort(key=lambda node: node["id"])
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    return workflow


def main() -> int:
    outputs = {
        WORKFLOW_DIR
        / "2026-08-22_H3_Motion_Recovery_Fullclip_Stock20_Advanced_EXP.json": build(False),
        WORKFLOW_DIR
        / "2026-08-22_H3_Motion_Recovery_Windowed_Stock20_Advanced_EXP.json": build(True),
    }
    for path, workflow in outputs.items():
        payload = json.dumps(workflow, ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")
        INSTALLED_DIR.mkdir(parents=True, exist_ok=True)
        (INSTALLED_DIR / path.name).write_text(payload, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
