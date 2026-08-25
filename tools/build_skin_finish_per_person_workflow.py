#!/usr/bin/env python3
from __future__ import annotations

import json

from build_skin_finish_p1_workflow import OUTPUT as P1_OUTPUT
from build_skin_finish_p1_workflow import _node, _note, _output, _socket, _widget


OUTPUT = P1_OUTPUT.with_name(
    "2026-08-25_H3_Skin_Finish_Per_Person_Advanced_EXP.json"
)
PROJECT_ROOT = OUTPUT.parents[3]
USER_OUTPUT = (
    PROJECT_ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "17-skin-finish"
    / OUTPUT.name
)
PARSER_MODEL = "facexlib_parsenet_v0.2.2_pinned"


def build() -> dict:
    nodes = [
        _node(
            1,
            "LoadVideo",
            "1. Final 24fps SDR source / 最终24fps SDR来源",
            (0, 0),
            (400, 130),
            [_widget("file", "COMBO")],
            [_output("VIDEO", "VIDEO")],
            ["replace_with_clear_two_or_three_person_24fps_source.mp4"],
            core=True,
        ),
        _node(
            2,
            "GetVideoComponents",
            "2. Decode exact source frames/audio / 解码同一来源画面音频",
            (460, 0),
            (410, 120),
            [_socket("video", "VIDEO")],
            [
                _output("images", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("fps", "FLOAT"),
                _output("bit_depth", "INT"),
            ],
            [],
            core=True,
        ),
        _node(
            3,
            "CheckpointLoaderSimple",
            "Native ComfyUI SAM3.1 multiplex / 原生SAM3.1多人模型",
            (0, 230),
            (400, 130),
            [_widget("ckpt_name", "COMBO")],
            [
                _output("MODEL", "MODEL"),
                _output("CLIP", "CLIP"),
                _output("VAE", "VAE"),
            ],
            ["sam3.1_multiplex_fp16.safetensors"],
            core=True,
        ),
        _node(
            4,
            "CLIPTextEncode",
            "SAM3.1 person prompt / SAM人物提示",
            (460, 210),
            (410, 170),
            [_widget("text", "STRING"), _socket("clip", "CLIP")],
            [_output("CONDITIONING", "CONDITIONING")],
            ["person"],
            core=True,
        ),
        _node(
            5,
            "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            "3. Per-shot tracks; offload SAM / 逐镜追踪后卸载SAM",
            (930, 0),
            (460, 530),
            [
                _socket("frames", "IMAGE"),
                _socket("model", "MODEL"),
                _socket("conditioning", "CONDITIONING"),
                _widget("fps", "FLOAT"),
                _widget("maximum_people", "INT"),
                _widget("detection_threshold", "FLOAT"),
                _widget("detect_interval", "INT"),
                _widget("scene_cut_threshold", "FLOAT"),
                _widget("analysis_max_side", "COMBO"),
                _widget("preview_stride", "INT"),
                _widget("release_policy", "COMBO"),
            ],
            [
                _output("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _output("colored_preview", "IMAGE"),
                _output("report_json", "STRING"),
                _output("shot_count", "INT"),
                _output("shot_local_track_count", "INT"),
            ],
            [24.0, 3, 0.35, 3, 0.28, 640, 8, "offload_sam31_after_track"],
        ),
        _node(
            6,
            "LoadImage",
            "Clear single-person reference for Character_A / A清晰单人参考",
            (0, 500),
            (400, 130),
            [_widget("image", "COMBO")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_clear_single_person_Character_A_reference.png"],
            core=True,
        ),
        _node(
            7,
            "LoadImage",
            "Clear single-person reference for Character_B / B清晰单人参考",
            (0, 730),
            (400, 130),
            [_widget("image", "COMBO")],
            [_output("IMAGE", "IMAGE"), _output("MASK", "MASK")],
            ["replace_with_clear_single_person_Character_B_reference.png"],
            core=True,
        ),
        _node(
            8,
            "MiniMaxH3FaceCharacterProfileT8Advanced",
            "4A. In-memory Character_A match profile / A角色匹配资料",
            (460, 500),
            (410, 210),
            [
                _widget("character_id", "STRING"),
                _socket("reference_images", "IMAGE"),
                _widget("reference_face_policy", "COMBO"),
            ],
            [
                _output("character_profile", "H3_T8_MULTIFACE_CHARACTER_PROFILE"),
                _output("reference_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            ["Character_A", "require_single_face"],
        ),
        _node(
            9,
            "MiniMaxH3FaceCharacterProfileT8Advanced",
            "4B. In-memory Character_B match profile / B角色匹配资料",
            (460, 760),
            (410, 210),
            [
                _widget("character_id", "STRING"),
                _socket("reference_images", "IMAGE"),
                _widget("reference_face_policy", "COMBO"),
            ],
            [
                _output("character_profile", "H3_T8_MULTIFACE_CHARACTER_PROFILE"),
                _output("reference_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            ["Character_B", "require_single_face"],
        ),
        _node(
            10,
            "MiniMaxH3FaceCastMergeT8Advanced",
            "Merge Character_A / 合并A角色",
            (930, 610),
            (410, 110),
            [_socket("profile", "H3_T8_MULTIFACE_CHARACTER_PROFILE")],
            [
                _output("face_cast", "H3_T8_MULTIFACE_CAST"),
                _output("reference_contact_sheet", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [],
        ),
        _node(
            11,
            "MiniMaxH3FaceCastMergeT8Advanced",
            "Merge Character_B into reviewed cast / 合并B角色",
            (1420, 600),
            (430, 135),
            [
                _socket("profile", "H3_T8_MULTIFACE_CHARACTER_PROFILE"),
                _socket("previous_cast", "H3_T8_MULTIFACE_CAST"),
            ],
            [
                _output("face_cast", "H3_T8_MULTIFACE_CAST"),
                _output("reference_contact_sheet", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [],
        ),
        _node(
            12,
            "MiniMaxH3FaceTrackAssignT8Advanced",
            "5. Review and bind every shot-local track / 审核每个镜头的人物绑定",
            (1420, 0),
            (520, 510),
            [
                _socket("frames", "IMAGE"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _socket("face_cast", "H3_T8_MULTIFACE_CAST"),
                _widget("identity_mode", "COMBO"),
                _widget("manual_assignments_json", "STRING"),
                _widget("minimum_similarity", "FLOAT"),
                _widget("minimum_margin", "FLOAT"),
                _widget("identity_samples_per_track", "INT"),
                _widget("strict_identity", "BOOLEAN"),
                _widget("preview_stride", "INT"),
            ],
            [
                _output("identity_assignment", "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT"),
                _output("assignment_preview", "IMAGE"),
                _output("report_json", "STRING"),
                _output("track_count", "INT"),
            ],
            [
                "sface_cpu_suggest",
                '{"0:0":"Character_A","0:1":"Character_B"}',
                0.40,
                0.05,
                3,
                True,
                8,
            ],
        ),
        _node(
            13,
            "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced",
            "6. Strict first, profile fallback / 严格优先、侧脸回退语义MASK",
            (2020, 0),
            (570, 810),
            [
                _socket("frames", "IMAGE"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _widget("parser_model", "COMBO"),
                _widget("detection_threshold", "FLOAT"),
                _widget("minimum_face_height_px", "FLOAT"),
                _widget("minimum_detail", "FLOAT"),
                _widget("minimum_person_overlap", "FLOAT"),
                _widget("minimum_track_quality", "FLOAT"),
                _widget("minimum_class_probability", "FLOAT"),
                _widget("feature_protection_px", "INT"),
                _widget("include_neck", "BOOLEAN"),
                _widget("minimum_skin_area_per_face", "FLOAT"),
                _widget("maximum_skin_area_per_frame", "FLOAT"),
                _widget("maximum_alignment_rms", "FLOAT"),
                _widget("profile_crop_expansion", "FLOAT"),
                _widget("minimum_ready_frame_fraction", "FLOAT"),
                _widget("preview_count", "INT"),
                _socket("identity_assignment", "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT"),
            ],
            [
                _output("semantic_skin_mask", "MASK"),
                _output("mask_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [
                PARSER_MODEL,
                0.45,
                32.0,
                0.010,
                0.20,
                0.10,
                0.55,
                3,
                False,
                0.00005,
                0.35,
                0.08,
                1.45,
                0.50,
                6,
            ],
        ),
        _node(
            14,
            "MiniMaxH3SkinFinishPersonProfileT8Advanced",
            "7A. Character_A: subtle, conservative / A人物保守参数",
            (2020, 900),
            (520, 390),
            [
                _widget("selector_type", "COMBO"),
                _widget("selector", "STRING"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("tone_adjust", "FLOAT"),
            ],
            [
                _output("profiles", "H3_T8_SKIN_FINISH_PERSON_PROFILES"),
                _output("report_json", "STRING"),
            ],
            ["character_id", "Character_A", "subtle", 0.30, 0.92, 0.30, 0.0],
        ),
        _node(
            15,
            "MiniMaxH3SkinFinishPersonProfileT8Advanced",
            "7B. Character_B: stronger oil control / B人物稍强控油",
            (2600, 900),
            (520, 420),
            [
                _widget("selector_type", "COMBO"),
                _widget("selector", "STRING"),
                _widget("preset", "COMBO"),
                _widget("amount", "FLOAT"),
                _widget("texture_keep", "FLOAT"),
                _widget("shine_control", "FLOAT"),
                _widget("tone_adjust", "FLOAT"),
                _socket("previous_profiles", "H3_T8_SKIN_FINISH_PERSON_PROFILES"),
            ],
            [
                _output("profiles", "H3_T8_SKIN_FINISH_PERSON_PROFILES"),
                _output("report_json", "STRING"),
            ],
            ["character_id", "Character_B", "oil_control", 0.40, 0.90, 0.45, 0.0],
        ),
        _node(
            16,
            "MiniMaxH3SkinFinishPerPersonT8Advanced",
            "8. Per-person/per-shot source-safe candidate / 逐人物逐镜头候选",
            (2700, 0),
            (600, 850),
            [
                _socket("frames", "IMAGE"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _socket("semantic_skin_mask", "MASK"),
                _widget("semantic_report_json", "STRING"),
                _widget("default_policy", "COMBO"),
                _widget("default_preset", "COMBO"),
                _widget("default_amount", "FLOAT"),
                _widget("default_texture_keep", "FLOAT"),
                _widget("default_shine_control", "FLOAT"),
                _widget("default_tone_adjust", "FLOAT"),
                _widget("execution_mode", "COMBO"),
                _widget("accept_candidate", "BOOLEAN"),
                _widget("chunk_frames", "INT"),
                _widget("proxy_long_side", "INT"),
                _widget("preview_count", "INT"),
                _socket("profiles", "H3_T8_SKIN_FINISH_PERSON_PROFILES"),
                _socket("identity_assignment", "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("used_skin_mask", "MASK"),
                _output("rejected_skin_mask", "MASK"),
                _output("ownership_preview", "IMAGE"),
                _output("skin_finish_state", "H3_T8_SKIN_FINISH_STATE"),
                _output("report_json", "STRING"),
            ],
            [
                "",
                "source_unmatched",
                "subtle",
                0.35,
                0.90,
                0.35,
                0.0,
                "candidate_only",
                False,
                2,
                640,
                6,
            ],
        ),
        _node(
            17,
            "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "9. Texture Guard remains source-relative / 纹理护栏仍相对原片",
            (3390, 0),
            (520, 650),
            [
                _socket("source_frames", "IMAGE"),
                _socket("candidate_frames", "IMAGE"),
                _socket("used_skin_mask", "MASK"),
                _widget("shadow_protection", "FLOAT"),
                _widget("highlight_protection", "FLOAT"),
                _widget("transition_width", "FLOAT"),
                _widget("minimum_texture_ratio", "FLOAT"),
                _widget("minimum_reference_texture", "FLOAT"),
                _widget("maximum_new_clipped_fraction", "FLOAT"),
                _widget("clipping_epsilon", "FLOAT"),
                _widget("texture_radius", "INT"),
                _widget("chunk_frames", "INT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("audio", "AUDIO"),
            ],
            [
                _output("guarded_candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("selected", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("effective_mask", "MASK"),
                _output("rejected_mask", "MASK"),
                _output("difference", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [0.10, 0.94, 0.06, 0.78, 0.003, 0.0005, 1.0 / 255.0, 1, 2, False],
        ),
        _node(
            18,
            "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            "11. Explicit final accept; copy source audio packets / 最终人工接受",
            (4620, 0),
            (520, 330),
            [
                _socket("source_video", "VIDEO"),
                _socket("processed_frames", "IMAGE"),
                _widget("filename_prefix", "STRING"),
                _widget("crf", "FLOAT"),
                _widget("accept_candidate", "BOOLEAN"),
            ],
            [
                _output("video", "VIDEO"),
                _output("saved_path", "STRING"),
                _output("report_json", "STRING"),
            ],
            ["MiniMaxH3/SkinFinish/per_person_reviewed", 18.0, False],
        ),
        _node(19, "PreviewImage", "SAM tracks are shot-local / SAM轨迹仅镜头内", (930, 1000), (430, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(20, "PreviewImage", "Reviewed identity assignment / 已审核人物绑定", (1420, 1000), (430, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(21, "PreviewImage", "ParseNet: green skin, red features / 绿皮肤红五官", (2020, 1380), (520, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(22, "PreviewImage", "Ownership colors; red stays source / 参数归属颜色；红区回原片", (2700, 1380), (560, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(23, "PreviewImage", "Per-person candidate / 逐人物候选", (3390, 760), (500, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(24, "PreviewImage", "Texture-guarded candidate / 纹理护栏候选", (3980, 430), (500, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _node(25, "MaskToImage", "Final processed skin mask / 最终处理皮肤MASK", (3390, 930), (300, 100), [_socket("mask", "MASK")], [_output("IMAGE", "IMAGE")], [], core=True),
        _node(26, "PreviewImage", "Processed mask only / 仅实际处理区域", (3740, 930), (330, 110), [_socket("images", "IMAGE")], [_output("images", "IMAGE")], [], core=True),
        _note(
            27,
            "START HERE / 使用顺序",
            "## 逐人物 Skin Finish 完整链\n\n只在最终解码、Face Refine/Motion Recovery之后使用：`SAM3.1逐镜追踪 -> 清晰单人参考建立Character -> 审核轨迹绑定 -> 严格五点/侧脸裁切ParseNet -> 逐人物参数 -> Texture Guard -> 人工接受后封装`。SAM文本建议用单数`person`，让multiplex检测返回多个独立实例；不要写`two people`这类可能被当成一个整体的数量短语。0.35是本次清晰双人素材的召回起点，误检时再提高。字幕、整体调色和Tape-FX放在后面。",
            (0, 1600),
            (1000, 390),
        ),
        _note(
            28,
            "IDENTITY IS REVIEWED ROUTING / 身份必须人工审核",
            "## 参考图如何传入\n\n每个Character使用一张或多张清晰、单人、五官可见的授权参考图。SFace只给匹配建议，不证明身份；每次切镜SAM track会重新编号。先查看彩色轨迹和assignment预览，必要时修改`manual_assignments_json`，例如`{\"0:0\":\"Character_A\",\"1:1\":\"Character_A\"}`。`strict_identity=true`会拒绝未绑定轨迹。",
            (1080, 1600),
            (1120, 430),
        ),
        _note(
            29,
            "PROFILE PRECEDENCE / 参数优先级",
            "## Character与shot:track可以同时用\n\n默认Profile节点使用`character_id`，同一人物可跨已审核镜头复用参数。如某一镜头光线特殊，再追加`selector_type=shot_track`、`selector=1:0`的节点；优先级固定为`shot:track > character_id > default`。链式Profile最多8个，重复或未知selector会失败回原片。`tone_adjust`只是中间调曝光式调整，不是自动肤色匹配，默认保持0。",
            (2280, 1600),
            (1140, 450),
        ),
        _note(
            30,
            "CROSSING + UNMATCHED SAFETY / 交叉与未匹配安全",
            "## 不靠猜测决定重叠像素属于谁\n\n`default_policy=source_unmatched`是推荐默认：没有显式Profile的人物保持原片。两个人物SAM MASK在皮肤像素发生重叠时，该重叠区也保持原片并在ownership预览标红，避免不同人物参数串色。彩色区域表示实际分配到某套参数，红色表示未匹配、歧义或交叉回退；报告给出逐track路线和像素计数。",
            (3500, 1600),
            (1150, 450),
        ),
        _note(
            31,
            "PARAMETERS / 建议参数",
            "## 先克制，再逐人微调\n\n真人建议从`subtle, amount 0.25~0.40, texture_keep 0.88~0.95, shine 0.25~0.45, tone 0`开始；明显油光才用`oil_control`。语义节点始终先尝试`maximum_alignment_rms=0.08`的严格五点对齐，仅失败姿态才使用`profile_crop_expansion=1.45`原侧脸裁切；它不会把侧脸变正脸。不要用低texture_keep追求锐化，它不会生成毛孔或修复模糊/崩脸。每次只改一个人物并看完整视频，特别检查说话嘴唇、眼睛、手遮脸、侧脸、肤色跳变和镜头切换。",
            (4700, 1600),
            (1100, 420),
        ),
        _note(
            32,
            "MEMORY + AUDIO + ACCEPT / 内存、音频与接受",
            "## 默认不会覆盖原片\n\nSAM完成后选择性卸载；YuNet释放后才加载CPU ParseNet，Profile执行不重跑模型并默认2帧chunk。多人物ParseNet仍可能较慢，长片请先分成已接受片段。视觉节点原对象旁路AUDIO；最终Video Finalize仅在兼容时逐包复制来源音频payload。Per-Person、Texture Guard和Finalize接受开关默认全为false，必须先审所有预览再只打开最终需要的门。",
            (5880, 1600),
            (1120, 430),
        ),
        _node(
            33,
            "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
            "10. Final hard-failure audit / 最终硬失败审计",
            (3980, 0),
            (570, 700),
            [
                _socket("source_frames", "IMAGE"),
                _socket("candidate_frames", "IMAGE"),
                _socket("used_skin_mask", "MASK"),
                _widget("audit_scope", "COMBO"),
                _widget("temporal_policy", "COMBO"),
                _widget("maximum_mean_abs_change", "FLOAT"),
                _widget("maximum_peak_abs_change", "FLOAT"),
                _widget("maximum_temporal_effect_jump", "FLOAT"),
                _widget("maximum_track_leak_fraction", "FLOAT"),
                _widget("minimum_temporal_pixels", "INT"),
                _widget("scene_cut_reset_threshold", "FLOAT"),
                _widget("accept_candidate", "BOOLEAN"),
                _socket("track_plan", "H3_T8_SAM31_MULTIFACE_TRACK_PLAN"),
                _socket("audio_source", "AUDIO"),
                _socket("audio_passthrough", "AUDIO"),
            ],
            [
                _output("selected", "IMAGE"),
                _output("gated_candidate", "IMAGE"),
                _output("source", "IMAGE"),
                _output("audio", "AUDIO"),
                _output("hard_gate_pass", "BOOLEAN"),
                _output("failed_frame_count", "INT"),
                _output("failure_preview", "IMAGE"),
                _output("report_json", "STRING"),
            ],
            [
                "unique_track_owner",
                "hard_gate",
                0.08,
                0.30,
                0.04,
                0.001,
                64,
                0.20,
                False,
            ],
        ),
        _node(
            34,
            "PreviewImage",
            "Hard-gate preview: red fail, green pass / 红失败绿通过",
            (4620, 430),
            (460, 110),
            [_socket("images", "IMAGE")],
            [_output("images", "IMAGE")],
            [],
            core=True,
        ),
        _note(
            35,
            "FINAL SAFETY AUDIT / 最终安全审计",
            "## 只会自动拒绝，不会自动判美\n\n多人示例固定使用`audit_scope=unique_track_owner`和`temporal_policy=hard_gate`：检查实际处理像素是否越出SAM人物轨迹、是否落在两人重叠歧义区、蒙版外/五官保护区是否被改动、逐人物处理强度是否突然跳变，以及AUDIO PCM是否一致。红色预览表示硬失败，绿色只代表机械门通过，不代表更好看。`gated_candidate`失败时自动回原片；最终仍只需在Video Finalize中人工打开接受开关。",
            (7080, 1600),
            (1180, 460),
        ),
    ]

    links = []
    node_map = {node["id"]: node for node in nodes}

    def connect(source, source_slot, target, target_slot, value_type):
        link_id = len(links) + 1
        links.append([link_id, source, source_slot, target, target_slot, value_type])
        output = node_map[source]["outputs"][source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
        node_map[target]["inputs"][target_slot]["link"] = link_id

    connect(1, 0, 2, 0, "VIDEO")
    connect(3, 1, 4, 1, "CLIP")
    connect(2, 0, 5, 0, "IMAGE")
    connect(3, 0, 5, 1, "MODEL")
    connect(4, 0, 5, 2, "CONDITIONING")
    connect(6, 0, 8, 1, "IMAGE")
    connect(7, 0, 9, 1, "IMAGE")
    connect(8, 0, 10, 0, "H3_T8_MULTIFACE_CHARACTER_PROFILE")
    connect(9, 0, 11, 0, "H3_T8_MULTIFACE_CHARACTER_PROFILE")
    connect(10, 0, 11, 1, "H3_T8_MULTIFACE_CAST")
    connect(2, 0, 12, 0, "IMAGE")
    connect(5, 0, 12, 1, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(11, 0, 12, 2, "H3_T8_MULTIFACE_CAST")
    connect(2, 0, 13, 0, "IMAGE")
    connect(5, 0, 13, 1, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(12, 0, 13, 15, "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
    connect(14, 0, 15, 7, "H3_T8_SKIN_FINISH_PERSON_PROFILES")
    connect(2, 0, 16, 0, "IMAGE")
    connect(5, 0, 16, 1, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(13, 0, 16, 2, "MASK")
    connect(13, 2, 16, 3, "STRING")
    connect(15, 0, 16, 15, "H3_T8_SKIN_FINISH_PERSON_PROFILES")
    connect(12, 0, 16, 16, "H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
    connect(2, 1, 16, 17, "AUDIO")
    connect(16, 1, 17, 0, "IMAGE")
    connect(16, 0, 17, 1, "IMAGE")
    connect(16, 4, 17, 2, "MASK")
    connect(16, 3, 17, 13, "AUDIO")
    connect(1, 0, 18, 0, "VIDEO")
    connect(5, 1, 19, 0, "IMAGE")
    connect(12, 1, 20, 0, "IMAGE")
    connect(13, 1, 21, 0, "IMAGE")
    connect(16, 6, 22, 0, "IMAGE")
    connect(16, 0, 23, 0, "IMAGE")
    connect(17, 0, 24, 0, "IMAGE")
    connect(17, 4, 25, 0, "MASK")
    connect(25, 0, 26, 0, "IMAGE")
    connect(16, 1, 33, 0, "IMAGE")
    connect(17, 0, 33, 1, "IMAGE")
    connect(17, 4, 33, 2, "MASK")
    connect(5, 0, 33, 12, "H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
    connect(2, 1, 33, 13, "AUDIO")
    connect(17, 3, 33, 14, "AUDIO")
    connect(33, 1, 18, 1, "IMAGE")
    connect(33, 6, 34, 0, "IMAGE")

    return {
        "id": "999536e5-5ff4-47cb-ab17-e638951c72f6",
        "revision": 0,
        "last_node_id": max(node_map),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Skin Finish: reviewed identities -> per-person/per-shot parameters",
                "bounding": [-50, -80, 4580, 1580],
                "color": "#536d4d",
                "font_size": 24,
                "flags": {},
            }
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.38, "offset": [100, 80]},
            "workflow_title": "2026-08-25 H3 Skin Finish Per Person Advanced EXP",
            "t8_skin_finish": {
                "scope": "reviewed per-character and exact shot-local Skin Finish parameters",
                "default_selection": "source",
                "profile_precedence": "shot_track_over_character_id_over_optional_default",
                "overlap_policy": "source_on_any_multi_track_overlap",
                "runtime_download": False,
                "automatic_accept": False,
            },
        },
        "version": 0.4,
    }


def main() -> int:
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    for path in (OUTPUT, USER_OUTPUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(OUTPUT)
    print(USER_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
