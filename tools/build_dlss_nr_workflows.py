from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "workflows" / "25-dlss-nr"
USER_OUTPUT = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "25-dlss-nr"
)

RUNTIME = "MiniMaxH3DLSSNRRuntimeAuditT8Advanced"
IMAGE = "MiniMaxH3DLSSNRImageSuperResolutionT8Advanced"
VIDEO_FRAMES = "MiniMaxH3DLSSNRVideoFramesT8Advanced"
VIDEO_FILE = "MiniMaxH3DLSSNRVideoFileT8Advanced"

FILENAMES = {
    "runtime": "2026-09-04_H3_DLSS_NR_Runtime_Audit_Advanced.json",
    "image": "2026-09-04_H3_DLSS_NR_Image_2x_Standard_Advanced.json",
    "video_frames": (
        "2026-09-04_H3_DLSS_NR_Video_Frames_2x_Standard_Advanced.json"
    ),
    "video_file": (
        "2026-09-04_H3_DLSS_NR_Video_File_2x_Standard_Advanced.json"
    ),
}

RUNTIME_WIDGETS = ["1.3", False, "feature_probe_1_frame", 0, 0]
QUALITY_WIDGETS = [
    "standard",
    "default",
    "0 Default",
    "0 Default",
    1.5,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    False,
    False,
]

COMMON_NOTE = (
    "## MiniMax H3 DLSS-NR v1.3 后处理\n\n"
    "- 仅支持 Windows + NVIDIA RTX。完整外部运行时由用户自行取得并放到 "
    "`ComfyUI/models/DLSS-NR/1.3/`；本节点不下载、不安装，也不分发 EXE/DLL。\n"
    "- 先阅读并接受外部许可，再把 Runtime Audit 的许可开关打开。保持关闭时工作流会明确拒绝执行。\n"
    "- 默认 Standard + 2x；它是经固定素材盲测通过的保守起点，不代表普遍优于其他超分方法。\n"
    "- 只处理 SDR 8-bit。视频不插帧；帧序列路线原样传递 AUDIO，文件路线严格复制并校验源音频。\n"
    "- 超分不会修复源片已有的身份、口型或真实纹理问题。请看完整候选，再决定是否采用。"
)


def _node(
    node_id: int,
    node_type: str,
    title: str,
    pos: list[int],
    size: list[int],
    order: int,
    inputs: list[dict],
    outputs: list[dict],
    widgets: list,
) -> dict:
    cnr_id = (
        "comfy-core"
        if node_type
        in {
            "MarkdownNote",
            "LoadImage",
            "LoadVideo",
            "GetVideoComponents",
            "PreviewImage",
            "CreateVideo",
            "SaveVideo",
        }
        else "minimax-h3-audio-T8"
    )
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"cnr_id": cnr_id, "Node name for S&R": node_type},
        "widgets_values": widgets,
    }


def _workflow(name: str, nodes: list[dict], links: list[list]) -> dict:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8-dlss-nr-{name}-v1.3")),
        "revision": 0,
        "last_node_id": max(node["id"] for node in nodes),
        "last_link_id": max((link[0] for link in links), default=0),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.82, "offset": [110, 80]},
            "frontendVersion": "1.24.3",
        },
        "version": 0.4,
    }


def _runtime_node(node_id: int, order: int, *, links: list[int] | None) -> dict:
    return _node(
        node_id,
        RUNTIME,
        "1. DLSS-NR v1.3 运行时审计（确认许可后开启）",
        [0, 360],
        [500, 250],
        order,
        [],
        [
            {"name": "dlss_nr_runtime", "type": "T8_DLSS_NR_RUNTIME", "links": links},
            {"name": "ready", "type": "BOOLEAN", "links": None},
            {"name": "status", "type": "STRING", "links": None},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        list(RUNTIME_WIDGETS),
    )


def _runtime_workflow() -> dict:
    nodes = [
        _node(
            1,
            "MarkdownNote",
            "使用说明 / Read first",
            [0, 0],
            [960, 300],
            0,
            [],
            [],
            [
                COMMON_NOTE
                + "\n\n本工作流只检查运行时。打开许可开关后运行；READY 才能连接到三份执行工作流。"
            ],
        ),
        _runtime_node(2, 1, links=None),
    ]
    return _workflow("runtime", nodes, [])


def _image_workflow() -> dict:
    nodes = [
        _node(
            1,
            "MarkdownNote",
            "使用说明 / Read first",
            [0, 0],
            [1040, 300],
            0,
            [],
            [],
            [COMMON_NOTE + "\n\n本工作流把每张图片独立处理，不在 batch 图片之间继承时序状态。"],
        ),
        _runtime_node(2, 1, links=[1]),
        _node(
            3,
            "LoadImage",
            "2. 选择要超分的图片",
            [0, 680],
            [500, 360],
            2,
            [],
            [
                {"name": "IMAGE", "type": "IMAGE", "links": [2]},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            ["replace_with_source_image.png", "image"],
        ),
        _node(
            4,
            IMAGE,
            "3. Standard 2x 图片超分",
            [600, 390],
            [570, 720],
            3,
            [
                {"name": "dlss_nr_runtime", "type": "T8_DLSS_NR_RUNTIME", "link": 1},
                {"name": "images", "type": "IMAGE", "link": 2},
            ],
            [
                {"name": "candidate", "type": "IMAGE", "links": [3]},
                {"name": "source", "type": "IMAGE", "links": [4]},
                {"name": "report_json", "type": "STRING", "links": None},
            ],
            ["sr_nr", "2.0", *QUALITY_WIDGETS],
        ),
        _node(
            5,
            "PreviewImage",
            "原图 / Source",
            [1260, 390],
            [420, 300],
            4,
            [{"name": "images", "type": "IMAGE", "link": 4}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            6,
            "PreviewImage",
            "DLSS-NR 2x 候选",
            [1260, 750],
            [420, 300],
            5,
            [{"name": "images", "type": "IMAGE", "link": 3}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
    ]
    links = [
        [1, 2, 0, 4, 0, "T8_DLSS_NR_RUNTIME"],
        [2, 3, 0, 4, 1, "IMAGE"],
        [3, 4, 0, 6, 0, "IMAGE"],
        [4, 4, 1, 5, 0, "IMAGE"],
    ]
    return _workflow("image", nodes, links)


def _video_frames_workflow() -> dict:
    nodes = [
        _node(
            1,
            "MarkdownNote",
            "使用说明 / Read first",
            [0, 0],
            [1080, 300],
            0,
            [],
            [],
            [
                COMMON_NOTE
                + "\n\n本工作流先解成 IMAGE batch；适合 H3 短片。长视频请用独立的 Video File 工作流。"
            ],
        ),
        _runtime_node(2, 1, links=[2]),
        _node(
            3,
            "LoadVideo",
            "2. 选择H3短片",
            [0, 680],
            [500, 150],
            2,
            [],
            [{"name": "VIDEO", "type": "VIDEO", "links": [1]}],
            ["replace_with_h3_video.mp4"],
        ),
        _node(
            4,
            "GetVideoComponents",
            "3. 解出帧、原音频和帧率",
            [580, 680],
            [470, 190],
            3,
            [{"name": "video", "type": "VIDEO", "link": 1}],
            [
                {"name": "images", "type": "IMAGE", "links": [3]},
                {"name": "audio", "type": "AUDIO", "links": [5]},
                {"name": "fps", "type": "FLOAT", "links": [4, 10]},
                {"name": "bit_depth", "type": "COMBO", "links": [11]},
                {"name": "color_space", "type": "COMBO", "links": None},
            ],
            [],
        ),
        _node(
            5,
            VIDEO_FRAMES,
            "4. Standard 2x 时序视频超分",
            [1140, 350],
            [620, 800],
            4,
            [
                {"name": "dlss_nr_runtime", "type": "T8_DLSS_NR_RUNTIME", "link": 2},
                {"name": "frames", "type": "IMAGE", "link": 3},
                {
                    "name": "fps",
                    "type": "FLOAT",
                    "link": 4,
                    "widget": {"name": "fps"},
                },
                {"name": "audio", "type": "AUDIO", "link": 5, "shape": 7},
            ],
            [
                {"name": "candidate_frames", "type": "IMAGE", "links": [6, 8]},
                {"name": "source_frames", "type": "IMAGE", "links": [7]},
                {"name": "audio", "type": "AUDIO", "links": [9]},
                {"name": "report_json", "type": "STRING", "links": None},
            ],
            [24.0, "sr_nr", "2.0", *QUALITY_WIDGETS, "auto"],
        ),
        _node(
            6,
            "PreviewImage",
            "原片帧 / Source",
            [1850, 350],
            [390, 180],
            5,
            [{"name": "images", "type": "IMAGE", "link": 7}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            7,
            "PreviewImage",
            "DLSS-NR 候选帧",
            [1850, 580],
            [390, 180],
            6,
            [{"name": "images", "type": "IMAGE", "link": 6}],
            [{"name": "images", "type": "IMAGE", "links": None}],
            [],
        ),
        _node(
            8,
            "CreateVideo",
            "5. 用原帧率和原音频合成",
            [1850, 820],
            [440, 280],
            7,
            [
                {"name": "images", "type": "IMAGE", "link": 8},
                {"name": "audio", "type": "AUDIO", "link": 9, "shape": 7},
                {
                    "name": "fps",
                    "type": "FLOAT",
                    "link": 10,
                    "widget": {"name": "fps"},
                },
                {
                    "name": "bit_depth",
                    "type": "COMBO",
                    "link": 11,
                    "widget": {"name": "bit_depth"},
                    "shape": 7,
                },
            ],
            [{"name": "VIDEO", "type": "VIDEO", "links": [12]}],
            [24.0, 8, "sRGB"],
        ),
        _node(
            9,
            "SaveVideo",
            "6. 保存候选视频",
            [2380, 820],
            [420, 240],
            8,
            [{"name": "video", "type": "VIDEO", "link": 12}],
            [{"name": "video", "type": "VIDEO", "links": None}],
            ["MiniMaxH3/DLSS-NR/video_frames_standard_2x"],
        ),
    ]
    links = [
        [1, 3, 0, 4, 0, "VIDEO"],
        [2, 2, 0, 5, 0, "T8_DLSS_NR_RUNTIME"],
        [3, 4, 0, 5, 1, "IMAGE"],
        [4, 4, 2, 5, 2, "FLOAT"],
        [5, 4, 1, 5, 3, "AUDIO"],
        [6, 5, 0, 7, 0, "IMAGE"],
        [7, 5, 1, 6, 0, "IMAGE"],
        [8, 5, 0, 8, 0, "IMAGE"],
        [9, 5, 2, 8, 1, "AUDIO"],
        [10, 4, 2, 8, 2, "FLOAT"],
        [11, 4, 3, 8, 3, "COMBO"],
        [12, 8, 0, 9, 0, "VIDEO"],
    ]
    return _workflow("video-frames", nodes, links)


def _video_file_workflow() -> dict:
    nodes = [
        _node(
            1,
            "MarkdownNote",
            "使用说明 / Read first",
            [0, 0],
            [1100, 300],
            0,
            [],
            [],
            [
                COMMON_NOTE
                + "\n\n本工作流不把长片完整物化为 IMAGE batch；只接受未裁切的 SDR 8-bit Rec.709 CFR 文件视频。"
            ],
        ),
        _runtime_node(2, 1, links=[2]),
        _node(
            3,
            "LoadVideo",
            "2. 选择未裁切的CFR成片",
            [0, 680],
            [500, 150],
            2,
            [],
            [{"name": "VIDEO", "type": "VIDEO", "links": [1]}],
            ["replace_with_h3_video.mp4"],
        ),
        _node(
            4,
            VIDEO_FILE,
            "3. Standard 2x 文件流式超分",
            [620, 360],
            [680, 850],
            3,
            [
                {"name": "dlss_nr_runtime", "type": "T8_DLSS_NR_RUNTIME", "link": 2},
                {"name": "source_video", "type": "VIDEO", "link": 1},
            ],
            [
                {"name": "candidate_video", "type": "VIDEO", "links": None},
                {"name": "source_video", "type": "VIDEO", "links": None},
                {"name": "saved_path", "type": "STRING", "links": None},
                {"name": "report_json", "type": "STRING", "links": None},
            ],
            [
                "sr_nr",
                "2.0",
                *QUALITY_WIDGETS,
                "auto",
                "MiniMaxH3/DLSS-NR/video_file_standard_2x",
                18.0,
            ],
        ),
    ]
    links = [
        [1, 3, 0, 4, 1, "VIDEO"],
        [2, 2, 0, 4, 0, "T8_DLSS_NR_RUNTIME"],
    ]
    return _workflow("video-file", nodes, links)


def build_workflows() -> dict[str, dict]:
    return {
        "runtime": _runtime_workflow(),
        "image": _image_workflow(),
        "video_frames": _video_frames_workflow(),
        "video_file": _video_file_workflow(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    USER_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, workflow in build_workflows().items():
        path = OUTPUT / FILENAMES[name]
        path.write_bytes(
            (json.dumps(workflow, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            )
        )
        mirror = USER_OUTPUT / path.name
        shutil.copy2(path, mirror)
        print(path)
        print(mirror)


if __name__ == "__main__":
    main()
