#!/usr/bin/env python3
"""Build formal OpenVDN-H3 T2VA and native multimodal frontend workflows."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_openvdn_h3_multimodal_validation as validation  # noqa: E402


SOURCE_NAME = "2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json"
OUTPUT_NAMES = {
    "t2va": "2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced.json",
    "i2va": "2026-09-03_H3_OpenVDN_DMD8_I2VA_Advanced.json",
    "l2va": "2026-09-03_H3_OpenVDN_DMD8_L2VA_0p5MP_Advanced.json",
    "fl2va": "2026-09-03_H3_OpenVDN_DMD8_FL2VA_0p5MP_Advanced.json",
    "ref2va": "2026-09-03_H3_OpenVDN_DMD8_Ref2VA_Image_0p5MP_Advanced.json",
    "multi_ref_images": "2026-09-03_H3_OpenVDN_DMD8_Ref2VA_MultiImage_Advanced.json",
    "ref_video_audio": "2026-09-03_H3_OpenVDN_DMD8_Ref2VA_VideoAudio_Advanced.json",
    "ref_audio": "2026-09-03_H3_OpenVDN_DMD8_Ref2VA_Audio_Advanced.json",
    "hybrid_first_audio": "2026-09-03_H3_OpenVDN_DMD8_Hybrid_FirstAudio_Advanced.json",
}
WORKFLOW_IDS = {
    "t2va": "40f1cae6-3ad6-45db-bda1-6d7ae1797600",
    "i2va": "d406be1e-ee15-47ab-92d3-07aa8ea4495c",
    "l2va": "c87aad24-b020-4b93-a61a-6656610754ce",
    "fl2va": "c0a924ee-b62e-42b0-be64-9480dedb58f0",
    "ref2va": "2cde16e6-d690-4883-b044-bbbd21164a99",
    "multi_ref_images": "8823999d-9583-48bc-8110-902e8e14ed95",
    "ref_video_audio": "731bb082-35ff-4f02-b110-765802882e77",
    "ref_audio": "19161f95-0d90-49c6-9823-220038c59bea",
    "hybrid_first_audio": "62e7c4f2-b4d6-4f8c-ad1f-867c959a4d9e",
}
PROFILES = {
    "t2va": (960, 512, 73),
    "i2va": (736, 416, 73),
    "l2va": (960, 544, 39),
    "fl2va": (960, 544, 39),
    "ref2va": (960, 544, 39),
    "multi_ref_images": (736, 416, 39),
    "ref_video_audio": (736, 416, 39),
    "ref_audio": (736, 416, 39),
    "hybrid_first_audio": (736, 416, 39),
}

T2VA_PROMPT = (
    "One continuous locked-off cinematic medium close-up of one adult East Asian woman "
    "facing the camera in a quiet, softly lit concert room. She says exactly once in "
    "Mandarin Chinese: <d>[Chinese] 你在哪里</d>. Clean classical chamber music plays "
    "underneath: warm solo cello and soft acoustic piano, with the voice clear in the "
    "foreground. Stable face and anatomy, one continuous shot. No extra words, singing, "
    "subtitles, cuts, hiss, static, crackle, clipping or distortion."
)


def _node(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def _shift_orders(workflow: dict[str, Any], amount: int) -> None:
    for node in workflow["nodes"]:
        if int(node.get("order", 0)) >= 4:
            node["order"] = int(node["order"]) + amount


def _convert_native_save(workflow: dict[str, Any], variant: str) -> None:
    create = _node(workflow, 12)
    create.update(
        {
            "type": "CreateVideo",
            "title": "Create exact 24fps OpenVDN audio/video",
            "size": [380, 264],
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": 17},
                {"name": "audio", "type": "AUDIO", "link": 18, "shape": 7},
            ],
            "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": [19]}],
            "properties": {
                "cnr_id": "comfy-core",
                "Node name for S&R": "CreateVideo",
            },
            "widgets_values": [24.0, 8, "sRGB"],
        }
    )
    note = _node(workflow, 13)
    note["order"] = 13
    note["title"] = "OpenVDN MiniMax H3 multimodal support boundary"
    note["widgets_values"] = [
        "## OpenVDN H3 · formal T8 multimodal route\n\n"
        "This workflow uses the pinned OpenVDN branch/adapters with native H3 PackedLayout "
        "conditioning. T8 has real-validated T2VA, first frame, last frame, first+last, "
        "single/multiple reference images, reference video with matching audio, standalone "
        "reference audio, and first-frame+audio Hybrid. Upstream OpenVDN itself declares "
        "T2VA; the other modes are T8-validated extensions.\n\n"
        "Do not stack EMA_B/Turbo LoRA, SLA, VSA, Sol-Attn, BlockCache or another MODEL/attention "
        "patch. DMD requires the full-width 2688-column AdaLN INT8/ConvRot base; do not use an "
        "adaln_t_table curve-basis/pruned checkpoint because its 8-column AdaLN cannot accept "
        "the published turbo adapter's 51 AdaLN LoRA targets. The supplied base is not claimed "
        "byte-identical to the upstream BF16 base. OpenVDN weights use the MiniMax H3 Community "
        "License and its territory restrictions must be reviewed before use.\n\n"
        "The 16GB validation profiles run close to the VRAM limit. Close other GPU applications "
        "or reduce canvas size if necessary."
    ]
    save = {
        "id": 14,
        "type": "SaveVideo",
        "title": f"Save OpenVDN {variant.upper()} result",
        "pos": [2590, 420],
        "size": [380, 238],
        "flags": {},
        "order": 12,
        "mode": 0,
        "inputs": [
            {"name": "video", "type": "VIDEO", "link": 19},
            {"name": "format", "type": "COMFY_DYNAMICCOMBO_V3", "link": None},
            {
                "name": "codec",
                "type": "COMFY_DYNAMICCOMBO_V3",
                "link": None,
                "shape": 7,
            },
        ],
        "outputs": [{"name": "video", "type": "VIDEO", "links": None}],
        "properties": {
            "cnr_id": "comfy-core",
            "Node name for S&R": "SaveVideo",
        },
        "widgets_values": [f"MiniMaxH3/OpenVDN/{variant}"],
    }
    workflow["nodes"].append(save)
    workflow["links"].append([19, 12, 0, 14, 0, "VIDEO"])
    workflow["last_node_id"] = 14
    workflow["last_link_id"] = 19


def _add_image(
    workflow: dict[str, Any], *, title: str, filename: str, target_name: str
) -> None:
    node_id = int(workflow["last_node_id"]) + 1
    link_id = int(workflow["last_link_id"]) + 1
    conditioning = _node(workflow, 6)
    existing = next(
        (item for item in conditioning["inputs"] if item["name"] == target_name),
        None,
    )
    if existing is None:
        existing = {
            "name": target_name,
            "type": "IMAGE",
            "link": None,
            "shape": 7,
        }
        conditioning["inputs"].append(existing)
    existing["link"] = link_id
    target_slot = conditioning["inputs"].index(existing)
    workflow["nodes"].append(
        {
            "id": node_id,
            "type": "LoadImage",
            "title": title,
            "pos": [-120, 730 + 330 * (node_id - 15)],
            "size": [390, 310],
            "flags": {},
            "order": 4 + (node_id - 15),
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [link_id]},
                {"name": "MASK", "type": "MASK", "links": None},
            ],
            "properties": {
                "cnr_id": "comfy-core",
                "Node name for S&R": "LoadImage",
            },
            "widgets_values": [filename, "image"],
        }
    )
    workflow["links"].append(
        [link_id, node_id, 0, 6, target_slot, "IMAGE"]
    )
    workflow["last_node_id"] = node_id
    workflow["last_link_id"] = link_id


def _add_reference_video(
    workflow: dict[str, Any], *, include_frames: bool, include_audio: bool
) -> None:
    first_node = int(workflow["last_node_id"]) + 1
    first_link = int(workflow["last_link_id"]) + 1
    load_id, slice_id, components_id = first_node, first_node + 1, first_node + 2
    order_offset = first_node - 15
    load_to_slice, slice_to_components = first_link, first_link + 1
    next_link = first_link + 2
    conditioning = _node(workflow, 6)
    image_link = next_link if include_frames else None
    audio_link = next_link + int(include_frames) if include_audio else None

    workflow["nodes"].extend(
        [
            {
                "id": load_id,
                "type": "LoadVideo",
                "title": "Replace with 2-15 second reference video",
                "pos": [-120, 760],
                "size": [390, 124],
                "flags": {},
                "order": 4 + order_offset,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "VIDEO", "type": "VIDEO", "links": [load_to_slice]}
                ],
                "properties": {
                    "cnr_id": "comfy-core",
                    "Node name for S&R": "LoadVideo",
                },
                "widgets_values": ["replace_with_reference_video.mp4"],
            },
            {
                "id": slice_id,
                "type": "Video Slice",
                "title": "Use first exact two seconds (48 frames at 24fps)",
                "pos": [330, 760],
                "size": [390, 238],
                "flags": {},
                "order": 5 + order_offset,
                "mode": 0,
                "inputs": [
                    {"name": "video", "type": "VIDEO", "link": load_to_slice}
                ],
                "outputs": [
                    {
                        "name": "VIDEO",
                        "type": "VIDEO",
                        "links": [slice_to_components],
                    }
                ],
                "properties": {
                    "cnr_id": "comfy-core",
                    "Node name for S&R": "Video Slice",
                },
                "widgets_values": [0.0, 2.0, True],
            },
            {
                "id": components_id,
                "type": "GetVideoComponents",
                "title": "Reference frames and matching soundtrack",
                "pos": [780, 760],
                "size": [390, 110],
                "flags": {},
                "order": 6 + order_offset,
                "mode": 0,
                "inputs": [
                    {
                        "name": "video",
                        "type": "VIDEO",
                        "link": slice_to_components,
                    }
                ],
                "outputs": [
                    {"name": "images", "type": "IMAGE", "links": [image_link] if image_link else None},
                    {"name": "audio", "type": "AUDIO", "links": [audio_link] if audio_link else None},
                    {"name": "fps", "type": "FLOAT", "links": None},
                    {"name": "bit_depth", "type": "INT", "links": None},
                ],
                "properties": {
                    "cnr_id": "comfy-core",
                    "Node name for S&R": "GetVideoComponents",
                },
                "widgets_values": [],
            },
        ]
    )
    workflow["links"].extend(
        [
            [load_to_slice, load_id, 0, slice_id, 0, "VIDEO"],
            [slice_to_components, slice_id, 0, components_id, 0, "VIDEO"],
        ]
    )
    if include_frames:
        target_name = "ref_videos.ref_video_0"
        conditioning["inputs"].append(
            {"name": target_name, "type": "IMAGE", "link": image_link, "shape": 7}
        )
        workflow["links"].append(
            [
                image_link,
                components_id,
                0,
                6,
                len(conditioning["inputs"]) - 1,
                "IMAGE",
            ]
        )
    if include_audio:
        target_name = (
            "ref_video_audios.ref_video_audio_0"
            if include_frames
            else "ref_audios.ref_audio_0"
        )
        conditioning["inputs"].append(
            {"name": target_name, "type": "AUDIO", "link": audio_link, "shape": 7}
        )
        workflow["links"].append(
            [
                audio_link,
                components_id,
                1,
                6,
                len(conditioning["inputs"]) - 1,
                "AUDIO",
            ]
        )
    workflow["last_node_id"] = components_id
    workflow["last_link_id"] = max(
        link for link in (slice_to_components, image_link, audio_link) if link is not None
    )


def build_variant(source: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in OUTPUT_NAMES:
        raise ValueError(f"unknown OpenVDN workflow variant: {variant}")
    workflow = copy.deepcopy(source)
    workflow["id"] = WORKFLOW_IDS[variant]
    workflow["revision"] = 0
    workflow.setdefault("extra", {})["workflow_name"] = (
        f"MiniMax H3 OpenVDN DMD8 {variant} (Advanced)"
    )
    _convert_native_save(workflow, variant)

    width, height, frames = PROFILES[variant]
    conditioning = _node(workflow, 6)
    prompt = T2VA_PROMPT if variant == "t2va" else validation.VARIANTS[variant].prompt
    task_type = "T2VA" if variant == "t2va" else validation.VARIANTS[variant].task_type
    conditioning["title"] = (
        f"3. OpenVDN {task_type} · {width}x{height} · {frames} frames"
    )
    conditioning["widgets_values"] = [
        prompt,
        width,
        height,
        frames,
        task_type,
        "native",
        1.0,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
    ]

    loader_count = {
        "t2va": 0,
        "i2va": 1,
        "l2va": 1,
        "fl2va": 2,
        "ref2va": 1,
        "multi_ref_images": 2,
        "ref_video_audio": 3,
        "ref_audio": 3,
        "hybrid_first_audio": 4,
    }[variant]
    _shift_orders(workflow, loader_count)

    if variant == "i2va":
        _add_image(
            workflow,
            title="Replace with first frame",
            filename="replace_with_first_frame.png",
            target_name="first_frame",
        )
    elif variant == "l2va":
        _add_image(
            workflow,
            title="Replace with final frame",
            filename="replace_with_final_frame.png",
            target_name="last_frame",
        )
    elif variant == "fl2va":
        _add_image(
            workflow,
            title="Replace with first frame",
            filename="replace_with_first_frame.png",
            target_name="first_frame",
        )
        _add_image(
            workflow,
            title="Replace with final frame",
            filename="replace_with_final_frame.png",
            target_name="last_frame",
        )
    elif variant == "ref2va":
        _add_image(
            workflow,
            title="Replace with visual reference",
            filename="replace_with_reference_image.png",
            target_name="ref_images.ref_image_0",
        )
    elif variant == "multi_ref_images":
        for index in range(2):
            _add_image(
                workflow,
                title=f"Replace with identity reference {index + 1}",
                filename=f"replace_with_reference_image_{index + 1}.png",
                target_name=f"ref_images.ref_image_{index}",
            )
    elif variant == "ref_video_audio":
        _add_reference_video(workflow, include_frames=True, include_audio=True)
    elif variant == "ref_audio":
        _add_reference_video(workflow, include_frames=False, include_audio=True)
    elif variant == "hybrid_first_audio":
        _add_image(
            workflow,
            title="Replace with first frame",
            filename="replace_with_first_frame.png",
            target_name="first_frame",
        )
        _add_reference_video(workflow, include_frames=False, include_audio=True)

    _node(workflow, 12)["pos"] = [2160, 420]
    _node(workflow, 13)["pos"] = [360, 760 + 340 * max(1, loader_count)]
    return workflow


def build_all(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: build_variant(source, name) for name in OUTPUT_NAMES}


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "examples" / "workflows" / "10-speed" / SOURCE_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "examples" / "workflows" / "10-speed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, workflow in build_all(source).items():
        path = args.output_dir / OUTPUT_NAMES[variant]
        path.write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
