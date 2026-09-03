#!/usr/bin/env python3
"""Run repeatable real OpenVDN-H3 multimodal DMD8 validation renders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_openvdn_h3_validation as base  # noqa: E402


SCHEMA = "t8.minimax_h3.openvdn.multimodal_real_validation.v1"
DEFAULT_IMAGE = "ComfyUI_00031_pkyly_1787418403.png"
DEFAULT_FIRST_IMAGE = "codex_prompt_relay_fl2va_first.png"
DEFAULT_LAST_IMAGE = "codex_prompt_relay_fl2va_last.png"
DEFAULT_REF_IMAGE_1 = "t8_multiface_2women_dark_reference_imagegen_v2.png"
DEFAULT_REF_IMAGE_2 = "t8_multiface_2women_blonde_reference_imagegen_v2.png"
DEFAULT_REF_VIDEO = "openvdn_h3_reference_48f_320x192.mp4"


@dataclass(frozen=True)
class Variant:
    task_type: str
    prompt: str
    first: bool = False
    last: bool = False
    ref_images: int = 0
    ref_video_audio: bool = False
    ref_audio: bool = False


VARIANTS = {
    "i2va": Variant(
        task_type="I2VA",
        first=True,
        prompt=(
            "Animate the supplied first frame as one continuous cinematic close-up. The same "
            "young cybernetic woman remains beside the same white robotic rabbit, with identical "
            "face, grey eyes, black hair, white-and-pink mechanical clothing and white background. "
            "She gently breathes, blinks once and slowly raises her head a few degrees. Stable "
            "camera, coherent motion and crisp details. Clean room ambience only; no speech, "
            "singing, music, cuts, identity change, blur, hiss, static or distortion."
        ),
    ),
    "l2va": Variant(
        task_type="L2VA",
        last=True,
        prompt=(
            "One continuous cinematic crane shot on the same rain-soaked neon street. Begin with "
            "a medium view of one woman in a red coat, then rise and pull back smoothly, ending "
            "exactly on the supplied final aerial street frame. Preserve her red coat, wet road, "
            "cyan and magenta signs and night lighting. Clean city ambience only; no speech, music, "
            "cuts, teleporting, duplicate person, blur, flicker, hiss, static or distortion."
        ),
    ),
    "fl2va": Variant(
        task_type="FL2VA",
        first=True,
        last=True,
        prompt=(
            "Create one continuous cinematic transition from the supplied first street portrait "
            "to the supplied final aerial street view. The same woman in the red coat turns away "
            "and walks forward while the camera cranes upward and backward. Preserve the wet neon "
            "street, red coat and cyan-magenta night palette, matching both endpoint frames exactly. "
            "Clean city ambience only; no speech, music, cuts, teleporting, duplicate person, blur, "
            "flicker, hiss, static or distortion."
        ),
    ),
    "ref2va": Variant(
        task_type="Ref2VA",
        ref_images=1,
        prompt=(
            "Use <Picture 1> as the only visual reference. Create one continuous cinematic medium "
            "close-up of the same young cybernetic woman in a clean futuristic repair studio. "
            "Preserve her face, grey eyes, black hair, white-and-pink mechanical clothing and small "
            "white robotic rabbit. She looks toward the camera and blinks once. Clean room ambience "
            "only; no speech, music, cuts, identity change, blur, hiss, static or distortion."
        ),
    ),
    "multi_ref_images": Variant(
        task_type="Ref2VA",
        ref_images=2,
        prompt=(
            "Use <Picture 1> and <Picture 2> as two separate identity references. Show both adult "
            "women together in one stable cinematic two-shot inside a quiet modern gallery: the "
            "curly dark-haired woman on the left and the blonde woman with black glasses on the "
            "right. Preserve each face, hair and eyewear without merging identities. They glance "
            "toward one another naturally. Clean room ambience only; no speech, music, cuts, extra "
            "people, face fusion, identity swap, blur, hiss, static or distortion."
        ),
    ),
    "ref_video_audio": Variant(
        task_type="Ref2VA",
        ref_video_audio=True,
        prompt=(
            "Use <Video 1> as the motion and scene reference and <Audio 1> as its matching soundtrack "
            "reference. Create one continuous coherent shot that preserves the reference movement "
            "rhythm and audiovisual atmosphere while keeping natural anatomy and stable detail. "
            "No cuts, duplicate subjects, frozen motion, blur, hiss, static, crackle or distortion."
        ),
    ),
    "ref_audio": Variant(
        task_type="Ref2VA",
        ref_audio=True,
        prompt=(
            "Use <Audio 1> as the only audio reference. Create one continuous cinematic medium shot "
            "of a solo dancer in a dark rehearsal studio moving naturally with the reference rhythm. "
            "Preserve a clean version of the referenced sound without adding speech or vocals. "
            "Stable camera and anatomy; no cuts, duplicate person, frozen motion, hiss, static, "
            "crackle, clipping or distortion."
        ),
    ),
    "hybrid_first_audio": Variant(
        task_type="Hybrid",
        first=True,
        ref_audio=True,
        prompt=(
            "Animate the supplied first frame while using <Audio 1> as the only audio reference. "
            "Preserve the cybernetic woman, white robotic rabbit, face, clothing and background. "
            "She makes one subtle head movement in time with the clean referenced sound. No speech, "
            "cuts, identity change, duplicate subject, blur, hiss, static, crackle or distortion."
        ),
    ),
}


def _load_image(graph: dict[str, Any], node_id: int, filename: str) -> list[Any]:
    graph[str(node_id)] = {
        "class_type": "LoadImage",
        "inputs": {"image": filename},
        "_meta": {"title": f"OpenVDN validation image: {filename}"},
    }
    return [str(node_id), 0]


def _load_reference_av(
    graph: dict[str, Any], node_id: int, filename: str
) -> tuple[list[Any], list[Any]]:
    graph[str(node_id)] = {
        "class_type": "LoadVideo",
        "inputs": {"file": filename},
        "_meta": {"title": "OpenVDN 48-frame reference video"},
    }
    graph[str(node_id + 1)] = {
        "class_type": "GetVideoComponents",
        "inputs": {"video": [str(node_id), 0]},
        "_meta": {"title": "Reference frames and matching soundtrack"},
    }
    return [str(node_id + 1), 0], [str(node_id + 1), 1]


def build_variant_prompt(
    args: argparse.Namespace, run_id: str, *, variant_name: str
) -> dict[str, Any]:
    variant = VARIANTS[variant_name]
    graph = _BASE_BUILD_PROMPT(args, run_id)
    inputs = graph["6"]["inputs"]
    inputs["prompt"] = variant.prompt
    inputs["task_type"] = variant.task_type
    node_id = 20

    if variant.first:
        filename = args.image if variant_name in {"i2va", "hybrid_first_audio"} else args.first_image
        inputs["first_frame"] = _load_image(graph, node_id, filename)
        node_id += 1
    if variant.last:
        inputs["last_frame"] = _load_image(graph, node_id, args.last_image)
        node_id += 1
    if variant.ref_images:
        filenames = (
            [args.image]
            if variant.ref_images == 1
            else [args.ref_image_1, args.ref_image_2]
        )
        for index, filename in enumerate(filenames):
            inputs[f"ref_images.ref_image_{index}"] = _load_image(
                graph, node_id, filename
            )
            node_id += 1
    if variant.ref_video_audio or variant.ref_audio:
        frames, audio = _load_reference_av(graph, node_id, args.ref_video)
        if variant.ref_video_audio:
            inputs["ref_videos.ref_video_0"] = frames
            inputs["ref_video_audios.ref_video_audio_0"] = audio
        if variant.ref_audio:
            inputs["ref_audios.ref_audio_0"] = audio

    graph["16"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["6", 5]},
    }
    graph["17"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["12", 0],
            "audio": ["12", 1],
            "fps": float(base.FPS),
            "bit_depth": 8,
        },
        "_meta": {"title": "Create strict native OpenVDN validation video"},
    }
    graph["18"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["17", 0],
            "filename_prefix": (
                f"MiniMaxH3_OpenVDN_Native/{run_id}_{variant_name}_"
                f"{args.width}x{args.height}_{args.frame_count}f_dmd8"
            ),
            "format": "mp4",
            "codec": "h264",
        },
        "_meta": {"title": "Save native OpenVDN validation result"},
    }
    return graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comfy-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--frame-count", type=int, default=39)
    parser.add_argument("--base-model", default=base.BASE_MODEL)
    parser.add_argument("--seed", type=int, default=2609032401)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--reserve-vram-gib", type=float, default=0.5)
    parser.add_argument("--lowvram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--first-image", default=DEFAULT_FIRST_IMAGE)
    parser.add_argument("--last-image", default=DEFAULT_LAST_IMAGE)
    parser.add_argument("--ref-image-1", default=DEFAULT_REF_IMAGE_1)
    parser.add_argument("--ref-image-2", default=DEFAULT_REF_IMAGE_2)
    parser.add_argument("--ref-video", default=DEFAULT_REF_VIDEO)
    parser.add_argument("--artifact-root", type=Path)
    return parser


def _required_inputs(args: argparse.Namespace) -> list[str]:
    variant = VARIANTS[args.variant]
    required: list[str] = []
    if variant.first:
        required.append(
            args.image
            if args.variant in {"i2va", "hybrid_first_audio"}
            else args.first_image
        )
    if variant.last:
        required.append(args.last_image)
    if variant.ref_images == 1:
        required.append(args.image)
    elif variant.ref_images == 2:
        required.extend([args.ref_image_1, args.ref_image_2])
    if variant.ref_video_audio or variant.ref_audio:
        required.append(args.ref_video)
    return required


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_root = args.comfy_root.resolve() / "input"
    missing = [name for name in _required_inputs(args) if not (input_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing ComfyUI input asset(s): {', '.join(missing)}")

    artifact_root = args.artifact_root
    if artifact_root is None:
        artifact_root = Path(
            f"artifacts/openvdn-h3-{args.variant.replace('_', '-')}-validation-20260903"
        )

    base.SCHEMA = SCHEMA
    base.build_prompt = lambda _inner_args, run_id: build_variant_prompt(
        args, run_id, variant_name=args.variant
    )
    base_args = [
        "--project-root",
        str(args.project_root),
        "--comfy-root",
        str(args.comfy_root),
        "--python",
        str(args.python),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frame-count",
        str(args.frame_count),
        "--base-model",
        args.base_model,
        "--seed",
        str(args.seed),
        "--server-start-timeout",
        str(args.server_start_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--min-free-vram-mib",
        str(args.min_free_vram_mib),
        "--reserve-vram-gib",
        str(args.reserve_vram_gib),
        "--artifact-root",
        str(artifact_root),
        "--lowvram" if args.lowvram else "--no-lowvram",
    ]
    rc = base.main(base_args)

    resolved_artifact_root = (
        artifact_root
        if artifact_root.is_absolute()
        else args.project_root.resolve() / artifact_root
    )
    reports = sorted(
        resolved_artifact_root.glob("*/validation_report.json"),
        key=lambda item: item.stat().st_mtime_ns,
    )
    if reports:
        report_path = reports[-1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        variant = VARIANTS[args.variant]
        report["schema"] = SCHEMA
        report["contract"]["task"] = variant.task_type
        report["contract"]["validation_variant"] = args.variant
        report["contract"]["input_assets"] = _required_inputs(args)
        report["support_claim"] = (
            "T8 native-packed-layout extension validation; upstream OpenVDN declares T2VA."
        )
        stderr_path = Path(report["run_root"]) / "logs" / "openvdn_h3.stderr.log"
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        report.setdefault("composition_checks", {}).update(
            base.adapter_integrity_checks(report.get("composition", {}), stderr_text)
        )
        report["adapter_runtime"] = {
            "stderr_log": str(stderr_path.resolve()),
            "lora_error_count": stderr_text.count("ERROR lora"),
        }
        native_output_root = (
            Path(report["run_root"]) / "output" / "MiniMaxH3_OpenVDN_Native"
        )
        native_outputs = sorted(
            native_output_root.glob("*.mp4"),
            key=lambda item: item.stat().st_mtime_ns,
        )
        if native_outputs:
            native_video = native_outputs[-1]
            report["vhs_output_video"] = report.get("output_video")
            report["vhs_media"] = report.get("media")
            media = base.stable_media_report(
                native_video,
                ffmpeg=str(report["preflight"]["ffmpeg"]),
                ffprobe=str(report["preflight"]["ffprobe"]),
            )
            audio = base.pdd._audio_numeric(
                native_video, str(report["preflight"]["ffmpeg"])
            )
            media_checks = base.pdd._media_checks(
                media,
                audio,
                width=int(report["contract"]["width"]),
                height=int(report["contract"]["height"]),
                frame_count=int(report["contract"]["frame_count"]),
            )
            native_contact = Path(report["run_root"]) / "native_contact_sheet.png"
            base.pdd._contact_sheet(
                native_video,
                native_contact,
                str(report["preflight"]["ffmpeg"]),
                width=int(report["contract"]["width"]),
                height=int(report["contract"]["height"]),
                frame_count=int(report["contract"]["frame_count"]),
            )
            report["output_video"] = str(native_video.resolve())
            report["contact_sheet"] = str(native_contact.resolve())
            report["media"] = media
            report["audio_numeric"] = audio
            report["media_checks"] = media_checks
            report["native_save"] = True
            core_passed = all(media_checks.values()) and all(
                report.get("composition_checks", {}).values()
            )
            headroom_passed = all(report.get("resource_checks", {}).values())
            if core_passed:
                report["status"] = (
                    "MECHANICAL_PASS_HUMAN_REVIEW_PENDING"
                    if headroom_passed
                    else "MECHANICAL_PASS_LOW_HEADROOM_HUMAN_REVIEW_PENDING"
                )
                rc = 0
            else:
                report["status"] = "FAIL_MECHANICAL"
                rc = 1
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return rc


_BASE_BUILD_PROMPT = base.build_prompt


if __name__ == "__main__":
    raise SystemExit(main())
