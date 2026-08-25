#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

import av
from PIL import Image, ImageDraw, ImageFont

import run_skin_finish_live_sam31_validation as base


DEFAULT_SOURCE = Path(r"C:\ComfyUI_00001_qirzb_1786961596.mp4")
DEFAULT_OUTPUT = (
    base.ROOT
    / "artifacts"
    / "skin-finish-per-person-profile-live-validation-20260825"
)
STRICT_BASELINE_REPORT = (
    base.ROOT
    / "artifacts"
    / "skin-finish-per-person-live-validation-v2-20260825"
    / "20260825-011242-f640feee"
    / "validation_report.json"
)
TARGET_WIDTH = 960
TARGET_HEIGHT = 704
SAMPLE_FRAMES = (0, 17, 34, 51, 68)
PREVIEW_START_ID = 22
PREVIEW_SOURCE_COUNT = 4
AUDIT_NODE_ID = PREVIEW_START_ID + len(SAMPLE_FRAMES) * 2 * PREVIEW_SOURCE_COUNT
AUDIT_REPORT_NODE_ID = AUDIT_NODE_ID + 1
AUDIT_PREVIEW_NODE_ID = AUDIT_NODE_ID + 2


def _sample_previews(
    prompt: dict[str, Any],
    *,
    first_id: int,
    image_source: list[Any],
) -> int:
    node_id = first_id
    for frame_index in SAMPLE_FRAMES:
        prompt[str(node_id)] = {
            "class_type": "ImageFromBatch",
            "inputs": {
                "image": image_source,
                "batch_index": frame_index,
                "length": 1,
            },
        }
        prompt[str(node_id + 1)] = {
            "class_type": "PreviewImage",
            "inputs": {"images": [str(node_id), 0]},
        }
        node_id += 2
    return node_id


def _build_prompt(source_name: str) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": source_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": base.SAM_MODEL.name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "person",
                "clip": ["3", 1],
            },
        },
        "5": {
            "class_type": "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            "inputs": {
                "frames": ["2", 0],
                "model": ["3", 0],
                "conditioning": ["4", 0],
                "fps": 24.0,
                "maximum_people": 2,
                "detection_threshold": 0.35,
                "detect_interval": 3,
                "scene_cut_threshold": 0.28,
                "analysis_max_side": 640,
                "preview_stride": 6,
                "release_policy": "offload_sam31_after_track",
            },
        },
        "6": {
            "class_type": (
                "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
            ),
            "inputs": {
                "frames": ["2", 0],
                "track_plan": ["5", 0],
                "parser_model": "facexlib_parsenet_v0.2.2_pinned",
                "detection_threshold": 0.45,
                "minimum_face_height_px": 32.0,
                "minimum_detail": 0.01,
                "minimum_person_overlap": 0.20,
                "minimum_track_quality": 0.10,
                "minimum_class_probability": 0.55,
                "feature_protection_px": 3,
                "include_neck": False,
                "minimum_skin_area_per_face": 0.00005,
                "maximum_skin_area_per_frame": 0.35,
                "maximum_alignment_rms": 0.08,
                "profile_crop_expansion": 1.45,
                "minimum_ready_frame_fraction": 0.50,
                "preview_count": 5,
            },
        },
        "7": {
            "class_type": "MiniMaxH3SkinFinishPersonProfileT8Advanced",
            "inputs": {
                "selector_type": "shot_track",
                "selector": "0:0",
                "preset": "subtle",
                "amount": 0.25,
                "texture_keep": 0.95,
                "shine_control": 0.25,
                "tone_adjust": 0.0,
            },
        },
        "8": {
            "class_type": "MiniMaxH3SkinFinishPersonProfileT8Advanced",
            "inputs": {
                "selector_type": "shot_track",
                "selector": "0:1",
                "preset": "oil_control",
                "amount": 0.55,
                "texture_keep": 0.88,
                "shine_control": 0.60,
                "tone_adjust": 0.0,
                "previous_profiles": ["7", 0],
            },
        },
        "9": {
            "class_type": "MiniMaxH3SkinFinishPerPersonT8Advanced",
            "inputs": {
                "frames": ["2", 0],
                "track_plan": ["5", 0],
                "semantic_skin_mask": ["6", 0],
                "semantic_report_json": ["6", 2],
                "default_policy": "source_unmatched",
                "default_preset": "subtle",
                "default_amount": 0.35,
                "default_texture_keep": 0.90,
                "default_shine_control": 0.35,
                "default_tone_adjust": 0.0,
                "execution_mode": "candidate_only",
                "accept_candidate": False,
                "chunk_frames": 2,
                "proxy_long_side": 640,
                "preview_count": 5,
                "profiles": ["8", 0],
                "audio": ["2", 1],
            },
        },
        "10": {
            "class_type": "MiniMaxH3SkinFinishTextureGuardT8Advanced",
            "inputs": {
                "source_frames": ["9", 1],
                "candidate_frames": ["9", 0],
                "used_skin_mask": ["9", 4],
                "shadow_protection": 0.10,
                "highlight_protection": 0.94,
                "transition_width": 0.06,
                "minimum_texture_ratio": 0.78,
                "minimum_reference_texture": 0.003,
                "maximum_new_clipped_fraction": 0.0005,
                "clipping_epsilon": 1.0 / 255.0,
                "texture_radius": 1,
                "chunk_frames": 2,
                "accept_candidate": False,
                "audio": ["9", 3],
            },
        },
        "11": {
            "class_type": "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            "inputs": {
                "source_video": ["1", 0],
                "processed_frames": ["10", 0],
                "filename_prefix": "MiniMaxH3/SkinFinish/per_person_profile_live",
                "crf": 18.0,
                "accept_candidate": True,
            },
        },
        "12": {"class_type": "PreviewImage", "inputs": {"images": ["5", 1]}},
        "13": {"class_type": "PreviewImage", "inputs": {"images": ["6", 1]}},
        "14": {"class_type": "PreviewImage", "inputs": {"images": ["9", 6]}},
        "15": {"class_type": "MaskToImage", "inputs": {"mask": ["9", 4]}},
        "16": {"class_type": "PreviewImage", "inputs": {"images": ["15", 0]}},
        "17": {"class_type": "PreviewAny", "inputs": {"source": ["5", 2]}},
        "18": {"class_type": "PreviewAny", "inputs": {"source": ["5", 3]}},
        "19": {"class_type": "PreviewAny", "inputs": {"source": ["6", 2]}},
        "20": {"class_type": "PreviewAny", "inputs": {"source": ["9", 8]}},
        "21": {"class_type": "PreviewAny", "inputs": {"source": ["10", 7]}},
    }
    next_id = PREVIEW_START_ID
    for image_source in (["2", 0], ["9", 0], ["10", 0], ["10", 6]):
        next_id = _sample_previews(
            prompt,
            first_id=next_id,
            image_source=image_source,
        )
    if next_id != AUDIT_NODE_ID:
        raise RuntimeError("Safety Audit prompt node allocation drifted")
    prompt["11"]["inputs"]["processed_frames"] = [str(AUDIT_NODE_ID), 1]
    prompt[str(AUDIT_NODE_ID)] = {
        "class_type": "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
        "inputs": {
            "source_frames": ["10", 1],
            "candidate_frames": ["10", 0],
            "used_skin_mask": ["10", 4],
            "audit_scope": "unique_track_owner",
            "temporal_policy": "hard_gate",
            "maximum_mean_abs_change": 0.08,
            "maximum_peak_abs_change": 0.30,
            "maximum_temporal_effect_jump": 0.04,
            "maximum_track_leak_fraction": 0.001,
            "minimum_temporal_pixels": 64,
            "scene_cut_reset_threshold": 0.20,
            "accept_candidate": False,
            "track_plan": ["5", 0],
            "audio_source": ["2", 1],
            "audio_passthrough": ["10", 3],
        },
    }
    prompt[str(AUDIT_REPORT_NODE_ID)] = {
        "class_type": "PreviewAny",
        "inputs": {"source": [str(AUDIT_NODE_ID), 7]},
    }
    prompt[str(AUDIT_PREVIEW_NODE_ID)] = {
        "class_type": "PreviewImage",
        "inputs": {"images": [str(AUDIT_NODE_ID), 6]},
    }
    return prompt


def _prepare_source(source: Path, target: Path) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(source),
        "-vf",
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True, timeout=300)
    probe = base._probe(target)
    video = next(item for item in probe["streams"] if item["codec_type"] == "video")
    if (
        int(video["width"]) != TARGET_WIDTH
        or int(video["height"]) != TARGET_HEIGHT
        or int(video["nb_frames"]) != 69
    ):
        raise RuntimeError(f"prepared source contract mismatch: {probe}")


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_selected_frames(path: Path) -> dict[int, Image.Image]:
    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        for index, frame in enumerate(container.decode(stream)):
            if index in SAMPLE_FRAMES:
                selected[index] = frame.to_image().convert("RGB")
    if set(selected) != set(SAMPLE_FRAMES):
        raise RuntimeError(f"could not decode all contact-sheet frames from {path}")
    return selected


def _make_contact_sheet(source: Path, candidate: Path, output: Path) -> None:
    source_frames = _read_selected_frames(source)
    candidate_frames = _read_selected_frames(candidate)
    tile_width = 384
    tile_height = 282
    label_height = 34
    canvas = Image.new(
        "RGB",
        (tile_width * len(SAMPLE_FRAMES), (tile_height + label_height) * 2),
        (28, 30, 34),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    for column, frame_index in enumerate(SAMPLE_FRAMES):
        for row, (label, collection) in enumerate(
            (("SOURCE", source_frames), ("PER-PERSON", candidate_frames))
        ):
            x = column * tile_width
            y = row * (tile_height + label_height)
            frame = collection[frame_index].resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            canvas.paste(frame, (x, y + label_height))
            draw.text(
                (x + 8, y + 5),
                f"{label}  F{frame_index}",
                font=font,
                fill=(245, 245, 245),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _make_review(source: Path, candidate: Path, output: Path) -> None:
    font = r"C\:/Windows/Fonts/arial.ttf"
    video_filter = (
        f"[0:v]drawbox=x=0:y=0:w=iw:h=54:color=black@0.65:t=fill,"
        f"drawtext=fontfile='{font}':text='SOURCE':x=18:y=13:fontsize=30:"
        "fontcolor=white[left];"
        f"[1:v]drawbox=x=0:y=0:w=iw:h=54:color=black@0.65:t=fill,"
        f"drawtext=fontfile='{font}':text='PER-PERSON SKIN FINISH':x=18:y=13:"
        "fontsize=30:fontcolor=white[right];[left][right]hstack=inputs=2[v]"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-threads",
        "1",
        "-i",
        str(source),
        "-i",
        str(candidate),
        "-filter_complex",
        video_filter,
        "-map",
        "[v]",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-c:a",
        "copy",
        "-shortest",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, timeout=300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=base.DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    gpu = base._gpu_sample()
    strict_baseline = None
    if STRICT_BASELINE_REPORT.is_file():
        strict_baseline = json.loads(
            STRICT_BASELINE_REPORT.read_text(encoding="utf-8")
        )
    current_source_sha = base._sha256(source) if source.is_file() else None
    baseline_source_sha = (
        strict_baseline.get("source", {}).get("original_sha256")
        if isinstance(strict_baseline, dict)
        else None
    )
    preflight = {
        "schema": "h3_t8_skin_finish_per_person_profile_live_validation/preflight-v1",
        "created_at": base._utc_now(),
        "source": str(source),
        "source_exists": source.is_file(),
        "python": str(args.python.resolve()),
        "python_exists": args.python.is_file(),
        "ffmpeg_exists": shutil.which("ffmpeg") is not None,
        "models": {
            "sam31": {"path": str(base.SAM_MODEL), "exists": base.SAM_MODEL.is_file()},
            "yunet": {"path": str(base.YUNET_MODEL), "exists": base.YUNET_MODEL.is_file()},
            "parsenet": {
                "path": str(base.PARSENET_MODEL),
                "exists": base.PARSENET_MODEL.is_file(),
            },
        },
        "strict_baseline": {
            "path": str(STRICT_BASELINE_REPORT),
            "exists": STRICT_BASELINE_REPORT.is_file(),
            "source_sha256": baseline_source_sha,
            "matches_current_source": bool(
                current_source_sha and current_source_sha == baseline_source_sha
            ),
        },
        "target_port_free": not base._port_is_listening(args.host, args.port),
        "user_port_8188_observed_only": base._port_is_listening(args.host, 8188),
        "gpu": gpu,
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "confirmed": bool(args.confirm_run),
    }
    preflight["ready"] = bool(
        preflight["source_exists"]
        and preflight["python_exists"]
        and preflight["ffmpeg_exists"]
        and all(item["exists"] for item in preflight["models"].values())
        and preflight["strict_baseline"]["matches_current_source"]
        and preflight["target_port_free"]
        and gpu.get("available")
        and int(gpu.get("free_mib", 0)) >= args.minimum_free_vram_mib
        and args.confirm_run
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base._json_write(output_root / "preflight.json", preflight)
    if not preflight["ready"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = output_root / run_id
    for name in ("input", "output", "temp", "user", "logs", "prompt", "review"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    prepared_source = run_root / "input" / "two_person_960x704x69.mp4"
    _prepare_source(source, prepared_source)
    prompt = _build_prompt(prepared_source.name)
    base._json_write(run_root / "prompt" / "prompt.json", prompt)

    monitor = base.GpuMonitor()
    server_url = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    history: dict[str, Any] | None = None
    prompt_id = None
    server_pid = None
    try:
        monitor.start()
        with base.IsolatedComfy(
            python=args.python.resolve(),
            host=args.host,
            port=args.port,
            run_root=run_root,
            start_timeout=args.server_start_timeout,
        ) as isolated:
            server_pid = int(isolated.process.pid) if isolated.process else None
            object_info = base._request_json(
                "GET",
                f"{server_url}/object_info/MiniMaxH3SkinFinishPerPersonT8Advanced",
            )
            if "MiniMaxH3SkinFinishPerPersonT8Advanced" not in object_info:
                raise RuntimeError("Skin Finish per-person node is not registered")
            profile_object_info = base._request_json(
                "GET",
                f"{server_url}/object_info/"
                "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced",
            )
            if (
                "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
                not in profile_object_info
            ):
                raise RuntimeError("Skin Finish profile semantic node is not registered")
            audit_object_info = base._request_json(
                "GET",
                f"{server_url}/object_info/MiniMaxH3SkinFinishSafetyAuditT8Advanced",
            )
            if "MiniMaxH3SkinFinishSafetyAuditT8Advanced" not in audit_object_info:
                raise RuntimeError("Skin Finish safety audit node is not registered")
            response = base._request_json(
                "POST",
                f"{server_url}/prompt",
                {"prompt": prompt, "client_id": f"skin-finish-per-person-{run_id}"},
            )
            prompt_id = str(response["prompt_id"])
            history = base._wait_for_history(
                server_url,
                prompt_id,
                args.prompt_timeout,
            )
            base._json_write(run_root / "history.json", history)
            errors = base._history_errors(history)
            if errors:
                raise RuntimeError(f"ComfyUI prompt failed: {errors[-1]}")
    finally:
        monitor.stop()

    candidates = sorted((run_root / "output").rglob("*.mp4"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one finalized MP4, found {len(candidates)}")
    candidate = candidates[0]
    assert history is not None
    track_report = json.loads(base._history_text(history, "17"))
    shot_count = int(base._history_text(history, "18"))
    parser_report = json.loads(base._history_text(history, "19"))
    executor_report = json.loads(base._history_text(history, "20"))
    guard_report = json.loads(base._history_text(history, "21"))
    safety_report = json.loads(
        base._history_text(history, str(AUDIT_REPORT_NODE_ID))
    )

    review_video = (
        run_root / "review" / "source_vs_per_person_profile_skin_finish.mp4"
    )
    contact_sheet = (
        run_root / "review" / "source_vs_per_person_profile_contact.png"
    )
    _make_review(prepared_source, candidate, review_video)
    _make_contact_sheet(prepared_source, candidate, contact_sheet)
    source_packet = base._audio_packet_payload(prepared_source)
    candidate_packet = base._audio_packet_payload(candidate)
    source_pcm = base._decoded_audio_sha(prepared_source)
    candidate_pcm = base._decoded_audio_sha(candidate)
    preview_files = sorted((run_root / "temp").rglob("*.png"))
    route_count = len(executor_report.get("routing", {}).get("routes", []))
    processed_pixels = int(
        executor_report.get("routing", {}).get("processed_pixels", 0)
    )
    accepted_track_total = sum(
        int(frame.get("accepted_track_count", 0))
        for frame in parser_report.get("frames", [])
    )
    fallback_total = sum(
        int(value)
        for value in parser_report.get("selection", {})
        .get("profile_crop_ready_counts", {})
        .values()
    )
    baseline_semantic = strict_baseline.get("semantic_report", {})
    baseline_accepted_total = sum(
        int(frame.get("accepted_track_count", 0))
        for frame in baseline_semantic.get("frames", [])
    )
    checks = {
        "track_plan_ready": track_report.get("status")
        == "sam31_shot_local_tracks_ready",
        "one_shot_detected": shot_count == 1,
        "two_tracks_detected": track_report.get("objects_per_shot") == [2],
        "semantic_parser_ready": parser_report.get("status") == "READY",
        "strict_first_profile_policy_locked": parser_report.get("alignment", {}).get(
            "policy"
        )
        == "five_point_then_profile_crop",
        "profile_crop_fallback_used": fallback_total > 0,
        "profile_track_coverage_improved_over_bound_strict_baseline": (
            accepted_track_total > baseline_accepted_total
        ),
        "per_person_candidate_ready": executor_report.get("status")
        == "CANDIDATE_READY",
        "two_profile_routes_present": route_count == 2,
        "profiled_skin_pixels_present": processed_pixels > 0,
        "outside_mask_bit_exact": executor_report.get("mechanical_gates", {}).get(
            "outside_mask_bit_exact"
        )
        is True,
        "alpha_preserved": executor_report.get("mechanical_gates", {}).get(
            "alpha_or_aux_channels_preserved"
        )
        is True,
        "safety_audit_passed": safety_report.get("status") == "PASS_HARD_GATES"
        and safety_report.get("summary", {}).get("hard_gate_pass") is True,
        "safety_audit_zero_failed_frames": safety_report.get("summary", {}).get(
            "failed_frame_count"
        )
        == 0,
        "safety_audit_source_bound_track_plan": safety_report.get(
            "track_plan", {}
        ).get("valid")
        is True,
        "safety_audit_unique_track_owner_hard_gate": safety_report.get(
            "parameters", {}
        ).get("audit_scope")
        == "unique_track_owner"
        and safety_report.get("parameters", {}).get("temporal_policy")
        == "hard_gate",
        "safety_audit_audio_exact": safety_report.get("audio", {}).get("exact")
        is True,
        "safety_audit_never_auto_accepts": safety_report.get("summary", {}).get(
            "automatic_accept"
        )
        is False
        and safety_report.get("summary", {}).get("candidate_selected") is False,
        "source_candidate_audio_packet_exact": source_packet == candidate_packet,
        "source_candidate_decoded_audio_exact": source_pcm == candidate_pcm,
        "candidate_strict_decode": base._strict_decode(candidate, "video")["passed"]
        and base._strict_decode(candidate, "audio")["passed"],
        "review_strict_decode": base._strict_decode(review_video, "video")["passed"]
        and base._strict_decode(review_video, "audio")["passed"],
        "preview_files_present": bool(preview_files),
        "review_files_present": review_video.is_file() and contact_sheet.is_file(),
        "server_stopped": not base._port_is_listening(args.host, args.port),
        "user_8188_untouched": preflight["user_port_8188_observed_only"]
        == base._port_is_listening(args.host, 8188),
    }
    report = {
        "schema": "h3_t8_skin_finish_per_person_profile_live_validation/v1",
        "created_at": base._utc_now(),
        "run_id": run_id,
        "server_pid": server_pid,
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "source": {
            "original_path": str(source),
            "original_sha256": base._sha256(source),
            "prepared_path": str(prepared_source),
            "prepared_sha256": base._sha256(prepared_source),
            "probe": base._probe(prepared_source),
            "audio_packet_payload": source_packet,
            "decoded_audio": source_pcm,
        },
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": base._sha256(candidate),
            "probe": base._probe(candidate),
            "audio_packet_payload": candidate_packet,
            "decoded_audio": candidate_pcm,
        },
        "profiles": [
            {
                "selector": "0:0",
                "preset": "subtle",
                "amount": 0.25,
                "texture_keep": 0.95,
                "shine_control": 0.25,
            },
            {
                "selector": "0:1",
                "preset": "oil_control",
                "amount": 0.55,
                "texture_keep": 0.88,
                "shine_control": 0.60,
            },
        ],
        "semantic_coverage_comparison": {
            "strict_baseline_report": str(STRICT_BASELINE_REPORT),
            "strict_baseline_source_sha256": baseline_source_sha,
            "strict_baseline_accepted_track_total": baseline_accepted_total,
            "profile_accepted_track_total": accepted_track_total,
            "profile_crop_fallback_total": fallback_total,
            "profile_crop_ready_counts": parser_report.get("selection", {}).get(
                "profile_crop_ready_counts", {}
            ),
        },
        "track_report": track_report,
        "semantic_report": parser_report,
        "executor_report": executor_report,
        "guard_report": guard_report,
        "safety_audit_report": safety_report,
        "checks": checks,
        "gpu": monitor.report(),
        "review": {
            "video": str(review_video.resolve()),
            "video_sha256": base._sha256(review_video),
            "contact_sheet": str(contact_sheet.resolve()),
            "contact_sheet_sha256": base._sha256(contact_sheet),
        },
        "preview_files": [
            {"path": str(path.resolve()), "sha256": base._sha256(path)}
            for path in preview_files
        ],
        "boundary": (
            "One 960x704x69 real two-person clip runs native SAM3.1, strict-first "
            "profile-crop pinned CPU ParseNet, two explicit shot-local Skin Finish "
            "profiles and the pre-encode unique-owner hard Safety Audit, compared to "
            "the source-bound prior strict report. "
            "It is not an identity, fairness, crossing-person, long-video, pressure, "
            "universal 16GiB or human aesthetic proof."
        ),
    }
    report["passed"] = all(checks.values())
    base._json_write(run_root / "validation_report.json", report)
    base._json_write(
        output_root / "latest.json",
        {"run_id": run_id, "report": str(run_root / "validation_report.json")},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
