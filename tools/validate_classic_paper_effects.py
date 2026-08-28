from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _load_project_module(project_root: Path, name: str):
    comfy_root = project_root.parents[1]
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    package_name = "h3_t8_classic_effect_validation"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(project_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{name}")


def _read_video(
    path: Path,
    *,
    start_frame: int,
    frame_count: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> tuple[torch.Tensor, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frames: list[np.ndarray] = []
    try:
        for _ in range(int(frame_count)):
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if target_width is not None and target_height is not None:
                source_ratio = rgb.shape[1] / rgb.shape[0]
                target_ratio = target_width / target_height
                if source_ratio > target_ratio:
                    crop_width = max(1, int(round(rgb.shape[0] * target_ratio)))
                    offset = max(0, (rgb.shape[1] - crop_width) // 2)
                    rgb = rgb[:, offset : offset + crop_width]
                elif source_ratio < target_ratio:
                    crop_height = max(1, int(round(rgb.shape[1] / target_ratio)))
                    offset = max(0, (rgb.shape[0] - crop_height) // 2)
                    rgb = rgb[offset : offset + crop_height]
                rgb = cv2.resize(
                    rgb,
                    (int(target_width), int(target_height)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(rgb)
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"video yielded only {len(frames)} frames")
    value = torch.from_numpy(np.stack(frames)).to(torch.float32).div_(255.0)
    return value, fps


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    output = rgb.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (8, 8, 8), -1)
    cv2.putText(
        output,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _encode_rgb_video(
    frames: list[np.ndarray],
    *,
    output: Path,
    fps: float,
    source_audio: Path | None,
    ffmpeg: str,
) -> None:
    if not frames:
        raise ValueError("cannot encode an empty frame list")
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
    ]
    if source_audio is not None:
        command.extend(["-i", str(source_audio), "-map", "0:v:0", "-map", "1:a:0?"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-threads",
            "1",
        ]
    )
    if source_audio is not None:
        command.extend(["-c:a", "copy", "-shortest"])
    command.append(str(output))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("all encoded frames must share one canvas")
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg encode failed ({return_code}): {stderr[-2000:]}")


def _contact_sheet(frames: list[np.ndarray], output: Path, *, columns: int = 4) -> None:
    if not frames:
        raise ValueError("contact sheet requires frames")
    rows = int(math.ceil(len(frames) / columns))
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _gold_subject_box(frame: torch.Tensor) -> tuple[int, int, int, int]:
    rgb = frame.mul(255.0).round().byte().numpy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((4, 45, 45)), np.array((40, 255, 255)))
    height, width = mask.shape
    allowed = np.zeros_like(mask)
    # The reviewed bird clip has a warm reflective tabletop below the subject.
    # Anchor detection deliberately uses only the upper body, then expands the
    # reviewed box; otherwise a color threshold joins the bird to the floor.
    allowed[int(height * 0.18) : int(height * 0.72), int(width * 0.03) : int(width * 0.97)] = 255
    mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area < max(40.0, height * width * 0.00008):
            continue
        if box_width < 8 or box_height < 8 or box_width / max(box_height, 1) > 5.0:
            continue
        if box_width > width * 0.55 or box_height > height * 0.55:
            continue
        score = area * (1.0 + 0.3 * (1.0 - y / max(height, 1)))
        candidates.append((score, (x, y, box_width, box_height)))
    if not candidates:
        raise RuntimeError("could not locate the reviewed gold subject for RAFT anchor masks")
    _, (x, y, box_width, box_height) = max(candidates, key=lambda item: item[0])
    pad_x = max(6, int(round(box_width * 0.18)))
    pad_y = max(10, int(round(box_height * 0.55)))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + box_width + pad_x),
        min(height, y + box_height + pad_y),
    )


def _reviewed_subject_mask(
    frame: torch.Tensor, box: tuple[int, int, int, int]
) -> torch.Tensor:
    rgb = frame.mul(255.0).round().byte().numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = box
    grabcut_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        bgr,
        grabcut_mask,
        (x0, y0, max(2, x1 - x0), max(2, y1 - y0)),
        background,
        foreground,
        7,
        cv2.GC_INIT_WITH_RECT,
    )
    subject = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    subject = cv2.morphologyEx(
        subject, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    if int(subject.sum()) < 64:
        raise RuntimeError("reviewed GrabCut anchor did not retain a usable foreground")
    return torch.from_numpy(subject).to(torch.float32)


def _tensor_rgb8(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()


def _strict_decode(path: Path, ffmpeg: str) -> bool:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def run_raft(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    module = _load_project_module(project, "optical_flow_advanced")
    frames, fps = _read_video(
        args.source.resolve(),
        start_frame=args.start_frame,
        frame_count=args.frame_count,
    )
    model = args.model.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_preview, audit_json, mean_motion, p95_motion, cut_count = module.audit_optical_flow(
        frames=frames,
        model_path=model,
        model_name=model.name,
        model_type="raft_small",
        precision="fp32",
        analysis_max_side=args.analysis_max_side,
        pair_batch_size=1,
        consistency_check=True,
        scene_cut_threshold=0.28,
        release_policy="offload_after",
    )
    anchor_indices = list(range(0, len(frames), 12))
    if anchor_indices[-1] != len(frames) - 1:
        anchor_indices.append(len(frames) - 1)
    anchor_boxes = [_gold_subject_box(frames[index]) for index in anchor_indices]
    anchors = torch.stack(
        [
            _reviewed_subject_mask(frames[index], box)
            for index, box in zip(anchor_indices, anchor_boxes, strict=True)
        ]
    )
    masks, confidence, mask_preview, propagation_json = module.propagate_keyframe_masks(
        frames=frames,
        keyframe_masks=anchors,
        keyframe_indices=",".join(str(index) for index in anchor_indices),
        model_path=model,
        model_name=model.name,
        model_type="raft_small",
        precision="fp32",
        analysis_max_side=args.analysis_max_side,
        pair_batch_size=1,
        scene_cut_threshold=0.28,
        consistency_threshold=max(1.0, p95_motion * 0.15),
        minimum_confidence=0.08,
        extend_edges=False,
        release_policy="clear_after",
    )
    source8 = _tensor_rgb8(frames[..., :3])
    mask_np = masks.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()
    review_frames: list[np.ndarray] = []
    for index, frame in enumerate(source8):
        alpha = np.clip(mask_np[index][..., None] * 0.48, 0.0, 0.48)
        green = np.zeros_like(frame, dtype=np.float32)
        green[..., 0] = 20
        green[..., 1] = 242
        green[..., 2] = 90
        overlay = frame.astype(np.float32) * (1.0 - alpha) + green * alpha
        review_frames.append(
            np.concatenate(
                (
                    _label(frame, "SOURCE"),
                    _label(overlay.round().astype(np.uint8), "RAFT PROPAGATED MASK"),
                ),
                axis=1,
            )
        )
    video_path = output / "raft_mask_propagation_side_by_side.mp4"
    _encode_rgb_video(
        review_frames,
        output=video_path,
        fps=fps,
        source_audio=args.source.resolve(),
        ffmpeg=args.ffmpeg,
    )
    indices = np.linspace(0, len(review_frames) - 1, min(8, len(review_frames))).round().astype(int)
    _contact_sheet([review_frames[index] for index in indices], output / "contact_sheet.png")
    _contact_sheet(
        [_label(_tensor_rgb8(item), f"FLOW {index + 1}") for index, item in enumerate(audit_preview)],
        output / "flow_preview.png",
    )
    _contact_sheet(
        [_label(_tensor_rgb8(item), f"MASK SAMPLE {index + 1}") for index, item in enumerate(mask_preview)],
        output / "mask_preview.png",
    )
    report = {
        "status": "pass" if _strict_decode(video_path, args.ffmpeg) else "fail",
        "source": str(args.source.resolve()),
        "source_frames": len(frames),
        "source_fps": fps,
        "anchor_indices": anchor_indices,
        "anchor_boxes_xyxy": [list(box) for box in anchor_boxes],
        "mean_motion_px_at_analysis_scale": mean_motion,
        "p95_motion_px_at_analysis_scale": p95_motion,
        "scene_cut_count": cut_count,
        "mean_propagation_confidence": float(confidence_np.mean()),
        "nonzero_mask_frame_count": int((mask_np.reshape(len(mask_np), -1).max(axis=1) > 0).sum()),
        "audit": json.loads(audit_json),
        "propagation": json.loads(propagation_json),
        "review_video": str(video_path),
        "strict_decode": _strict_decode(video_path, args.ffmpeg),
        "claim_boundary": "real-input optical-flow and mask-transport effect; not a restoration or quality-superiority claim",
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _sharpness(frames: np.ndarray) -> float:
    values = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        values.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    return float(np.mean(values))


def _temporal_delta(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    value = np.abs(np.diff(frames.astype(np.float32), axis=0))
    return float(value.mean() / 255.0)


def run_realbasicvsr(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    module = _load_project_module(project, "realbasicvsr_advanced")
    frames, fps = _read_video(
        args.source.resolve(),
        start_frame=args.start_frame,
        frame_count=args.frame_count,
        target_width=args.width,
        target_height=args.height,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_marker: dict[str, Any] = {"source_media": str(args.source.resolve())}
    candidate, source, audio_out, node_report_json = module.restore_realbasicvsr(
        frames,
        audio_marker,
        model_path=args.model.resolve(),
        model_name=args.model.name,
        output_mode="native_size_restore",
        strength=args.strength,
        chunk_frames=args.chunk_frames,
        overlap_frames=args.overlap_frames,
        precision="auto",
        checkpoint_branch="prefer_ema",
        release_policy="clear_after",
    )
    source8 = _tensor_rgb8(source[..., :3])
    candidate8 = _tensor_rgb8(candidate[..., :3])
    review_frames = [
        np.concatenate(
            (
                _label(source8[index], "SOURCE DOWNSCALE"),
                _label(candidate8[index], f"REALBASICVSR {args.strength:.2f}"),
            ),
            axis=1,
        )
        for index in range(len(source8))
    ]
    video_path = output / "realbasicvsr_native_restore_side_by_side.mp4"
    _encode_rgb_video(
        review_frames,
        output=video_path,
        fps=fps,
        source_audio=args.source.resolve(),
        ffmpeg=args.ffmpeg,
    )
    indices = np.linspace(0, len(review_frames) - 1, min(8, len(review_frames))).round().astype(int)
    _contact_sheet([review_frames[index] for index in indices], output / "contact_sheet.png")
    delta = np.abs(candidate8.astype(np.float32) - source8.astype(np.float32)) / 255.0
    report = {
        "status": "pass" if _strict_decode(video_path, args.ffmpeg) else "fail",
        "source": str(args.source.resolve()),
        "source_frames": len(frames),
        "processed_resolution": [args.width, args.height],
        "source_fps": fps,
        "mean_absolute_change": float(delta.mean()),
        "p95_absolute_change": float(np.quantile(delta, 0.95)),
        "source_laplacian_variance": _sharpness(source8),
        "candidate_laplacian_variance": _sharpness(candidate8),
        "source_temporal_delta": _temporal_delta(source8),
        "candidate_temporal_delta": _temporal_delta(candidate8),
        "audio_object_identity_preserved": audio_out is audio_marker,
        "node_report": json.loads(node_report_json),
        "review_video": str(video_path),
        "strict_decode": _strict_decode(video_path, args.ffmpeg),
        "claim_boundary": "single real H3 clip post-process review; cannot prove recovered identity, lip sync, or universal superiority",
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_trajectory(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    module = _load_project_module(project, "trajectory_control_advanced")
    keyframes = [
        {"frame": 0, "object_id": "subject_A", "x": 0.05, "y": 0.24, "width": 0.16, "height": 0.34, "strength": 1.0},
        {"frame": 62, "object_id": "subject_A", "x": 0.42, "y": 0.12, "width": 0.18, "height": 0.38, "strength": 0.9},
        {"frame": 123, "object_id": "subject_A", "x": 0.78, "y": 0.25, "width": 0.14, "height": 0.30, "strength": 0.75},
        {"frame": 0, "object_id": "subject_B", "x": 0.80, "y": 0.53, "width": 0.12, "height": 0.24, "strength": 0.65},
        {"frame": 62, "object_id": "subject_B", "x": 0.46, "y": 0.52, "width": 0.14, "height": 0.28, "strength": 0.85},
        {"frame": 123, "object_id": "subject_B", "x": 0.08, "y": 0.46, "width": 0.16, "height": 0.32, "strength": 1.0},
    ]
    plan, plan_preview, plan_report_json, object_count = module.build_trajectory_control_plan(
        keyframes_json=json.dumps(keyframes, ensure_ascii=False),
        width=1152,
        height=640,
        length=124,
        fps=24.0,
        easing="smoothstep",
        clip_policy="reject_outside",
    )
    control, mask, render_preview, render_report_json = module.render_trajectory_control(
        trajectory_plan=plan,
        render_mode="box_outline",
        feather=0.002,
        line_width=10,
        background_level=0.02,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control8 = _tensor_rgb8(control)
    review_frames = [_label(frame, "TWO-OBJECT TRAJECTORY CONTROL") for frame in control8]
    video_path = output / "trajectory_control_1152x640_124f.mp4"
    _encode_rgb_video(
        review_frames,
        output=video_path,
        fps=24.0,
        source_audio=None,
        ffmpeg=args.ffmpeg,
    )
    indices = np.linspace(0, len(review_frames) - 1, 12).round().astype(int)
    _contact_sheet([review_frames[index] for index in indices], output / "contact_sheet.png")
    _contact_sheet(
        [_label(_tensor_rgb8(frame), f"PLAN {index + 1}") for index, frame in enumerate(plan_preview)],
        output / "plan_preview.png",
    )
    _contact_sheet(
        [_label(_tensor_rgb8(frame), f"RENDER {index + 1}") for index, frame in enumerate(render_preview)],
        output / "render_preview.png",
    )
    mask_np = mask.detach().cpu().numpy()
    report = {
        "status": "pass" if _strict_decode(video_path, args.ffmpeg) else "fail",
        "object_count": object_count,
        "frame_count": int(control.shape[0]),
        "resolution": [int(control.shape[2]), int(control.shape[1])],
        "nonzero_mask_frames": int((mask_np.reshape(len(mask_np), -1).max(axis=1) > 0).sum()),
        "plan": json.loads(plan_report_json),
        "render": json.loads(render_report_json),
        "review_video": str(video_path),
        "strict_decode": _strict_decode(video_path, args.ffmpeg),
        "claim_boundary": "creator-facing control-video rendering only; trajectory adherence still requires a real Fun Control H3 generation",
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_seam_drift(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    module = _load_project_module(project, "long_video_seam_drift_advanced")
    frames, fps = _read_video(
        args.source.resolve(),
        start_frame=args.start_frame,
        frame_count=args.frame_count,
        target_width=args.width,
        target_height=args.height,
    )
    boundary = int(args.boundary)
    if not 1 <= boundary < len(frames):
        raise ValueError("local boundary must stay inside the decoded frame window")
    _, actual_status, actual_report_json = module.process_long_video_seam_drift(
        frames,
        json.dumps([boundary]),
        mode="report_only",
        color_contract="sdr_rec709_0_to_1",
    )
    controlled_source = args.controlled_source.resolve() if args.controlled_source else args.source.resolve()
    if controlled_source == args.source.resolve():
        controlled_frames = frames
        controlled_fps = fps
    else:
        controlled_frames, controlled_fps = _read_video(
            controlled_source,
            start_frame=args.controlled_start_frame,
            frame_count=args.frame_count,
            target_width=args.width,
            target_height=args.height,
        )
    controlled = controlled_frames.clone()
    controlled[boundary:] = (controlled[boundary:] * 0.95).clamp(0.0, 1.0)
    candidate, candidate_status, candidate_report_json = module.process_long_video_seam_drift(
        controlled,
        json.dumps([boundary]),
        mode="bounded_candidate_exp",
        color_contract="sdr_rec709_0_to_1",
        analysis_window_frames=3,
        transition_frames=min(16, len(frames) - boundary),
        scene_cut_threshold=0.18,
        minimum_drift=0.008,
        maximum_gain=1.08,
        maximum_offset=0.04,
        maximum_frame_change=0.06,
        maximum_texture_ratio_deviation=0.12,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    controlled8 = _tensor_rgb8(controlled)
    candidate8 = _tensor_rgb8(candidate)
    review_frames = [
        np.concatenate(
            (
                _label(controlled8[index], "CONTROLLED SEAM DRIFT"),
                _label(candidate8[index], "BOUNDED CORRECTION"),
            ),
            axis=1,
        )
        for index in range(len(controlled8))
    ]
    video_path = output / "seam_drift_bounded_side_by_side.mp4"
    _encode_rgb_video(
        review_frames,
        output=video_path,
        fps=controlled_fps,
        source_audio=controlled_source,
        ffmpeg=args.ffmpeg,
    )
    indices = sorted(
        {
            0,
            max(0, boundary - 2),
            boundary - 1,
            boundary,
            min(len(frames) - 1, boundary + 1),
            min(len(frames) - 1, boundary + 5),
            len(frames) - 1,
        }
    )
    _contact_sheet([review_frames[index] for index in indices], output / "contact_sheet.png")
    candidate_report = json.loads(candidate_report_json)
    boundary_report = candidate_report["boundaries"][0]
    delta = np.abs(candidate8.astype(np.float32) - controlled8.astype(np.float32)) / 255.0
    report = {
        "status": "pass"
        if _strict_decode(video_path, args.ffmpeg)
        and candidate_status == "candidate_applied"
        else "fail",
        "source": str(args.source.resolve()),
        "actual_source_status": actual_status,
        "actual_source_report": json.loads(actual_report_json),
        "controlled_source": str(controlled_source),
        "controlled_probe": "real decoded H3 frames with a fixed 5% display-domain gain drift injected after the review boundary",
        "candidate_status": candidate_status,
        "seam_mad_before": boundary_report["seam_mad_before"],
        "seam_mad_after": boundary_report["seam_mad_after"],
        "corrected_frames": boundary_report["corrected_frames"],
        "maximum_observed_change": boundary_report["maximum_observed_change"],
        "mean_absolute_candidate_change": float(delta.mean()),
        "candidate_report": candidate_report,
        "review_video": str(video_path),
        "strict_decode": _strict_decode(video_path, args.ffmpeg),
        "claim_boundary": "controlled real-frame seam correction; not evidence that every natural H3 seam should be modified",
    }
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-load real-effect validation for classic-paper nodes")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raft = subparsers.add_parser("raft")
    raft.add_argument("--source", type=Path, required=True)
    raft.add_argument("--model", type=Path, required=True)
    raft.add_argument("--output", type=Path, required=True)
    raft.add_argument("--start-frame", type=int, default=0)
    raft.add_argument("--frame-count", type=int, default=48)
    raft.add_argument("--analysis-max-side", type=int, default=640)
    raft.set_defaults(function=run_raft)

    vsr = subparsers.add_parser("realbasicvsr")
    vsr.add_argument("--source", type=Path, required=True)
    vsr.add_argument("--model", type=Path, required=True)
    vsr.add_argument("--output", type=Path, required=True)
    vsr.add_argument("--start-frame", type=int, default=24)
    vsr.add_argument("--frame-count", type=int, default=32)
    vsr.add_argument("--width", type=int, default=416)
    vsr.add_argument("--height", type=int, default=232)
    vsr.add_argument("--strength", type=float, default=0.65)
    vsr.add_argument("--chunk-frames", type=int, default=4)
    vsr.add_argument("--overlap-frames", type=int, default=1)
    vsr.set_defaults(function=run_realbasicvsr)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--output", type=Path, required=True)
    trajectory.set_defaults(function=run_trajectory)

    seam = subparsers.add_parser("seam-drift")
    seam.add_argument("--source", type=Path, required=True)
    seam.add_argument("--controlled-source", type=Path)
    seam.add_argument("--output", type=Path, required=True)
    seam.add_argument("--start-frame", type=int, default=112)
    seam.add_argument("--controlled-start-frame", type=int, default=24)
    seam.add_argument("--frame-count", type=int, default=32)
    seam.add_argument("--boundary", type=int, default=12)
    seam.add_argument("--width", type=int, default=576)
    seam.add_argument("--height", type=int, default=320)
    seam.set_defaults(function=run_seam_drift)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
