"""Explicit pinned CPU pose extraction on an exact real source-frame timeline.

Own YOLOX/SimCC decoding and primitive raster, adapted from the project's H07
helper. No external OpenPose drawing implementation is copied/imported. This
prepares media; it is neither an H3 sampler nor a pose/identity certificate.
"""
from __future__ import annotations

import argparse
import colorsys
from fractions import Fraction
import hashlib
import importlib.util
from itertools import chain
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def resolve_core_root():
    """Find the real Core, including a source checkout outside custom_nodes."""
    explicit = os.environ.get('T8_H21_CORE')
    if explicit:
        candidate = Path(explicit).resolve()
        if not (candidate / 'folder_paths.py').is_file() or not (candidate / 'comfy').is_dir():
            raise ValueError('T8_H21_CORE must point to the actual ComfyUI directory')
        return candidate
    for candidate in ROOT.parents:
        if (candidate / 'folder_paths.py').is_file() and (candidate / 'comfy').is_dir():
            return candidate
    spec = importlib.util.find_spec('folder_paths')
    if spec is not None and spec.origin:
        candidate = Path(spec.origin).resolve().parent
        if (candidate / 'comfy').is_dir():
            return candidate
    return None


CORE = resolve_core_root()
EVIDENCE = ROOT / "artifacts/development/h21-implementation-20261021"
YOLOX = Path("G:/T8-H07/n12-yolox-source-v1/yolox_l.torchscript.pt")
YOLOX_SHA = "80bc14b13c260c24b3014cd42c02994bf52296ab8fa2d80a60b6afe08c93ef42"
DWPOSE = (CORE or ROOT) / "custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt"
DWPOSE_SHA = "d86a0b2b59fddc0901a7076e9f59c9f8602602133ed72511c693fd11eea23d91"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rgb_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def encode_rgb_movie(path, arrays, rate):
    """Mux exact Pillow PNG packets into MOV; never invoke PyAV PNG encoding.

    The retained real192 case exposed invalid deflate packets from the previous
    host PNG encoder, reproduced by independent FFmpeg. Muxed existing PNG bytes
    passed all192 packet/RGB/PTS checks. No lower-level cause is inferred here.
    """
    iterator = iter(arrays)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("No actual RGB frames to encode") from error
    rate = Fraction(rate)
    if rate <= 0:
        raise ValueError("Positive exact frame rate required")
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected actual uint8 RGB frames")
    with av.open(str(path), mode="w", format="mov") as movie:
        stream = movie.add_stream("png", rate=rate)
        stream.width, stream.height, stream.pix_fmt = first.shape[1], first.shape[0], "rgb24"
        stream.time_base = stream.codec_context.time_base = 1 / rate
        for number, rgb in enumerate(chain((first,), iterator)):
            if rgb.dtype != np.uint8 or rgb.shape != first.shape:
                raise ValueError("RGB sequence geometry/dtype changed")
            png = BytesIO()
            Image.fromarray(rgb).save(png, format="PNG")
            packet = av.Packet(png.getvalue())
            packet.pts = packet.dts = number
            packet.duration, packet.time_base, packet.stream = 1, 1 / rate, stream
            movie.mux(packet)


def source_indices(source_rate, target_rate, duration):
    source_rate, target_rate, duration = map(Fraction, (source_rate, target_rate, duration))
    count = target_rate * duration
    if min(source_rate, target_rate, duration) <= 0 or count.denominator != 1:
        raise ValueError("Positive exact rates/duration and an integral target frame count are required")
    indices = [round(Fraction(index, 1) * source_rate / target_rate) for index in range(int(count))]
    if len(set(indices)) != len(indices):
        raise ValueError("This real-source qualification requires unique source indices, never repeated or synthesized poses")
    return indices


def inspect_source(source, expected_sha, *, duration=Fraction(8), target_rate=Fraction(24)):
    """Decode ALL original RGB/PTS first; keep hashes, not a full resident movie."""
    source = Path(source).resolve(strict=True)
    before = sha(source)
    if before != expected_sha:
        raise ValueError("Original source SHA differs")
    hashes, timestamps = [], []
    full = hashlib.sha256()
    with av.open(str(source)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Select a source with one explicit video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or rate <= 0:
            raise ValueError("Actual source needs an observed finite positive CFR rate")
        indices = source_indices(rate, target_rate, duration)
        width, height = stream.width, stream.height
        for number, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.pts * frame.time_base != Fraction(number, 1) / rate:
                raise ValueError(f"Original source is not exact zero-origin CFR at frame{number}")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (height, width, 3):
                raise ValueError("Original decoded RGB geometry changed")
            full.update(rgb.tobytes())
            hashes.append(rgb_sha(rgb))
            timestamps.append(str(frame.pts * frame.time_base))
        if not hashes or indices[-1] >= len(hashes):
            raise ValueError("Real source is too short; no held/looped/invented tail is allowed")
    if sha(source) != before:
        raise ValueError("Original source changed during complete decode")
    return {"path": str(source), "sha256": before, "bytes": source.stat().st_size,
            "frames": len(hashes), "width": width, "height": height, "fps": str(rate),
            "all_RGB_sha256": full.hexdigest(), "all_frame_RGB_sha256": hashes,
            "all_PTS": timestamps, "all_CFR_PTS_verified": True,
            "target_fps": str(target_rate), "duration": str(duration),
            "source_indices": indices, "target_frames": len(indices),
            "selected_frame_RGB_sha256": [hashes[index] for index in indices],
            "mapping": "exact Fraction round-half-to-even(k*source_fps/target_fps)",
            "original_source_modified": False}


def resize_source(rgb, width, height):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape) <= 0):
        raise ValueError("Expected actual uint8 RGB [H,W,3]")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Pose dimensions must be positive integers")
    if width * rgb.shape[0] != height * rgb.shape[1]:
        raise ValueError("Explicit pose canvas must preserve the actual source aspect, not squeeze/crop")
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def materialize_source_pngs(source, plan, output):
    """Finish pure source decode BEFORE any model load or encoder is opened.

    Own copied PNG pixels isolate decoder buffer lifetimes and make all later
    model/encoder inputs independently byte-verifiable, not shared AV buffers.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    wanted = {index: number for number, index in enumerate(plan["source_indices"])}
    rows = []
    with av.open(str(source)) as incoming:
        for source_index, frame in enumerate(incoming.decode(video=0)):
            # Keep the exact prefix of inspect_source's conversion sequence.
            # A real-source counterexample showed that selecting first changed
            # later decoded RGB in this host; converting all frames matched all
            # 192 canonical hashes. The lower-level codec cause is not claimed.
            rgb = frame.to_ndarray(format="rgb24").copy()
            if source_index not in wanted:
                continue
            number = wanted[source_index]
            if rgb_sha(rgb) != plan["selected_frame_RGB_sha256"][number] or str(frame.pts * frame.time_base) != plan["all_PTS"][source_index]:
                raise ValueError("Source RGB/PTS differs from full inspection; not evidence that file bytes changed")
            path = output / f"{number:05d}.png"
            Image.fromarray(rgb).save(path)
            with Image.open(path) as saved:
                actual = np.array(saved)
            if not np.array_equal(actual, rgb):
                raise ValueError("Selected original PNG did not preserve exact RGB")
            rows.append({"frame": number, "source_frame": source_index, "path": str(path), "sha256": sha(path), "RGB_sha256": rgb_sha(actual)})
            if number == len(wanted) - 1:
                break
    if len(rows) != plan["target_frames"] or sha(source) != plan["sha256"]:
        raise ValueError("Incomplete selected source PNGs or original file changed")
    return rows


def person_boxes(prediction, ratio):
    """Actual pinned YOLOX640 raw head, person NMS; no invented/truncated people."""
    if prediction.shape != (1, 8400, 85) or not np.isfinite(prediction).all():
        raise ValueError("Pinned YOLOX output shape/data differs")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Detector letterbox ratio must be finite and positive")
    rows = prediction[0].copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        yy, xx = np.mgrid[:640 // stride, :640 // stride]
        grids.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
        strides.append(np.full((xx.size, 1), stride))
    scale = np.concatenate(strides)
    try:
        with np.errstate(over="raise", invalid="raise"):
            centers = (rows[:, :2] + np.concatenate(grids)) * scale / ratio
            sizes = np.exp(rows[:, 2:4]) * scale / ratio
            scores = rows[:, 4] * rows[:, 5]
            boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    except FloatingPointError as error:
        raise ValueError("Nonfinite decoded detector coordinates") from error
    if not np.isfinite(boxes).all():
        raise ValueError("Nonfinite decoded detector coordinates")
    order = np.flatnonzero(scores > .3)
    order = order[np.argsort(-scores[order], kind="stable")]
    kept = []
    while len(order):
        first, *rest = order.tolist()
        kept.append(first)
        rest = np.asarray(rest, dtype=int)
        if not len(rest):
            break
        inter = np.maximum(0, np.minimum(boxes[first, 2:], boxes[rest, 2:]) - np.maximum(boxes[first, :2], boxes[rest, :2]) + 1).prod(axis=1)
        area = np.maximum(0, boxes[:, 2:] - boxes[:, :2] + 1).prod(axis=1)
        iou = inter / np.maximum(area[first] + area[rest] - inter, 1e-6)
        order = rest[iou <= .45]
    if not 1 <= len(kept) <= 5:
        raise ValueError(f"Actual detector found{len(kept)} people; pinned pose export has fixedBS5, no invention/truncation/skip allowed")
    return boxes[kept].astype(np.float32), scores[kept].astype(np.float32)


def crop_transform(box):
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError("Invalid actual detector box")
    center = (box[:2] + box[2:]) / 2
    scale = (box[2:] - box[:2]) * 1.25
    scale = np.array([max(scale[0], scale[1] * .75), max(scale[1], scale[0] / .75)], dtype=np.float32)
    matrix = np.array([[288 / scale[0], 0, 144 - center[0] * 288 / scale[0]],
                       [0, 384 / scale[1], 192 - center[1] * 384 / scale[1]]], dtype=np.float32)
    return matrix, center, scale


def infer_frame(rgb, detector, estimator, *, width, height):
    """Two CPU calls on real frame; injected synthetic callables are TEST ONLY."""
    import torch
    small = resize_source(rgb, width, height)
    bgr = small[:, :, ::-1].copy()
    ratio = min(640 / height, 640 / width)
    resized = cv2.resize(bgr, (int(width * ratio), int(height * ratio)))
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    inp = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    boxes, detector_scores = person_boxes(detector(inp).detach().float().cpu().numpy(), ratio)
    crops, centers, scales = [], [], []
    for box in boxes:
        matrix, center, scale = crop_transform(box)
        crop = cv2.warpAffine(bgr, matrix, (288, 384), flags=cv2.INTER_LINEAR)
        crop = (crop.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
        crops.append(crop.transpose(2, 0, 1))
        centers.append(center)
        scales.append(scale)
    batch = np.zeros((5, 3, 384, 288), np.float32)
    batch[:len(crops)] = np.stack(crops)
    outputs = estimator(torch.from_numpy(batch))
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError("Pinned DWPose must return actual x/y SimCC tensors")
    sx, sy = [value.detach().float().cpu().numpy() for value in outputs]
    if sx.shape != (5, 133, 576) or sy.shape != (5, 133, 768) or not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("Pinned DWPose SimCC output shape/data differs")
    sx, sy = sx[:len(crops)], sy[:len(crops)]
    points = np.stack((sx.argmax(axis=-1), sy.argmax(axis=-1)), axis=-1).astype(np.float32) / 2
    points = points / np.array([288, 384]) * np.asarray(scales)[:, None] + np.asarray(centers)[:, None] - np.asarray(scales)[:, None] / 2
    scores = np.minimum(sx.max(axis=-1), sy.max(axis=-1))
    if not np.isfinite(points).all() or not (scores[:, :17] >= .3).any(axis=1).all():
        raise ValueError("An actual detected person has no confident body joints; do not invent or skip it")
    return small, boxes, detector_scores, points, scores


def raster(points, scores, *, width, height):
    """Own PIL line/ellipse/point primitives, conventional133-keypoint data."""
    if (points.ndim != 3 or points.shape[1:] != (133, 2) or scores.shape != points.shape[:2]
            or not np.isfinite(points).all() or not np.isfinite(scores).all()):
        raise ValueError("Raster requires finite actual133-keypoint arrays")
    canvas = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(canvas)
    mapping = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    limbs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
    palette = [tuple(round(v * 255) for v in colorsys.hsv_to_rgb(index / 18, 1, 1)) for index in range(18)]
    for person, confidence in zip(points, scores):
        coords = np.concatenate((person[:17], person[[5, 6]].mean(axis=0)[None]))[mapping]
        conf = np.concatenate((confidence[:17], [min(confidence[5], confidence[6])]))[mapping]
        for index, (a, b) in enumerate(limbs):
            if conf[a] >= .3 and conf[b] >= .3:
                draw.line([tuple(coords[a]), tuple(coords[b])], fill=palette[index], width=2)
        for index, xy in enumerate(coords):
            if conf[index] >= .3:
                draw.ellipse((xy[0] - 1, xy[1] - 1, xy[0] + 1, xy[1] + 1), fill=palette[index])
        for start in (91, 112):
            for finger in range(5):
                chain = [start] + list(range(start + 1 + finger * 4, start + 5 + finger * 4))
                for a, b in zip(chain, chain[1:]):
                    if confidence[a] >= .3 and confidence[b] >= .3:
                        draw.line([tuple(person[a]), tuple(person[b])], fill=palette[(finger * 3) % 18], width=1)
        for index in range(23, 91):
            if confidence[index] >= .3:
                draw.point(tuple(person[index]), fill="white")
    return canvas


def _cpu_runtime(threads):
    if type(threads) is not int or threads <= 0:
        raise ValueError("CPU thread count must be positive")
    if CORE is None:
        raise ValueError('Set T8_H21_CORE to the actual ComfyUI directory before CPU pose extraction')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(CORE))
    from comfy.cli_args import args
    args.cpu = True
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation cannot continue with initialized CUDA")
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    return torch


def prepare(source, output, *, expected_source_sha, allow_pinned_cpu_models=False,
            duration=Fraction(8), target_rate=Fraction(24), width=288, height=512, threads=4):
    output = Path(output).resolve()
    if not allow_pinned_cpu_models:
        raise ValueError("Explicit --allow-pinned-CPU-models opt-in is required")
    if output.exists() or output == EVIDENCE.resolve() or not output.is_relative_to(EVIDENCE.resolve()):
        raise ValueError("Fresh contained owned output required; prior evidence is never overwritten")
    source = Path(source).resolve(strict=True)
    plan = inspect_source(source, expected_source_sha, duration=duration, target_rate=target_rate)
    if width * plan["height"] != height * plan["width"] or min(width, height) <= 0:
        raise ValueError("Pose dimensions must explicitly preserve the source aspect")
    models = [(YOLOX, YOLOX_SHA), (DWPOSE, DWPOSE_SHA)]
    for path, digest in models:
        if sha(path) != digest:
            raise ValueError("Pinned trusted TorchScript resource SHA differs; no replacement is selected")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pose").mkdir()
    (output / "source-images").mkdir()
    write_new(output / "source-clock.json", plan)
    detector = estimator = torch = None
    rows = []
    source_images = {}
    started = time.perf_counter()
    selected_movie = output / "selected_rgb_24fps_lossless.mov"
    try:
        selected_pngs = materialize_source_pngs(source, plan, output / "selected-original-png")
        write_new(output / "selected-original-png.json", selected_pngs)
        torch = _cpu_runtime(threads)
        detector = torch.jit.load(str(YOLOX), map_location="cpu").eval()
        estimator = torch.jit.load(str(DWPOSE), map_location="cpu").eval()
        assert all(parameter.device.type == "cpu" for model in (detector, estimator) for parameter in model.parameters())
        wanted = {index: number for number, index in enumerate(plan["source_indices"])}
        sample_roles = {0: "first", (len(wanted) - 1) // 2: "middle", len(wanted) - 1: "last"}
        with (output / "frames.jsonl").open("x", encoding="utf-8") as journal:
            with torch.inference_mode():
                for number, selected in enumerate(selected_pngs):
                    source_index = selected["source_frame"]
                    with Image.open(selected["path"]) as image:
                        rgb = np.array(image)
                    if sha(selected["path"]) != selected["sha256"] or rgb_sha(rgb) != plan["selected_frame_RGB_sha256"][number]:
                        raise ValueError("Independent original RGB PNG changed before model inference")
                    small, boxes, detection_scores, points, confidence = infer_frame(rgb, detector, estimator, width=width, height=height)
                    image = raster(points, confidence, width=width, height=height)
                    path = output / "pose" / f"{number:05d}.png"
                    image.save(path)
                    row = {"frame": number, "source_frame": source_index, "source_PTS": plan["all_PTS"][source_index],
                           "target_time": str(Fraction(number, 1) / target_rate), "source_RGB_sha256": rgb_sha(rgb),
                           "estimator_RGB_sha256": rgb_sha(small), "path": str(path), "sha256": sha(path),
                           "pose_RGB_sha256": rgb_sha(np.asarray(image)), "boxes_pose_pixels": boxes.tolist(),
                           "detector_scores": detection_scores.tolist(), "keypoints_pose_pixels": points.tolist(),
                           "keypoint_scores": confidence.tolist(), "actual_people": len(boxes),
                           "fixed_batch5_filled": len(boxes), "fixed_batch5_padding": 5 - len(boxes),
                           "padding_is_not_additional_people": True, "detector_forward_calls": 1, "pose_forward_calls": 1}
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    if number in sample_roles:
                        target = output / "source-images" / (sample_roles[number] + ".png")
                        Image.fromarray(rgb).save(target)
                        source_images[sample_roles[number]] = {"path": str(target), "sha256": sha(target), "source_frame": source_index, "target_frame": number, "RGB_sha256": rgb_sha(rgb)}
                    if number % 8 == 0 or number == len(wanted) - 1:
                        print(f"Actual CPU pose {number + 1}/{len(wanted)} sourceF{source_index}, people{len(boxes)}, {time.perf_counter() - started:.1f}s", flush=True)
        if len(rows) != plan["target_frames"] or torch.cuda.is_initialized():
            raise ValueError("Incomplete actual frame extraction or unexpected CUDA initialization")
        if sha(source) != expected_source_sha:
            raise ValueError("Original source changed during extraction")
        detector = estimator = None

        def original_arrays():
            for selected in selected_pngs:
                with Image.open(selected["path"]) as image:
                    array = np.array(image)
                if sha(selected["path"]) != selected["sha256"] or rgb_sha(array) != selected["RGB_sha256"]:
                    raise ValueError("Original RGB PNG changed before encoding")
                yield array

        encode_rgb_movie(selected_movie, original_arrays(), target_rate)
        verified = []
        with av.open(str(selected_movie)) as movie:
            for number, frame in enumerate(movie.decode(video=0)):
                if number >= len(rows) or frame.pts * frame.time_base != Fraction(number, 1) / target_rate or rgb_sha(frame.to_ndarray(format="rgb24")) != rows[number]["source_RGB_sha256"]:
                    raise ValueError("Lossless selected RGB source/time verification failed")
                verified.append(number)
        if len(verified) != len(rows):
            raise ValueError("Selected RGB movie is incomplete")
        write_new(output / "keypoints.json", rows)
        report = {"schema": "t8.local.h21.actual-CPU-pose-source.v1", "status": "actual_CPU_pose_complete_NOT_H3_GPU_or_quality_qualified",
                  "source": str(selected_movie), "source_sha256": sha(selected_movie), "original_source": plan,
                  "frames": len(rows), "fps": int(target_rate) if target_rate.denominator == 1 else str(target_rate),
                  "width": width, "height": height, "geometry": "actual source aspect preserved by explicit INTER_AREA resize; no crop, time interpolation, looping or held frames",
                  "model_sources": [{"path": str(path), "sha256": digest, "bytes": path.stat().st_size} for path, digest in models],
                  "actual_detector_and_pose_forward_frames": len(rows), "actual_detector_forward_calls": len(rows), "actual_pose_forward_calls": len(rows),
                  "fixed_pose_export_batch": 5, "padding_discarded_not_people": True, "device": "cpu",
                  "elapsed_seconds": time.perf_counter() - started, "source_images": source_images,
                  "selected_RGB_movie_all_original_selected_pixels_and_CFR_exact": True, "selected_RGB_movie_has_audio": False,
                  "selected_RGB_codec": "native_rgb24_PNG_in_MOV_no_pixel_format_conversion",
                  "phase_isolation": "all_original_decode_and_verified_PNGs_then_model_only_PNG_reads_then_encoding_then_decode_verification; no interleaved source decoder/model/encoder",
                  "selected_original_PNG_manifest_sha256": sha(output / "selected-original-png.json"),
                  "original_audio": "Original file untouched; no audio adoption, normalization or generated sound qualification",
                  "keypoints_sha256": sha(output / "keypoints.json"), "helper_sha256": sha(Path(__file__)),
                  "images": [{"frame": row["frame"], "path": row["path"], "sha256": row["sha256"], "source_frame": row["source_frame"]} for row in rows],
                  "CUDA_initialized": False, "DiT_NFE": 0, "queued": False, "H3_model_executed": False,
                  "identity_tracking_masks_or_pose_effect_claimed": False, "external_plugin_or_restricted_draw_code_imported": False,
                  "human_accepted": False, "source_clock_sha256": sha(output / "source-clock.json")}
        write_new(output / "report.json", report)
        media_spec = importlib.util.spec_from_file_location("h21_pose_media", Path(__file__).with_name("prepare_h21_control_media.py"))
        media = importlib.util.module_from_spec(media_spec)
        media_spec.loader.exec_module(media)
        _, _, _, _, rate, arrays, pngs = media._manifest(output / "report.json", sha(output / "report.json"))
        media_dir = output / "media"
        media_dir.mkdir()
        control_movie = media_dir / "control_pose_cfr_rgb_lossless.mov"
        encode_rgb_movie(control_movie, arrays, rate)
        checked = media.verify_video(control_movie, arrays, rate)
        companion_images, audio_streams = media._source_frames(selected_movie, media_dir, len(rows), rate)
        prepared = {"schema": "t8.local.h21.control-media.v1", "status": "CPU_lossless_RGB_and_CFR_verified_not_H3_qualified",
                    "manifest": {"path": str(output / "report.json"), "sha256": sha(output / "report.json")},
                    "raw_rgb": {"path": str(selected_movie), "sha256": sha(selected_movie), "frames": len(rows), "fps": str(rate), "audio_streams": audio_streams},
                    "original_source": {"path": str(source), "sha256": expected_source_sha, "source_clock": str(output / "source-clock.json")},
                    "control_video": {"path": str(control_movie), "sha256": sha(control_movie), "bytes": control_movie.stat().st_size, "frames": len(rows), "fps": str(rate),
                                      "width": width, "height": height, "codec": "png_native_rgb24_in_MOV", "audio": False},
                    "source_images": companion_images, "input_pngs": pngs, "decoded_control_frames": checked,
                    "manifest_contract_actual_prepare_h21_control_media_manifest_reader_verified": True,
                    "original_media_helper_not_modified": True, "old_H264_reencode_not_used_or_claimed_verified": True,
                    "all_actual_source_RGB_and_pose_PNGs_retained": True, "GPU_or_H3_model_executed": False, "quality_accepted": False}
        write_new(media_dir / "report.json", prepared)
        for role, saved in companion_images.items():
            if saved["rgb_sha256"] != source_images[role]["RGB_sha256"]:
                raise ValueError("Source-image companion preparation changed selected original RGB")
        if torch.cuda.is_initialized() or sha(source) != expected_source_sha:
            raise ValueError("CPU-only/original-source postcondition failed")
        write_new(output / "completion.json", {"status": "complete_CPU_source_only", "frames": len(rows), "report_sha256": sha(output / "report.json"),
                  "media_report_sha256": sha(output / "media/report.json"), "control_movie": prepared["control_video"], "CUDA_initialized": False,
                  "pose_movie_all_original_PNG_RGB_and_CFR_verified": True, "H3_or_GPU_execution": False, "quality_accepted": False})
        return report
    except Exception as error:
        write_new(output / "failure.json", {"status": "CPU_pose_preparation_failed_not_published", "completed_frames": len(rows),
                  "exception": type(error).__name__, "reason": str(error), "CUDA_initialized": torch.cuda.is_initialized() if torch is not None else False,
                  "original_source_sha256_now": sha(source), "queued": False, "H3_NFE": 0})
        raise
    finally:
        del detector, estimator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=Fraction, default=Fraction(8))
    parser.add_argument("--target-fps", type=Fraction, default=Fraction(24))
    parser.add_argument("--pose-width", type=int, default=288)
    parser.add_argument("--pose-height", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-pinned-CPU-models", action="store_true")
    args = parser.parse_args(argv)
    report = prepare(args.source, args.output, expected_source_sha=args.source_sha256, allow_pinned_cpu_models=args.allow_pinned_CPU_models,
                     duration=args.duration_seconds, target_rate=args.target_fps, width=args.pose_width, height=args.pose_height, threads=args.cpu_threads)
    print(json.dumps({"status": report["status"], "frames": report["frames"], "CUDA_initialized": report["CUDA_initialized"]}), flush=True)


if __name__ == "__main__":
    main()
