from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .long_video_delivery import (
    _decode_audio_exact,
    _resolve_inside,
    _safe_token,
    _sha256_file,
    _verify_accepted_files,
    apply_cosine_bridge,
    compose_accepted_long_video,
)
from .repair_execution_advanced import (
    REPAIR_EXECUTION_SCHEMA,
    _base_snapshot,
    _candidate_contract_mismatches,
    load_repair_overlay,
)


def _compose_manifest(
    manifest: Mapping[str, Any],
    root: Path,
    output_path: Path,
    audio_seam_policy: str,
    bridge_ms: float,
    crf: int,
) -> dict[str, Any]:
    if audio_seam_policy not in {"none", "cosine_bridge"}:
        raise ValueError("audio_seam_policy must be none or cosine_bridge")
    if not 0 <= float(bridge_ms) <= 50:
        raise ValueError("bridge_ms must be between 0 and 50")
    verified = _verify_accepted_files(dict(manifest), root)
    first = manifest["segments"][0]
    fps = int(first["fps"])
    sample_rate = int(first["sample_rate"])
    width = int(first["width"])
    height = int(first["height"])
    total_frames = int(manifest["segments"][-1]["timeline_end_frame"])
    total_audio_samples = int(manifest["segments"][-1]["audio_end_sample"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import av

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".mp4.tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    encoded_frames = 0
    encoded_audio_samples = 0
    previous_last = None
    seams = []
    segment_reports = []
    try:
        with av.open(str(temporary), mode="w", format="mp4") as output:
            video_stream = output.add_stream("libx264", rate=Fraction(fps, 1))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": str(int(crf)), "preset": "medium"}
            audio_stream = output.add_stream("aac", rate=sample_rate, layout="stereo")
            for segment, path in verified:
                local_frames = 0
                with av.open(str(path), mode="r") as source:
                    if not source.streams.video:
                        raise ValueError(f"Repair source has no video stream: {path}")
                    for decoded_frame in source.decode(source.streams.video[0]):
                        if local_frames >= int(segment["frame_count"]):
                            raise ValueError(
                                "Repair source contains more frames than declared"
                            )
                        rgb = decoded_frame.to_ndarray(format="rgb24")
                        output.mux(
                            video_stream.encode(
                                av.VideoFrame.from_ndarray(rgb, format="rgb24")
                            )
                        )
                        local_frames += 1
                        encoded_frames += 1
                if local_frames != int(segment["frame_count"]):
                    raise ValueError(
                        "Repair source frame count does not match the manifest"
                    )
            output.mux(video_stream.encode(None))

            for segment, path in verified:
                expected = int(segment["audio_end_sample"]) - int(
                    segment["audio_start_sample"]
                )
                audio_array, decode_report = _decode_audio_exact(
                    path, sample_rate, expected
                )
                if previous_last is not None:
                    if audio_seam_policy == "cosine_bridge":
                        audio_array, seam = apply_cosine_bridge(
                            previous_last,
                            audio_array,
                            round(float(bridge_ms) * sample_rate / 1000),
                        )
                    else:
                        jump = float(np.max(np.abs(audio_array[:, 0] - previous_last)))
                        seam = {
                            "bridge_samples": 0,
                            "jump_before": jump,
                            "jump_after": jump,
                            "clipped_sample_values": 0,
                        }
                    seam.update(
                        {
                            "boundary_before_segment": int(segment["index"]),
                            "absolute_sample": int(segment["audio_start_sample"]),
                        }
                    )
                    seams.append(seam)
                previous_last = audio_array[:, -1].copy()
                frame = av.AudioFrame.from_ndarray(
                    audio_array,
                    format="fltp",
                    layout="stereo",
                )
                frame.sample_rate = sample_rate
                frame.pts = encoded_audio_samples
                frame.time_base = Fraction(1, sample_rate)
                output.mux(audio_stream.encode(frame))
                encoded_audio_samples += expected
                segment_reports.append(
                    {
                        "index": int(segment["index"]),
                        "candidate_id": segment["candidate_id"],
                        "frame_count": int(segment["frame_count"]),
                        "audio_samples": expected,
                        **decode_report,
                    }
                )
            output.mux(audio_stream.encode(None))
        if encoded_frames != total_frames:
            raise RuntimeError(
                "Repair composition frame total does not match absolute boundaries"
            )
        if encoded_audio_samples != total_audio_samples:
            raise RuntimeError(
                "Repair composition sample total does not match absolute boundaries"
            )
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "fps": fps,
        "frame_count": total_frames,
        "sample_rate": sample_rate,
        "audio_samples": total_audio_samples,
        "video_duration_seconds": total_frames / fps,
        "audio_duration_seconds": total_audio_samples / sample_rate,
        "absolute_sample_accounting": True,
        "video_reencoded_h264": True,
        "audio_reencoded_aac": True,
        "streaming_memory_scope": "one decoded video frame plus one segment PCM buffer",
        "segments": segment_reports,
        "seams": seams,
    }


def compose_repair_overlay(
    chain_id: str,
    repair_manifest_path: str,
    compose_mode: str,
    filename_prefix: str,
    require_final_segment: bool,
    audio_seam_policy: str,
    bridge_ms: float,
    crf: int,
) -> tuple[str, dict[str, Any]]:
    if compose_mode not in {"repair_overlay", "base_rollback"}:
        raise ValueError("compose_mode must be repair_overlay or base_rollback")
    if compose_mode == "base_rollback":
        path, report_json = compose_accepted_long_video(
            chain_id,
            filename_prefix,
            require_final_segment,
            audio_seam_policy,
            bridge_ms,
            crf,
        )
        report = json.loads(report_json)
        report.update(
            {
                "schema": REPAIR_EXECUTION_SCHEMA,
                "compose_mode": compose_mode,
                "repair_overlay_used": False,
                "rollback_used": True,
            }
        )
        return path, report

    overlay, base, root = load_repair_overlay(chain_id, repair_manifest_path)
    if not base["segments"]:
        raise ValueError("Accepted base manifest contains no segments")
    if require_final_segment and not bool(base["segments"][-1].get("is_final_segment")):
        raise ValueError("The last accepted base segment is not marked final")

    composed_manifest = deepcopy(base)
    replacement_indices = sorted(int(key) for key in overlay["replacements"])
    for index in replacement_indices:
        if index < 0 or index >= len(composed_manifest["segments"]):
            raise ValueError(
                "Repair overlay replacement index is outside the base manifest"
            )
        source = composed_manifest["segments"][index]
        replacement = deepcopy(overlay["replacements"][str(index)])
        mismatches = _candidate_contract_mismatches(source, replacement)
        if mismatches:
            raise ValueError(
                "Repair overlay no longer preserves exact timeline/media fields: "
                + ", ".join(mismatches)
            )
        composed_manifest["segments"][index] = replacement

    unselected = [
        segment
        for segment in base["segments"]
        if int(segment["index"]) not in replacement_indices
    ]
    before_hashes = {
        int(segment["index"]): _sha256_file(
            _resolve_inside(root, segment["video_path"])
        )
        for segment in unselected
    }
    safe_prefix = _safe_token(filename_prefix, fallback_prefix="H3_Repair")
    resolved_manifest = _resolve_inside(root, repair_manifest_path)
    output_path = (
        resolved_manifest.parent
        / "assembled"
        / (
            f"{safe_prefix}_overlay_r{int(overlay['revision']):04d}_{audio_seam_policy}.mp4"
        )
    )
    report = _compose_manifest(
        composed_manifest,
        root,
        output_path,
        audio_seam_policy,
        bridge_ms,
        crf,
    )
    after_hashes = {
        int(segment["index"]): _sha256_file(
            _resolve_inside(root, segment["video_path"])
        )
        for segment in unselected
    }
    if before_hashes != after_hashes:
        raise RuntimeError(
            "An unselected accepted source asset changed during repair composition"
        )
    report.update(
        {
            "schema": REPAIR_EXECUTION_SCHEMA,
            "compose_mode": compose_mode,
            "chain_id": base["chain_id"],
            "base_manifest_revision": base["revision"],
            "base_manifest_sha256": _base_snapshot(base)["sha256"],
            "repair_overlay_revision": overlay["revision"],
            "repair_manifest_path": str(resolved_manifest),
            "replacement_indices": replacement_indices,
            "unselected_indices": sorted(before_hashes),
            "unselected_source_files_sha256_unchanged": True,
            "base_manifest_mutated": False,
            "rollback_available": True,
            "final_container_reencoded": True,
            "bit_exact_final_container_claim": False,
        }
    )
    return str(output_path), report
