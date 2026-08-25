from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.multiface_refine_advanced import _hash_json, _json_safe
from h3_audio_t8_pkg.nodes_skin_finish_p1 import (
    MiniMaxH3SkinFinishMultiPersonT8Advanced,
    MiniMaxH3SkinFinishVideoFinalizeT8Advanced,
    MiniMaxH3SkinFinishVideoStreamT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_p1 import (
    SKIN_FINISH_SDR_VIDEO_CONTRACT,
    SKIN_FINISH_SEQUENCE_REPORT_SCHEMA,
    SKIN_FINISH_VIDEO_REPORT_SCHEMA,
    SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA,
    _validate_sdr_video_stream,
    finalize_skin_finish_video,
    run_multiface_skin_finish,
    stream_skin_finish_video,
)


def _fake_video_stream(
    *,
    pixel_format: str = "yuv420p",
    component_bits: int = 8,
    primaries=2,
    transfer=2,
    colorspace=2,
):
    video_format = SimpleNamespace(
        name=pixel_format,
        components=[SimpleNamespace(bits=component_bits) for _ in range(3)],
    )
    codec_context = SimpleNamespace(
        format=video_format,
        pix_fmt=pixel_format,
        bits_per_raw_sample=0,
        color_primaries=primaries,
        color_trc=transfer,
        colorspace=colorspace,
    )
    return SimpleNamespace(codec_context=codec_context)


def test_sdr_video_contract_accepts_unmarked_8bit_and_reports_evidence():
    report = _validate_sdr_video_stream(
        _fake_video_stream(),
        reported_bit_depth=8,
    )

    assert report["contract"] == SKIN_FINISH_SDR_VIDEO_CONTRACT
    assert report["pixel_format"] == "yuv420p"
    assert report["detected_bit_depth"] == 8
    assert report["component_bits"] == [8, 8, 8]
    assert report["color_primaries"]["code"] == 2
    assert report["transfer_characteristic"]["code"] == 2
    assert report["matrix_colorspace"]["code"] == 2
    assert report["explicit_hdr_or_wide_gamut"] is False


@pytest.mark.parametrize(
    ("kwargs", "reported_bit_depth", "message"),
    [
        ({}, 10, "8-bit SDR"),
        ({"component_bits": 10}, 8, "8-bit SDR"),
        ({"pixel_format": "p010le"}, 8, "8-bit SDR"),
        ({"primaries": 9}, 8, "color primaries"),
        ({"primaries": "bt2020"}, 8, "color primaries"),
        ({"transfer": 16}, 8, "transfer characteristic"),
        ({"transfer": 18}, 8, "transfer characteristic"),
        ({"transfer": "smpte2084"}, 8, "transfer characteristic"),
        ({"colorspace": 14}, 8, "matrix colorspace"),
    ],
)
def test_sdr_video_contract_rejects_high_depth_hdr_and_wide_gamut(
    kwargs,
    reported_bit_depth,
    message,
):
    with pytest.raises(ValueError, match=message):
        _validate_sdr_video_stream(
            _fake_video_stream(**kwargs),
            reported_bit_depth=reported_bit_depth,
        )


def _frames(frame_count: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(24824)
    return torch.rand((frame_count, 64, 96, 3), generator=generator) * 0.55 + 0.20


def _track_plan(frames: torch.Tensor) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    count, height, width, _ = frames.shape
    masks = torch.zeros((count, 2, 16, 24), dtype=torch.bool)
    masks[:, 0, 2:15, 1:12] = True
    masks[:, 1, 2:15, 12:23] = True
    packed = pack_masks(masks)
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": int(count) - 1,
        "frame_count": int(count),
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": "test",
        "mask_size": [16, 24],
    }
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": {
            "frame_count": int(count),
            "height": int(height),
            "width": int(width),
            "fps": 24.0,
            "proxy_sha256": "test-source",
        },
        "analysis": {"height": 16, "width": 24},
        "sam31": {"track_identity_scope": "shot_local_only"},
        "shots": [shot],
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 0,
        "max_scene_delta": 0.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    return plan


def _fake_detections(frames, *_args, **_kwargs):
    detections = []
    for _ in range(int(frames.shape[0])):
        detections.append(
            [
                {
                    "box": [8.0, 13.0, 35.0, 50.0],
                    "confidence": 0.94,
                    "landmarks_xy": [
                        [15.0, 25.0],
                        [28.0, 25.0],
                        [21.5, 32.0],
                        [17.0, 40.0],
                        [26.0, 40.0],
                    ],
                },
                {
                    "box": [60.0, 13.0, 87.0, 50.0],
                    "confidence": 0.93,
                    "landmarks_xy": [
                        [67.0, 25.0],
                        [80.0, 25.0],
                        [73.5, 32.0],
                        [69.0, 40.0],
                        [78.0, 40.0],
                    ],
                },
            ]
        )
    return detections, {"backend": "deterministic_test_yunet", "frame_count": len(detections)}


def test_multiface_stateful_chunks_match_one_whole_pass(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_p1 as module

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", _fake_detections)
    frames = _frames()
    plan = _track_plan(frames)
    whole = run_multiface_skin_finish(
        frames,
        plan,
        minimum_face_height_px=12.0,
        minimum_detail=0.001,
        accept_candidate=True,
    )
    first = run_multiface_skin_finish(
        frames[:5],
        plan,
        minimum_face_height_px=12.0,
        minimum_detail=0.001,
        accept_candidate=True,
    )
    second = run_multiface_skin_finish(
        frames[3:],
        plan,
        absolute_start_frame=3,
        previous_state=first[7],
        minimum_face_height_px=12.0,
        minimum_detail=0.001,
        accept_candidate=True,
    )
    stitched = torch.cat([first[0], second[0]], dim=0)
    assert torch.equal(stitched, whole[0])
    assert second[9:] == (5, 3)
    assert second[3] is None
    assert torch.count_nonzero(whole[4]) > 0
    parsed = json.loads(second[8])
    assert parsed["schema"] == SKIN_FINISH_SEQUENCE_REPORT_SCHEMA
    assert parsed["chunk"]["leading_overlap_frames_verified_and_discarded"] == 2
    assert parsed["mask_contract"]["sam31_reloaded"] is False
    assert parsed["mask_contract"]["rgb_temporal_averaging"] is False
    assert parsed["color_contract"]["per_person_hue_or_saturation_shift"] is False


def test_multiface_overlap_from_another_source_fails_closed(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_p1 as module

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", _fake_detections)
    frames = _frames()
    plan = _track_plan(frames)
    first = run_multiface_skin_finish(
        frames[:5], plan, minimum_face_height_px=12.0, minimum_detail=0.001
    )
    changed = frames[3:].clone()
    changed[0] = 0.0
    with pytest.raises(ValueError, match="overlap source mismatch"):
        run_multiface_skin_finish(
            changed,
            plan,
            absolute_start_frame=3,
            previous_state=first[7],
            minimum_face_height_px=12.0,
            minimum_detail=0.001,
        )


def _write_source(path: Path, frames: int = 5) -> None:
    import av

    with av.open(str(path), mode="w", format="mp4") as output:
        video = output.add_stream("libx264", rate=Fraction(24, 1))
        video.width = 96
        video.height = 64
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("aac", rate=48000, layout="stereo")
        for index in range(frames):
            array = np.full((64, 96, 3), 40 + index * 5, dtype=np.uint8)
            output.mux(video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        output.mux(video.encode(None))
        wave = np.linspace(-0.2, 0.2, round(frames / 24 * 48000), dtype=np.float32)
        planar = np.stack([wave, -wave])
        frame = av.AudioFrame.from_ndarray(planar, format="fltp", layout="stereo")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def test_video_finalize_is_source_by_default_and_packet_copies_audio(monkeypatch, tmp_path):
    import folder_paths
    from comfy_api.latest import InputImpl
    import h3_audio_t8_pkg.skin_finish_p1 as module

    source_path = tmp_path / "source.mp4"
    _write_source(source_path)
    source_video = InputImpl.VideoFromFile(str(source_path))
    components = source_video.get_components()
    frames = components.images.clone()
    frames[:, 20:40, 30:65] = (frames[:, 20:40, 30:65] * 0.95).clamp(0.0, 1.0)
    unchanged = finalize_skin_finish_video(
        source_video, frames, accept_candidate=False
    )
    assert unchanged[0] is source_video
    assert unchanged[1] == ""
    assert json.loads(unchanged[2])["status"] == "SOURCE_SELECTED_NO_FILE_WRITTEN"

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    strictly_validated = []
    monkeypatch.setattr(
        module,
        "_strict_validate_encoded_video",
        lambda path: strictly_validated.append(Path(path)),
    )
    video, path, report, saved = finalize_skin_finish_video(
        source_video,
        frames,
        filename_prefix="skin-finish-test",
        crf=18.0,
        accept_candidate=True,
    )
    parsed = json.loads(report)
    assert Path(path).is_file()
    assert saved is not None
    assert video.get_frame_count() == 5
    assert parsed["schema"] == SKIN_FINISH_VIDEO_REPORT_SCHEMA
    assert parsed["audio"]["packet_payload_exact"] is True
    assert parsed["audio"]["source"] == parsed["audio"]["output"]
    assert parsed["video"]["encoder_threads"] == 1
    assert parsed["video"]["strict_decode_policy"] == "ffmpeg_single_thread_xerror_v1"
    assert parsed["video"]["source_contract"]["contract"] == (
        SKIN_FINISH_SDR_VIDEO_CONTRACT
    )
    assert parsed["video"]["source_contract"]["detected_bit_depth"] == 8
    assert parsed["video"]["output_color_metadata"] == {
        "color_primaries": 2,
        "color_range": 0,
        "color_trc": 2,
        "colorspace": 2,
    }
    assert parsed["source_overwritten"] is False
    assert len(strictly_validated) == 1


def test_p1_node_schemas_are_review_gated_and_append_only():
    multi = MiniMaxH3SkinFinishMultiPersonT8Advanced.define_schema()
    finalizer = MiniMaxH3SkinFinishVideoFinalizeT8Advanced.define_schema()
    stream = MiniMaxH3SkinFinishVideoStreamT8Advanced.define_schema()
    multi_inputs = {item.id: item for item in multi.inputs}
    final_inputs = {item.id: item for item in finalizer.inputs}
    stream_inputs = {item.id: item for item in stream.inputs}
    assert multi.is_experimental is True
    assert multi_inputs["accept_candidate"].default is False
    assert multi_inputs["include_neck"].default is False
    assert multi_inputs["maximum_overlap_frames"].default == 8
    assert finalizer.is_experimental is True
    assert finalizer.is_output_node is True
    assert final_inputs["accept_candidate"].default is False
    assert stream.is_experimental is True
    assert stream.is_output_node is True
    assert stream_inputs["accept_candidate"].default is False
    assert stream_inputs["chunk_frames"].default == 4


def test_video_stream_default_is_exact_source_without_analysis(monkeypatch, tmp_path):
    from comfy_api.latest import InputImpl
    import h3_audio_t8_pkg.skin_finish_p1 as module

    source_path = tmp_path / "source-default.mp4"
    _write_source(source_path)
    source_video = InputImpl.VideoFromFile(str(source_path))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default source selection must not analyze or write")

    monkeypatch.setattr(module, "_analyze_stream_faces", forbidden)
    video, path, report, saved = stream_skin_finish_video(
        source_video, accept_candidate=False
    )
    parsed = json.loads(report)
    assert video is source_video
    assert path == ""
    assert saved is None
    assert parsed["schema"] == SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA
    assert parsed["two_pass_executed"] is False
    assert parsed["full_image_batch_materialized"] is False


def test_video_stream_two_pass_is_chunk_bounded_and_packet_copies_audio(
    monkeypatch, tmp_path
):
    import folder_paths
    from comfy_api.latest import InputImpl
    import h3_audio_t8_pkg.skin_finish_p1 as module

    source_path = tmp_path / "source-stream.mp4"
    _write_source(source_path)
    source_video = InputImpl.VideoFromFile(str(source_path))
    detection = {
        "box": [24.0, 10.0, 72.0, 58.0],
        "confidence": 0.95,
        "landmarks_xy": [
            [36.0, 25.0],
            [60.0, 25.0],
            [48.0, 34.0],
            [39.0, 45.0],
            [57.0, 45.0],
        ],
    }
    monkeypatch.setattr(
        module,
        "_create_pinned_yunet",
        lambda *_args, **_kwargs: (
            object(),
            {"backend": "deterministic_test_yunet", "model_sha256": "test"},
        ),
    )
    monkeypatch.setattr(module, "_detect_yunet_rgb", lambda *_args: [detection])
    monkeypatch.setattr(
        module,
        "_quality_weight",
        lambda *_args, **_kwargs: (0.80, {"quality_weight": 0.80}),
    )
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    strictly_validated = []
    monkeypatch.setattr(
        module,
        "_strict_validate_encoded_video",
        lambda path: strictly_validated.append(Path(path)),
    )

    video, path, report, saved = stream_skin_finish_video(
        source_video,
        minimum_face_height_px=12.0,
        minimum_detail=0.001,
        chunk_frames=2,
        filename_prefix="stream-test",
        accept_candidate=True,
    )
    parsed = json.loads(report)
    assert saved is not None
    assert Path(path).is_file()
    assert video.get_frame_count() == 5
    assert parsed["schema"] == SKIN_FINISH_VIDEO_STREAM_REPORT_SCHEMA
    assert parsed["status"] == "CANDIDATE_TWO_PASS_STREAM_FINALIZED"
    assert parsed["execution"]["full_image_batch_materialized"] is False
    assert parsed["execution"]["peak_chunk_frames"] == 2
    assert parsed["execution"]["used_mask_frame_count"] == 5
    assert parsed["execution"]["source_proxy_equal_between_passes"] is True
    assert parsed["audio"]["packet_payload_exact"] is True
    assert parsed["audio"]["source"] == parsed["audio"]["output"]
    assert parsed["video"]["encoder_threads"] == 1
    assert parsed["video"]["strict_decode_policy"] == "ffmpeg_single_thread_xerror_v1"
    assert parsed["video"]["source_contract"]["contract"] == (
        SKIN_FINISH_SDR_VIDEO_CONTRACT
    )
    assert parsed["video"]["source_contract"]["detected_bit_depth"] == 8
    assert parsed["video"]["output_color_metadata"] == {
        "color_primaries": 2,
        "color_range": 0,
        "color_trc": 2,
        "colorspace": 2,
    }
    assert parsed["source_overwritten"] is False
    assert len(strictly_validated) == 1


def test_strict_video_validation_uses_single_thread_and_explode(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.skin_finish_p1 as module

    path = tmp_path / "candidate.mp4"
    path.write_bytes(b"fixture")
    captured = {}

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(module.shutil, "which", lambda name: "ffmpeg" if name == "ffmpeg" else None)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._strict_validate_encoded_video(path)
    command = captured["command"]
    assert command[command.index("-threads") + 1] == "1"
    assert "-xerror" in command
    assert command[command.index("-err_detect") + 1] == "explode"
    assert command[command.index("-map") + 1] == "0:v:0"


def test_p1_frontend_workflow_has_notes_and_safe_accept_defaults():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-24_H3_Skin_Finish_MultiPerson_Video_Finalize_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = [node["type"] for node in workflow["nodes"]]
    assert types.count("MiniMaxH3SAM31MultiPersonTrackT8Advanced") == 1
    assert types.count("MiniMaxH3SkinFinishMultiPersonT8Advanced") == 1
    assert types.count("MiniMaxH3SkinFinishVideoFinalizeT8Advanced") == 1
    assert types.count("MarkdownNote") == 5
    multi = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishMultiPersonT8Advanced"
    )
    finalizer = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishVideoFinalizeT8Advanced"
    )
    assert multi["widgets_values"][-1] is False
    assert finalizer["widgets_values"][-1] is False
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "不重复加载SAM" in note_text
    assert "payload" in note_text
    assert "不对RGB做时间平均" in note_text


def test_two_pass_stream_workflow_avoids_full_image_decode_and_has_notes():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-24_H3_Skin_Finish_Two_Pass_Video_Stream_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = [node["type"] for node in workflow["nodes"]]
    assert types.count("LoadVideo") == 1
    assert types.count("MiniMaxH3SkinFinishVideoStreamT8Advanced") == 1
    assert "GetVideoComponents" not in types
    assert types.count("MarkdownNote") == 5
    stream = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishVideoStreamT8Advanced"
    )
    assert stream["widgets_values"][-1] is False
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "不要先经过`GetVideoComponents`" in note_text
    assert "第一遍逐帧解码" in note_text
    assert "packet payload" in note_text
