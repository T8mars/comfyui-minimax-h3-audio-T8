from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.core import nested_av_parts
from h3_audio_t8_pkg.native_latent_timeline_advanced import (
    CONTINUATION_CONCAT_SCHEMA,
    RESUME_MANIFEST_SCHEMA,
    audit_native_h3_av_latent_resume_manifest,
    concat_native_h3_av_continuation,
    concat_native_h3_av_latents,
)


def _latent(frames: int, value: float, *, mask=False, width=8, metadata="same"):
    n = (frames - 5) // 17
    video_t = n * 5 + 2
    audio_t = round(frames / 24 * 40)
    video = torch.full((1, 24, video_t, 4, width), value, dtype=torch.float32)
    audio = torch.full((1, 32, 2, audio_t), value, dtype=torch.float32)
    result = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "source_metadata": metadata,
    }
    if mask:
        result["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.full_like(video, value + 10), torch.full_like(audio, value + 20))
        )
    return result


def _continuation_proofs(
    *,
    segment_index: int,
    timeline_frames: int,
    context_frames: int = 22,
    render_frames: int = 124,
    final_frame_count: int | None = None,
    hidden_tail_frames: int = 0,
    chain_id: str = "chain-a",
    context_audio: str = "video_and_audio",
):
    physical_append = render_frames - context_frames
    if final_frame_count is None:
        final_frame_count = physical_append - hidden_tail_frames
    is_final = hidden_tail_frames > 0
    planner = {
        "schema": 1,
        "chain_id": chain_id,
        "segment_index": segment_index,
        "render_frames": render_frames,
        "context_frames": context_frames,
        "trim_start_seconds": context_frames / 24,
        "final_frame_count": final_frame_count,
        "hidden_tail_frames": hidden_tail_frames,
        "timeline_start_seconds": timeline_frames / 24,
        "timeline_end_seconds": (timeline_frames + final_frame_count) / 24,
        "is_final_segment": is_final,
        "save_context": not is_final,
    }
    conditioning = {
        "schema": 1,
        "segment_index": segment_index,
        "context_active": True,
        "context_frames": context_frames,
        "context_audio": context_audio,
        "task": "i2va-motion",
        "render_frames": render_frames,
        "motion_keyframes": {5: 2, 22: 7, 39: 12}[context_frames],
        "timeline_audio_ref": context_audio == "video_and_audio",
    }
    return json.dumps(planner), json.dumps(conditioning)


def test_two_22_frame_segments_become_exact_39_frame_native_grid():
    first = _latent(22, 1.0)
    second = _latent(22, 2.0)
    output, frames, count, audio_t, report_json = concat_native_h3_av_latents(
        first, second, "cpu", False
    )
    video, audio = nested_av_parts(output)
    report = json.loads(report_json)
    assert frames == 39 and count == 2
    assert video.shape == (1, 24, 12, 4, 8)
    assert audio.shape == (1, 32, 2, 65)
    assert audio_t == 65
    assert torch.all(video[:, :, :7] == 1)
    assert torch.all(video[:, :, 7:] == 2)
    assert report["segments"][1]["drop_video_t"] == 2
    assert report["segments"][1]["drop_audio_t"] == 9
    assert report["sampling_executed"] is False
    assert report["vae_decode_executed"] is False


def test_three_124_frame_segments_preserve_cumulative_audio_rounding_phase():
    first = _latent(124, 1.0)
    second = _latent(124, 2.0)
    third = _latent(124, 3.0)
    output, frames, count, audio_t, report_json = concat_native_h3_av_latents(
        first,
        second,
        "cpu",
        False,
        {"segment_0": third},
    )
    video, audio = nested_av_parts(output)
    report = json.loads(report_json)
    assert frames == 362
    assert count == 3
    assert video.shape[2] == 107
    assert audio_t == round(362 / 24 * 40) == 603
    assert audio.shape[-1] == 603
    assert [segment["drop_audio_t"] for segment in report["segments"]] == [0, 9, 9]


def test_nested_noise_masks_follow_the_same_video_and_audio_phase():
    output = concat_native_h3_av_latents(
        _latent(22, 1.0, mask=True), _latent(22, 2.0, mask=True)
    )[0]
    video, audio = nested_av_parts(output)
    video_mask, audio_mask = tuple(output["noise_mask"].unbind())
    assert video_mask.shape == video.shape
    assert audio_mask.shape == audio.shape
    assert torch.all(video_mask[:, :, :7] == 11)
    assert torch.all(video_mask[:, :, 7:] == 12)
    assert torch.all(audio_mask[..., :37] == 21)
    assert torch.all(audio_mask[..., 37:] == 22)


def test_proven_22_frame_long_video_context_is_removed_on_both_native_clocks():
    first = _latent(124, 1.0, mask=True)
    continuation = _latent(124, 2.0, mask=True)
    planner, conditioning = _continuation_proofs(segment_index=1, timeline_frames=124)
    output, physical, visible, trim_tail, audio_t, report_json = (
        concat_native_h3_av_continuation(
            first,
            continuation,
            planner,
            conditioning,
        )
    )
    video, audio = nested_av_parts(output)
    video_mask, audio_mask = tuple(output["noise_mask"].unbind())
    report = json.loads(report_json)
    assert physical == visible == 226 and trim_tail == 0
    assert video.shape == (1, 24, 67, 4, 8)
    assert audio.shape == (1, 32, 2, round(226 / 24 * 40))
    assert audio_t == 377
    assert torch.all(video[:, :, :37] == 1)
    assert torch.all(video[:, :, 37:] == 2)
    assert torch.all(video_mask[:, :, :37] == 11)
    assert torch.all(video_mask[:, :, 37:] == 12)
    assert torch.all(audio_mask[..., :207] == 21)
    assert torch.all(audio_mask[..., 207:] == 22)
    assert report["schema"] == CONTINUATION_CONCAT_SCHEMA
    assert report["context_removed"] == {
        "frames": 22,
        "video_latent_steps": 7,
        "audio_latent_steps": 37,
        "audio_context_mode": "video_and_audio",
    }
    assert report["segment_count"] == 2 and report["last_segment_index"] == 1
    assert report["chain_closed"] is False


def test_continuation_concat_chains_39_frame_context_and_preserves_audio_phase():
    first = _latent(124, 1.0)
    second = _latent(124, 2.0)
    proof_1 = _continuation_proofs(segment_index=1, timeline_frames=124)
    timeline = concat_native_h3_av_continuation(first, second, *proof_1)[0]

    third = _latent(124, 3.0)
    proof_2 = _continuation_proofs(
        segment_index=2,
        timeline_frames=226,
        context_frames=39,
    )
    output, physical, visible, trim_tail, audio_t, report_json = (
        concat_native_h3_av_continuation(timeline, third, *proof_2)
    )
    video, audio = nested_av_parts(output)
    report = json.loads(report_json)
    assert physical == visible == 311 and trim_tail == 0
    assert video.shape[2] == 92 and audio.shape[-1] == audio_t == round(311 / 24 * 40)
    assert report["context_removed"]["frames"] == 39
    assert report["context_removed"]["video_latent_steps"] == 12
    assert report["segment_count"] == 3 and report["last_segment_index"] == 2


def test_continuation_concat_five_frame_context_uses_cumulative_audio_phase():
    planner, conditioning = _continuation_proofs(
        segment_index=1,
        timeline_frames=124,
        context_frames=5,
    )
    output, physical, visible, trim_tail, audio_t, report_json = (
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            planner,
            conditioning,
        )
    )
    video, audio = nested_av_parts(output)
    report = json.loads(report_json)
    assert physical == visible == 243 and trim_tail == 0
    assert video.shape[2] == 72
    assert audio.shape[-1] == audio_t == round(243 / 24 * 40) == 405
    assert report["context_removed"] == {
        "frames": 5,
        "video_latent_steps": 2,
        "audio_latent_steps": 9,
        "audio_context_mode": "video_and_audio",
    }


def test_final_hidden_tail_is_reported_for_one_decode_trim_and_closes_chain():
    planner, conditioning = _continuation_proofs(
        segment_index=1,
        timeline_frames=124,
        final_frame_count=100,
        hidden_tail_frames=2,
    )
    output, physical, visible, trim_tail, _audio_t, report_json = (
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            planner,
            conditioning,
        )
    )
    report = json.loads(report_json)
    assert physical == 226 and visible == 224 and trim_tail == 2
    assert report["chain_closed"] is True
    next_planner, next_conditioning = _continuation_proofs(
        segment_index=2,
        timeline_frames=226,
    )
    with pytest.raises(ValueError, match="already closed"):
        concat_native_h3_av_continuation(
            output,
            _latent(124, 3.0),
            next_planner,
            next_conditioning,
        )


def test_exact_grid_final_segment_without_hidden_tail_still_closes_chain():
    planner_json, conditioning = _continuation_proofs(
        segment_index=1,
        timeline_frames=124,
        context_frames=5,
    )
    planner = json.loads(planner_json)
    planner["is_final_segment"] = True
    planner["save_context"] = False
    output, physical, visible, trim_tail, _audio_t, report_json = (
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            json.dumps(planner),
            conditioning,
        )
    )
    report = json.loads(report_json)
    assert physical == visible == 243 and trim_tail == 0
    assert report["chain_closed"] is True
    next_planner, next_conditioning = _continuation_proofs(
        segment_index=2,
        timeline_frames=243,
    )
    with pytest.raises(ValueError, match="already closed"):
        concat_native_h3_av_continuation(
            output,
            _latent(124, 3.0),
            next_planner,
            next_conditioning,
        )


def test_continuation_concat_requires_matching_long_video_proofs_and_audio_policy():
    planner, conditioning = _continuation_proofs(segment_index=1, timeline_frames=124)
    wrong_conditioning = json.loads(conditioning)
    wrong_conditioning["motion_keyframes"] = 2
    with pytest.raises(ValueError, match="motion_keyframes"):
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            planner,
            json.dumps(wrong_conditioning),
        )

    video_only_planner, video_only_conditioning = _continuation_proofs(
        segment_index=1,
        timeline_frames=124,
        context_audio="video_only",
    )
    with pytest.raises(ValueError, match="audio_context_policy"):
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            video_only_planner,
            video_only_conditioning,
        )
    result = concat_native_h3_av_continuation(
        _latent(124, 1.0),
        _latent(124, 2.0),
        video_only_planner,
        video_only_conditioning,
        audio_context_policy="allow_video_only",
    )
    assert json.loads(result[-1])["context_removed"]["audio_context_mode"] == "video_only"

    stale_planner = json.loads(planner)
    stale_planner["timeline_start_seconds"] = 0
    with pytest.raises(ValueError, match="timeline_start_seconds"):
        concat_native_h3_av_continuation(
            _latent(124, 1.0),
            _latent(124, 2.0),
            json.dumps(stale_planner),
            conditioning,
        )


def test_mixed_masks_canvas_dtype_audio_grid_and_metadata_fail_closed():
    with pytest.raises(ValueError, match="either carry"):
        concat_native_h3_av_latents(_latent(22, 1.0, mask=True), _latent(22, 2.0))
    with pytest.raises(ValueError, match="canvas mismatch"):
        concat_native_h3_av_latents(_latent(22, 1.0), _latent(22, 2.0, width=9))
    wrong_dtype = _latent(22, 2.0)
    video, audio = nested_av_parts(wrong_dtype)
    wrong_dtype["samples"] = comfy.nested_tensor.NestedTensor((video.half(), audio))
    with pytest.raises(ValueError, match="dtype mismatch"):
        concat_native_h3_av_latents(_latent(22, 1.0), wrong_dtype)
    wrong_audio = _latent(22, 2.0)
    video, audio = nested_av_parts(wrong_audio)
    wrong_audio["samples"] = comfy.nested_tensor.NestedTensor((video, audio[..., :-1]))
    with pytest.raises(ValueError, match="does not match"):
        concat_native_h3_av_latents(_latent(22, 1.0), wrong_audio)
    with pytest.raises(ValueError, match="metadata differs"):
        concat_native_h3_av_latents(
            _latent(22, 1.0, metadata="a"),
            _latent(22, 2.0, metadata="b"),
            require_identical_metadata=True,
        )


def test_resume_manifest_baseline_and_exact_match_are_chunk_size_invariant():
    latent = _latent(124, 1.0, mask=True, width=128, metadata={"shot": 3})
    status, verified, digest, baseline_json = audit_native_h3_av_latent_resume_manifest(
        latent,
        checkpoint_id="shot_003",
        hash_chunk_megabytes=1,
    )
    baseline = json.loads(baseline_json)
    assert status == "BASELINE_CREATED"
    assert verified is False
    assert len(digest) == 64 and digest == digest.upper()
    assert baseline["schema"] == RESUME_MANIFEST_SCHEMA
    assert baseline["frame_count"] == 124
    assert baseline["noise_mask_present"] is True
    assert baseline["files_written"] is False
    assert baseline["sampling_executed"] is False
    assert baseline["vae_decode_executed"] is False

    status, verified, second_digest, matched_json = (
        audit_native_h3_av_latent_resume_manifest(
            latent,
            checkpoint_id="shot_003",
            expected_manifest_json=baseline_json,
            mismatch_policy="error",
            hash_chunk_megabytes=8,
        )
    )
    matched = json.loads(matched_json)
    assert status == "MATCH" and verified is True
    assert second_digest == digest
    assert matched["comparison"]["mismatches"] == []


def test_resume_manifest_detects_content_mask_metadata_and_checkpoint_mismatch():
    baseline_latent = _latent(22, 1.0, mask=True, metadata={"shot": "a"})
    baseline_json = audit_native_h3_av_latent_resume_manifest(
        baseline_latent,
        checkpoint_id="shot_a",
    )[3]

    changed = _latent(22, 1.0, mask=True, metadata={"shot": "a"})
    changed_video, _changed_audio = nested_av_parts(changed)
    changed_video[0, 0, 0, 0, 0] = 1.25
    with pytest.raises(ValueError, match="content_sha256"):
        audit_native_h3_av_latent_resume_manifest(
            changed,
            checkpoint_id="shot_a",
            expected_manifest_json=baseline_json,
        )

    status, verified, changed_digest, report_json = (
        audit_native_h3_av_latent_resume_manifest(
            changed,
            checkpoint_id="wrong_shot",
            expected_manifest_json=baseline_json,
            mismatch_policy="report_only",
        )
    )
    report = json.loads(report_json)
    assert status == "MISMATCH" and verified is False
    assert changed_digest != json.loads(baseline_json)["content_sha256"]
    assert {item["field"] for item in report["comparison"]["mismatches"]} == {
        "checkpoint_id",
        "content_sha256",
    }

    changed_mask = _latent(22, 1.0, mask=True, metadata={"shot": "a"})
    changed_mask_parts = tuple(changed_mask["noise_mask"].unbind())
    changed_mask_parts[1][0, 0, 0, 0] = 99
    assert (
        audit_native_h3_av_latent_resume_manifest(changed_mask, checkpoint_id="shot_a")[2]
        != json.loads(baseline_json)["content_sha256"]
    )

    changed_metadata = _latent(22, 1.0, mask=True, metadata={"shot": "b"})
    assert (
        audit_native_h3_av_latent_resume_manifest(changed_metadata, checkpoint_id="shot_a")[2]
        != json.loads(baseline_json)["content_sha256"]
    )


def test_resume_manifest_excludes_only_declared_volatile_reports_and_fails_closed():
    baseline = _latent(22, 1.0)
    baseline["t8_native_latent_timeline_concat"] = {"output_device": "cpu"}
    first = audit_native_h3_av_latent_resume_manifest(baseline)[2]
    baseline["t8_native_latent_timeline_concat"] = {"output_device": "cuda:0"}
    assert audit_native_h3_av_latent_resume_manifest(baseline)[2] == first

    metadata_string = _latent(22, 1.0)
    metadata_string["typed"] = "abc"
    metadata_bytes = _latent(22, 1.0)
    metadata_bytes["typed"] = b'"abc"'
    assert (
        audit_native_h3_av_latent_resume_manifest(metadata_string)[2]
        != audit_native_h3_av_latent_resume_manifest(metadata_bytes)[2]
    )

    unsupported = _latent(22, 1.0)
    unsupported["opaque"] = object()
    with pytest.raises(ValueError, match="unsupported type object"):
        audit_native_h3_av_latent_resume_manifest(unsupported)
    with pytest.raises(ValueError, match="invalid JSON"):
        audit_native_h3_av_latent_resume_manifest(
            _latent(22, 1.0), expected_manifest_json="{not-json"
        )
    with pytest.raises(ValueError, match="must use schema"):
        audit_native_h3_av_latent_resume_manifest(
            _latent(22, 1.0), expected_manifest_json='{"schema":"wrong"}'
        )


def test_native_latent_frontend_workflow_is_importable_and_has_formula_notes():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-22_H3_Native_Latent_Timeline_Concat_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    concat = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3NativeLatentTimelineConcatT8Advanced"
    )
    notes = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert concat["widgets_values"] == ["cpu", False]
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "next_audio_T" in notes and "362帧/603音频步" in notes
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == 0 and workflow["links"] == []


def test_native_latent_resume_manifest_workflow_is_importable_and_fail_closed():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-23_H3_Native_Latent_Resume_Manifest_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    audit = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3NativeLatentResumeManifestT8Advanced"
    )
    notes = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert audit["widgets_values"] == ["timeline_checkpoint", "", "error", 8]
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 4
    assert "resume_verified=true" in notes
    assert "不保存扩散器内部迭代状态" in notes
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == 0 and workflow["links"] == []


def test_native_latent_continuation_workflow_wires_both_long_video_proofs_directly():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-23_H3_Native_Latent_Continuation_Concat_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    planner = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3LongVideoPlannerT8"
    )
    conditioning = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    concat = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3NativeLatentContinuationConcatT8Advanced"
    )
    links = {link[0]: link for link in workflow["links"]}
    assert concat["widgets_values"] == ["cpu", "require_video_and_audio"]
    assert planner["widgets_values"][1:6] == [1, 4.25, 22, 124, 5.166666667]
    assert links[concat["inputs"][2]["link"]][1:5] == [planner["id"], 9, concat["id"], 2]
    assert links[concat["inputs"][3]["link"]][1:5] == [
        conditioning["id"],
        6,
        concat["id"],
        3,
    ]
    assert concat["inputs"][0]["link"] is None
    assert concat["inputs"][1]["link"] is None
    notes = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 4
    assert "124帧时间线 + 124帧续段" in notes
    assert "require_video_and_audio" in notes
    assert "trim_tail_frames_after_decode" in notes
    assert "不恢复运行中的NFE" in notes
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
