from __future__ import annotations

import json
from pathlib import Path
import types

import pytest
import torch

from h3_audio_t8_pkg.conditioning import build_packed_layout
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.long_video import (
    LONG_VIDEO_CONDITIONING_KEY,
    LONG_VIDEO_SCHEMA,
    MOTION_AUDIO_END_FRAME,
    MOTION_FRAME_INDEX,
    build_long_video_conditioning,
    context_fingerprint,
    load_context_state,
    make_long_video_plan,
    patch_long_video_model,
    repair_long_video_payload,
    save_context_state,
)
from h3_audio_t8_pkg.nodes_long_video_exp import MiniMaxH3LongVideoContextLoadT8
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE


def make_context(width=128, height=128, source_segment_index=0):
    return {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": torch.arange(
            1 * 24 * 12 * (height // 16) * (width // 16), dtype=torch.float32
        ).reshape(1, 24, 12, height // 16, width // 16),
        "audio_tail": torch.arange(1 * 32 * 2 * 65, dtype=torch.float32).reshape(1, 32, 2, 65),
        "metadata": {
            "source_segment_index": source_segment_index,
            "target_segment_index": source_segment_index + 1,
            "max_context_frames": 39,
            "audio_overhang": 1 / 3,
        },
    }


def test_segment_planner_keeps_segment_zero_legacy_and_adds_later_overlap():
    first = make_long_video_plan("测试长视频", 0, 5.0, 22)
    assert first.render_frames == 124
    assert first.context_frames == 0
    assert first.trim_start_seconds == 0
    assert first.final_frame_count == 124
    assert first.final_duration_seconds == pytest.approx(124 / 24)
    assert first.save_context is True

    second = make_long_video_plan("测试长视频", 1, 4.25, 22)
    assert second.render_frames == 124
    assert second.context_frames == 22
    assert second.trim_start_seconds == pytest.approx(22 / 24)
    assert second.final_frame_count == 102
    assert second.timeline_start_seconds == pytest.approx(124 / 24)
    report = json.loads(second.report())
    assert report["video_condition_row_ratio"] == pytest.approx(7 / 37)


def test_final_segment_may_trim_exactly_but_cannot_save_a_continuation_tail():
    final = make_long_video_plan("chain", 2, 4.0, 22, is_final_segment=True)
    assert final.render_frames == 124
    assert final.final_frame_count == 96
    assert final.final_duration_seconds == 4.0
    assert final.hidden_tail_frames == 6
    assert final.save_context is False

    path, report = save_context_state(
        empty_av_latent(128, 128, 124)[0],
        "chain",
        2,
        save_context=False,
    )
    assert path == ""
    assert json.loads(report)["saved"] is False


def test_segment_planner_rejects_unsafe_identity_and_context_values():
    sanitized = make_long_video_plan("../", 0, 4.0, 22)
    assert sanitized.chain_id == "_"
    with pytest.raises(ValueError, match="5, 22, or 39"):
        make_long_video_plan("chain", 1, 4.0, 6)
    with pytest.raises(ValueError, match="negative"):
        make_long_video_plan("chain", -1, 4.0, 22)


def test_context_loader_fingerprint_tolerates_unresolved_dynamic_inputs():
    value = MiniMaxH3LongVideoContextLoadT8.fingerprint_inputs(None, None)
    assert value.startswith("unresolved:")


def test_tail_state_roundtrip_is_bounded_retry_safe_and_cache_aware(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.long_video as long_video

    monkeypatch.setattr(long_video.folder_paths, "get_output_directory", lambda: str(tmp_path))
    latent, _ = empty_av_latent(128, 128, 124)
    path, save_report = save_context_state(latent, "chain-A", 0, "test-model", "4-step")
    assert Path(path).is_file()
    assert not list(Path(path).parent.glob("*.tmp"))
    assert json.loads(save_report)["max_context_frames"] == 39

    context, has_context, load_report = load_context_state("chain-A", 1)
    assert has_context is True
    assert context["video_tail"].shape == (1, 24, 12, 8, 8)
    assert context["audio_tail"].shape[-1] == 65
    assert context["metadata"]["source_segment_index"] == 0
    assert json.loads(load_report)["checksums_valid"] is True
    assert ":missing:" not in context_fingerprint("chain-A", 1)

    # A same-index re-roll atomically replaces only slot 0; segment 1 still reads slot 0.
    reroll, _ = empty_av_latent(128, 128, 124)
    reroll_video, reroll_audio = reroll["samples"].unbind()
    reroll_video.add_(1)
    reroll_audio.add_(1)
    save_context_state(reroll, "chain-A", 0, "test-model", "4-step-reroll")
    context_after, _, _ = load_context_state("chain-A", 1)
    assert torch.all(context_after["video_tail"] == 1)
    assert torch.all(context_after["audio_tail"] == 1)

    empty, has_context, _ = load_context_state("chain-A", 0)
    assert empty["empty"] is True
    assert has_context is False
    with pytest.raises(FileNotFoundError, match="segment 1"):
        load_context_state("chain-A", 2)


def test_loaded_context_rejects_wrong_segment_and_canvas():
    context = make_context(source_segment_index=3)
    with pytest.raises(ValueError, match="immediately previous"):
        build_long_video_conditioning(
            FakeClip(), FakeVideoVAE(), FakeAudioVAE(), context, 1, 22,
            "video_only", "continue", 128, 128, 124,
        )
    with pytest.raises(ValueError, match="same canvas"):
        build_long_video_conditioning(
            FakeClip(), FakeVideoVAE(), FakeAudioVAE(), make_context(), 1, 22,
            "video_only", "continue", 256, 128, 124,
        )


def test_long_conditioning_uses_direct_latent_tail_and_preserves_user_refs():
    clip = FakeClip()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    context = make_context()
    initial_frame = torch.ones((1, 128, 128, 3))
    ref_image = torch.zeros((1, 128, 128, 3))

    conditioning, latent, _, _, media_map, report = build_long_video_conditioning(
        clip,
        video_vae,
        audio_vae,
        context,
        1,
        22,
        "video_and_audio",
        "Continue the motion and sound.",
        128,
        128,
        124,
        "auto",
        "native",
        0.35,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
        None,
        None,
        initial_frame,
        None,
        {"ref_image_1": ref_image},
    )
    metadata = conditioning[0][1]
    assert metadata[LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert len(metadata["minimax_keyframes"]) == 7
    assert [item[MOTION_FRAME_INDEX] for item in metadata["minimax_keyframes"]] == [0, 1, 5, 9, 13, 17, 18]
    assert [item["kind"] for item in metadata["minimax_refs"]] == ["image", "audio"]
    assert MOTION_AUDIO_END_FRAME in metadata["minimax_refs"][-1]
    assert all(item["kind"] != "t8_keyframe_latent" for item in metadata["minimax_refs"])
    assert len(video_vae.encode_calls) == 1  # reference image only; no video VAE round trip for motion
    assert json.loads(media_map)["pictures"] == {"1": "ref_image_1"}
    report_data = json.loads(report)
    assert report_data["motion_keyframes"] == 7
    assert report_data["user_reference_blocks"] == 1
    assert any("first_frame was ignored" in warning for warning in report_data["warnings"])
    assert len(latent["samples"].unbind()) == 2


def test_segment_zero_long_conditioning_keeps_initial_first_frame():
    clip = FakeClip()
    video_vae = FakeVideoVAE()
    context = {"schema": LONG_VIDEO_SCHEMA, "empty": True, "chain_id": "chain", "target_segment_index": 0}
    first = torch.ones((1, 128, 128, 3))
    conditioning, *_ = build_long_video_conditioning(
        clip,
        video_vae,
        FakeAudioVAE(),
        context,
        0,
        0,
        "video_and_audio",
        "Start from the image.",
        128,
        128,
        124,
        "I2VA",
        "native",
        0.35,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
        None,
        None,
        first,
    )
    assert conditioning[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert len(video_vae.encode_calls) == 1


def test_continuation_can_reuse_first_frame_as_persistent_identity_reference():
    clip = FakeClip()
    video_vae = FakeVideoVAE()
    first = torch.ones((1, 128, 128, 3))
    conditioning, _, _, conditioned_prompt, media_map, report = build_long_video_conditioning(
        clip,
        video_vae,
        FakeAudioVAE(),
        make_context(),
        1,
        22,
        "video_and_audio",
        "Keep <Picture 1> as the same person while motion continues.",
        128,
        128,
        124,
        "auto",
        "native",
        0.35,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
        None,
        None,
        first,
        None,
        None,
        None,
        None,
        None,
        "persistent_identity_reference",
    )

    metadata = conditioning[0][1]
    assert [item["kind"] for item in metadata["minimax_refs"]] == ["image", "audio"]
    assert len(metadata["minimax_keyframes"]) == 7
    assert len(video_vae.encode_calls) == 1
    assert conditioned_prompt == "Keep <Picture 1> as the same person while motion continues."
    assert json.loads(media_map)["pictures"] == {
        "1": "first_frame (persistent identity reference)"
    }
    report_data = json.loads(report)
    assert report_data["task"] == "hybrid"
    assert report_data["first_frame_reuse"] == "persistent_identity_reference"
    assert report_data["persistent_identity_reference"] is True
    assert report_data["persistent_identity_strategy"] == "single_reference"
    assert report_data["persistent_identity_source"] == "first_frame"
    assert report_data["persistent_identity_sources"] == ["first_frame"]
    assert report_data["persistent_identity_reference_count"] == 1
    assert report_data["persistent_identity_image_connected"] is False
    assert report_data["user_reference_blocks"] == 0
    assert report_data["reference_blocks_total"] == 1
    assert any("not a guarantee" in warning for warning in report_data["warnings"])


def test_continuation_prefers_dedicated_identity_image_without_changing_segment_zero():
    first = torch.zeros((1, 128, 128, 3))
    identity = torch.ones((1, 96, 96, 3))

    segment_zero_vae = FakeVideoVAE()
    _, _, _, _, segment_zero_map, segment_zero_report = build_long_video_conditioning(
        FakeClip(),
        segment_zero_vae,
        FakeAudioVAE(),
        {"schema": LONG_VIDEO_SCHEMA, "empty": True, "chain_id": "chain", "target_segment_index": 0},
        0,
        0,
        "video_and_audio",
        "Start from the image.",
        128,
        128,
        124,
        task_type="I2VA",
        first_frame=first,
        first_frame_reuse="persistent_identity_reference",
        persistent_identity_image=identity,
    )
    assert len(segment_zero_vae.encode_calls) == 1
    assert torch.equal(segment_zero_vae.encode_calls[0], first)
    assert json.loads(segment_zero_map)["pictures"] == {"1": "first_frame (exact frame 0)"}
    segment_zero_data = json.loads(segment_zero_report)
    assert segment_zero_data["persistent_identity_reference"] is False
    assert segment_zero_data["persistent_identity_source"] == "none"
    assert segment_zero_data["persistent_identity_image_connected"] is True

    continuation_vae = FakeVideoVAE()
    conditioning, _, _, _, media_map, report = build_long_video_conditioning(
        FakeClip(),
        continuation_vae,
        FakeAudioVAE(),
        make_context(),
        1,
        22,
        "video_and_audio",
        "Keep <Picture 1> as the same person while motion continues.",
        128,
        128,
        124,
        first_frame=first,
        first_frame_reuse="persistent_identity_reference",
        persistent_identity_image=identity,
    )
    assert len(continuation_vae.encode_calls) == 1
    assert torch.equal(continuation_vae.encode_calls[0], identity)
    assert [item["kind"] for item in conditioning[0][1]["minimax_refs"]] == ["image", "audio"]
    assert json.loads(media_map)["pictures"] == {
        "1": "persistent_identity_image (persistent identity reference)"
    }
    report_data = json.loads(report)
    assert report_data["persistent_identity_reference"] is True
    assert report_data["persistent_identity_strategy"] == "single_reference"
    assert report_data["persistent_identity_source"] == "persistent_identity_image"
    assert report_data["persistent_identity_sources"] == ["persistent_identity_image"]
    assert report_data["persistent_identity_reference_count"] == 1
    assert report_data["persistent_identity_image_connected"] is True


def test_dedicated_identity_image_is_inert_when_persistent_policy_is_disabled():
    video_vae = FakeVideoVAE()
    conditioning, _, _, _, media_map, report = build_long_video_conditioning(
        FakeClip(),
        video_vae,
        FakeAudioVAE(),
        make_context(),
        1,
        22,
        "video_and_audio",
        "Continue motion.",
        128,
        128,
        124,
        first_frame=torch.zeros((1, 128, 128, 3)),
        persistent_identity_image=torch.ones((1, 96, 96, 3)),
    )
    assert len(video_vae.encode_calls) == 0
    assert [item["kind"] for item in conditioning[0][1]["minimax_refs"]] == ["audio"]
    assert json.loads(media_map)["pictures"] == {}
    report_data = json.loads(report)
    assert report_data["persistent_identity_reference"] is False
    assert report_data["persistent_identity_source"] == "none"
    assert any(
        "first_frame_reuse is segment0_only" in warning
        for warning in report_data["warnings"]
    )


def test_scene_plus_identity_supplies_two_continuation_only_references():
    video_vae = FakeVideoVAE()
    first = torch.zeros((1, 128, 128, 3))
    identity = torch.ones((1, 96, 96, 3))
    conditioning, _, _, _, media_map, report = build_long_video_conditioning(
        FakeClip(),
        video_vae,
        FakeAudioVAE(),
        make_context(),
        1,
        22,
        "video_and_audio",
        "Continue with the same person and scene.",
        128,
        128,
        124,
        first_frame=first,
        first_frame_reuse="persistent_identity_reference",
        persistent_identity_image=identity,
        persistent_identity_strategy="scene_plus_identity",
    )
    assert len(video_vae.encode_calls) == 2
    assert torch.equal(video_vae.encode_calls[0], first)
    assert torch.equal(video_vae.encode_calls[1], identity)
    assert [item["kind"] for item in conditioning[0][1]["minimax_refs"]] == [
        "image",
        "image",
        "audio",
    ]
    assert json.loads(media_map)["pictures"] == {
        "1": "first_frame (persistent identity reference)",
        "2": "persistent_identity_image (persistent identity reference)",
    }
    report_data = json.loads(report)
    assert report_data["persistent_identity_strategy"] == "scene_plus_identity"
    assert report_data["persistent_identity_source"] == (
        "first_frame+persistent_identity_image"
    )
    assert report_data["persistent_identity_reference_count"] == 2
    assert report_data["user_reference_blocks"] == 0


def test_persistent_identity_interval_skips_only_selected_continuation_segments():
    first = torch.zeros((1, 128, 128, 3))
    identity = torch.ones((1, 96, 96, 3))

    skipped_vae = FakeVideoVAE()
    skipped, _, _, _, skipped_map, skipped_report = build_long_video_conditioning(
        FakeClip(),
        skipped_vae,
        FakeAudioVAE(),
        make_context(source_segment_index=1),
        2,
        22,
        "video_and_audio",
        "Continue the same action.",
        128,
        128,
        124,
        first_frame=first,
        first_frame_reuse="persistent_identity_reference",
        persistent_identity_image=identity,
        persistent_identity_strategy="scene_plus_identity",
        persistent_identity_interval=2,
    )
    assert len(skipped_vae.encode_calls) == 0
    assert [item["kind"] for item in skipped[0][1]["minimax_refs"]] == ["audio"]
    assert json.loads(skipped_map)["pictures"] == {}
    skipped_data = json.loads(skipped_report)
    assert skipped_data["persistent_identity_requested"] is True
    assert skipped_data["persistent_identity_reference"] is False
    assert skipped_data["persistent_identity_due"] is False
    assert skipped_data["persistent_identity_interval"] == 2
    assert any("skipped on segment 2" in warning for warning in skipped_data["warnings"])

    due_vae = FakeVideoVAE()
    due, _, _, _, due_map, due_report = build_long_video_conditioning(
        FakeClip(),
        due_vae,
        FakeAudioVAE(),
        make_context(source_segment_index=2),
        3,
        22,
        "video_and_audio",
        "Continue with <Picture 1> and <Picture 2>.",
        128,
        128,
        124,
        first_frame=first,
        first_frame_reuse="persistent_identity_reference",
        persistent_identity_image=identity,
        persistent_identity_strategy="scene_plus_identity",
        persistent_identity_interval=2,
    )
    assert len(due_vae.encode_calls) == 2
    assert [item["kind"] for item in due[0][1]["minimax_refs"]] == [
        "image",
        "image",
        "audio",
    ]
    assert set(json.loads(due_map)["pictures"]) == {"1", "2"}
    due_data = json.loads(due_report)
    assert due_data["persistent_identity_reference"] is True
    assert due_data["persistent_identity_due"] is True
    assert due_data["persistent_identity_interval"] == 2


def test_persistent_identity_interval_rejects_zero():
    with pytest.raises(ValueError, match="persistent_identity_interval must be at least 1"):
        build_long_video_conditioning(
            FakeClip(),
            FakeVideoVAE(),
            FakeAudioVAE(),
            make_context(),
            1,
            22,
            "video_only",
            "Continue.",
            128,
            128,
            124,
            first_frame=torch.zeros((1, 128, 128, 3)),
            first_frame_reuse="persistent_identity_reference",
            persistent_identity_interval=0,
        )


def test_scene_plus_identity_fails_closed_without_dedicated_image():
    with pytest.raises(ValueError, match="requires a connected persistent_identity_image"):
        build_long_video_conditioning(
            FakeClip(),
            FakeVideoVAE(),
            FakeAudioVAE(),
            make_context(),
            1,
            22,
            "video_only",
            "Continue.",
            128,
            128,
            124,
            first_frame=torch.zeros((1, 128, 128, 3)),
            first_frame_reuse="persistent_identity_reference",
            persistent_identity_strategy="scene_plus_identity",
        )


def test_persistent_identity_reference_fails_closed_without_image_or_with_too_many_refs():
    common = (
        FakeClip(), FakeVideoVAE(), FakeAudioVAE(), make_context(), 1, 22,
        "video_only", "continue", 128, 128, 124,
    )
    with pytest.raises(ValueError, match="requires a connected first_frame"):
        build_long_video_conditioning(
            *common,
            first_frame_reuse="persistent_identity_reference",
        )

    first = torch.ones((1, 128, 128, 3))
    refs = {f"ref_image_{index}": torch.zeros((1, 128, 128, 3)) for index in range(9)}
    with pytest.raises(ValueError, match="must not exceed 9 pictures"):
        build_long_video_conditioning(
            *common,
            first_frame=first,
            ref_images=refs,
            first_frame_reuse="persistent_identity_reference",
        )


def test_persistent_identity_reference_keeps_explicit_fl2va_fail_closed():
    with pytest.raises(ValueError, match="FL2VA continuation requires last_frame and no references"):
        build_long_video_conditioning(
            FakeClip(), FakeVideoVAE(), FakeAudioVAE(), make_context(), 1, 22,
            "video_only", "continue", 128, 128, 124,
            task_type="FL2VA",
            first_frame=torch.ones((1, 128, 128, 3)),
            last_frame=torch.zeros((1, 128, 128, 3)),
            first_frame_reuse="persistent_identity_reference",
        )


class _CondConstant:
    def __init__(self, cond):
        self.cond = cond


def test_local_payload_repair_handles_multiple_refs_without_global_patch():
    text_len, latent_t, latent_h, latent_w, audio_t = 7, 37, 8, 8, 207
    frame_count = 124
    keyframes = [
        {
            "resolved_frame_index": 0,
            MOTION_FRAME_INDEX: offset,
            "latent": torch.full((1, 24, 1, 8, 8), float(index)),
        }
        for index, offset in enumerate([0, 1, 5, 9, 13, 17, 18])
    ]
    refs = [
        {
            "kind": "image",
            "latent_h": 8,
            "latent_w": 8,
            "latent": torch.full((1,), 10.0),
        },
        {"kind": "audio", "ref_audio_t": 3, "audio_latent": torch.full((1,), 11.0)},
        {
            "kind": "audio",
            "ref_audio_t": 37,
            "audio_latent": torch.full((1,), 12.0),
            MOTION_AUDIO_END_FRAME: 22.0,
        },
    ]
    layout = build_packed_layout(
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=keyframes,
        refs=refs,
        frame_count=frame_count,
    )
    payload = {"layout": layout, "frame_count": frame_count}
    out = {"minimax_payload": _CondConstant(payload)}
    kwargs = {
        LONG_VIDEO_CONDITIONING_KEY: LONG_VIDEO_SCHEMA,
        "minimax_keyframes": keyframes,
        "minimax_refs": refs,
        "minimax_frame_count": frame_count,
    }
    repair_long_video_payload(out, kwargs)

    assert len(payload["cond_video_latents"]) == 8
    assert len(payload["cond_audio_latents"]) == 2
    assert payload["t8_long_video_patch_version"] == 1
    total_ref_advance = 1 + 3 + 37
    first_cond = next((a, b) for a, b, kind in layout.segments if kind == "cond")
    assert torch.all(layout.position_ids[first_cond[0]:first_cond[1], 0] == text_len + total_ref_advance)

    ref_audio_segments = [(a, b) for a, b, kind in layout.segments if kind == "ref_audio"]
    marked_start, marked_stop = ref_audio_segments[-1]
    expected_start = text_len + total_ref_advance + (5 / 3) * 22 - 37
    assert float(layout.position_ids[marked_start, 0]) == pytest.approx(expected_start)
    assert marked_stop - marked_start == 74


def test_payload_repair_is_a_noop_for_existing_stable_conditioning():
    marker = object()
    out = {"unchanged": marker}
    assert repair_long_video_payload(out, {}) is out
    assert out["unchanged"] is marker


class _FakeModelPatcher:
    def __init__(self, model, object_patches=None):
        self.model = model
        self.object_patches = dict(object_patches or {})

    def clone(self):
        return _FakeModelPatcher(self.model, self.object_patches)

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name))

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


def test_model_patch_is_attached_only_to_clone_and_repairs_extra_conds_output():
    keyframes = [
        {
            "resolved_frame_index": 0,
            MOTION_FRAME_INDEX: offset,
            "latent": torch.full((1, 24, 1, 8, 8), float(index)),
        }
        for index, offset in enumerate([0, 1, 5, 9, 13, 17, 18])
    ]
    refs = [
        {"kind": "image", "latent_h": 8, "latent_w": 8, "latent": torch.full((1,), 10.0)},
        {
            "kind": "audio",
            "ref_audio_t": 37,
            "audio_latent": torch.full((1,), 12.0),
            MOTION_AUDIO_END_FRAME: 22.0,
        },
    ]

    def extra_conds(_self, **kwargs):
        layout = build_packed_layout(
            7,
            37,
            8,
            8,
            207,
            keyframes=kwargs["minimax_keyframes"],
            refs=kwargs["minimax_refs"],
            frame_count=kwargs["minimax_frame_count"],
        )
        payload = {"layout": layout, "frame_count": kwargs["minimax_frame_count"]}
        return {"minimax_payload": _CondConstant(payload)}

    model_type = type("MiniMaxH3", (), {})
    base_model = model_type()
    base_model.extra_conds = types.MethodType(extra_conds, base_model)
    original = _FakeModelPatcher(base_model)

    patched = patch_long_video_model(original)
    assert "extra_conds" not in original.object_patches
    assert "extra_conds" in patched.object_patches
    assert base_model.extra_conds.__func__ is extra_conds

    out = patched.object_patches["extra_conds"](
        **{
            LONG_VIDEO_CONDITIONING_KEY: LONG_VIDEO_SCHEMA,
            "minimax_keyframes": keyframes,
            "minimax_refs": refs,
            "minimax_frame_count": 124,
        }
    )
    payload = out["minimax_payload"].cond
    assert payload["t8_long_video_patch_version"] == 1
    assert len(payload["cond_video_latents"]) == 8
    assert len(payload["cond_audio_latents"]) == 1
