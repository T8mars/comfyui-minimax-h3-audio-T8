from __future__ import annotations

import json

import pytest
import torch

import h3_audio_t8_pkg.conditioning as conditioning_module
from h3_audio_t8_pkg.conditioning import (
    HYBRID_KEYFRAME_SENTINEL,
    HYBRID_LAYOUT_LEGACY_SENTINEL,
    HYBRID_LAYOUT_NATIVE_CONCAT,
    assert_hybrid_layout_contract,
    build_conditioning,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def base_args():
    return {
        "clip": FakeClip(),
        "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(),
        "prompt": "Audio 1 drives the performance",
        "width": 128,
        "height": 128,
        "length": 124,
        "task_type": "auto",
        "audio_mode": "lock_source",
        "drive_audio": make_audio(),
    }


def test_source_audio_is_encoded_once_reused_and_locked():
    args = base_args()
    conditioning, latent, output_audio, prompt, media_map, report = build_conditioning(**args)
    assert len(args["audio_vae"].encode_calls) == 1
    video, audio = latent["samples"].unbind()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    assert torch.all(audio[..., :200] == 0.25)
    assert torch.all(audio[..., 200:] == 0)
    assert torch.all(video_mask == 1)
    assert torch.all(audio_mask == 0)
    assert prompt == "<Audio 1> drives the performance"
    assert json.loads(media_map)["source_audio_ordinal"] == 1
    assert output_audio is args["drive_audio"]


def test_video_soundtracks_shift_primary_source_audio_tag():
    args = base_args()
    args.update({
        "ref_videos": {"ref_video_1": torch.zeros((48, 64, 64, 3)),
                       "ref_video_2": torch.zeros((48, 64, 64, 3))},
        "ref_video_audios": {"ref_video_audio_1": make_audio(2),
                             "ref_video_audio_2": make_audio(2)},
        "ref_audios": {"ref_audio_1": make_audio(1)},
    })
    conditioning, _, _, prompt, media_map, _ = build_conditioning(**args)
    media = json.loads(media_map)
    assert prompt.startswith("<Audio 3>")
    assert media["source_audio_ordinal"] == 3
    assert media["audios"] == {
        "1": "ref_video_audio_1", "2": "ref_video_audio_2",
        "3": "drive_audio (primary source)", "4": "ref_audio_1",
    }
    refs = conditioning[0][1]["minimax_refs"]
    assert [item["kind"] for item in refs] == ["video_audio", "video_audio", "audio", "audio"]


def test_soundtrack_pairing_uses_autogrow_ordinal_not_dense_position():
    args = base_args()
    args.update({
        "ref_videos": {"ref_video_1": torch.zeros((48, 64, 64, 3)),
                       "ref_video_2": torch.zeros((48, 64, 64, 3))},
        "ref_video_audios": {"ref_video_audio_2": make_audio(2)},
    })
    conditioning, *_ = build_conditioning(**args)
    refs = conditioning[0][1]["minimax_refs"]
    assert refs[0]["kind"] == "video"
    assert refs[1]["kind"] == "video_audio"


def test_exact_keyframe_and_audio_reference_use_guarded_hybrid_payload():
    hybrid_route = assert_hybrid_layout_contract()
    args = base_args()
    args["first_frame"] = torch.zeros((1, 128, 128, 3))
    conditioning, _, _, _, media_map, report = build_conditioning(**args)
    metadata = conditioning[0][1]
    assert metadata["minimax_keyframes"][0]["resolved_frame_index"] == 0
    if hybrid_route == HYBRID_LAYOUT_LEGACY_SENTINEL:
        assert metadata["minimax_refs"][0]["kind"] == HYBRID_KEYFRAME_SENTINEL
    else:
        assert [item["kind"] for item in metadata["minimax_refs"]] == ["audio"]
    tokenize_kwargs = args["clip"].tokenize_calls[0][1]
    assert [item["type"] for item in tokenize_kwargs["minimax_ref_items"]] == ["image", "audio"]
    assert json.loads(media_map)["pictures"]["1"].startswith("first_frame")


def test_hybrid_contract_accepts_semantically_compatible_external_wrapper(monkeypatch):
    original = conditioning_module.MiniMaxH3BaseModel.extra_conds

    def compatible_wrapper(self, **kwargs):
        out = original(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes")
        refs = kwargs.get("minimax_refs")
        if keyframes and refs:
            payload = out["minimax_payload"].cond
            payload["cond_video_latents"] = [
                *[kf["latent"] for kf in keyframes],
                *[ref["latent"] for ref in refs if "latent" in ref],
            ]
        return out

    compatible_wrapper._minimax_kfref_patched = True
    monkeypatch.setattr(
        conditioning_module.MiniMaxH3BaseModel,
        "extra_conds",
        compatible_wrapper,
    )

    assert assert_hybrid_layout_contract() == HYBRID_LAYOUT_NATIVE_CONCAT


def test_hybrid_contract_preserves_legacy_overwrite_route(monkeypatch):
    original = conditioning_module.MiniMaxH3BaseModel.extra_conds

    def legacy_wrapper(self, **kwargs):
        out = original(self, **kwargs)
        refs = kwargs.get("minimax_refs")
        if refs is not None:
            payload = out["minimax_payload"].cond
            payload["cond_video_latents"] = [
                ref["latent"] for ref in refs if "latent" in ref
            ]
        return out

    monkeypatch.setattr(
        conditioning_module.MiniMaxH3BaseModel,
        "extra_conds",
        legacy_wrapper,
    )

    assert assert_hybrid_layout_contract() == HYBRID_LAYOUT_LEGACY_SENTINEL


def test_hybrid_contract_rejects_incompatible_external_layout_patch(monkeypatch):
    original = conditioning_module.PackedLayout.__init__

    def incompatible_wrapper(self, *args, **kwargs):
        return original(self, *args, frame_count=None, **kwargs)

    monkeypatch.setattr(
        conditioning_module.PackedLayout,
        "__init__",
        incompatible_wrapper,
    )

    with pytest.raises(RuntimeError, match="external custom-node layout patch"):
        assert_hybrid_layout_contract()


def test_hybrid_contract_bypasses_verified_obsolete_painter_layout_patch(monkeypatch):
    original = conditioning_module.PackedLayout.__init__

    def obsolete_painter_wrapper(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=None,
        refs=None,
        frame_count=None,
    ):
        return original(
            self,
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=keyframes,
            refs=refs,
            frame_count=frame_count,
        )

    obsolete_painter_wrapper._minimax_kfref_layout_patched = True
    monkeypatch.setattr(
        conditioning_module.PackedLayout,
        "__init__",
        obsolete_painter_wrapper,
    )

    assert assert_hybrid_layout_contract() == HYBRID_LAYOUT_NATIVE_CONCAT
    assert conditioning_module.PackedLayout.__init__ is original


def test_reference_only_keeps_blank_target_audio():
    args = base_args()
    args["audio_mode"] = "reference_only"
    _, latent, *_ = build_conditioning(**args)
    _, target_audio = latent["samples"].unbind()
    assert torch.all(target_audio == 0)
    assert "noise_mask" not in latent


def test_native_t2va_needs_no_source_audio():
    args = base_args()
    args.update({"audio_mode": "native", "drive_audio": None, "add_source_as_reference": False,
                 "prompt": "A person speaks naturally"})
    conditioning, latent, output_audio, prompt, media_map, report = build_conditioning(**args)
    assert output_audio is None
    assert "minimax_refs" not in conditioning[0][1]
    assert "task=t2va" in report


def test_plain_disconnected_media_number_is_preserved_as_text_and_reported():
    args = base_args()
    args.update({
        "audio_mode": "native",
        "drive_audio": None,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "prompt": "Video 2 opens with a quiet landscape and Audio 3 fades in.",
    })
    _conditioning, _latent, _output_audio, prompt, _media_map, report = build_conditioning(
        **args
    )
    assert prompt == "Video 2 opens with a quiet landscape and Audio 3 fades in."
    assert "warning: <Video 2> is not connected" in report
    assert "warning: <Audio 3> is not connected" in report


def test_explicit_disconnected_media_tag_still_fails_closed():
    args = base_args()
    args.update({
        "audio_mode": "native",
        "drive_audio": None,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "prompt": "Use <Picture 1> as the identity reference.",
    })
    with pytest.raises(ValueError, match="Connect the referenced media"):
        build_conditioning(**args)


def test_orphan_video_soundtrack_is_rejected():
    args = base_args()
    args["ref_video_audios"] = {"ref_video_audio_2": make_audio(2)}
    with pytest.raises(ValueError, match="no same-numbered video"):
        build_conditioning(**args)


def test_canvas_above_reference_area_is_warning_only_without_opt_in():
    args = base_args()
    args.update({
        "width": 1920,
        "height": 1088,
        "length": 22,
        "audio_mode": "native",
        "prompt": "A quiet cinematic landscape",
        "drive_audio": None,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
    })
    _, latent, *_ = build_conditioning(**args)
    video, _audio = latent["samples"].unbind()
    assert video.shape[-2:] == (68, 120)

    # Exact canvas from the user report: 2,396,160 pixels, formerly rejected.
    args.update({"width": 2080, "height": 1152})
    result = build_conditioning(**args, return_details=True)
    _conditioning, latent, *_rest, report, details = result
    video, _audio = latent["samples"].unbind()
    assert video.shape[-2:] == (72, 130)
    assert "execution remains allowed" in report
    assert details["exceeds_reference_area"] is True
    assert details["allow_above_reference_area"] is False
    assert details["reference_area_policy"] == "warning_only_no_area_gate"

    args["allow_above_reference_area"] = True
    _conditioning, latent_opt_in, *_rest = build_conditioning(**args)
    video_opt_in, _audio = latent_opt_in["samples"].unbind()
    assert video_opt_in.shape == video.shape
