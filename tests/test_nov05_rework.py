"""Incremental repair consumers; separate latent, raw RGB and final composite evidence."""
import json

import pytest
import torch
import comfy.nested_tensor

from h3_audio_t8_pkg import lanpaint_av_advanced as repair
from h3_audio_t8_pkg.nodes_long_video_seam_drift_advanced import MiniMaxH3LongVideoSeamDriftT8Advanced as Drift
from test_long_video_seam_drift_advanced import _textured
from h3_audio_t8_pkg.audio_ops import decode_av_latent
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model, sample
from test_modular_rf_restart import source_latent
from test_h21_core_vae_contracts import real_tiny_vae_owners  # noqa: F401


def test_bounded_color_candidate_stops_before_later_hard_cut():
    frames = torch.cat([_textured(.4, 4), _textured(.43, 4), _textured(.8, 8)])
    output, status, raw = Drift.execute(frames=frames, boundary_frames_json="[4]",
        mode="bounded_candidate_exp", transition_frames=12).result
    assert status == "candidate_applied"
    assert not torch.equal(output[4:8], frames[4:8])
    assert torch.equal(output[8:], frames[8:])  # Later shot is not part of the seam.
    row = json.loads(raw)["boundaries"][0]
    assert row["transition_stop_frame"] == 8


@pytest.mark.parametrize("broken", ["mask", "candidate"])
def test_real_composite_rejects_nonfinite_pixels_even_outside_active_region(broken):
    source = torch.zeros(5, 32, 32, 3)
    candidate = torch.ones_like(source)
    mask = torch.zeros(5, 32, 32)
    mask[:, 10:20, 10:20] = 1
    sound = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
    if broken == "mask":
        mask[2, 0, 0] = float("nan")
    else:
        candidate[2, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        repair.composite_lanpaint_av(source, candidate, sound, sound, "[]", 1, 0., mask)


def test_latent_mask_transport_rejects_nonfinite_before_core_sampling():
    mask = torch.zeros(5, 32, 64)
    mask[1, 2, 3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        repair._video_mask_to_latent(mask, torch.zeros(1, 24, 2, 2, 4), 5)


@pytest.mark.parametrize("kind", ["black", "white", "quarter", "thin", "occlusion"])
def test_real_core_mask_sampling_raw_decode_and_final_composite_are_separate(
    real_tiny_vae_owners, kind, tmp_path,  # noqa: F811
):
    video_vae, audio_vae = real_tiny_vae_owners
    source = source_latent()
    video, audio = source["samples"].unbind()
    with torch.no_grad():
        original_frames, original_audio, *_ = decode_av_latent(source, video_vae, audio_vae)
        mask = torch.zeros(original_frames.shape[:3])
        if kind == "white":
            mask.fill_(1)
        elif kind == "quarter":
            mask.fill_(.25)
        elif kind == "thin":
            mask[:, 20:40, 30:31] = 1
        elif kind == "occlusion":
            mask[:, 20:40, 25:45] = 1
            mask[2].zero_()
        video_mask = repair._video_mask_to_latent(mask, video, len(mask))
        source["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(audio)))
        before = [value.clone() for value in source["samples"].unbind()]
        bare = model()
        prepared, sampler, sigmas = setup_dual_clock_sampling(bare, source, 2, 12., 3.)
        sampled = sample(prepared, sampler, sigmas, source)[0]
        actual_video, actual_audio = sampled["samples"].unbind()
        protected = video_mask == 0
        torch.testing.assert_close(actual_video[protected], video[protected], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_audio, audio, rtol=1e-5, atol=1e-6)
        raw_candidate, candidate_audio, *_ = decode_av_latent(sampled, video_vae, audio_vae)
        final, final_audio, _ = repair.composite_lanpaint_av(original_frames, raw_candidate,
            original_audio, candidate_audio, "[]", 1, 0., mask)
    outside = (mask == 0)[..., None].expand_as(original_frames)
    assert torch.equal(final[outside], original_frames[outside])
    assert torch.equal(final_audio["waveform"], original_audio["waveform"])
    if kind == "quarter":
        assert bool((video_mask == .25).all())
        torch.testing.assert_close(final, original_frames*.75 + raw_candidate*.25, rtol=0, atol=0)
    if kind == "occlusion":
        assert torch.equal(final[2], original_frames[2])
    for old, current in zip(before, source["samples"].unbind()):
        assert torch.equal(old, current)
    evidence = {"mask": kind, "actual_consumer": "Core Euler sampler and tiny random Core VAEs",
        "trained_model": False, "human_quality_reviewed": False,
        "latent_protected_max_abs_error": float((actual_video-video)[protected].abs().max()) if bool(protected.any()) else None,
        "raw_candidate_outside_mask_max_abs_error": float((raw_candidate-original_frames)[outside].abs().max()) if bool(outside.any()) else None,
        "final_outside_mask_bit_exact": True, "source_pcm_bit_exact": True,
        "mask_transport": repair.video_mask_transport_report(mask, video, len(mask), video_mask)}
    (tmp_path / "mask-layer-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    assert not torch.cuda.is_initialized()


def test_report_only_keeps_intended_color_changes_and_explicit_boundaries_stop_candidates():
    frames = torch.cat([_textured(.4, 4), _textured(.43, 4), _textured(.46, 8)])
    original, status, _ = Drift.execute(frames=frames, boundary_frames_json="[4,8]").result
    assert original is frames and status == "source_identity_report_only"
    _, _, raw = Drift.execute(frames=frames, boundary_frames_json="[4,8]",
        mode="bounded_candidate_exp", transition_frames=12).result
    first = json.loads(raw)["boundaries"][0]
    assert first["transition_stop_frame"] == 8
    assert first["transition_stop_reason"] == "declared_next_boundary"


def test_two_subject_crossing_occlusion_hard_cut_and_actual_core_composite(tmp_path):
    from comfy_extras.nodes_mask import ImageCompositeMasked
    from h3_audio_t8_pkg import face_refine_window_advanced as window
    frames = torch.full((56, 32, 64, 3), .1)
    frames[28:] = .7
    target = torch.zeros(frames.shape[:3])
    protected = torch.zeros_like(target)
    for index in range(56):
        x = 5 + index % 28
        other_x = 36 - index % 28
        target[index, 8:22, x:x+8] = 1
        protected[index, 8:22, other_x:other_x+8] = 1
        frames[index, 8:22, x:x+8] = frames.new_tensor([.8, .1, .2])
        frames[index, 8:22, other_x:other_x+8] = frames.new_tensor([.1, .7, .3])
    target *= 1-protected  # Target behind protected subject, explicitly authored.
    args = (frames, 24., "6-21,34-49", "frames_inclusive", 2, 2, 22, 56, .18, "reject", "reject", True)
    planned = window.build_face_refine_window_plan(*args)[0]
    assert len(planned["shots"]) == 2
    with pytest.raises(ValueError, match="hard cut"):
        window.build_face_refine_window_plan(frames, 24., "24-32", "frames_inclusive",
            2, 2, 22, 56, .18, "reject", "reject", True)
    for index in (0, 1):
        rendered, _, mapping, *_ = window.extract_face_refine_window(frames, planned, index, "reject")
        mask = torch.stack([target[row["source_frame"]] for row in mapping["frame_map"]])
        raw = torch.full_like(rendered, .6)  # Deliberately modifies the other subject too.
        refused = window.apply_face_refine_manual_review(frames, raw, mask, mapping,
            "accept_selected", "", True, 0)
        assert json.loads(refused[-1])["status"] == "rejected_contract"
        assert torch.equal(refused[1], frames)
        composited = ImageCompositeMasked.execute(rendered, raw, 0, 0, False, mask).result[0]
        accepted = window.apply_face_refine_manual_review(frames, composited, mask, mapping,
            "accept_selected", "", True, 0)
        assert json.loads(accepted[-1])["status"] == "accept_selected"
        protected_pixels = protected[..., None].bool().expand_as(frames)
        assert torch.equal(accepted[1][protected_pixels], frames[protected_pixels])
        other_shot = slice(28, None) if index == 0 else slice(None, 28)
        assert torch.equal(accepted[1][other_shot], frames[other_shot])
