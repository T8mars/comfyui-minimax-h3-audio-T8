"""Separated P7 delivery must form a real, auditable two-segment chain."""
import json
from pathlib import Path

from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import pytest

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_delivery as candidate
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
from test_modular_hyperflow import inputs as hyperflow_inputs
from test_modular_hyperflow_p7_storage import fake_lift_for


def completed(contexts, length, monkeypatch, upscaler):
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    low_phase = p7.prepare_phase(contexts, "low", clip=clip, video_vae=video_vae,
                                 audio_vae=audio_vae, prompt="LOW moving scene", length=length)
    high_phase = p7.prepare_phase(contexts, "high", clip=clip, video_vae=video_vae,
                                  audio_vae=audio_vae, prompt="HIGH moving scene", length=length)
    low_model, high_model, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    model, sampler, sigmas, source, stage, positive, _, _ = p7.setup_low(
        low_phase, low_model, low_phase.result[0], low_phase.result[0])
    first = sample_stage(RandomNoise.execute(91).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, stage)[2]
    low = p7.bind_low(low_phase, first)
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise", lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    lift = p7.lift_low(low, upscaler.name)
    handoff = p7.handoff_high(lift, high_phase, high_phase.result[0], high_phase.result[0])
    model, sampler, sigmas, source, stage, positive, _, _ = p7.setup_high(handoff, high_model)
    second = sample_stage(RandomNoise.execute(92).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, stage)[2]
    return p7.bind_high(handoff, second), video_vae, audio_vae


@pytest.fixture
def case(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(output))
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 delivery tiny routing model")
    root = delivery.long_video_chain_root("p7_delivery")
    contexts = p7.capture_initial(root, chain_id="p7_delivery", context_frames=5,
        width=128, height=64, low_width=64, low_height=32)
    high, video_vae, audio_vae = completed(contexts, 39, monkeypatch, upscaler)
    return root, high, video_vae, audio_vae, upscaler


def test_segment_zero_candidate_accept_parent_and_final_continuation(case, monkeypatch):
    root, first, video_vae, audio_vae, upscaler = case
    path, movie, job, report = candidate.save_candidate(first, video_vae, audio_vae,
        candidate_id="seg0", model_id="tiny-chain", color_match=False)
    assert Path(movie).is_file() and json.loads(report)["accepted"] is False
    descriptor, audit, _ = candidate.verify_candidate(path, job)
    assert descriptor["frame_count"] == 39 and descriptor["timeline_end_frame"] == 39
    assert audit["sampling_plan"]["dual_model"]["low_context"]["audio_source"] == "completed_second_pass_output"
    assert candidate.accept_candidate(path, job, accept=False)[1] is False
    accepted_video, accepted, manifest_path, parent_id, revision, _ = candidate.accept_candidate(
        path, job, accept=True)
    assert accepted and Path(accepted_video).is_file() and Path(manifest_path).is_file()
    assert candidate.accept_candidate(path, job, accept=True)[4] == revision
    parent = p7.capture_parent(root, chain_id="p7_delivery", segment_index=1,
        parent_candidate_id=parent_id, parent_revision=revision, previous_job_sha256=job,
        context_frames=5, width=128, height=64, low_width=64, low_height=32)
    next_contexts = parent.prepare_contexts(FakeVideoVAE())
    second, video_vae, audio_vae = completed(next_contexts, 22, monkeypatch, upscaler)
    path2, movie2, job2, _ = candidate.save_candidate(second, video_vae, audio_vae,
        candidate_id="seg1", is_final_segment=True, color_match=False)
    descriptor2, audit2, _ = candidate.verify_candidate(path2, job2)
    assert descriptor2["frame_count"] == 17 and descriptor2["timeline_start_frame"] == 39
    assert descriptor2["timeline_end_frame"] == 56 and Path(movie2).is_file()
    assert audit2["sampling_plan"]["dual_model"]["low_context"] is None
    assert candidate.accept_candidate(path2, job2, accept=True)[1]
    manifest, _ = delivery.load_delivery_manifest("p7_delivery")
    assert [entry["frame_count"] for entry in manifest["segments"]] == [39, 17]
    assert manifest["segments"][1]["is_final_segment"] is True
    assembled, compose_json = delivery.compose_accepted_long_video(
        "p7_delivery", "p7_modular_tiny", True, "cosine_bridge", 5.0, 28)
    compose = json.loads(compose_json)
    assert Path(assembled).is_file() and compose["strict_decode_validated"] is True
    assert compose["frame_count"] == 56 and compose["segment_count"] == 2


def test_incomplete_or_changed_sidecar_cannot_be_accepted(case):
    _, high, video_vae, audio_vae, _ = case
    path, _, job, _ = candidate.save_candidate(high, video_vae, audio_vae,
        candidate_id="seg0", color_match=False)
    with pytest.raises(ValueError, match="does not match this job"):
        candidate.accept_candidate(path, "f" * 64, accept=True)
    audit_path = Path(path).parent / "effects_audit.json"
    audit_path.unlink()
    with pytest.raises(ValueError, match="audit sidecar"):
        candidate.accept_candidate(path, job, accept=True)
    audit_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="audit SHA-256"):
        candidate.accept_candidate(path, job, accept=True)


def test_duplicate_candidate_is_never_overwritten(case):
    _, high, video_vae, audio_vae, _ = case
    path, movie, _, _ = candidate.save_candidate(high, video_vae, audio_vae,
        candidate_id="seg0", color_match=False)
    before = delivery._sha256_file(Path(movie))
    with pytest.raises(FileExistsError, match="already exists"):
        candidate.save_candidate(high, video_vae, audio_vae,
            candidate_id="seg0", color_match=False)
    assert delivery._sha256_file(Path(movie)) == before
    assert Path(path).is_file()


def test_missing_low_context_cannot_be_accepted(case):
    root, high, video_vae, audio_vae, _ = case
    path, _, job, _ = candidate.save_candidate(high, video_vae, audio_vae,
        candidate_id="seg0", color_match=False)
    (Path(path).parent / "low.context.safetensors").unlink()
    with pytest.raises(ValueError, match="LOW context"):
        candidate.accept_candidate(path, job, accept=True)
    assert not (root / delivery.MANIFEST_NAME).exists()


def test_sidecar_write_failure_leaves_only_an_unacceptable_unique_candidate(case, monkeypatch):
    root, high, video_vae, audio_vae, _ = case
    def interrupted(*_args, **_kwargs):
        raise OSError("injected P7 audit write interruption")
    monkeypatch.setattr(candidate, "_write_effects_audit", interrupted)
    with pytest.raises(OSError, match="injected P7 audit"):
        candidate.save_candidate(high, video_vae, audio_vae,
            candidate_id="interrupted", color_match=False)
    path = root / "candidates" / "segment_00000" / "interrupted" / "candidate.json"
    assert path.is_file() and not (path.parent / "effects_audit.json").exists()
    with pytest.raises(ValueError, match="audit sidecar"):
        candidate.accept_candidate(path, high.verify()["sha256"], accept=True)
    with pytest.raises(FileExistsError, match="already exists"):
        candidate.save_candidate(high, video_vae, audio_vae,
            candidate_id="interrupted", color_match=False)
    assert not (root / delivery.MANIFEST_NAME).exists()


def test_exact_124_plus_68_frame_grid_uses_original_22_frame_trim(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery.folder_paths, "get_output_directory",
                        lambda: str(tmp_path / "output"))
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 exact-frame-grid tiny routing model")
    root = delivery.long_video_chain_root("p7_exact_8s")
    initial = p7.capture_initial(root, chain_id="p7_exact_8s", context_frames=22,
        width=128, height=64, low_width=64, low_height=32)
    first, video_vae, audio_vae = completed(initial, 124, monkeypatch, upscaler)
    path, _, first_job, _ = candidate.save_candidate(first, video_vae, audio_vae,
        candidate_id="first", color_match=False)
    _, _, _, parent_id, revision, _ = candidate.accept_candidate(path, first_job, accept=True)
    parent = p7.capture_parent(root, chain_id="p7_exact_8s", segment_index=1,
        parent_candidate_id=parent_id, parent_revision=revision, previous_job_sha256=first_job,
        context_frames=22, width=128, height=64, low_width=64, low_height=32)
    contexts = parent.prepare_contexts(FakeVideoVAE())
    second, video_vae, audio_vae = completed(contexts, 124, monkeypatch, upscaler)
    path2, _, second_job, _ = candidate.save_candidate(second, video_vae, audio_vae,
        candidate_id="second", is_final_segment=True, final_frame_count=68,
        color_match=False)
    descriptor, audit, _ = candidate.verify_candidate(path2, second_job)
    assert descriptor["timeline_start_frame"] == 124
    assert descriptor["frame_count"] == 68 and descriptor["timeline_end_frame"] == 192
    assert audit["trim"]["start_frame"] == 22 and audit["trim"]["frame_count"] == 68
    assert candidate.accept_candidate(path2, second_job, accept=True)[1]
    movie, report_json = delivery.compose_accepted_long_video(
        "p7_exact_8s", "p7_exact_tiny", True, "cosine_bridge", 5.0, 28)
    assert Path(movie).is_file() and json.loads(report_json)["frame_count"] == 192
