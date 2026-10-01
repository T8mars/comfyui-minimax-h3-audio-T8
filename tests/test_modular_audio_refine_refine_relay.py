"""The opt-in refinement Relay is sampled, not just drawn in a workflow."""
import json

import comfy.model_management
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
import pytest
import torch

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling.audio_refine_stage import (
    audit_audio_refine_stage, audit_audio_refine_tail_delivery,
    bind_audio_refine_stage,
)
from test_audio_refine_advanced import _audit, _audio, _runtime
from test_fast_h3_v2_core_sampler import model
from test_progressive_relay import paired
from test_progressive_sampling_runtime import latent
from test_prompt_relay_long_video_advanced import (
    NativeLikeFakeClip, _allow_fixture_core_contract, _global_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from h3_audio_t8_pkg.long_video import LONG_VIDEO_SCHEMA
from h3_audio_t8_pkg.long_video import patch_long_video_model
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    build_prompt_relay_long_video_conditioning,
    project_prompt_relay_plan_to_long_video_window,
)


def test_separate_refine_relay_runs_four_real_tiny_core_forwards(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    bound, positive, _ = paired(model())
    original = latent()
    audit, _, _ = _audit(
        model=bound, positive=positive, av_latent=original,
        conditioning_report="task=T2VA\naudio_mode=native\nframes=5")
    routed_model, route, decision, _ = refine.route_audio_refine_compatibility(
        audit=audit, refine_model=bound, positive=positive,
        generation_profile="prompt_relay_turbo8", declared_first_pass_nfe=8)
    assert decision == "ALLOW"
    assert routed_model is bound
    plan, decision, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 2608290017)
    assert decision == "ALLOW"
    setup = refine.setup_audio_refine_compatibility(
        plan=plan, refine_model=routed_model, positive=positive,
        av_latent=original, runtime_snapshot_fn=_runtime)
    bound_stage = bind_audio_refine_stage(
        plan, original, setup.model, setup.noise, setup.guider,
        setup.sampler, setup.sigmas, setup.latent, setup.report_json,
        expected_variant="compatibility")
    calls = []
    previous = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
        return previous(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    candidate = SamplerCustomAdvanced.execute(
        bound_stage[1], bound_stage[2], bound_stage[3], bound_stage[4],
        bound_stage[5]).result[0]
    checked, report = audit_audio_refine_stage(
        bound_stage[6], plan, original, setup.latent, setup.report_json, candidate)
    assert checked is candidate
    assert json.loads(report)["status"] == "candidate_received"
    assert len(calls) == 4
    assert set(calls) == {route["runtime_contract"]["prompt_relay_binding_hash"]}
    delivered, delivery_report = audit_audio_refine_tail_delivery(
        bound_stage[6], report, setup.latent, candidate, 1.0, False, 0.0)
    assert json.loads(delivery_report)["node"] == "MiniMaxH3TwoPassAudioAuditT8Advanced"
    source_video, source_audio = original["samples"].unbind()
    candidate_video, candidate_audio = candidate["samples"].unbind()
    delivered_video, delivered_audio = delivered["samples"].unbind()
    assert not torch.equal(candidate_video, source_video)
    assert torch.equal(delivered_video, candidate_video)
    assert torch.equal(delivered_audio, candidate_audio)
    assert not torch.equal(candidate_audio, source_audio)
    decoded_original, decoded_candidate = _audio(), _audio()
    kept, kept_audio, accepted, decision, _ = refine.gate_audio_refine_candidate(
        original_av_latent=original, candidate_av_latent=delivered,
        original_audio=decoded_original, candidate_audio=decoded_candidate,
        accept_candidate=False)
    assert kept is original and kept_audio is decoded_original
    assert accepted is False and decision == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    selected, selected_audio, accepted, decision, gate_report = (
        refine.gate_audio_refine_candidate(
            original_av_latent=original, candidate_av_latent=delivered,
            original_audio=decoded_original, candidate_audio=decoded_candidate,
            accept_candidate=True))
    selected_video, selected_audio_latent = selected["samples"].unbind()
    assert torch.equal(selected_video, source_video)
    assert torch.equal(selected_audio_latent, candidate_audio)
    assert selected_audio is decoded_candidate
    assert accepted is True and decision == "ACCEPT_CANDIDATE"
    assert json.loads(gate_report)["selected_video_relocked_exact"] is True
    assert not torch.cuda.is_initialized()


def test_refine_relay_rejects_cross_plan_model_conditioning_pair():
    bound, positive, plain = paired(model())
    audit, _, _ = _audit(
        model=bound, positive=positive, av_latent=latent(),
        conditioning_report="task=T2VA\naudio_mode=native\nframes=5")
    with pytest.raises(RuntimeError, match="authenticated binding|same plan"):
        refine.route_audio_refine_compatibility(
            audit=audit, refine_model=bound, positive=plain,
            generation_profile="prompt_relay_turbo8", declared_first_pass_nfe=8)


def test_long_video_refine_relay_authenticates_independent_projected_window(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    original_plan = _global_plan(345)
    refine_plan = build_prompt_relay_plan(
        global_prompt="Only the refinement stage uses this different scene instruction.",
        local_prompts="人物抬手并响起钟声\n人物奔跑，只有脚步声\n镜头拉远并响起雷声",
        length=345, timing_mode="auto_equal", time_ranges="",
        math_profile="paper_v1", epsilon=.1,
        allow_gaps=False, allow_overlaps=False)[0]
    assert refine_plan["plan_hash"] != original_plan["plan_hash"]
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        refine_plan, segment_index=1, length=124, context_frames=22,
        timeline_start_seconds=124 / 24, timeline_end_seconds=226 / 24)
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": torch.zeros((1, 24, 12, 8, 8)),
        "audio_tail": torch.zeros((1, 32, 2, 65)),
        "metadata": {"source_segment_index": 0, "target_segment_index": 1,
                     "max_context_frames": 39, "audio_overhang": 1 / 3},
    }
    source_model = model()
    base = source_model.model
    loaded_first_pass = patch_long_video_model(source_model)
    base.extra_conds = loaded_first_pass.get_model_object("extra_conds")
    try:
        patched, positive, stage_latent, _, prompt, media_map, report = (
            build_prompt_relay_long_video_conditioning(
                model=source_model, clip=NativeLikeFakeClip(),
                video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                context=context, prompt_relay_plan=projected,
                segment_index=1, context_frames=22, context_audio="video_and_audio",
                width=128, height=128, length=124, task_type="auto",
                audio_mode="native", audio_denoise_strength=.35,
                add_source_as_reference=False, prompt_primary_audio_ordinal=0,
                strict_prompt_tags=True, ref_image_size="match",
                reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
                query_chunk_rows=64))
        audit, _, _ = _audit(model=patched, positive=positive, av_latent=stage_latent,
                             conditioned_prompt=prompt, media_map_json=media_map,
                             conditioning_report=report)
        routed_model, route, decision, _ = refine.route_audio_refine_compatibility(
            audit=audit, refine_model=patched, positive=positive,
            generation_profile="long_video_prompt_relay_turbo8",
            declared_first_pass_nfe=8)
        assert routed_model is patched and decision == "ALLOW"
        assert route["runtime_contract"]["segment_index"] == 1
        assert route["runtime_contract"]["projected_plan_hash"] == projected["plan_hash"]
        with pytest.raises(RuntimeError, match="same plan|binding hashes differ"):
            refine.route_audio_refine_compatibility(
                audit=audit, refine_model=patched, positive=paired(model())[1],
                generation_profile="long_video_prompt_relay_turbo8",
                declared_first_pass_nfe=8)
    finally:
        del base.extra_conds
