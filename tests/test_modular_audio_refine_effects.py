"""Real tiny native CPU tails: separate EAV/Relay without changing old controls.

Random weights prove execution and ownership, never audio/video quality.
"""
from copy import deepcopy
import json

import comfy.model_management
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
import pytest
import torch

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import audio_refine_effects as effects
from h3_audio_t8_pkg.modular_sampling.audio_refine_stage import bind_audio_refine_stage
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from test_audio_refine_advanced import _audit, _audio, _runtime, _turbo4_metadata
from test_fast_h3_v2_core_sampler import model
from test_progressive_relay import paired
from test_progressive_sampling_runtime import conditioning, latent


def prepared(case="dual_clock", *, abstain=False):
    bare, positive, source = model(), conditioning(), latent()
    first = bare
    if case in ("base_without_turbo", "same_turbo_stack"):
        first = bare.clone()
        key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                           if key.startswith("diffusion_model.") and value.ndim == 2)
        first.add_patches({key: ("lora", (torch.full((weight.shape[0], 1), .02),
                                        torch.full((1, weight.shape[1]), .02), None, None, None))})
        first.set_attachments("lora_metadata", _turbo4_metadata())
        if case == "same_turbo_stack":
            bare = first.clone()
    if case == "prompt_relay_turbo8":
        bare, positive, _ = paired(bare)
        first = bare
    audit, decision, _ = _audit(model=first, positive=positive, av_latent=source,
        conditioning_report="task=T2VA\naudio_mode=" + ("lock_source" if abstain else "native") + "\nframes=5")
    if case != "prompt_relay_turbo8":
        assert decision == ("ABSTAIN" if abstain else "ALLOW")
    if case == "dual_clock":
        plan, _, _ = refine.plan_audio_refine(audit, 4, .5, 26092801)
        setup = refine.setup_audio_refine(plan=plan, model=bare, positive=positive,
            av_latent=source, runtime_snapshot_fn=_runtime)
    elif case in ("base_without_turbo", "same_turbo_stack"):
        _, route, decision, _ = refine.route_audio_refine_model(audit=audit, first_pass_model=first,
            refine_model=bare, route_strategy=case, declared_first_pass_nfe=4)
        assert decision == ("ABSTAIN" if abstain else "ALLOW")
        plan, _, _ = refine.plan_audio_refine_phase2(route, 4, .5, 26092801)
        setup = refine.setup_audio_refine_dual_model(plan=plan, refine_model=bare, positive=positive,
            av_latent=source, runtime_snapshot_fn=_runtime)
    else:
        _, route, decision, _ = refine.route_audio_refine_compatibility(audit=audit, refine_model=bare,
            positive=positive, generation_profile=case, declared_first_pass_nfe=8)
        assert decision == ("ABSTAIN" if abstain else "ALLOW")
        plan, _, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 26092801)
        setup = refine.setup_audio_refine_compatibility(plan=plan, refine_model=bare, positive=positive,
            av_latent=source, runtime_snapshot_fn=_runtime)
    boundary = bind_audio_refine_stage(plan, source, setup.model, setup.noise, setup.guider,
        setup.sampler, setup.sigmas, setup.latent, setup.report_json)[6]
    selected, context, report = effects.bind_tail_effects(boundary, setup.model, setup.noise,
        setup.guider, setup.sampler, setup.sigmas, setup.latent)
    return source, setup, boundary, selected, context, report


CASES = ["dual_clock", "base_without_turbo", "same_turbo_stack", "turbo8",
         "learned_latent_twopass_final8", "pdd8", "pdd4_plus4", "eav_turbo8", "prompt_relay_turbo8"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode,external_relay", [("disabled", False), ("report_only", False),
    ("apply_exp", False), ("disabled", True), ("apply_exp", True)])
def test_real_tail_effect_combinations_preserve_sampling_contract(monkeypatch, case, mode, external_relay):
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    source, setup, _, selected, context, _ = prepared(case)
    if external_relay and case == "prompt_relay_turbo8":
        # The original recipe is already paired. Keep that actual source plan;
        # other recipes bind their own separately encoded Relay after Setup.
        positive = None
    elif external_relay:
        selected, positive, _ = paired(selected)
    else:
        positive = None
    calls = []
    previous = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
        return previous(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    selected, runtime, _ = apply_stage_eav(selected, setup.sigmas, setup.latent, context,
        EAVConfig(mode, tau=.25, start_video_progress=0., end_video_progress=1.))
    guider, report = effects.tail_effects_guider(selected, context, setup.sigmas, setup.latent, positive)
    assert json.loads(report)["cfg"] == 1.
    candidate = SamplerCustomAdvanced.execute(setup.noise, guider, setup.sampler,
        setup.sigmas, setup.latent).result[0]
    _, report = audit_stage_eav(candidate, runtime)
    report = json.loads(report)
    assert report["status"] == {"disabled": "disabled_identity", "report_only": "observed_report_only",
                                "apply_exp": "observed_apply_exp"}[mode]
    assert report["completed_forwards"] == (0 if mode == "disabled" else 4)
    assert len(calls) == (4 if external_relay or case == "prompt_relay_turbo8" else 0)
    assert setup.guider.model_patcher is setup.model
    assert effects.KEY not in setup.model.attachments
    assert len(setup.sigmas) == 5 and setup.sampler.extra_options == {}
    video_mask, audio_mask = setup.latent["noise_mask"].unbind()
    assert video_mask.count_nonzero() == 0 and bool((audio_mask == 1).all())
    kept, _, accepted, decision, _ = refine.gate_audio_refine_candidate(
        original_av_latent=source, candidate_av_latent=candidate,
        original_audio=_audio(), candidate_audio=_audio(), accept_candidate=False)
    assert kept is source and not accepted and decision == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("case", ["dual_clock", "base_without_turbo", "turbo8"])
def test_effect_abstain_is_real_zero_forward_identity(monkeypatch, case):
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    source, setup, _, selected, context, report = prepared(case, abstain=True)
    assert json.loads(report)["status"] == "abstain_no_sample"
    selected, runtime, _ = apply_stage_eav(selected, setup.sigmas, setup.latent, context, EAVConfig("apply_exp"))
    guider, _ = effects.tail_effects_guider(selected, context, setup.sigmas, setup.latent)
    candidate = SamplerCustomAdvanced.execute(setup.noise, guider, setup.sampler,
        setup.sigmas, setup.latent).result[0]
    assert all(torch.equal(a, b) for a, b in zip(candidate["samples"].unbind(), source["samples"].unbind()))
    assert runtime.snapshot()["status"] == "abstain_no_sample"
    assert not selected.wrappers and runtime.completed_forwards == 0


def test_tail_source_controls_and_relay_pair_cannot_be_swapped():
    _, setup, boundary, selected, context, _ = prepared()
    with pytest.raises(ValueError, match="controls differ"):
        effects.bind_tail_effects(boundary, setup.model.clone(), setup.noise, setup.guider,
            setup.sampler, setup.sigmas, setup.latent)
    wrong = deepcopy(setup.latent)
    wrong["noise_mask"].tensors[0].fill_(1)
    with pytest.raises(ValueError, match="AV or masks"):
        apply_stage_eav(selected, setup.sigmas, wrong, context, EAVConfig())
    wrong_sigma = setup.sigmas.clone()
    wrong_sigma[0] = .5
    with pytest.raises(ValueError, match="SIGMAS"):
        effects.tail_effects_guider(selected, context, wrong_sigma, setup.latent)
    paired_model, positive, _ = paired(selected)
    with pytest.raises(ValueError, match="no paired MODEL"):
        effects.tail_effects_guider(selected, context, setup.sigmas, setup.latent, positive)
    with pytest.raises(ValueError, match="binding differ"):
        effects.tail_effects_guider(paired_model, context, setup.sigmas, setup.latent)
    positive[0][1]["minimax_frame_count"] = 100
    with pytest.raises(ValueError, match="source media or frame timeline"):
        effects.tail_effects_guider(paired_model, context, setup.sigmas, setup.latent, positive)


@pytest.mark.parametrize("segment,context_frames", [(0, 0), (1, 5), (1, 22), (1, 39)])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_long_relay_tail_real_forwards_keep_projected_motion_window(monkeypatch, segment, context_frames, mode):
    from helpers import FakeAudioVAE, FakeVideoVAE
    from test_prompt_relay_long_video_advanced import NativeLikeFakeClip, _allow_fixture_core_contract, _global_plan
    from h3_audio_t8_pkg.long_video import LONG_VIDEO_SCHEMA
    from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
        build_prompt_relay_long_video_conditioning, project_prompt_relay_plan_to_long_video_window)

    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    _allow_fixture_core_contract(monkeypatch)  # Byte-tokenizer/VAE doubles; native H3 executes below.
    frames = 124 if context_frames > 5 else 22
    global_plan = _global_plan(345 if context_frames > 5 else 39)
    start, end = ((frames / 24, (2 * frames - context_frames) / 24)
                  if segment else (0., frames / 24))
    projected, *_ = project_prompt_relay_plan_to_long_video_window(global_plan,
        segment_index=segment, length=frames, context_frames=context_frames,
        timeline_start_seconds=start, timeline_end_seconds=end)
    context = {"schema": LONG_VIDEO_SCHEMA, "empty": not bool(segment),
        "video_tail": torch.zeros((1, 24, 12, 8, 8)),
        "audio_tail": torch.zeros((1, 32, 2, 65)),
        "metadata": {"source_segment_index": segment - 1, "target_segment_index": segment,
                     "max_context_frames": 39, "audio_overhang": 1 / 3}}
    patched, positive, source, _, prompt, media, report = build_prompt_relay_long_video_conditioning(
        model=model(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        context=context, prompt_relay_plan=projected, segment_index=segment, context_frames=context_frames,
        context_audio="video_and_audio", width=128, height=128, length=frames,
        task_type="auto", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp", query_chunk_rows=64)
    audit, _, _ = _audit(model=patched, positive=positive, av_latent=source,
        conditioned_prompt=prompt, media_map_json=media, conditioning_report=report)
    _, route, decision, _ = refine.route_audio_refine_compatibility(audit=audit, refine_model=patched,
        positive=positive, generation_profile="long_video_prompt_relay_turbo8", declared_first_pass_nfe=8)
    assert decision == "ALLOW"
    plan, _, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 26092802)
    setup = refine.setup_audio_refine_compatibility(plan=plan, refine_model=patched,
        positive=positive, av_latent=source, runtime_snapshot_fn=_runtime)
    boundary = bind_audio_refine_stage(plan, source, setup.model, setup.noise, setup.guider,
        setup.sampler, setup.sigmas, setup.latent, setup.report_json)[6]
    selected, stage, _ = effects.bind_tail_effects(boundary, setup.model, setup.noise, setup.guider,
        setup.sampler, setup.sigmas, setup.latent, segment, context_frames)
    selected, runtime, _ = apply_stage_eav(selected, setup.sigmas, setup.latent, stage,
        EAVConfig(mode, tau=.25, start_video_progress=0., end_video_progress=1.))
    guider, _ = effects.tail_effects_guider(selected, stage, setup.sigmas, setup.latent)
    candidate = SamplerCustomAdvanced.execute(setup.noise, guider, setup.sampler,
        setup.sigmas, setup.latent).result[0]
    _, raw = audit_stage_eav(candidate, runtime)
    result = json.loads(raw)
    assert result["status"] == {"disabled": "disabled_identity", "report_only": "observed_report_only",
                                "apply_exp": "observed_apply_exp"}[mode]
    if mode != "disabled":
        assert result["relay_attention_calls"] == result["completed_forwards"] == 4
        assert {entry["task"] for entry in result["feta"]["forwards"]} == {
            "LongVideoMotion" if segment else "LongVideoSegment0"}
    assert effects.capture_owner(selected).long_contract["segment_index"] == segment
    assert not torch.cuda.is_initialized()


def test_report_only_and_disabled_do_not_change_real_tail_math(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    _, setup, _, selected, context, _ = prepared()
    baseline = SamplerCustomAdvanced.execute(setup.noise, setup.guider, setup.sampler,
        setup.sigmas, setup.latent).result[0]
    for mode in ("disabled", "report_only"):
        branch, runtime, _ = apply_stage_eav(selected, setup.sigmas, setup.latent, context,
            EAVConfig(mode, tau=.25, start_video_progress=0., end_video_progress=1.))
        guider, _ = effects.tail_effects_guider(branch, context, setup.sigmas, setup.latent)
        candidate = SamplerCustomAdvanced.execute(setup.noise, guider, setup.sampler,
            setup.sigmas, setup.latent).result[0]
        assert all(torch.equal(a, b) for a, b in zip(baseline["samples"].unbind(), candidate["samples"].unbind()))
        assert runtime.snapshot()["quality_accepted"] is False


def test_tail_retains_original_eav_gain_guard(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    _, setup, _, selected, context, _ = prepared()
    selected, runtime, _ = apply_stage_eav(selected, setup.sigmas, setup.latent, context,
        EAVConfig("apply_exp", start_video_progress=0., end_video_progress=1.))
    guider, _ = effects.tail_effects_guider(selected, context, setup.sigmas, setup.latent)
    with pytest.raises(RuntimeError, match="hard limit"):
        SamplerCustomAdvanced.execute(setup.noise, guider, setup.sampler, setup.sigmas, setup.latent)
    assert runtime.snapshot()["status"] == "aborted"


def test_new_effect_nodes_append_after_the_complete_legacy_modular_and_hypervae_prefix():
    import asyncio
    import h3_audio_t8_pkg
    from h3_audio_t8_pkg.modular_sampling.audio_refine_effect_nodes import NODES
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    before = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
              *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes]
    assert classes[:len(before) + len(NODES)] == before + NODES
    assert len(classes) == len({node.define_schema().node_id for node in classes})
    assert [node.define_schema().node_id for node in NODES] == [
        "MiniMaxH3AudioRefineEffectsBindEXPT8", "MiniMaxH3AudioRefineEffectsGuiderEXPT8"]
    assert all(node.GET_NODE_INFO_V1()["input"] for node in NODES)


def test_original_guider_cfg_cannot_change_after_effect_binding():
    _, setup, _, selected, context, _ = prepared()
    setup.guider.cfg = 2.
    with pytest.raises(ValueError, match="Setup guider changed"):
        effects.tail_effects_guider(selected, context, setup.sigmas, setup.latent)
