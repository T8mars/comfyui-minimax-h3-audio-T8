"""Fresh native spans/local clock with real tiny Core Relay/EAV forwards."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import sampling, chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.temporal_dialogue_bank import prepare_window_bank
from h3_audio_t8_pkg.speed_advanced import H3ModalityStableNoise
from h3_audio_t8_pkg.modular_sampling.chunked_v5 import lift_standard_joint, prepare_standard_joint
from h3_audio_t8_pkg.modular_sampling.chunked_v5_effects import bind_v5_eav, audit_v5_eav
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling.temporal_chunked_v5 import sample_scoped_v5_window
from h3_audio_t8_pkg.modular_sampling.temporal_chunked_relay import (
    ScopedRelayConfig, _local_plan, bind_scoped_v5_relay, audit_scoped_relay,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_chunked_two_pass_parity import _plan
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip
from test_temporal_dialogue_scope import user_plan


@pytest.fixture
def native_case(monkeypatch):
    monkeypatch.setattr(legacy, 'learned_upscale_h3_av_latent', _fake_lift)
    source = {'samples': comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 87, 2, 2), torch.zeros(1, 32, 2, 501)))}
    plan = _plan(temporal_chunk_frames=187, temporal_overlap_frames=34)
    clip = NativeLikeFakeClip()
    recipe = build_conditioning(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt='native', width=64, height=64, length=294, audio_mode='native',
        return_text_recipe=True)[-1]['text_recipe']
    bank, _ = prepare_window_bank(recipe, user_plan(), source, plan)
    noise = H3ModalityStableNoise(2609174402)
    lifted, receipt, _ = lift_standard_joint(source, plan)
    prepared, _ = prepare_standard_joint(source, lifted, receipt, plan, noise)
    model, sampler, _ = sampling.setup_dual_clock_sampling(_model(), lifted, 8, 12., 3.)
    from h3_audio_t8_pkg.chunked_two_pass_parity import UPSTREAM_REFINE_VIDEO_SIGMAS
    sigmas = torch.tensor(UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    return model, clip, source, lifted, prepared, plan, sigmas, bank, sampler, noise


def sample(h, index, previous=None, model=None, positive=None):
    raw, _clip, source, lifted, prepared, plan, sigmas, bank, sampler, noise = h
    return sample_scoped_v5_window(raw if model is None else model, source, lifted, prepared,
        plan, noise, sampler, sigmas, bank, index, previous, positive)


def test_scope_relay_bypasses_single_event_then_binds_new_native_second_window(native_case):
    h = native_case
    model, clip, source, lifted, prepared, plan, sigmas, bank, _sampler, _noise = h
    config = ScopedRelayConfig(query_chunk_rows=32)
    selected, positive, bypass, report = bind_scoped_v5_relay(model, clip, source, lifted,
        prepared, plan, sigmas, bank, 0, config)
    assert selected is model and positive is bank.encoded[0].positive
    assert json.loads(report)['status'] == 'passthrough_single_event'
    _, _, first, _ = sample(h, 0, model=selected, positive=positive)
    assert not json.loads(audit_scoped_relay(first, source, plan, prepared, bank, bypass)[1])['attention_patch_applied']
    local = _local_plan(bank.encoded[1], config)
    projected = bank.encoded[1].compiled.dialogue[0]
    assert local['events'][0]['start_frame'] == projected.local_start_frame
    selected, paired, runtime, report = bind_scoped_v5_relay(model, clip, source, lifted,
        prepared, plan, sigmas, bank, 1, config, first)
    binding = paired[0][1]['minimax_prompt_relay_binding']
    assert len(binding['events']) == 2 and '我没看过。' not in bank.encoded[1].prepared_prompt
    assert binding['events'][0]['midpoint'] == local['events'][0]['midpoint']
    assert json.loads(report)['clock_projection_count'] == 1
    final, _, result, _ = sample(h, 1, first, selected, paired)
    audit = json.loads(audit_scoped_relay(result, source, plan, prepared, bank, runtime)[1])
    assert audit['actual_calls'] == audit['expected_calls'] and audit['actual_calls']['completed_forwards'] == 4
    assert audit['status'] == 'observed_relay_calls_quality_unverified' and not audit['hard_time_isolation']
    assert torch.equal(final['samples'].tensors[1][..., :first.base.output_latent['samples'].tensors[1].shape[-1]],
                       first.base.output_latent['samples'].tensors[1])


def test_scoped_second_window_external_relay_and_eav_actual_combined_calls(native_case):
    h = native_case
    model, clip, source, lifted, prepared, plan, sigmas, bank, _sampler, _noise = h
    _, _, first, _ = sample(h, 0)
    selected, positive, _, _ = bind_scoped_v5_relay(model, clip, source, lifted,
        prepared, plan, sigmas, bank, 1, ScopedRelayConfig(query_chunk_rows=32), first)
    selected, runtime, _, _ = bind_v5_eav(selected, sigmas, source, lifted, prepared, plan, 1,
        first.base, EAVConfig(mode='apply_exp', start_video_progress=0.1, g_hard_limit=3.), positive)
    _, base, _, _ = sample(h, 1, first, selected, positive)
    report = json.loads(audit_v5_eav(base, prepared, runtime)[1])
    assert report['relay_required'] and report['relay_attention_calls'] > 0
    assert report['completed_forwards'] == report['planned_forwards'] == 4


def test_scoped_relay_pair_and_bank_ownership_errors_not_silently_ignored(native_case):
    h = native_case
    model, clip, source, lifted, prepared, plan, sigmas, bank, _sampler, _noise = h
    _, _, first, _ = sample(h, 0)
    selected, paired, _, _ = bind_scoped_v5_relay(model, clip, source, lifted,
        prepared, plan, sigmas, bank, 1, ScopedRelayConfig(query_chunk_rows=32), first)
    with pytest.raises(ValueError, match='CONDITIONING'):
        sample(h, 1, first, selected, bank.encoded[0].positive)
    with pytest.raises(ValueError, match='MODEL and CONDITIONING'):
        sample(h, 1, first, selected, bank.encoded[1].positive)
    with pytest.raises(ValueError, match='raw HIGH'):
        bind_scoped_v5_relay(selected, clip, source, lifted, prepared, plan, sigmas, bank,
                             1, ScopedRelayConfig(), first)
