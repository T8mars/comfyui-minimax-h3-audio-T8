"""H16 new opt-in exact accepted AV prefix; released H16 behavior is separate."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy, sampling
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.speed_advanced import H3ModalityStableNoise
from h3_audio_t8_pkg.temporal_dialogue_bank import prepare_window_bank
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import lift_chunked_segment, prepare_chunked_pass2
from h3_audio_t8_pkg.modular_sampling.temporal_h16 import sample_scoped_h16
from h3_audio_t8_pkg.modular_sampling.temporal_h16_effects import bind_h16_scoped_effects, audit_h16_scoped_effects
from h3_audio_t8_pkg.modular_sampling.temporal_h16_storage import save_h16_scoped_window, load_h16_scoped_window
from h3_audio_t8_pkg.modular_sampling.temporal_chunked_relay import ScopedRelayConfig
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
from test_modular_h16_stages import _no_op_lift, _native_like_piece
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip
from test_temporal_dialogue_scope import user_plan


def make_case(monkeypatch, clip):
    monkeypatch.setattr(legacy, 'learned_upscale_h3_av_latent', _no_op_lift)
    source = {'samples': comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 87, 2, 2), torch.zeros(1, 32, 2, 501)))}
    # Legal tiny v4 contract: the old H16 convenience planner fixes minimum
    # tile=256, so it cannot construct this 32px CPU-only fixture. Keep that
    # production guard; use the original general v4 builder with explicit 32.
    plan, _ = legacy.build_chunked_two_pass_masked_low_sigma_plan(
        model_name='minimax_h3_latent_upscaler_3d_fp16.safetensors',
        target_width=32, target_height=32, temporal_chunk_frames=187, temporal_overlap_frames=34,
        anchor_strength=0.999, tile_width=32, tile_height=32, spatial_overlap=0, spatial_fade=0,
        minimum_tile_size=32, overlap_blend='smoothstep', precision='fp16', release_policy='offload_after',
        spatial_strategy='full_frame_safe', temporal_strategy='guarded_overlap_exp',
        second_pass_audio_policy='joint_av_preserve_input', video_mask_policy='inherit_if_present_else_generate_all')
    recipe = build_conditioning(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt='native', width=32, height=32, length=294, audio_mode='native',
        return_text_recipe=True)[-1]['text_recipe']
    bank, _ = prepare_window_bank(recipe, user_plan(), source, plan)
    noise = H3ModalityStableNoise(2609174402)
    context, _ = prepare_chunked_pass2(source, plan, noise)
    return source, plan, bank, noise, context


def window(h, index):
    source, plan, _bank, _noise, context = h
    segment, spec, _ = slice_chunked_source(source, plan, index)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    return segment, lifted, spec


def test_h16_next_window_consumes_real_refined_audio_with_mask_zero_and_exact_append(monkeypatch):
    import h3_audio_t8_pkg.modular_sampling.temporal_h16 as scoped
    monkeypatch.setattr(scoped, 'rebind_dual_clock_sampler', lambda _model, _piece, sampler: sampler)
    observed = []
    def sample(piece, *args, **kwargs):
        observed.append(piece)
        return _native_like_piece(piece, *args, **kwargs)
    monkeypatch.setattr(legacy, 'sample_piece', sample)
    h = make_case(monkeypatch, FakeClip())
    source, plan, bank, noise, context = h
    previous = None
    for index in range(2):
        segment, lifted, spec = window(h, index)
        output, result, report = sample_scoped_h16(object(), segment, lifted, spec,
            context, plan, noise, object(), torch.tensor([0.3, 0.]), bank, previous)
        if previous is not None:
            old_video, old_audio = previous.output_latent['samples'].unbind()
            video, audio = output['samples'].unbind()
            assert torch.equal(video[:, :, :old_video.shape[2]], old_video)
            assert torch.equal(audio[..., :old_audio.shape[-1]], old_audio)
            assert torch.equal(observed[-1]['samples'].tensors[1][..., :57], old_audio[..., spec.audio_start:])
            assert bool((observed[-1]['noise_mask'].tensors[1][..., :57] == 0).all())
        previous = result
    audio = output['samples'].tensors[1]
    assert audio.shape[-1] == 501 and torch.equal(audio[..., 490:], source['samples'].tensors[1][..., 490:])
    assert not torch.equal(audio[..., :490], source['samples'].tensors[1][..., :490])
    assert json.loads(report)['audio_merge'] == 'exact_append_source_padding_retained'
    assert previous.dialogue_receipt.window.audio_padding_policy == 'retain_source_tail'


@pytest.mark.parametrize('with_eav', [False, True])
def test_h16_scoped_native_relay_and_optional_external_eav_actual_tiny_forwards(monkeypatch, with_eav):
    clip = NativeLikeFakeClip()
    h = make_case(monkeypatch, clip)
    source, plan, bank, noise, context = h
    model, sampler, _ = sampling.setup_dual_clock_sampling(_model(), source, 3, 12., 3.)
    sigmas = torch.tensor([0.3, 0.])
    segment, lifted, spec = window(h, 0)
    _, first, _ = sample_scoped_h16(model, segment, lifted, spec, context, plan, noise, sampler, sigmas, bank)
    segment, lifted, spec = window(h, 1)
    selected, positive, runtime, _ = bind_h16_scoped_effects(model, segment, lifted, spec, context,
        plan, sigmas, bank, first, clip=clip, relay_config=ScopedRelayConfig(query_chunk_rows=32),
        eav_config=EAVConfig(mode='apply_exp', start_video_progress=0.1, g_hard_limit=3.) if with_eav else None)
    _, result, _ = sample_scoped_h16(selected, segment, lifted, spec, context, plan, noise,
        sampler, sigmas, bank, first, positive)
    report = json.loads(audit_h16_scoped_effects(result, runtime)[1])
    if with_eav:
        assert report['effects']['eav']['relay_required']
        assert report['effects']['eav']['relay_attention_calls'] > 0
        assert report['effects']['eav']['completed_forwards'] == 1
    else:
        assert report['effects']['relay']['actual_calls'] == report['effects']['relay']['expected_calls']
        assert report['effects']['relay']['actual_calls']['completed_forwards'] == 1


def test_h16_literal_scoped_cold_restores_real_prefix_without_resampling(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.modular_sampling.temporal_h16 as scoped
    monkeypatch.setattr(scoped, 'rebind_dual_clock_sampler', lambda _model, _piece, sampler: sampler)
    monkeypatch.setattr(legacy, 'sample_piece', _native_like_piece)
    h = make_case(monkeypatch, FakeClip())
    source, plan, bank, noise, context = h
    segment, lifted, spec = window(h, 0)
    sigmas = torch.tensor([0.3, 0.])
    _, first, _ = sample_scoped_h16(object(), segment, lifted, spec, context, plan, noise, object(), sigmas, bank)
    _, _, path, digest, _ = save_h16_scoped_window(first, segment, lifted, spec, context, plan, bank, tmp_path)
    _, loaded, report = load_h16_scoped_window(segment, lifted, spec, context, plan, bank, tmp_path, path, digest)
    assert loaded.dialogue_receipt == first.dialogue_receipt and not json.loads(report)['automatic_cache_reuse']
    segment, lifted, spec = window(h, 1)
    expected, _, _ = sample_scoped_h16(object(), segment, lifted, spec, context, plan, noise, object(), sigmas, bank, first)
    actual, _, _ = sample_scoped_h16(object(), segment, lifted, spec, context, plan, noise, object(), sigmas, bank, loaded)
    for left, right in zip(actual['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(left, right)
    with pytest.raises(ValueError, match='window differs'):
        load_h16_scoped_window(segment, lifted, spec, context, plan, bank, tmp_path, path, digest)
    with (tmp_path/path).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='SHA'):
        load_h16_scoped_window(segment, lifted, spec, context, plan, bank, tmp_path, path, digest)


def test_h16_public_integrated_separated_and_explicit_save_slots(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.modular_sampling.temporal_h16 as scoped
    import h3_audio_t8_pkg.nodes_temporal_h16 as nodes
    monkeypatch.setattr(scoped, 'rebind_dual_clock_sampler', lambda _model, _piece, sampler: sampler)
    monkeypatch.setattr(legacy, 'sample_piece', _native_like_piece)
    monkeypatch.setattr(nodes, '_root', lambda: tmp_path)
    h = make_case(monkeypatch, FakeClip())
    source, plan, bank, noise, context = h
    sigmas = torch.tensor([0.3, 0.])
    previous = None
    for index in range(2):
        segment, lifted, spec = window(h, index)
        expected, previous, _ = nodes.MiniMaxH3TemporalH16WindowEXPT8.execute(object(), segment,
            lifted, spec, context, plan, bank, noise, object(), sigmas, previous).result
        unsaved = nodes.MiniMaxH3TemporalH16WindowSaveEXPT8.execute(previous, segment,
            lifted, spec, context, plan, bank).result
        assert unsaved[2:4] == ('', '') and not list(tmp_path.iterdir())
    actual, result, report = nodes.MiniMaxH3TemporalH16JointPass2EXPT8.execute(object(), source,
        plan, bank, noise, object(), sigmas).result
    assert result.index == 1 and not json.loads(report)['quality_accepted']
    for left, right in zip(actual['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(left, right)
    saved = nodes.MiniMaxH3TemporalH16WindowSaveEXPT8.execute(previous, segment,
        lifted, spec, context, plan, bank, True).result
    loaded = nodes.MiniMaxH3TemporalH16WindowLoadEXPT8.execute(segment, lifted, spec,
        context, plan, bank, saved[2], saved[3]).result
    assert loaded[1].output_identity == previous.output_identity
    fingerprint = nodes.MiniMaxH3TemporalH16WindowLoadEXPT8.fingerprint_inputs(saved[2], saved[3])
    assert isinstance(fingerprint, tuple) and len(fingerprint) == 2
