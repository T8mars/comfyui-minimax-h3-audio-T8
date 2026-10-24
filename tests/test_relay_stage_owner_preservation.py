"""Portable Relay-owner/stage-report contracts; random tiny Core, never trained/GPU.

The numerical oracle delegates directly to the unchanged native segment
sampler. No historical experiment, private builder or saved model is loaded.
"""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import comfy.model_management as mm
from comfy.ldm.modules import attention
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrapperExecutor

from h3_audio_t8_pkg import h3_core_compat, prompt_relay_advanced as relay
from h3_audio_t8_pkg import long_video_dual_model_stages as original_stage
from h3_audio_t8_pkg.relay_kj_backend import KJRelayBackend
from test_prompt_relay_core_compat import bound_layout, model_fixture

# Optional pre-promotion source selection; an ordinary run uses public code.
target = Path(os.environ.get("H22_STAGE_SOURCE", original_stage.__file__)).resolve(strict=True)
if target == Path(original_stage.__file__).resolve():
    stages = original_stage
else:
    spec = importlib.util.spec_from_file_location("h3_audio_t8_pkg._relay_stage_test_candidate", target)
    stages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stages)


@pytest.fixture
def case(monkeypatch):
    """Real random-weight H3 HIGH4 at 37 video cells/207 audio positions.

    Text and unused VAE objects are explicitly synthetic. This is the same
    context/pairing/reconcile/prefix/dual-clock path as the original tiny
    regression, expressed only through public APIs and literal small inputs.
    No LOW, learned upscaler, cached stage, historical asset or file is read.
    """
    from comfy.ldm.minimax import model as native
    from h3_audio_t8_pkg import core, sampling, long_video
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    from h3_audio_t8_pkg import long_video_dual_model_runner as runner
    from h3_audio_t8_pkg import prompt_relay_long_video_advanced as long_relay
    from helpers import FakeAudioVAE, FakeVideoVAE
    from test_fast_h3_v2_core_sampler import model as actual_tiny_model
    from test_long_video import make_context
    from test_prompt_relay_long_video_advanced import NativeLikeFakeClip

    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    torch.set_num_threads(1)
    bare = actual_tiny_model()
    source_plan = relay.build_prompt_relay_plan(global_prompt="g",
        local_prompts="move\nstop", length=193, timing_mode="percent",
        time_ranges="0-50\n50-100", math_profile="paper_v1", epsilon=.1,
        allow_gaps=False, allow_overlaps=False)[0]
    projected = long_relay.project_prompt_relay_plan_to_long_video_window(source_plan,
        segment_index=1, length=124, context_frames=22,
        timeline_start_seconds=124 / 24, timeline_end_seconds=8)[0]
    source, frames = core.empty_av_latent(32, 32, 124)
    assert frames == 124
    generator = torch.Generator().manual_seed(21102107)
    source["samples"] = NestedTensor(tuple(torch.randn(item.shape, generator=generator) * .03
        for item in source["samples"].unbind()))
    context = make_context(32, 32)
    context["video_tail"] *= .0001
    context["audio_tail"] *= .0001
    context["metadata"]["chain_id"] = "r7-tiny"
    with torch.inference_mode():
        model, positive, template, *_ = stages.bind_stage_conditioning(bare,
            long_relay.build_prompt_relay_long_video_conditioning, clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), context=context,
            prompt_relay_plan=projected, task_type="auto", context_audio="video_and_audio",
            audio_mode="native", audio_denoise_strength=.35, add_source_as_reference=True,
            prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
            reference_video_policy="official_2_to_15s", first_frame_reuse="segment0_only",
            persistent_identity_strategy="single_reference", persistent_identity_interval=1,
            segment_index=1, context_frames=22, width=32, height=32, length=124,
            execution_mode="apply_exp", query_chunk_rows=64)
        model = long_video.patch_long_video_model(model)
        engine = runner.DualModelSegmentRunner(model, model, contract={}, low_width=448,
            low_height=224, upscaler_model="NOT_LOADED_IN_THIS_HIGH_ONLY_TEST",
            coarse_steps=4, refine_steps=4, second_audio_source="legacy_policy",
            second_audio_strength=0.)
        prepared, positive, _ = engine._reconcile(source, template, positive)
        prepared, _ = runner.lock_high_video_prefix(prepared, context,
            chain_id="r7-tiny", segment_index=1, context_frames=22, mode="high_native_mask_exp")
        model, sampler, _ = sampling.setup_dual_clock_sampling(model, prepared, 8, 12., 3.,
            "dual_clock_euler", "native_flow")
        _, sigmas, _ = learned.build_learned_two_pass_parity_plan(model, 8, 4, 4)
    video, audio = prepared["samples"].unbind()
    vm, am = prepared["noise_mask"].unbind()
    assert list(video.shape) == [1, 24, 37, 2, 2]
    assert list(audio.shape) == [1, 32, 2, 207]
    assert torch.equal(video[:, :, :7], context["video_tail"][:, :, -7:])
    assert torch.equal(audio, source["samples"].unbind()[1])
    assert torch.all(vm[:, :, :7] == 0) and torch.all(vm[:, :, 7:] == 1) and torch.all(am == 1)
    assert sigmas.tolist() == [.9035000205039978, .800000011920929,
        .631600022315979, .3158000111579895, 0.]
    assert not bare.wrappers and not bare.object_patches
    return SimpleNamespace(model=model, positive=positive, prepared=prepared,
        sampler=sampler, sigmas=sigmas, plan={"seed": 2609032102})

def latent():
    return {'samples': NestedTensor((torch.zeros(1, 24, 3, 2, 2), torch.zeros(1, 32, 2, 11)))}


def qkv_and_oracle(layout, binding):
    generator = torch.Generator().manual_seed(220041)
    q, k, v = [torch.randn(1, 2, layout.seq_len, 4, generator=generator) for _ in range(3)]
    route = relay._runtime_route(layout, binding, torch.device('cpu'))
    bias = torch.zeros(layout.seq_len, layout.seq_len)
    for segment in route['query_segments']:
        for event in binding['events']:
            outside = ((segment['query_times'] - event['midpoint']).abs() - event['window']).clamp_min(0)
            bias[segment['start']:segment['end'], event['text_key_start']:event['text_key_end']] += (
                -.5 * (outside / event['sigma']).square())[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return (q, k, v), expected.transpose(1, 2).reshape(1, layout.seq_len, 8)


def install_real_attention_sample(monkeypatch, binding, layout, qkv):
    seen = []
    def sample(observed, *args, **kwargs):
        options = observed.model_options['transformer_options']
        entry = {'observed': observed, 'override': options.get('optimized_attention_override')}
        seen.append(entry)
        def diffusion(x, timestep, context, transformer_options, **kw):
            entry['QKV_output'] = attention.attention_pytorch(*qkv, 2,
                skip_reshape=True, transformer_options=transformer_options)
            return latent()
        def apply():
            return WrapperExecutor.new_executor(diffusion, observed.get_all_wrappers('diffusion_model')).execute(
                [torch.zeros(1)], None, None, options, minimax_payload={'layout': layout},
                **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding['binding_hash']})
        output = WrapperExecutor.new_executor(apply,
            observed.get_wrappers('apply_model', 't8_dual_stage_network_observer')).execute()
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
        return output
    monkeypatch.setattr(stages, '_sample_prepared_segment', sample)
    return sample, seen


def run(model):
    return stages.sample_model_stage(model, [], latent(), sampler=None,
        sigmas=torch.tensor([1., 0.]), seed=17, output_kind='zero_sigma_output')


def test_exact_Relay_owner_kept_no_observer_warning_and_real_QKV_math(monkeypatch, caplog):
    binding, layout = bound_layout('joint_av_exp')
    model, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    owner = model.model_options['transformer_options']['optimized_attention_override']
    qkv, expected = qkv_and_oracle(layout, binding)
    direct, seen = install_real_attention_sample(monkeypatch, binding, layout, qkv)
    direct(model)
    caplog.clear()
    with caplog.at_level('WARNING'):
        _, report = run(model)
    assert not caplog.records
    assert seen[-1]['override'] is owner
    assert relay.prompt_relay_model_contract(seen[-1]['observed'])['attention_owner_verified'] is True
    assert model.model_options['transformer_options']['optimized_attention_override'] is owner
    torch.testing.assert_close(seen[-1]['QKV_output'], seen[0]['QKV_output'], rtol=0, atol=0)
    torch.testing.assert_close(seen[-1]['QKV_output'], expected, rtol=1e-5, atol=1e-6)
    assert report['prompt_relay_execution'] == {'completed_forwards': 1, 'routed_attention_calls': 1}
    assert report['backend'] == {'status': 'not_measured_by_plain_selector_observer'}
    assert not torch.cuda.is_initialized()


def test_real_captured_KJ_backend_cpu_counts_are_stage_deltas_not_lifetime(monkeypatch, caplog):
    def forbidden_cuda_kernel(*args, **kwargs):
        pytest.fail('CPU evidence must never claim/invoke Sage CUDA')
    backend = KJRelayBackend('auto', forbidden_cuda_kernel, 'private_CPU_fixture', False, None)
    binding, layout = bound_layout('joint_av_exp')
    bare = model_fixture()
    model, _ = relay._install_prompt_relay_model(bare, binding, 32,
        relay._assert_core_contract(bare), attention_backend=backend)
    owner = model.model_options['transformer_options']['optimized_attention_override']
    qkv, expected = qkv_and_oracle(layout, binding)
    _direct, seen = install_real_attention_sample(monkeypatch, binding, layout, qkv)
    caplog.clear()
    with caplog.at_level('WARNING'):
        _, first = run(model)
        lifetime_first = deepcopy(backend.report())
        _, second = run(model)
    assert not caplog.records
    assert first['backend']['kind'] == second['backend']['kind'] == 'audited_kj_selector'
    assert first['backend']['completed_calls'] == second['backend']['completed_calls']
    assert first['backend']['completed_calls']['pytorch:non_cuda'] > 0
    assert not any(key.startswith('sage:') for key in first['backend']['completed_calls'])
    assert backend.report()['completed_calls']['pytorch:non_cuda'] == 2 * lifetime_first['completed_calls']['pytorch:non_cuda']
    assert first['backend']['completed_calls'] == lifetime_first['completed_calls']
    assert 'this stage' in second['backend']['counter_scope']
    assert all(item['override'] is owner for item in seen)
    for item in seen:
        torch.testing.assert_close(item['QKV_output'], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(seen[0]['QKV_output'], seen[1]['QKV_output'], rtol=0, atol=0)


def test_unknown_before_binding_remains_real_unverified_delegate_with_per_stage_counts(monkeypatch, caplog):
    calls = []
    def unknown(original, *args, **kwargs):
        calls.append(kwargs.get('mask'))
        return attention.attention_pytorch(*args, **{**kwargs, '_inside_attn_wrapper': True})
    bare = model_fixture()
    bare.model_options['transformer_options']['optimized_attention_override'] = unknown
    binding, layout = bound_layout('joint_av_exp')
    model, _ = relay.patch_prompt_relay_model(bare, binding, 32)
    contract = relay.prompt_relay_model_contract(model)
    real_backend = contract['attention_backend']
    qkv, expected = qkv_and_oracle(layout, binding)
    install_real_attention_sample(monkeypatch, binding, layout, qkv)
    caplog.clear()
    _, first = run(model)
    lifetime_first = deepcopy(real_backend.report())
    _, second = run(model)
    assert not caplog.records
    assert calls and any(mask is not None for mask in calls)
    assert first['backend']['kernel_verified'] is second['backend']['kernel_verified'] is False
    assert first['backend']['kind'] == second['backend']['kind'] == 'user_selected_unverified_delegate'
    assert first['backend']['completed_calls'] == second['backend']['completed_calls'] == lifetime_first['completed_calls']
    assert real_backend.report()['completed_calls']['delegate:completed'] == 2 * first['backend']['completed_calls']['delegate:completed']
    assert real_backend.override is unknown


@pytest.mark.parametrize('forwards_Relay', [True, False])
def test_later_unknown_delegate_still_warns_and_executes_as_selected(monkeypatch, caplog, forwards_Relay):
    binding, layout = bound_layout('joint_av_exp')
    model, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    original = model.model_options['transformer_options']['optimized_attention_override']
    calls = []
    def later(original_attention, *args, **kwargs):
        calls.append(1)
        if forwards_Relay:
            return original(original_attention, *args, **kwargs)
        return attention.attention_pytorch(*args, **{**kwargs, '_inside_attn_wrapper': True})
    model.model_options['transformer_options']['optimized_attention_override'] = later
    qkv, _expected = qkv_and_oracle(layout, binding)
    install_real_attention_sample(monkeypatch, binding, layout, qkv)
    caplog.clear()
    with caplog.at_level('WARNING'):
        _, report = run(model)
    assert calls == [1]
    assert model.model_options['transformer_options']['optimized_attention_override'] is later
    assert 'later user-selected attention override' in caplog.text
    assert 'replaced after the MODEL was bound' in caplog.text
    assert report['prompt_relay_execution'] == {'completed_forwards': 1, 'routed_attention_calls': int(forwards_Relay)}
    assert report['backend']['kernel_verified'] is False


@pytest.mark.parametrize('error', [RuntimeError('actual delegate failure'), TypeError('actual delegate body TypeError'),
    mm.InterruptProcessingException()])
def test_later_delegate_actual_failure_or_cancel_propagates_once_and_cleans_observation(monkeypatch, error):
    binding, layout = bound_layout('joint_av_exp')
    model, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    calls = []
    def later(*args, **kwargs):
        calls.append(1)
        raise error
    model.model_options['transformer_options']['optimized_attention_override'] = later
    qkv, _expected = qkv_and_oracle(layout, binding)
    _sample, seen = install_real_attention_sample(monkeypatch, binding, layout, qkv)
    with pytest.raises(type(error)) as actual:
        run(model)
    assert actual.value is error and calls == [1]
    assert model.model_options['transformer_options']['optimized_attention_override'] is later
    assert not seen[-1]['observed'].get_wrappers('apply_model', 't8_dual_stage_network_observer')
    assert not seen[-1]['observed'].get_wrappers('prepare_sampling', 't8_dual_stage_prepare_timer')


def test_plain_native_selector_path_keeps_actual_math_and_counter(monkeypatch):
    binding, layout = bound_layout('joint_av_exp')
    model = model_fixture()
    h3_core_compat.set_h3_attention_backend(model, attention.attention_pytorch)
    selected = model.model_options['transformer_options']['optimized_attention_override']
    qkv, _expected = qkv_and_oracle(layout, binding)
    install_real_attention_sample(monkeypatch, binding, layout, qkv)
    _, report = run(model)
    assert report['prompt_relay_execution'] is None
    assert report['backend']['kind'] == 'native_core_pytorch_selector'
    assert report['backend']['completed_calls'] == {'pytorch:completed': 1}
    assert model.model_options['transformer_options']['optimized_attention_override'] is selected


def test_nested_report_delta_only_changes_call_counts_not_originals_or_flags():
    before = {'kind': 'synthetic_report_not_kernel_evidence', 'completed_calls': {'outer': 11},
        'fallback': {'completed_calls': {'inner': 9}, 'kernel_verified': False}, 'configuration': {'strength': .4}}
    after = {'kind': 'synthetic_report_not_kernel_evidence', 'completed_calls': {'outer': 13},
        'fallback': {'completed_calls': {'inner': 12, 'new': 1}, 'kernel_verified': False}, 'configuration': {'strength': .4}}
    originals = deepcopy((before, after))
    delta = stages._stage_backend_call_deltas(after, before)
    assert delta['completed_calls'] == {'outer': 2}
    assert delta['fallback']['completed_calls'] == {'inner': 3, 'new': 1}
    assert delta['fallback']['kernel_verified'] is False
    assert delta['configuration'] == after['configuration']
    assert (before, after) == originals
    delta['configuration']['strength'] = 99
    assert after['configuration']['strength'] == .4


def test_random_weight_real_Core_AV_bitexact_and_repeated_stage_Relay_counts(case, caplog):
    owner = case.model.model_options['transformer_options']['optimized_attention_override']
    with torch.inference_mode():
        direct = original_stage._sample_prepared_segment(case.model, case.positive, case.prepared,
            sampler=case.sampler, sigmas=case.sigmas, seed=case.plan['seed'], segment_index=1, preview_state={})
        caplog.clear()
        results = [stages.sample_model_stage(case.model, case.positive, case.prepared,
            sampler=case.sampler, sigmas=case.sigmas, seed=case.plan['seed'], segment_index=1,
            output_kind='zero_sigma_output') for _ in range(2)]
    assert not caplog.records
    for output, report in results:
        for a, b in zip(direct['samples'].unbind(), output['samples'].unbind()):
            assert torch.equal(a, b)
        assert report['completed_network_forwards'] == 4
        assert report['prompt_relay_execution'] == {'completed_forwards': 4, 'routed_attention_calls': 4}
    assert case.model.model_options['transformer_options']['optimized_attention_override'] is owner
    assert relay.prompt_relay_model_contract(case.model)['attention_owner_verified'] is True
    assert not torch.cuda.is_initialized()


def test_real_FastH3V2_owner_native_eight_step_path_not_promoted_to_Relay_branch():
    from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
    from test_fast_h3_v2_core_sampler import model as actual_model
    from test_progressive_sampling_runtime import conditioning, latent as source_latent
    source = source_latent()
    prepared, sampler, sigmas, _report = v2.build_fast_h3_v2_setup(actual_model(), source, 'dense_compat_exp')
    with torch.inference_mode():
        direct = original_stage._sample_prepared_segment(prepared, conditioning(), source,
            sampler=sampler, sigmas=sigmas, seed=220045, segment_index=0, preview_state={})
        output, report = stages.sample_model_stage(prepared, conditioning(), source,
            sampler=sampler, sigmas=sigmas, seed=220045, output_kind='zero_sigma_output')
    for a, b in zip(direct['samples'].unbind(), output['samples'].unbind()):
        assert torch.equal(a, b)
    assert report['completed_network_forwards'] == 8
    assert report['prompt_relay_execution'] is None
    assert 'fasth3_v2' in report['backend']
    assert 'counter_scope' not in report['backend']
    assert v2.capture_fast_h3_v2_owner(prepared) is not None
    assert not torch.cuda.is_initialized()
