"""NAG uses actual tiny Core attention; disabled stays exact identity."""
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
import comfy.samplers

from h3_audio_t8_pkg.nag_h3 import apply_nag, NAG_WRAPPER_KEY
from h3_audio_t8_pkg.nodes_nag_h3 import MiniMaxH3NAGEXPT8 as Apply, MiniMaxH3NAGAuditEXPT8 as Audit
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent
from test_semantic_bridge import weight_file  # noqa: F401


def sample(selected):
    positive = conditioning()
    positive[0][0].copy_(torch.randn(positive[0][0].shape, generator=torch.Generator().manual_seed(830)))
    guider = BasicGuider.execute(selected,positive).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(17).result[0],guider,
        comfy.samplers.ksampler("euler"),torch.tensor([1.,.5,0.]),latent()).result[0]


def test_off_and_scale_one_preserve_exact_model_and_sampler_output():
    selected = model()
    disabled, runtime, report = Apply.execute(selected, [], enabled=False).result
    assert disabled is selected and runtime is None
    assert apply_nag(selected, [], 1.,2.5,.25,0.) is selected
    assert apply_nag(selected, [], 5.,2.5,0.,0.) is selected
    expected, result = sample(selected), sample(disabled)
    assert all(torch.equal(a,b) for a,b in zip(expected["samples"].unbind(),result["samples"].unbind()))


def test_empty_encoded_negative_is_active_and_differs_from_off_and_negative():
    bare = model()
    empty = conditioning()
    empty[0][0].zero_()
    negative = conditioning()
    negative[0][0].fill_(2.)
    results = []
    for value in (empty,negative):
        selected,runtime,_ = Apply.execute(bare,value,enabled=True).result
        output = sample(selected)
        report = json.loads(Audit.execute(output,runtime).result[1])
        assert report["active_diffusion_calls"] == 2 and report["guided_blocks"] == 2
        assert report["failed_calls"] == 0 and not report["state_retained"]
        assert not runtime.negative_embeds
        assert all(torch.isfinite(p).all() for p in output["samples"].unbind())
        results.append(output["samples"].unbind()[0])
    baseline = sample(bare)["samples"].unbind()[0]
    assert not torch.equal(results[0],baseline)
    assert not torch.equal(results[0],results[1])
    assert not bare.wrappers and not bare.get_attachment(NAG_WRAPPER_KEY)


def test_reapply_replaces_own_patch_keeps_foreign_owner_and_sigma_cutoff():
    base = model()
    calls = []
    def foreign(args,extra):
        calls.append(args["img"].shape[0])
        return extra["original_block"](args)
    base.set_model_patch_replace(foreign,"dit","double_block",0)
    first = apply_nag(base,conditioning(),5.,2.5,.25,0.)
    second = apply_nag(first,conditioning(),3.,2.,.1,.75)
    patch = second.model_options["transformer_options"]["patches_replace"]["dit"][("double_block",0)]
    assert patch.previous is foreign
    sample(second)
    stats = second.get_attachment(NAG_WRAPPER_KEY).snapshot()
    assert stats["active_diffusion_calls"] == 1 and stats["bypassed_diffusion_calls"] == 1
    assert len(calls) == 2
    assert first.get_attachment(NAG_WRAPPER_KEY).snapshot()["active_diffusion_calls"] == 0


def test_compatible_alternate_diffusion_class_delegates():
    base = model()
    actual = base.model.diffusion_model
    class OtherImplementation(type(actual)):
        pass
    actual.__class__ = OtherImplementation
    selected = apply_nag(base,conditioning(),5.,2.5,.25,0.)
    output = sample(selected)
    assert torch.isfinite(output["samples"].unbind()[0]).all()


def test_invalid_negative_batch_is_not_silently_truncated():
    negative = conditioning()
    negative[0][0] = negative[0][0].repeat(2,1,1)
    with pytest.raises(ValueError,match="batch-1"):
        apply_nag(model(),negative,5.,2.5,.25,0.)


def test_grouped_memory_and_foreign_positive_attention_are_really_consumed():
    from h3_audio_t8_pkg.h3_memory_advanced import configure_low_vram_attention, configure_chunk_feed_forward
    base = model()
    grouped, _ = configure_low_vram_attention(base, 3)
    grouped, _ = configure_chunk_feed_forward(grouped, 2, 256)
    calls = []
    def foreign(original, q, k, v, heads, **kwargs):
        calls.append((heads, q.shape[-2]))
        return original(q, k, v, heads, **kwargs)
    grouped.model_options['transformer_options']['optimized_attention_override'] = foreign
    selected = apply_nag(grouped, conditioning(), 5., 2.5, .25, 0.)
    output = sample(selected)
    stats = selected.get_attachment(NAG_WRAPPER_KEY).snapshot()
    assert stats['guided_blocks'] == 2
    assert stats['attention_kernel_calls'] == 6  # 3 real head groups x 2 denoiser calls
    assert len(calls) >= 6 and all(heads == 1 for heads, _ in calls[-6:])
    assert all(torch.isfinite(t).all() for t in output['samples'].unbind())
    assert not stats['state_retained']


def test_actual_sampler_failure_cleans_nag_state_and_subsequent_run_works():
    base = model()
    def fail(args, extra):
        raise RuntimeError('test interrupted block')
    base.set_model_patch_replace(fail, 'dit', 'double_block', 0)
    selected = apply_nag(base, conditioning(), 5., 2.5, .25, 0.)
    runtime = selected.get_attachment(NAG_WRAPPER_KEY)
    with pytest.raises(RuntimeError, match='interrupted block'):
        sample(selected)
    assert runtime.state is None and not runtime.negative_embeds
    assert runtime.failed_calls == 1
    selected.model_options['transformer_options']['patches_replace']['dit'][('double_block', 0)].previous = None
    assert torch.isfinite(sample(selected)['samples'].unbind()[0]).all()


@pytest.mark.parametrize('groups', [1, 2, 3])
def test_grouped_nag_preserves_full_hidden_norm(groups):
    from h3_audio_t8_pkg.nag_h3 import delegated_nag_attention
    from h3_audio_t8_pkg.h3_memory_advanced import configure_low_vram_attention
    base = model()
    attn = base.model.diffusion_model.blocks[0].attn
    x = torch.randn(17, 24, generator=torch.Generator().manual_seed(123))
    # Independent SDPA/full-vector formula, before any head grouping.
    q, k, v = attn.qkv_proj(x).chunk(3, dim=-1)
    q = attn.q_norm(q.reshape(17, attn.heads, attn.head_dim)).transpose(0, 1)
    k = attn.k_norm(k.reshape(17, attn.heads, attn.head_dim)).transpose(0, 1)
    v = v.reshape(17, attn.heads, attn.head_dim).transpose(0, 1)
    def attend(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(0, 1).flatten(1)
    positive = attend(q[:, :12], k[:, :12], v[:, :12])
    negative = attend(q[:, 3:], k[:, 3:], v[:, 3:])
    guidance = 5. * positive[3:] - 4. * negative[:9]
    ratio = guidance.abs().sum(-1, keepdim=True).clamp_min(1e-6) / positive[3:].abs().sum(-1, keepdim=True).clamp_min(1e-6)
    mixed = .8 * guidance * ratio.clamp(max=.5) / ratio + .2 * positive[3:]
    wanted = attn.out_proj(torch.cat((positive[:3], mixed, negative[9:])))
    grouped, _ = configure_low_vram_attention(base, groups)
    grouped.patch_model(load_weights=False)
    try:
        got = delegated_nag_attention(x.clone(), forward=attn, seq_len=12, text_len=3,
                                      scale=5., tau=.5, alpha=.8, observed=lambda: None)
    finally:
        grouped.unpatch_model(unpatch_weights=False)
    torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-5)


def test_preferred_backend_is_consumed_and_projection_hook_removed_on_failure():
    from h3_audio_t8_pkg.nag_h3 import delegated_nag_attention
    from comfy.ldm.modules.attention import attention_pytorch
    attn = model().model.diffusion_model.blocks[0].attn
    calls = []
    def preferred(q, k, v, heads, **kwargs):
        calls.append((q.shape[-2], k.shape[-2]))
        return attention_pytorch(q, k, v, heads, **kwargs)
    attn.comfy_attention.function = preferred
    x = torch.randn(17, 24)
    delegated_nag_attention(x, forward=attn, seq_len=12, text_len=3,
                           scale=5., tau=2.5, alpha=.25, observed=lambda: None)
    assert calls == [(12, 12), (14, 14)]
    assert not attn.out_proj._forward_pre_hooks
    def fail(*args, **kwargs):
        raise RuntimeError('failed preferred attention')
    attn.comfy_attention.function = fail
    with pytest.raises(RuntimeError, match='failed preferred'):
        delegated_nag_attention(x, forward=attn, seq_len=12, text_len=3,
                               scale=5., tau=2.5, alpha=.25, observed=lambda: None)
    assert not attn.out_proj._forward_pre_hooks


def test_preprocessing_failure_restores_contextvar(monkeypatch):
    selected = apply_nag(model(), conditioning(), 5., 2.5, .25, 0.)
    runtime = selected.get_attachment(NAG_WRAPPER_KEY)
    def fail(*args):
        raise RuntimeError('negative preprocess interrupted')
    monkeypatch.setattr(runtime, 'get_negative_embeds', fail)
    with pytest.raises(RuntimeError, match='negative preprocess'):
        sample(selected.clone())
    assert runtime.state is None and runtime.failed_calls == 1


def test_disabled_later_node_preserves_and_reports_upstream_nag():
    first = apply_nag(model(), conditioning(), 5., 2.5, .25, 0.)
    selected, runtime, report = Apply.execute(first, [], enabled=False).result
    assert selected is first and runtime is first.get_attachment(NAG_WRAPPER_KEY)
    assert json.loads(report)['upstream_nag_retained']
    sample(selected)
    assert runtime.snapshot()['guided_blocks'] == 2


def test_unexposed_kernel_is_a_real_interface_error_not_silent_dense_fallback():
    from h3_audio_t8_pkg.nag_h3 import delegated_nag_attention
    attn = model().model.diffusion_model.blocks[0].attn
    calls = []
    def custom_producer(x, **kwargs):
        calls.append(True)
        # Simulates a producer exposing its output projection but directly
        # dispatching a private kernel instead of Core's attention interface.
        return attn.out_proj(torch.zeros(x.shape[0], attn.heads * attn.head_dim))
    with pytest.raises(ValueError, match='without an optimized attention consumer'):
        delegated_nag_attention(torch.zeros(17, 24), forward=custom_producer,
            projection=attn.out_proj, seq_len=12, text_len=3,
            scale=5., tau=2.5, alpha=.25, observed=lambda: None)
    assert calls == [True] and not attn.out_proj._forward_pre_hooks


def test_native_semantic_bridge_reference_and_negative_text_pipeline(weight_file):  # noqa: F811
    from test_semantic_bridge_native_pipeline import raw_text_model
    from h3_audio_t8_pkg import semantic_bridge as sb
    from h3_audio_t8_pkg.nodes_semantic_bridge import MiniMaxH3SemanticBridgeApplyT8
    native = torch.randn(1, 8, 5120, generator=torch.Generator().manual_seed(94))
    refs = [{'kind':'image', 'latent_t':1, 'latent_h':4, 'latent_w':8, 'ref_audio_t':0,
             'latent':torch.ones(1, 24, 1, 4, 8) * .2}]
    tags = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    positive, _ = MiniMaxH3SemanticBridgeApplyT8.execute(
        [[native, {'minimax_refs':refs, 'minimax_token_tags':tags}]],
        sb.BridgeConfig(str(weight_file), sb.file_sha(weight_file), device='cpu',
                        token_scope='text_only_preserve_reference')).result
    negative = [[torch.zeros(1, 3, 5120), {'minimax_token_tags':torch.ones(3, dtype=torch.int64)}]]
    selected = apply_nag(raw_text_model(), negative, 5., 2.5, .25, 0.)
    guider = BasicGuider.execute(selected, positive).result[0]
    output = SamplerCustomAdvanced.execute(RandomNoise.execute(17).result[0], guider,
        comfy.samplers.ksampler('euler'), torch.tensor([1., .5, 0.]), latent()).result[0]
    assert selected.get_attachment(NAG_WRAPPER_KEY).snapshot()['guided_blocks'] == 2
    assert positive[0][1]['minimax_refs'] is refs
    assert all(torch.isfinite(p).all() for p in output['samples'].unbind())


def test_real_eav_positive_override_is_preserved_with_nag():
    from test_modular_results import inputs
    from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav
    noise, guider, sampler, sigmas, source, context = inputs('high_4_8')
    eav, runtime, _ = apply_stage_eav(guider.model_patcher, sigmas, source, context,
        EAVConfig(mode='apply_exp', tau=.2, start_video_progress=0., end_video_progress=1.))
    selected = apply_nag(eav, conditioning(), 5., 2.5, .25, 0.)
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result[0]
    assert selected.get_attachment(NAG_WRAPPER_KEY).snapshot()['guided_blocks'] == 4
    assert runtime.snapshot()['completed_forwards'] == 4
    assert all(torch.isfinite(p).all() for p in output['samples'].unbind())
