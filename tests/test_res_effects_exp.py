"""Actual tiny RES/EAV/Relay behavior; not pretrained or CUDA qualification."""
from dataclasses import replace
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import res_effects_exp as effects, res_history_exp as history
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from h3_audio_t8_pkg.res_history_setup import setup_res_history_sampling
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from h3_audio_t8_pkg.res_stage_exp import sample_res_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent, conditioning
from test_progressive_relay import paired
from test_res_history_setup import options
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- actual installed source fixture
from test_res_sol_normalization import selected as selected_original_sol
from test_res_original_sol_identity import snapshot, assert_untouched


def setup(bare, source, tmp_path, *, mode="disabled", config=None):
    original = setup_res_history_sampling(bare, source, **options(tmp_path, mode=mode,
        confirm_checkpoint_write=mode == "checkpoint"))
    bound = effects.bind_effects(*original[:3], source)
    assert bound[1] is original[1] and bound[2] is original[2]
    assert bound[0].model is original[0].model
    if config is None:
        config = EAVConfig("report_only", tau=.2, start_video_progress=0.,
                           end_video_progress=1., g_hard_limit=3.)
    applied, runtime, _ = apply_stage_eav(bound[0], bound[2], source, bound[3], config)
    return applied, bound[1], bound[2], runtime, bound


def sample(selected, source, positive):
    return sample_res_stage(RandomNoise.execute(42).result[0],
        BasicGuider.execute(selected[0], positive).result[0], selected[1], selected[2], source)


def assert_equal(first, second):
    for a, b in zip(first[:2], second[:2], strict=True):
        for av_a, av_b in zip(a["samples"].unbind(), b["samples"].unbind(), strict=True):
            assert torch.equal(av_a, av_b)


def test_RES_external_report_only_actual_full8_is_exact_and_completion_freezes(tmp_path):
    bare, source = model(), latent()
    plain = setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    reference = sample(plain, source, conditioning())
    selected = setup(bare, source, tmp_path)
    actual = sample(selected, source, conditioning())
    assert_equal(reference, actual)
    report = json.loads(audit_stage_eav(actual[0], selected[3])[1])
    assert report["status"] == "observed_report_only"
    assert report["completed_forwards"] == report["selector_calls"] == 8
    assert report["clock_match"] and report["planned_integrator_intervals"] == 8
    assert actual[2].verify()["portable_identity"]
    path, digest, _ = save_stage(actual[2], tmp_path / "completed", "RES_effect")
    assert_equal(actual, load_stage(tmp_path / "completed", path, digest, "res_complete"))
    assert not bare.object_patches and not bare.wrappers and not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_RES_paired_external_Relay_EAV_actual_checkpoint_remaining4_is_exact(tmp_path, mode):
    bare, source = model(), latent()
    bound, positive, plain = paired(bare)
    config = EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.)
    first = setup(bound, source, tmp_path, mode="checkpoint", config=config)
    complete = sample(first, source, positive)
    checkpoint = history.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")
    assert complete[2].verify()["portable_identity"]
    second = setup(bound, source, tmp_path, mode="resume", config=config)
    remaining = sample(second, source, positive)
    assert_equal(complete, remaining)
    report = json.loads(audit_stage_eav(remaining[0], second[3])[1])
    assert report["status"] == ("observed_report_only" if mode == "report_only" else "observed_apply_exp")
    assert report["completed_forwards"] == report["selector_calls"] == report["relay_attention_calls"] == 4
    assert report["clock_match"] and report["planned_integrator_intervals"] == 4
    assert report["observed_absolute_sigmas"] == pytest.approx(second[4][3].trajectory_sigmas[4:8])
    assert remaining[2].verify()["portable_identity"]
    assert history.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == checkpoint["file_sha256"]
    with pytest.raises(ValueError, match="conditioning contract changed"):
        sample(second, source, plain)
    assert bound.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    assert not bare.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    assert not torch.cuda.is_initialized()


def test_RES_effect_descriptor_config_and_unknown_delegate_are_not_silently_reused(tmp_path):
    bare, source = model(), latent()
    selected = setup(bare, source, tmp_path, mode="checkpoint")
    sample(selected, source, conditioning())
    changed = setup(bare, source, tmp_path, mode="resume", config=EAVConfig("apply_exp", tau=.3))
    with pytest.raises(ValueError, match="contract|identity"):
        sample(changed, source, conditioning())
    owner = effects.capture_owner(selected[4][0])
    broken = selected[4][0].clone()
    broken.set_attachments(effects.KEY, replace(owner, context=replace(owner.context, audio_shift=4.)))
    with pytest.raises(ValueError, match="context|coordinates|trajectory"):
        effects.capture_owner(broken)
    calls = []
    def foreign(func, *args, **kwargs):
        calls.append(1)
        return func(*args, **kwargs)
    custom = bare.clone()
    custom.model_options["transformer_options"]["optimized_attention_override"] = foreign
    unknown = setup(custom, source, tmp_path)
    actual = sample(unknown, source, conditioning())
    assert len(calls) == 8 and actual[2].verify()["portable_identity"] is False
    assert unknown[4][0].model_options["transformer_options"]["optimized_attention_override"] is foreign
    assert not loaded_model_identity(unknown[0])["portable_cache_reuse"]
    assert not torch.cuda.is_initialized()


def test_RES_Relay_original_Sol_KJ_FFN_identity_is_source_bound_without_execution(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    bare = selected_original_sol(module, tmp_path)
    from comfy.ldm.minimax.model import PackedLayout
    original_constructor = PackedLayout.__init__
    bound, _, _ = paired(bare)
    assert PackedLayout.__init__ is original_constructor
    source = latent()
    original = setup_res_history_sampling(bound, source, **options(tmp_path, mode="disabled"))
    effect = effects.bind_effects(*original[:3], source)
    applied, _, _ = apply_stage_eav(effect[0], effect[2], source, effect[3], EAVConfig())
    before = snapshot(applied)
    first = loaded_model_identity(applied)
    assert first["automatic_loaded_weights_verified"]
    assert first["scoped_RES_history_content"]["content_reconstruction_verified"]
    assert not first["portable_cache_reuse"]
    assert first["RES_external_effects"]["effects"]["relay"]["selected_delegate"]["kind"] == "source_bound_original_Sol_Relay_delegate"
    assert first["RES_external_effects"]["effects"]["relay"]["selected_delegate"]["RES_memory_effects"]["actual_paths"]
    assert loaded_model_identity(applied) == first
    assert_untouched(applied, before)
    assert not torch.cuda.is_initialized()


def test_RES_memory_delegate_keeps_actual_bias_and_does_not_call_unmasked_kernel_on_CPU():
    from comfy.ldm.modules import attention
    from h3_audio_t8_pkg.res_memory_effects import RESMemoryDelegate
    from h3_audio_t8_pkg.relay_sol_backend import UserSelectedBackend
    calls = []
    def selected(func, q, k, v, heads, **kwargs):
        calls.append(kwargs["mask"])
        return attention.attention_pytorch(q, k, v, heads, **{**kwargs, "_inside_attn_wrapper": True})
    def unmasked_kernel(*args):
        raise AssertionError("unmasked CUDA-only kernel must not consume a bias or CPU input")
    backend = RESMemoryDelegate(UserSelectedBackend(selected), unmasked_kernel, "CPU_behavior_fixture_not_source_qualification")
    generator = torch.Generator().manual_seed(829)
    q, k, v = (torch.randn(1, 3, 11, 8, generator=generator) for _ in range(3))
    bias = torch.zeros(1, 1, 11, 11)
    bias[..., 2:6] = -2.5
    k_before = k.clone()
    actual = backend.attention(q, k, v, 3, mask=bias, skip_reshape=True)
    expected = attention.attention_pytorch(q, k, v, 3, mask=bias, skip_reshape=True, _inside_attn_wrapper=True)
    assert torch.equal(actual, expected) and torch.equal(k, k_before) and calls == [bias]
    assert backend.counters["selected_delegate:biased"] == 1 and not torch.cuda.is_initialized()


def test_RES_standalone_EAV_original_KJ_Sol_identity_runtime_and_disabled_unchanged(actual_source_chain, tmp_path):  # noqa: F811
    from h3_audio_t8_pkg import res_eav_memory, res_memory_effects
    from h3_audio_t8_pkg.relay_kj_memory import MEMORY_TOKEN_KEY
    from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
    module, _, _ = actual_source_chain
    bare = selected_original_sol(module, tmp_path)
    before = snapshot(bare)
    source = latent()
    original = setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    bound = effects.bind_effects(*original[:3], source)
    disabled, _, _ = apply_stage_eav(bound[0], bound[2], source, bound[3], EAVConfig("disabled"))
    assert disabled is bound[0] and disabled.get_attachment(res_eav_memory.KEY) is None
    applied, _, _ = apply_stage_eav(bound[0], bound[2], source, bound[3], EAVConfig("apply_exp"))
    assert not applied.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    owner = applied.get_attachment(res_eav_memory.KEY)
    assert owner is not None and applied.get_attachment(res_memory_effects.KEY) is not None
    route = {"seq_len": 1234}
    res_eav_memory.bind_runtime(applied, route)
    assert route == {"seq_len": 1234, MEMORY_TOKEN_KEY: owner["backend"].runtime_token}
    state_before = snapshot(applied)
    actual = loaded_model_identity(applied)
    assert actual["automatic_loaded_weights_verified"]
    assert actual["scoped_RES_history_content"]["content_reconstruction_verified"]
    assert actual["RES_external_effects"]["effects"]["eav"]["mode"] == "apply_exp"
    assert actual["RES_external_effects"]["effects"]["standalone_EAV_memory"]["synthetic_Relay_installed"] is False
    assert loaded_model_identity(applied) == actual
    assert_untouched(applied, state_before)
    assert_untouched(bare, before)
    changed = applied.clone()
    changed_owner = dict(owner)
    changed_owner["selector"] = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not invoke a foreign selector"))
    changed.set_attachments(res_eav_memory.KEY, changed_owner)
    with pytest.raises(UnverifiedModelStack, match="replaced"):
        res_eav_memory.bind_runtime(changed, {})
    class Foreign:
        @property
        def override(self):
            raise AssertionError("inspection must not invoke a foreign property")
    original_delegate = owner["backend"].delegate.original
    try:
        owner["backend"].delegate.original = Foreign()
        with pytest.raises(UnverifiedModelStack, match="source-bound"):
            res_eav_memory.bind_runtime(applied, {})
    finally:
        owner["backend"].delegate.original = original_delegate
    unknown = bare.clone()
    unknown.model_options["transformer_options"]["optimized_attention_override"] = changed_owner["selector"]
    assert res_eav_memory.prepare(unknown) is unknown
    assert not torch.cuda.is_initialized()


def test_foreign_packed_constructor_and_selected_Relay_delegate_are_not_executed(monkeypatch):
    from comfy.ldm.minimax.model import PackedLayout
    from h3_audio_t8_pkg.sla_attention_advanced import _required_parameters
    from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
    from h3_audio_t8_pkg.res_relay_identity import project_original_sol_delegate
    from h3_audio_t8_pkg.relay_sol_backend import UserSelectedBackend
    calls = []
    def foreign(*args, **kwargs):
        calls.append(1)
        raise AssertionError("inspection must not execute this foreign callable")
    monkeypatch.setattr(PackedLayout, "__init__", foreign)
    with pytest.raises(RuntimeError, match="lost PackedLayout parameters"):
        _required_parameters(foreign, {"keyframes", "refs"}, "PackedLayout parameters")
    with pytest.raises(UnverifiedModelStack):
        project_original_sol_delegate(model(), UserSelectedBackend(foreign))
    assert PackedLayout.__init__ is foreign and not calls and not torch.cuda.is_initialized()
