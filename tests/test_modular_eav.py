"""Real tiny Core stage effects. No pre-trained/GPU qualification is implied."""
from dataclasses import replace
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent
from test_progressive_relay import paired


def sample(prepared, sampler, sigmas, source, positive=None):
    positive = conditioning() if positive is None else positive
    guider = BasicGuider.execute(prepared, positive).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(123).result[0], guider, sampler, sigmas, source).result


def assert_equal(actual, expected):
    for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_actual_stage_forwards_absolute_clock_and_effect_math(stage, mode):
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, stage, "dense_compat_exp")
    original = sample(prepared, sampler, sigmas, source)[0]
    config = EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.)
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, config)
    actual = sample(applied, sampler, sigmas, source)[0]
    _, raw = audit_stage_eav(actual, runtime)
    report = json.loads(raw)
    if mode == "disabled":
        assert applied is prepared
        assert report["status"] == "disabled_identity"
    else:
        assert report["completed_forwards"] == 4
        assert report["selector_calls"] == 4
        assert report["clock_match"] is True
        assert report["feta"]["attention_measurement_count"] == 4
        assert report["status"] == ("observed_report_only" if mode == "report_only" else "observed_apply_exp")
        expected = list(context.trajectory_sigmas[context.start:context.end])
        assert report["observed_absolute_sigmas"] == pytest.approx(expected, abs=2e-6)
        # Model reuse resets the telemetry, not an accumulating 8-forward fiction.
        sample(applied, sampler, sigmas, source)
        assert runtime.snapshot()["completed_forwards"] == 4
    if mode != "apply_exp":
        assert_equal(actual, original)
    else:
        assert not torch.equal(actual["samples"].unbind()[0], original["samples"].unbind()[0])
    assert not torch.cuda.is_initialized()


def test_default_window_is_not_reset_at_each_stage():
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, "low_0_4", "dense_compat_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    sample(applied, sampler, sigmas, source)
    report = runtime.snapshot()
    assert report["status"] == "observed_no_steps_in_effect_window"
    assert report["feta"]["active_forward_count"] == 0


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_relay_and_eav_share_one_composed_owner_and_keep_real_bias(monkeypatch, stage, mode):
    bare, source = model(), latent()
    bound, positive, plain = paired(bare)
    prepared, sampler, sigmas, context, _ = build_stage(bound, source, stage, "dense_compat_exp")
    original = sample(prepared, sampler, sigmas, source, positive)[0]
    calls = []
    previous = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
        return previous(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context,
        EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    assert not applied.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    assert prepared.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    actual = sample(applied, sampler, sigmas, source, positive)[0]
    if mode == "report_only":
        assert_equal(actual, original)
    else:
        assert not torch.equal(actual["samples"].unbind()[0], original["samples"].unbind()[0])
    assert len(calls) == 4 and len(set(calls)) == 1
    assert runtime.snapshot()["relay_attention_calls"] == 4
    assert runtime.snapshot()["status"] == ("observed_report_only" if mode == "report_only" else "observed_apply_exp")
    with pytest.raises(RuntimeError, match="binding hashes differ"):
        sample(applied, sampler, sigmas, source, plain)


def test_native_video_and_audio_masks_are_audited_without_changing_them():
    source = latent()
    video, audio = source["samples"].unbind()
    vm, am = torch.ones_like(video), torch.ones_like(audio)
    vm[:, :, :1] = 0
    am[..., :4] = 0
    source["noise_mask"] = NestedTensor((vm, am))
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, "high_4_8", "dense_compat_exp")
    original = sample(prepared, sampler, sigmas, source)[0]
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context,
        EAVConfig(tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    actual = sample(applied, sampler, sigmas, source)[0]
    assert_equal(actual, original)
    assert source["noise_mask"].unbind()[0] is vm
    assert source["noise_mask"].unbind()[1] is am
    assert runtime.snapshot()["completed_forwards"] == 4


def test_unknown_attention_delegate_is_called_and_preserved():
    bare, source = model(), latent()
    calls = []

    def custom(func, *args, **kwargs):
        calls.append(True)
        return func(*args, **kwargs)

    bare.model_options["transformer_options"]["optimized_attention_override"] = custom
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, "high_4_8", "dense_compat_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    sample(applied, sampler, sigmas, source)
    assert len(calls) == 4
    assert bare.model_options["transformer_options"]["optimized_attention_override"] is custom
    assert prepared.model_options["transformer_options"]["optimized_attention_override"] is custom


def test_delegate_failure_is_recorded_and_retry_starts_a_new_run():
    bare, source = model(), latent()
    state = {"fail": True}

    def custom(func, *args, **kwargs):
        if state["fail"]:
            raise RuntimeError("intentional original backend failure")
        return func(*args, **kwargs)

    bare.model_options["transformer_options"]["optimized_attention_override"] = custom
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, "high_4_8", "dense_compat_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    with pytest.raises(RuntimeError, match="intentional original backend failure"):
        sample(applied, sampler, sigmas, source)
    with pytest.raises(RuntimeError, match="intentional original backend failure"):
        audit_stage_eav(source, runtime)
    assert runtime.snapshot()["status"] == "aborted"
    state["fail"] = False
    sample(applied, sampler, sigmas, source)
    assert runtime.snapshot()["status"] == "observed_report_only"
    assert runtime.snapshot()["completed_forwards"] == 4


def test_a_producer_bypassing_selector_is_preserved_and_cannot_claim_eav_success():
    bare, source = model(), latent()
    observed = []

    def producer(args, extra):
        observed.append(True)
        options = dict(args["transformer_options"])
        options.pop("optimized_attention_override", None)
        return extra["original_block"]({**args, "transformer_options": options})

    bare.set_model_patch_replace(producer, "dit", "double_block", 0)
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, "high_4_8", "dense_compat_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    sample(applied, sampler, sigmas, source)
    report = runtime.snapshot()
    assert len(observed) == 4
    assert report["completed_forwards"] == 4 and report["selector_calls"] == 0
    assert report["status"] == "unverified_incomplete_stage_coverage"
    assert applied.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is producer


def test_trained_vsa_owner_not_removed_or_replaced_with_dense():
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, "high_4_8", "trained_vsa_exp")
    original_dit = prepared.model_options["transformer_options"]["patches_replace"]["dit"].copy()
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    # Exact native producers are delegated through the bounded sparse adapter;
    # the source MODEL and its native eligibility/kernel remain untouched.
    assert prepared.model_options["transformer_options"]["patches_replace"]["dit"] == original_dit
    assert applied.model_options["transformer_options"]["patches_replace"]["dit"] != original_dit
    sample(applied, sampler, sigmas, source)
    report = runtime.snapshot()
    assert report["v2_dispatch"]["profile"] == "trained_vsa_exp"
    assert report["v2_dispatch"]["actual_vsa_dispatched"] is False  # CPU eligibility, not forced Dense.
    assert report["v2_dispatch"]["counts"]["dense"] == 4
    assert report["completed_forwards"] == 4


@pytest.mark.parametrize("mutation", ["sigmas", "shape", "stage", "profile", "shift"])
def test_wrong_stage_identity_is_not_treated_as_an_unverified_user_patch(mutation):
    source = latent()
    prepared, _, sigmas, context, _ = build_stage(model(), source, "low_0_4", "dense_compat_exp")
    if mutation == "sigmas":
        sigmas = sigmas.clone()
        sigmas[-1] = 0
    elif mutation == "shape":
        context = replace(context, video_shape=(1, 24, 2, 8, 8))
    elif mutation == "stage":
        context = replace(context, stage="high_4_8")
    elif mutation == "profile":
        context = replace(context, profile="trained_vsa_exp")
    else:
        context = replace(context, video_shift=12.)
    with pytest.raises(ValueError):
        apply_stage_eav(prepared, sigmas, source, context, EAVConfig())


def test_unsampled_runtime_is_not_success_and_both_stages_get_independent_tokens():
    source, bare, config = latent(), model(), EAVConfig()
    contexts = [build_stage(bare, source, stage, "dense_compat_exp") for stage in ("low_0_4", "high_4_8")]
    tokens = [apply_stage_eav(prepared, sigmas, source, context, config)[1]
              for prepared, _, sigmas, context, _ in contexts]
    assert tokens[0] is not tokens[1]
    assert tokens[0].config is tokens[1].config is config
    for token in tokens:
        assert token.snapshot()["status"] == "unverified_incomplete_stage_coverage"
        assert token.snapshot()["cache_reuse_authorized"] is False
