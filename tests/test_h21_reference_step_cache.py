"""Actual random-weight tiny Core CPU sampling, not trained/GPU qualification."""
from copy import deepcopy
import json

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import reference_step_cache as primitive
from h3_audio_t8_pkg import reference_step_cache_advanced as feature
from h3_audio_t8_pkg.nodes_reference_step_cache import MiniMaxH3ReferenceStepCacheEXPT8 as Apply
from h3_audio_t8_pkg.nodes_reference_step_cache import MiniMaxH3ReferenceStepCacheAuditEXPT8 as Audit
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_h21_reference_temporal_pack import positive
from test_progressive_sampling_runtime import latent, conditioning


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def sample(selected, pos, source=None, steps=4, **sampler_options):
    source = latent() if source is None else source
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, steps, 12., 3., "dual_clock_euler", "native_flow")
    sampler.extra_options.update(sampler_options)
    guider = BasicGuider.execute(prepared, pos).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(211025).result[0], guider,
                                         sampler, sigmas, source).result[0]


def equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a["samples"].unbind(), b["samples"].unbind(), strict=True))


def two_layers():
    selected = model()
    # A genuine second Core DiTBlock, not a simulated residual/cache kernel.
    # One block's ref input is the fixed embedding; after its actual attention
    # a second block observes target-dependent reference states.
    selected.model.diffusion_model.blocks.append(deepcopy(selected.model.diffusion_model.blocks[0]))
    return selected


def test_off_identity_and_interval1_are_real_full_recompute_exact():
    bare, pos = model(), positive()
    off, same, report = feature.configure_reference_step_cache(bare, pos)
    assert off is bare and same is pos and report["status"] == "disabled_identity"
    expected = sample(bare, pos)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 1)
    assert equal(sample(selected, paired), expected)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 4
    assert report["full_block_evaluations"] == 4 and report["approximate_block_hits"] == 0
    assert report["cleanup_observed"] and report["cached_bytes_after_request"] == 0
    assert not bare.wrappers and not bare.callbacks and not bare.attachments


@pytest.mark.parametrize("include_text", [False, True])
def test_real_four_evaluations_hit_only_steps1_3_measured_nonzero_error_and_clean_next_request(include_text, request):
    bare, pos = two_layers(), positive()
    original = deepcopy(pos)
    baseline = sample(bare, pos)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 2, include_text)
    output = sample(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 4
    assert report["approximate_block_hits"] == 4 and report["full_block_evaluations"] == 4
    assert [e["logical_step"] for e in report["evaluations"]] == [0, 1, 2, 3]
    assert [e["blocks"][0]["decision"] for e in report["evaluations"]] == [
        "full_refresh", "approximate_hit", "full_refresh", "approximate_hit"]
    assert not report["partial"] and report["peak_CPU_cache_bytes"] > 0
    assert report["reference_query_outproj_MLP_rows_skipped"] > 0
    assert report["cached_bytes_after_request"] == 0 and not report["request_active"]
    assert not report["portable_cache_identity"] and not report["bit_exact_claimed"]
    errors = [(x-y).abs().max().item() for x, y in zip(output["samples"].unbind(), baseline["samples"].unbind(), strict=True)]
    rms = [torch.mean((x.double()-y.double()).square()).sqrt().item()
           for x, y in zip(output["samples"].unbind(), baseline["samples"].unbind(), strict=True)]
    request.node.user_properties.append(("actual_tiny_error", json.dumps({"video_audio_max_abs": errors,
        "video_audio_RMS": rms, "approximate_hits": report["approximate_block_hits"],
        "full_blocks": report["full_block_evaluations"], "peak_CPU_cache_bytes": report["peak_CPU_cache_bytes"]})))
    assert all(torch.isfinite(x).all() for x in output["samples"].unbind())
    assert max(errors) > 0  # Explicit approximate algorithm, not metadata-only no-op.
    assert equal(sample(selected, paired), output)
    assert feature.audit_reference_step_cache(selected)["request_number"] == 2
    assert equal(sample(bare, pos), baseline)
    for before, after in zip(original[0][1]["minimax_refs"], pos[0][1]["minimax_refs"], strict=True):
        if before.get("latent") is not None:
            assert torch.equal(before["latent"], after["latent"])


def test_phase_and_geometry_are_independent_on_one_shared_network():
    from comfy.nested_tensor import NestedTensor
    bare = model()
    low, lp, _ = feature.configure_reference_step_cache(bare, positive(), "approximate", phase="LOW")
    hp = positive()
    hp[0][1]["minimax_keyframes"][0]["latent"] = torch.ones(1, 24, 1, 8, 12) * .2
    high, hp, _ = feature.configure_reference_step_cache(bare, hp, "approximate", phase="HIGH")
    high_source = {"samples": NestedTensor((torch.zeros(1, 24, 2, 8, 12), latent()["samples"].unbind()[1]))}
    sample(low, lp)
    sample(high, hp, high_source)
    a, b = feature.audit_reference_step_cache(low), feature.audit_reference_step_cache(high)
    assert a["binding_sha256"] != b["binding_sha256"]
    assert a["phase"] == "LOW" and b["phase"] == "HIGH"
    assert a["evaluations"][0]["blocks"][0]["packed_rows"] < b["evaluations"][0]["blocks"][0]["packed_rows"]
    assert a["approximate_block_hits"] == b["approximate_block_hits"] == 2
    sample(low, lp)
    assert feature.audit_reference_step_cache(low)["request_number"] == 2
    assert feature.audit_reference_step_cache(high)["request_number"] == 1


def test_selected_pair_and_known_condition_content_mutation_are_real_errors():
    bare = model()
    a, ap, _ = feature.configure_reference_step_cache(bare, positive(), "approximate", phase="LOW")
    b, bp, _ = feature.configure_reference_step_cache(bare, positive(), "approximate", phase="HIGH")
    with pytest.raises(ValueError, match="pair binding"):
        sample(a, bp)
    bp[0][1]["minimax_refs"][0]["audio_latent"].add_(.01)
    with pytest.raises(ValueError, match="conditioning content SHA"):
        sample(b, bp)
    assert all(torch.isfinite(x).all() for x in sample(a, ap)["samples"].unbind())


@pytest.mark.parametrize("target", ["embedding", "audio", "keyframe", "weight_data"])
def test_last_forward_content_changes_cannot_publish_success(target):
    bare, pos, count = model(), positive(), []
    pos[0][1]["opaque"] = object()
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    def mutate(*_):
        count.append(1)
        if len(count) == 4:
            tensor = {"embedding": paired[0][0], "audio": paired[0][1]["minimax_refs"][0]["audio_latent"],
                      "keyframe": paired[0][1]["minimax_keyframes"][0]["latent"],
                      "weight_data": bare.model.diffusion_model.final_layer.video_out.weight.data}[target]
            tensor.add_(.01)
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match="content SHA changed"):
            sample(selected, paired)
    finally:
        handle.remove()
    report = feature.audit_reference_step_cache(selected) if target == "weight_data" else None
    if report is not None:
        assert report["failed_model_evaluations"] == 1 and report["cached_bytes_after_request"] == 0


def test_foreign_block_owner_and_real_exception_delegate_once_then_next_request():
    bare, pos, calls, state = model(), positive(), [], {"fail": True}
    def foreign(args, extra):
        calls.append(1)
        if state["fail"]:
            raise TypeError("foreign actual kernel")
        return extra["original_block"](args)
    bare.set_model_patch_replace(foreign, "dit", "double_block", 0)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    with pytest.raises(TypeError, match="foreign actual kernel"):
        sample(selected, paired)
    assert calls == [1]
    assert not feature.audit_reference_step_cache(selected)["request_active"]
    state["fail"] = False
    output = sample(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert len(calls) == 5 and report["approximate_block_hits"] == 0 and report["partial"]
    assert "existing_block_owner_original_delegated" in report["reasons"]
    assert all(torch.isfinite(x).all() for x in output["samples"].unbind())
    assert bare.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is foreign


def test_unknown_forward_executes_without_model_name_ban_and_reports_no_hits():
    bare, calls = model(), []
    old = bare.model.diffusion_model._forward
    def unknown(*args, **kwargs):
        calls.append(1)
        return old(*args, **kwargs)
    bare.model.diffusion_model._forward = unknown
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), "approximate")
    output = sample(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert len(calls) == 4 and report["partial"] and report["approximate_block_hits"] == 0
    assert all(torch.isfinite(x).all() for x in output["samples"].unbind())
    assert bare.model.diffusion_model._forward is unknown


def test_unknown_model_default_and_explicit_observation_delegate_original():
    foreign, pos = object(), positive()
    result, output, report = feature.configure_reference_step_cache(foreign, pos, "approximate")
    assert result is foreign and output is pos and not report["portable_cache_identity"]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, float("nan")])
def test_interval_actual_input_contract_not_arbitrary_cap(bad):
    with pytest.raises(ValueError, match="positive logical-step"):
        primitive.ReferenceStepPlan(bad)
    assert primitive.ReferenceStepPlan(100000).refresh_every == 100000


def test_no_reference_and_registered_block_hooks_keep_full_original_computation():
    bare, pos = model(), conditioning()
    baseline = sample(bare, pos)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    assert equal(sample(selected, paired), baseline)
    assert feature.audit_reference_step_cache(selected)["approximate_block_hits"] == 0
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), "approximate")
    calls = []
    handle = bare.model.diffusion_model.blocks[0].mlp.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        sample(selected, paired)
    finally:
        handle.remove()
    assert len(calls) == 4
    assert feature.audit_reference_step_cache(selected)["approximate_block_hits"] == 0


def test_actual_schema_defaults_paired_outputs_and_audit_depend_on_sampler():
    schema = Apply.define_schema().get_v1_info(Apply)
    assert schema.input["required"]["mode"][1]["default"] == "off"
    selected, paired, raw = Apply.execute(model(), positive(), "approximate", 2, False, "SINGLE").result
    assert json.loads(raw)["completed_model_evaluations"] == 0
    actual = sample(selected, paired)
    report = json.loads(Audit.execute(selected, actual).result[0])
    assert report["approximate_block_hits"] == 2 and report["cached_bytes_after_request"] == 0
    assert Audit.define_schema().is_output_node


def test_repeated_sigma_real_core_evaluations_do_not_advance_refresh_or_replace_cache():
    import comfy.samplers
    from comfy.k_diffusion.sampling import sample_euler
    bare, pos, source = two_layers(), positive(), latent()
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    prepared, _, sigmas = setup_dual_clock_sampling(selected, source, 4, 12., 3., "dual_clock_euler", "native_flow")
    calls = []
    def twice(net, x, sigmas, extra_args=None, callback=None, disable=None):
        def evaluate(value, sigma, **kwargs):
            first = net(value, sigma, **kwargs)
            second = net(value, sigma, **kwargs)
            calls.append((sigma.clone(), [torch.isfinite(item).all().item() for item in second.unbind()]))
            return first
        return sample_euler(evaluate, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    guider = BasicGuider.execute(prepared, paired).result[0]
    output = SamplerCustomAdvanced.execute(RandomNoise.execute(211025).result[0], guider,
        comfy.samplers.KSAMPLER(twice), sigmas, source).result[0]
    report = feature.audit_reference_step_cache(selected)
    assert len(calls) == 4 and report["completed_model_evaluations"] == 8
    assert [e["logical_step"] for e in report["evaluations"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [e["blocks"][0]["decision"] for e in report["evaluations"]][1::2] == ["same_sigma_full"] * 4
    assert report["approximate_block_hits"] == 4 and report["full_block_evaluations"] == 12
    assert all(torch.isfinite(item).all() for item in output["samples"].unbind())


def test_two_selected_positive_rows_have_independent_real_context_cache_buckets():
    bare, pos = two_layers(), positive()
    second = deepcopy(pos[0])
    second[0] = second[0] + .2
    second[1]["strength"] = .4
    pos.append(second)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    output = sample(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 8
    assert report["approximate_block_hits"] == 8 and report["full_block_evaluations"] == 8
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())


def test_real_soft_video_mask_per_row_modulation_is_executed_and_audio_unchanged_by_configuration():
    source, bare, pos = latent(), two_layers(), positive()
    # Actual Core broadcasts a video-only mask; audio remains normal denoising.
    source["noise_mask"] = torch.linspace(.1, 1., 2 * 4 * 8).reshape(1, 1, 2, 4, 8)
    original = source["noise_mask"].clone()
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 1)
    assert equal(sample(selected, paired, source), sample(bare, pos, source))
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 2)
    result = sample(selected, paired, source)
    assert feature.audit_reference_step_cache(selected)["approximate_block_hits"] == 4
    assert torch.equal(source["noise_mask"], original)
    assert all(torch.isfinite(item).all() for item in result["samples"].unbind())


def test_real_cache_buffer_corruption_fails_then_fresh_request_does_not_reuse_it():
    bare, calls = two_layers(), []
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), "approximate")
    runtime = selected.get_attachment(feature.KEY)
    def corrupt(*_):
        calls.append(1)
        if len(calls) == 1:
            next(iter(runtime.buckets.values()))["blocks"][0][0].add_(1)
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(corrupt)
    try:
        with pytest.raises(ValueError, match="stored reference input SHA changed"):
            sample(selected, paired)
    finally:
        handle.remove()
    assert feature.audit_reference_step_cache(selected)["cached_bytes_after_request"] == 0
    assert all(torch.isfinite(item).all() for item in sample(selected, paired)["samples"].unbind())
    assert feature.audit_reference_step_cache(selected)["request_number"] == 2


def test_actual_interrupt_cleans_cache_without_retry_and_restores_next_plain_request():
    import comfy.model_management
    bare, pos, calls = two_layers(), positive(), []
    baseline = sample(bare, pos)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    def cancel(*_):
        calls.append(1)
        raise comfy.model_management.InterruptProcessingException()
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(cancel)
    try:
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            sample(selected, paired)
    finally:
        handle.remove()
    assert calls == [1]
    report = feature.audit_reference_step_cache(selected)
    assert report["failed_model_evaluations"] == 1 and report["cleanup_observed"]
    assert report["cached_bytes_after_request"] == 0
    assert equal(sample(bare, pos), baseline)
    sample(selected, paired)
    assert feature.audit_reference_step_cache(selected)["failed_model_evaluations"] == 0


def test_reselection_changes_only_own_slots_and_preserves_same_key_foreign_callback_order():
    bare, pos, callbacks = two_layers(), positive(), []
    first, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 2)
    from comfy.patcher_extension import CallbacksMP
    def foreign(_):
        callbacks.append(1)
    first.add_callback_with_key(CallbacksMP.ON_PRE_RUN, feature.KEY, foreign)
    old_prepare = first.get_attachment(feature.KEY).prepare_callback
    selected, output, _ = feature.configure_reference_step_cache(first, paired, "approximate", 3)
    values = selected.callbacks[CallbacksMP.ON_PRE_RUN][feature.KEY]
    assert values[-1] is foreign and values[0] is selected.get_attachment(feature.KEY).prepare_callback
    assert first.callbacks[CallbacksMP.ON_PRE_RUN][feature.KEY][0] is old_prepare
    assert first.callbacks[CallbacksMP.ON_PRE_RUN][feature.KEY][-1] is foreign
    sample(selected, output)
    report = feature.audit_reference_step_cache(selected)
    assert callbacks == [1] and report["plan"]["refresh_every"] == 3
    assert report["approximate_block_hits"] == 4
    sample(first, paired)
    assert feature.audit_reference_step_cache(first)["plan"]["refresh_every"] == 2


@pytest.mark.parametrize("target", ["plan", "phase", "helper"])
def test_explicit_own_plan_phase_and_helper_mutation_on_final_forward_is_not_accepted(monkeypatch, target):
    bare, calls = two_layers(), []
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), "approximate")
    runtime = selected.get_attachment(feature.KEY)
    def mutate(*_):
        calls.append(1)
        if len(calls) == 4:
            if target == "helper":
                monkeypatch.setattr(primitive, "cached_block", lambda *_: None)
            elif target == "plan":
                runtime.plan = primitive.ReferenceStepPlan(3)
            else:
                runtime.phase = "HIGH"
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match="binding changed|execution helper changed"):
            sample(selected, paired)
    finally:
        handle.remove()


def test_real_ordered_multiple_lora_weights_are_used_unmodified_then_alloff_restores_baseline():
    import comfy.lora
    bare, pos = two_layers(), positive()
    baseline = sample(bare, pos)
    patched = bare.clone()
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = bare.model.state_dict()[target]
    generator = torch.Generator().manual_seed(26102105)
    for strength in (-.75, 1.25):
        tensors = {"actual.lora_A.weight": torch.randn(2, weight.shape[1], generator=generator) * .1,
                   "actual.lora_B.weight": torch.randn(weight.shape[0], 2, generator=generator) * .1}
        adapters = comfy.lora.load_lora(tensors, {"actual": target}, log_missing=False)
        assert target in adapters
        assert patched.add_patches(adapters, strength_patch=strength) == [target]
    original = list(patched.patches[target])
    expected = sample(patched, pos)
    selected, paired, _ = feature.configure_reference_step_cache(patched, pos, "approximate", 1)
    assert equal(sample(selected, paired), expected)
    assert [row[0] for row in selected.patches[target]] == [-.75, 1.25]
    assert all(a[1] is b[1] for a, b in zip(original, selected.patches[target], strict=True))
    assert not equal(expected, baseline)
    assert equal(sample(bare, pos), baseline)


def test_unknown_attention_and_norm_owner_execute_without_approximate_substitution(monkeypatch):
    bare, pos, calls = two_layers(), positive(), []
    original = native.optimized_attention
    def foreign_attention(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(native, "optimized_attention", foreign_attention)
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    sample(selected, paired)
    assert len(calls) == 8 and feature.audit_reference_step_cache(selected)["approximate_block_hits"] == 0
    assert "unknown_global_attention_owner_original_delegated" in feature.audit_reference_step_cache(selected)["reasons"]
    monkeypatch.setattr(native, "optimized_attention", original)
    original_norm = bare.model.diffusion_model.blocks[0].norm1.forward
    count = []
    def foreign_norm(value):
        count.append(1)
        return original_norm(value)
    monkeypatch.setattr(bare.model.diffusion_model.blocks[0].norm1, "forward", foreign_norm)
    sample(selected, paired)
    assert count == [1] * 4
    report = feature.audit_reference_step_cache(selected)
    assert report["approximate_block_hits"] == 2 and report["partial"]


def test_true_invalid_weight_nan_and_reference_nan_are_not_unknown_bypass():
    bare, pos = two_layers(), positive()
    pos[0][1]["minimax_refs"][0]["audio_latent"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        feature.configure_reference_step_cache(bare, pos, "approximate")
    pos = positive()
    bare.model.diffusion_model.blocks[0].mlp.fc1.weight[0, 0] = float("nan")
    selected, paired, _ = feature.configure_reference_step_cache(bare, pos, "approximate")
    with pytest.raises(ValueError, match="non-finite weights"):
        sample(selected, paired)


def test_actual_diffusion_predecessor_two_evaluations_not_one_legacy_wrapper_counter():
    from comfy.patcher_extension import WrappersMP
    bare, calls = two_layers(), []
    def previous(executor, *args, **kwargs):
        first = executor(*args, **kwargs)
        second = executor(*args, **kwargs)
        assert all(torch.isfinite(item).all() for item in second)
        calls.append(1)
        return first
    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, feature.KEY, previous)
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), "approximate")
    sample(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert calls == [1] * 4
    assert report["completed_model_evaluations"] == 8
    assert [event["logical_step"] for event in report["evaluations"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [event["blocks"][0]["decision"] for event in report["evaluations"]][1::2] == ["same_sigma_full"] * 4


def test_actual_native_stage_can_execute_but_does_not_invent_portable_completion():
    from h3_audio_t8_pkg.modular_sampling import native_explicit
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity
    source = latent()
    selected, paired, _ = feature.configure_reference_step_cache(two_layers(), positive(), "approximate")
    prepared, sampler, sigmas = setup_dual_clock_sampling(selected, source, 4, 12., 3.)
    prepared, sampler, sigmas, context, _ = native_explicit.bind_stage(prepared, sampler, sigmas, source)
    assert selected_model_identity(prepared)["portable_cache_reuse"] is False
    guider = BasicGuider.execute(prepared, paired).result[0]
    result, _, stage, _ = sample_stage(RandomNoise.execute(211025).result[0], guider, sampler, sigmas, source, context)
    receipt = stage.verify()
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert receipt["portable_identity"] is False
    assert feature.audit_reference_step_cache(selected)["approximate_block_hits"] == 4
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())


def test_actual_relay_keeps_its_mask_and_executes_without_cache_substitution(monkeypatch):
    from test_progressive_relay import paired as make_relay
    from h3_audio_t8_pkg import prompt_relay_advanced as relay
    bare, pos, _ = make_relay(two_layers(), "i2va")
    before = deepcopy(relay.prompt_relay_model_contract(bare)["binding"])
    baseline = sample(bare, pos)
    before_counts = relay.prompt_relay_model_contract(bare)["execution_counts"]
    assert before_counts["routed_attention_calls"] == 8
    selected, output, _ = feature.configure_reference_step_cache(bare, pos, "approximate", 2, True)
    actual = sample(selected, output)
    assert equal(actual, baseline)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 4 and report["approximate_block_hits"] == 0
    assert report["partial"] and "unknown_attention_owner_original_delegated" in report["reasons"]
    after_counts = relay.prompt_relay_model_contract(selected)["execution_counts"]
    assert after_counts["routed_attention_calls"] - before_counts["routed_attention_calls"] == 8
    assert after_counts["completed_forwards"] - before_counts["completed_forwards"] == 4
    assert relay.prompt_relay_model_contract(selected)["binding"] == before


def test_actual_eav_attention_owner_remains_effective_and_is_not_called_plain_attention():
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    source, bare = latent(), two_layers()
    prepared, _, sigmas = setup_dual_clock_sampling(bare, source, 8, 12., 3.)
    enhanced, runtime, _ = eav.build_eav_model(prepared, sigmas, mode="apply_exp", tau=.2,
        start_video_progress=0., end_video_progress=1., max_workspace_mib=4, g_hard_limit=2.,
        sampling_profile="progressive_native8_exp", allowed_tasks=("Hybrid",), allow_reference_blocks=True,
        composer_profile="progressive_native_av_v1")
    selected, paired, _ = feature.configure_reference_step_cache(enhanced, positive(), "approximate")
    result = sample(selected, paired, steps=8)
    report = feature.audit_reference_step_cache(selected)
    observed = runtime.snapshot(consume=False)
    assert observed["model_forward_count"] == 8 and observed["active_forward_count"] == 8
    assert observed["attention_calls_per_active_forward"] == [2] * 8
    assert report["completed_model_evaluations"] == 8 and report["approximate_block_hits"] == 0
    assert report["partial"] and "unknown_attention_owner_original_delegated" in report["reasons"]
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())


def test_actual_core_api_graph_accepts_paired_apply_guider_sampler_and_output_audit(monkeypatch):
    import asyncio
    import execution
    import nodes
    from comfy_api.latest import io
    class Inputs(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="H21ReferenceStepTinyInputs", inputs=[], outputs=[
                io.Model.Output("model"), io.Conditioning.Output("positive"), io.Latent.Output("latent"),
                io.Custom("NOISE").Output("noise"), io.Custom("SAMPLER").Output("sampler"), io.Custom("SIGMAS").Output("sigmas")])
        @classmethod
        def execute(cls):
            source = latent()
            prepared, sampler, sigmas = setup_dual_clock_sampling(two_layers(), source, 4, 12., 3.)
            return io.NodeOutput(prepared, positive(), source, RandomNoise.execute(211025).result[0], sampler, sigmas)
    for cls in (Inputs, Apply, Audit, BasicGuider, SamplerCustomAdvanced):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    graph = {
        "1": {"class_type": "H21ReferenceStepTinyInputs", "inputs": {}},
        "2": {"class_type": Apply.__name__, "inputs": {"model": ["1", 0], "positive": ["1", 1],
            "mode": "approximate", "refresh_every": 2, "include_text": False, "phase": "SINGLE"}},
        "3": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["2", 1]}},
        "4": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["1", 3], "guider": ["3", 0],
            "sampler": ["1", 4], "sigmas": ["1", 5], "latent_image": ["1", 2]}},
        "5": {"class_type": Audit.__name__, "inputs": {"model": ["2", 0], "after_sampling": ["4", 0]}},
    }
    result = asyncio.run(execution.validate_prompt("h21-reference-step-cache", graph, None))
    assert result[0], result[1:]
    assert "5" in result[2]


def test_native_source_contract_authenticates_real_current_core_attention():
    from pathlib import Path
    from h3_audio_t8_pkg.source_code_identity import _FIELDS
    path = str(Path(attention.__file__).resolve())
    stat = Path(path).stat()
    actual = attention.attention_pytorch.__code__
    expected = primitive._source_codes(path, stat.st_mtime_ns, stat.st_size)[actual.co_qualname]
    differences = {key: (getattr(actual, key), getattr(expected, key)) for key in _FIELDS
                   if getattr(actual, key) != getattr(expected, key)}
    assert primitive.native_attention_backend(), differences


def test_actual_cfg_guider_unpaired_negative_is_full_recomputed_not_an_admission_ban():
    from comfy_extras.nodes_custom_sampler import CFGGuider
    base, pos, source = two_layers(), positive(), latent()
    negative = conditioning()
    negative[0][0].add_(.07)
    def cfg(selected, positive):
        prepared, sampler, sigmas = setup_dual_clock_sampling(selected, source, 4, 12., 3.)
        guider = CFGGuider.execute(prepared, positive, negative, 2.).result[0]
        return SamplerCustomAdvanced.execute(RandomNoise.execute(211025).result[0], guider,
            sampler, sigmas, source).result[0]
    selected, paired, _ = feature.configure_reference_step_cache(base, pos, "approximate", 1)
    expected, actual = cfg(base, pos), cfg(selected, paired)
    assert equal(actual, expected)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 8 and report["approximate_block_hits"] == 0
    assert report["completed_paired_model_evaluations"] == 4
    assert report["completed_unpaired_negative_evaluations"] == 4
    assert report["partial"] and "unpaired_negative_full_recompute" in report["reasons"]
    selected, paired, _ = feature.configure_reference_step_cache(base, pos, "approximate", 2)
    actual = cfg(selected, paired)
    report = feature.audit_reference_step_cache(selected)
    assert report["completed_model_evaluations"] == 8 and report["approximate_block_hits"] == 4
    assert report["completed_paired_model_evaluations"] == 4
    assert report["completed_unpaired_negative_evaluations"] == 4
    assert report["full_block_evaluations"] == 12
    assert all(torch.isfinite(value).all() for value in actual["samples"].unbind())


def test_unknown_callable_function_metadata_never_runs_user_equality_to_classify_it():
    base, pos = model(), positive()
    norm = base.model.diffusion_model.blocks[0].norm1
    original = norm.forward
    class CallableOwner:
        @property
        def __func__(self):
            return self
        def __eq__(self, other):
            raise AssertionError("user equality must not be invoked")
        def __call__(self, value):
            calls.append(value.shape[0])
            return original(value) + value.mean() * .125
    calls = []
    owner = CallableOwner()
    norm.forward = owner
    expected = sample(base, pos)
    original_calls = list(calls)
    calls.clear()
    selected, paired, _ = feature.configure_reference_step_cache(base, pos, "approximate")
    actual = sample(selected, paired)
    assert norm.forward is owner and calls == original_calls
    assert equal(expected, actual)
    report = feature.audit_reference_step_cache(selected)
    assert report["partial"] and report["approximate_block_hits"] == 0


def test_actual_finite_float64_weight_and_matmul_do_not_become_false_nonfinite_via_float32():
    linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    linear.weight.fill_(1e200)
    result = linear(torch.full((1, 2), 1e-200, dtype=torch.float64))
    assert torch.isfinite(result).all() and torch.equal(result, torch.full_like(result, 2.))
    assert len(primitive.weight_digest(linear)) == 64
