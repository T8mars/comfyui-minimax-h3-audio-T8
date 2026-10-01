"""Real Core ModelPatcher/wrapper delegation; no global load or sampler patch."""
import asyncio
from copy import deepcopy
import json

import pytest
import torch
from comfy.model_patcher import ModelPatcher
import comfy.patcher_extension as extension
import comfy.sampler_helpers

import h3_audio_t8_pkg
from h3_audio_t8_pkg.modular_sampling import ltx_load_policy as policy, ltx_load_policy_nodes as nodes


def fixture():
    model = ModelPatcher(torch.nn.Linear(4, 4), torch.device("cpu"), torch.device("cpu"))
    return model


def test_legacy_default_returns_the_same_opaque_MODEL_without_inspection():
    class Foreign:
        @property
        def model_options(self):
            raise AssertionError("Legacy bypass must not inspect options")
    model = Foreign()
    result, runtime, raw = policy.apply_ltx_load_policy(model)
    assert result is model and json.loads(raw)["status"] == "legacy_profile_unchanged"
    latent = {"samples": torch.ones(1, 128, 2, 2, 2)}
    assert policy.audit_ltx_load_policy(model, latent, runtime)[0] is latent


def test_explicit_profile_clones_preserves_weights_hooks_and_legacy_delegate():
    model = fixture()
    calls = []
    def legacy(executor, *args, **kwargs):
        calls.append("legacy")
        return executor(*args, **kwargs)
    model.add_wrapper_with_key(extension.WrappersMP.PREPARE_SAMPLING, "legacy", legacy)
    options = deepcopy(model.model_options)
    weights = [tensor.clone() for tensor in model.model.parameters()]
    rng = torch.get_rng_state().clone()
    selected, runtime, raw = policy.apply_ltx_load_policy(model, "consistent_streaming_exp")
    assert selected is not model and model.model_options == options
    assert selected.model is model.model and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before, after) for before, after in zip(weights, model.model.parameters(), strict=True))
    sentinel = object()
    def native(actual_model, actual_shape, actual_conds, **kwargs):
        assert actual_model is selected and actual_shape == [1, 128, 2, 2, 2] and actual_conds == {}
        assert kwargs["force_offload"] is True and kwargs["force_full_load"] is False
        calls.append("native")
        return sentinel
    active_options = comfy.model_patcher.create_model_options_clone(selected.model_options)
    comfy.sampler_helpers.prepare_model_patcher(selected, {}, active_options)
    wrappers = extension.get_all_wrappers(extension.WrappersMP.PREPARE_SAMPLING, active_options, is_model_options=True)
    result = extension.WrapperExecutor.new_executor(native, wrappers).execute(selected, [1, 128, 2, 2, 2], {},
                                          model_options=active_options, force_offload=False)
    assert result is sentinel and calls == ["legacy", "native"]
    report = json.loads(runtime.report(selected))
    assert report["native_preparation_delegations"] == 1 and report["quality_accepted"] is False
    assert report["portable_cache_reuse_authorized"] is False
    assert json.loads(raw)["status"] == "installed_not_executed"
    assert not torch.cuda.is_initialized()


def test_missing_owner_is_unverified_not_a_candidate_rejection():
    selected, runtime, _ = policy.apply_ltx_load_policy(fixture(), "consistent_streaming_exp")
    selected.remove_wrappers_with_key(extension.WrappersMP.PREPARE_SAMPLING, policy.KEY)
    latent = {"samples": torch.ones(1, 128, 2, 2, 2)}
    candidate, raw = policy.audit_ltx_load_policy(selected, latent, runtime)
    assert candidate is latent and json.loads(raw)["status"] == "unverified_wrapper_not_present"


def test_native_errors_and_conflicting_full_load_remain_real_errors():
    runtime = policy.LTXLoadPolicyRuntime("consistent_streaming_exp")
    with pytest.raises(ValueError, match="full-load"):
        runtime(lambda *args, **kwargs: None, fixture(), [1], {}, force_full_load=True)
    def failed(*args, **kwargs):
        raise RuntimeError("native loader error")
    with pytest.raises(RuntimeError, match="native loader error"):
        runtime(failed, fixture(), [1], {})
    with pytest.raises(ValueError, match="Unknown"):
        policy.apply_ltx_load_policy(fixture(), "wrong")
    with pytest.raises(TypeError, match="real ModelPatcher"):
        policy.apply_ltx_load_policy(object(), "consistent_streaming_exp")


def test_registration_preserves_entire_577_prefix_and_appends_only_two_policy_nodes():
    previous = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
        *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
        *h3_audio_t8_pkg._audio_refine_effect_node_classes, *h3_audio_t8_pkg._ltx_rgb_source_node_classes,
        *h3_audio_t8_pkg._serial_video_io_node_classes, *h3_audio_t8_pkg._ltx_effect_node_classes,
        *h3_audio_t8_pkg._ltx_relay_node_classes, *h3_audio_t8_pkg._veda_sparse_node_classes,
        *h3_audio_t8_pkg._veda_heuristic_node_classes, *h3_audio_t8_pkg._prepared_ltx_effect_node_classes,
        *h3_audio_t8_pkg._prepared_ltx_relay_cache_node_classes, *h3_audio_t8_pkg._semantic_bridge_extra_node_classes]
    actual = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    policy_prefix = previous + nodes.NODES
    assert len(previous) == 577 and len(policy_prefix) == 579
    assert actual[:579] == policy_prefix
    assert actual[579:] == h3_audio_t8_pkg._face_source_node_classes
    assert [node.define_schema().node_id for node in actual[579:]] == [
        "MiniMaxH3MultiFaceSourceSaveEXPT8", "MiniMaxH3MultiFaceSourceLoadEXPT8"]
    assert len(actual) == len({node.define_schema().node_id for node in actual})
