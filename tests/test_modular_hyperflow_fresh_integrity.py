"""No live mutation by identity projection; real tampering still fails closed."""
from dataclasses import replace
import json

import comfy.patcher_extension
import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow, hyperflow_fresh as fresh
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity, canonical, sha
from test_modular_hyperflow_fresh import setup
from test_modular_hyperflow import inputs, equal


def test_identity_projection_never_changes_live_loaded_network(monkeypatch):
    args, _, _, _ = setup(monkeypatch, fresh.STAGES[0])
    sample_stage(*args)
    model = args[1].model_patcher
    modules = {name: id(module) for name, module in model.model.named_modules()}
    installed = {path: model.model.get_submodule(path.removesuffix(".forward")).forward
                 for path in model.object_patches if path.endswith(".forward")}
    buffers = {name: value.data_ptr() for name, value in model.model.named_buffers()}
    assert hyperflow.model_selection(model)["portable_cache_reuse"]
    assert modules == {name: id(module) for name, module in model.model.named_modules()}
    assert buffers == {name: value.data_ptr() for name, value in model.model.named_buffers()}
    for path, value in installed.items():
        assert model.model.get_submodule(path.removesuffix(".forward")).forward is value


@pytest.mark.parametrize("change", ["live_forward", "sampling", "lora", "selected_forward"])
def test_actual_in_run_mutations_do_not_acquire_result(monkeypatch, change):
    args, _, _, _ = setup(monkeypatch, fresh.STAGES[2])
    model, context = args[1].model_patcher, args[5]
    def mutate(executor, *a, **kw):
        output = executor(*a, **kw)
        if fresh.runtime.active_step().absolute_index == context.end - 1:
            if change == "sampling":
                model.object_patches["model_sampling"].set_noise_scale(2.)
            elif change == "lora":
                key = next(iter(model.patches))
                model.patches[key].reverse()
                model.patches[key].append(model.patches[key][0])
            else:
                key = "diffusion_model.blocks.0.forward"
                original = model.object_patches[key]
                def altered(*a, **kw):
                    return original(*a, **kw)
                if change == "selected_forward":
                    model.object_patches[key] = altered
                else:
                    model.model.diffusion_model.blocks[0].forward = altered
        return output
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "test-mutation", mutate)
    with pytest.raises(ValueError, match="changed during|sampling object changed"):
        sample_stage(*args)
    if change != "sampling":
        assert not fresh.capture_owner(model).lock.locked()


@pytest.mark.parametrize("change", ["unknown_portable", "eav_forwards", "relay_forwards"])
def test_rehashed_receipt_cannot_promote_unverified_execution(monkeypatch, change):
    args, _, _, _ = setup(monkeypatch, fresh.STAGES[3], mode="apply_exp", with_relay=True)
    if change == "unknown_portable":
        def user(executor, *a, **kw):
            return executor(*a, **kw)
        args[1].model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user", user)
    result = sample_stage(*args)[2]
    receipt = json.loads(result.receipt_json)
    if change == "unknown_portable":
        assert receipt["portable_identity"] is False
        receipt["portable_identity"] = True
    else:
        receipt["execution"]["hyperflow_fresh"][change.split("_")[0]]["completed_forwards"] = 0
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha(receipt)
    with pytest.raises(ValueError, match="portable|coverage"):
        replace(result, receipt_json=canonical(receipt)).verify()


def test_new_loader_late_shared_residency_matches_cold_native_endpoint(monkeypatch):
    from test_hyperflow_advanced import _tiny_weights
    from test_modular_hyperflow_fresh_public import args_for
    from h3_audio_t8_pkg.modular_sampling import hyperflow_identity, hyperflow_fresh_nodes as public
    from h3_audio_t8_pkg import nodes_hyperflow_advanced as old
    low, _, _, _ = inputs(monkeypatch)
    base = low.parent
    weights = _tiny_weights(base.get_model_object("diffusion_model"))
    cold, _, _ = fresh.install(base, weights)
    expected_identity = hyperflow_identity.model_identity(cold)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.full((1, 24, 2, 4, 8), .1), torch.full((1, 32, 2, 8), -.2)))}
    # An original cold Loader and the new Loader have identical adapter/math.
    assert expected_identity == hyperflow_identity.model_identity(low)
    expected = sample_stage(*args_for(cold, source, fresh.STAGES[2], "none", 32))[2]
    sample_stage(*args_for(low, source, fresh.STAGES[0], "none", 31))
    live_time = base.model.diffusion_model.time_embedder.proj_in.weight.detach().clone()
    monkeypatch.setattr(old, "_resolve", lambda _: "test-selected-original")
    monkeypatch.setattr(old, "load_hyperflow_original", lambda _: weights)
    late, report = public.MiniMaxH3HyperFlowFreshLoaderEXPT8.execute(base, "test-selected-original").result
    assert json.loads(report)["shared_model_unloaded"] is False
    assert torch.equal(base.model.diffusion_model.time_embedder.proj_in.weight, live_time)
    assert hyperflow_identity.model_identity(late)["adapter_sha256"] == expected_identity["adapter_sha256"]
    args = args_for(late, source, fresh.STAGES[2], "none", 32)
    # Bind the same explicit sampling selection before comparing full identity;
    # a bare Loader has no phase owner to project its dormant sampling buffer.
    assert canonical(selected_model_identity(args[1].model_patcher)) == canonical(expected.verify()["request"]["model"])
    actual = sample_stage(*args)[0]
    equal(actual, expected.output)
