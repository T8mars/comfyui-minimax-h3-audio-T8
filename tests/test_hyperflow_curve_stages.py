"""Real tiny native Euler/curve/cold state, never a split-executor mock."""
from dataclasses import replace
import json

import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_curve_runtime_exp as runtime
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from h3_audio_t8_pkg.modular_sampling import hyperflow_curve as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_curve_storage as storage
from test_hyperflow_curve_identity import setup
from test_progressive_sampling_runtime import conditioning


def inputs(tmp_path, monkeypatch, *, distinct=False):
    _, diffusion, low, high, files = setup(tmp_path, monkeypatch)
    if distinct:
        key = "diffusion_model.blocks.0.attn.out_proj.weight"
        high.add_patches({key: ("diff", (torch.full_like(diffusion.blocks[0].attn.out_proj.weight, .002),))}, .2)
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    return low, high, source, conditioning(), files


def head(low, source, positive, *, split=4, **kwargs):
    return stages.sample_head(low, source, positive, [], comfy.sample.prepare_noise(source["samples"], 7),
        seed=7, split=split, **kwargs)


def equal(left, right):
    for a, b in zip(left["samples"].unbind(), right["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("split", [1, 4, 7])
def test_split_exact_true_native_single8_and_no_tail_fresh_noise(tmp_path, monkeypatch, split):
    low, high, source, positive, _ = inputs(tmp_path, monkeypatch)
    branch, sampler, sigmas = runtime.setup_sampler(low, source, runtime.build_plan(low))
    try:
        full = comfy.sample.sample_custom(branch, comfy.sample.prepare_noise(source["samples"], 7),
            1., sampler, sigmas, positive, [], source["samples"], disable_pbar=True, seed=7)
    finally:
        _restore_stage_objects(branch)
    callbacks = []
    boundary = head(low, source, positive, split=split, callback=lambda index, _p, _x, total: callbacks.append((index, total)))
    assert boundary.verify()["execution"]["actual_51_coverage"] is True
    assert boundary.verify()["execution"]["portable_identity"] is True
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A continuous TAIL cannot draw new noise or run HEAD")
    monkeypatch.setattr(comfy.sample, "prepare_noise", forbidden)
    monkeypatch.setattr(stages, "sample_head", forbidden)
    result, report = stages.sample_tail(boundary, high, positive, [], seed=7,
        callback=lambda index, _p, _x, total: callbacks.append((index, total)))
    equal({"samples": full}, result.output)
    assert callbacks == [(index, 8) for index in range(8)]
    assert json.loads(report)["execution"]["portable_identity"] is True
    assert runtime.ACTIVE.get() is None and runtime.full_runtime.active_step() is None
    assert not low.object_patches_backup and not high.object_patches_backup
    assert not torch.cuda.is_initialized()


def test_saved_exact_head_cold_tail_and_saved_delivery_zero_sampling(tmp_path, monkeypatch):
    low, high, source, positive, _ = inputs(tmp_path, monkeypatch, distinct=True)
    boundary = head(low, source, positive)
    path, digest, report = storage.save_boundary(boundary, str(tmp_path / "store"))
    loaded, status = storage.load_boundary(str(tmp_path / "store"), path, digest)
    assert json.loads(status)["sampling_calls"] == 0
    assert json.loads(report)["portable_identity"] is True
    torch.testing.assert_close(loaded.x_sigma, boundary.x_sigma, rtol=0, atol=0)
    assert loaded.receipt_json == boundary.receipt_json
    result, _ = stages.sample_tail(loaded, high, positive, [], seed=7)
    tail_path, tail_digest, _ = storage.save_result(result, str(tmp_path / "store"))
    monkeypatch.setattr(stages, "_run", lambda *_a, **_kw: pytest.fail("A frozen delivery cannot sample"))
    delivery, _ = storage.load_result(str(tmp_path / "store"), tail_path, tail_digest)
    equal(delivery.output, result.output)
    assert delivery.receipt_json == result.receipt_json
    assert storage.fingerprint(str(tmp_path / "store"), tail_path, "tail") == tail_digest
    with pytest.raises(ValueError, match="SHA mismatch"):
        storage.load_result(str(tmp_path / "store"), tail_path, "0" * 64)
    with pytest.raises(ValueError, match="exact completed"):
        storage.load_boundary(str(tmp_path / "store"), tail_path, tail_digest)


@pytest.mark.parametrize("fault", ["raw", "receipt", "clock", "base"])
def test_bad_boundary_or_another_basis_rejected_before_tail_forward(tmp_path, monkeypatch, fault):
    low, high, source, positive, files = inputs(tmp_path, monkeypatch)
    boundary = head(low, source, positive)
    with torch.inference_mode():
        if fault == "raw":
            boundary.x_sigma.flatten()[0] += .1
        elif fault == "receipt":
            data = json.loads(boundary.receipt_json)
            data["execution"]["callbacks"] = []
            boundary = replace(boundary, receipt_json=json.dumps(data))
        elif fault == "clock":
            high.set_attachments(runtime.KEY, replace(high.get_attachment(runtime.KEY), raw_sigmas=tuple([1.] * 9)))
        else:
            from test_hyperflow_curve_runtime_exp import prepare
            other = tmp_path / "changed-original-base"
            other.mkdir()
            base, network, _, _, _ = prepare(other, monkeypatch, portable=True)
            network.blocks[0].attn.out_proj.weight.add_(.1)
            details = low.get_attachment(runtime.DETAILS_KEY)
            high, _, _ = runtime.install_curve(base, details.weights, details.fit, files["base"])
    monkeypatch.setattr(stages, "_run", lambda *_a, **_kw: pytest.fail("Invalid boundary reached sampling"))
    with pytest.raises(ValueError):
        stages.sample_tail(boundary, high, positive, [], seed=7)


def test_unknown_user_delegate_executes_without_persistent_promotion(tmp_path, monkeypatch):
    low, _, source, positive, _ = inputs(tmp_path, monkeypatch)
    calls = []
    def wrapper(executor, *args, **kwargs):
        calls.append(runtime.full_runtime.active_step().absolute_index)
        return executor(*args, **kwargs)
    low.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user", wrapper)
    boundary = head(low, source, positive)
    assert calls == list(range(4))
    assert boundary.verify()["execution"]["portable_identity"] is False
    path, digest, report = storage.save_boundary(boundary, str(tmp_path / "store"))
    assert json.loads(report)["portable_identity"] is False
    with pytest.raises(ValueError, match="persistent curve reuse"):
        storage.load_boundary(str(tmp_path / "store"), path, digest)


def test_cancel_cleanup_keeps_source_and_current_own_model(tmp_path, monkeypatch):
    low, _, source, positive, _ = inputs(tmp_path, monkeypatch)
    def cancel(index, *_args):
        if index == 1:
            raise RuntimeError("actual curve cancellation")
    with pytest.raises(RuntimeError, match="actual curve cancellation"):
        head(low, source, positive, callback=cancel)
    assert not low.object_patches_backup and runtime.ACTIVE.get() is None
    assert runtime.full_runtime.active_step() is None
    assert not hasattr(low.model.diffusion_model, "time_embedder")
    assert all(not torch.count_nonzero(tensor) for tensor in source["samples"].unbind())
    assert head(low, source, positive).verify()["execution"]["portable_identity"] is True


def test_head_implementation_difference_rejected_before_any_tail_forward(tmp_path, monkeypatch):
    low, high, source, positive, _ = inputs(tmp_path, monkeypatch)
    boundary = head(low, source, positive)
    old = stages.implementation()
    monkeypatch.setattr(stages, "implementation", lambda: {**old, "actual_integrator_or_contract": "changed"})
    monkeypatch.setattr(stages, "_run", lambda *_a, **_kw: pytest.fail("Changed HEAD implementation reached sampling"))
    with pytest.raises(ValueError, match="producer implementation changed"):
        stages.sample_tail(boundary, high, positive, [], seed=7)
