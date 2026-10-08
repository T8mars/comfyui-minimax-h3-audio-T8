"""Actual installed custom-op and CPU derivations, no CUDA kernel launches."""
from dataclasses import replace
import importlib
import inspect

import pytest
import torch
import comfy_kitchen as ck

from h3_audio_t8_pkg import sol_attn_minimax_v2 as sol
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kitchen_identity import inspect_sol_kitchen
from h3_audio_t8_pkg.res_sol_cache_identity import inspect_sol_caches


def test_actual_custom_op_registry_native_binary_and_hot_CPU_dispatch_are_content_bound():
    eager = importlib.import_module("comfy_kitchen.backends.eager.sol_attn")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    selected = eager._op_sol_attn._backend_fns[None]
    before = (dict(ck.registry._backends), list(ck.registry._priority), set(ck.registry._disabled), cuda._C.sol_attn)
    contract = inspect_sol_kitchen(ck)
    assert contract == inspect_sol_kitchen(ck)
    assert contract["dispatch_content_verified"] and not contract["CUDA_execution_qualified"]
    assert contract["CUDA_binary"]["sha256"] == "d41adbb507c46868446beb1d756d3953a4fc5b9187970baf83f7abe5993c58b9"
    assert set(contract["candidates"]) == {"eager"}
    # One real tiny eager call establishes the native hot wrapper/cache state;
    # it is not a fake Sage function or any CUDA quality/performance evidence.
    q = torch.arange(4 * 128, dtype=torch.float32).reshape(1, 4, 1, 128) / 512
    output = ck.sol_attn(q, q, q)
    assert output.shape == q.shape and torch.isfinite(output).all()
    assert inspect_sol_kitchen(ck) == contract
    assert eager._op_sol_attn._backend_fns[None] is selected
    assert (dict(ck.registry._backends), list(ck.registry._priority), set(ck.registry._disabled), cuda._C.sol_attn) == before
    assert not torch.cuda.is_initialized()


def test_changed_selection_constraints_and_foreign_kernel_do_not_get_a_content_certificate(monkeypatch):
    contract = inspect_sol_kitchen(ck)
    with monkeypatch.context() as changed:
        changed.setattr(ck.registry, "_priority", list(reversed(ck.registry._priority)))
        assert inspect_sol_kitchen(ck)["sha256"] != contract["sha256"]
    with monkeypatch.context() as changed:
        key = "eager", "sol_attn"
        changed.setitem(ck.registry._constraints, key, replace(ck.registry._constraints[key], min_compute_capability=(9, 0)))
        assert inspect_sol_kitchen(ck)["sha256"] != contract["sha256"]
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("identity must not execute unknown computation")
    eager = importlib.import_module("comfy_kitchen.backends.eager.sol_attn")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    for owner, name in ((cuda._C, "sol_attn"), (cuda, "_block_lengths"), (eager._op_sol_attn, "_init_fn")):
        with monkeypatch.context() as changed:
            changed.setattr(owner, name, foreign)
            with pytest.raises(UnverifiedModelStack):
                inspect_sol_kitchen(ck)
    with monkeypatch.context() as changed:
        changed.setitem(eager._op_sol_attn._backend_fns, "cuda", foreign)
        with pytest.raises(UnverifiedModelStack):
            inspect_sol_kitchen(ck)
    assert not calls and not torch.cuda.is_initialized()


def test_actual_layout_and_Morton_cache_integrity_is_checked_without_mutation(monkeypatch):
    from comfy.ldm.minimax import model as core_h3
    sol._patch_packed_layout(core_h3)
    layout = core_h3.PackedLayout(3, 2, 4, 6, 5)
    span_key = id(layout.position_ids)
    before_spans = dict(sol._SPANS)
    with monkeypatch.context() as isolated:
        isolated.setattr(sol, "_PERM_CACHE", {})
        isolated.setattr(sol, "_DEVICE_CACHE", {})
        isolated.setattr(sol, "_SOL_KERNEL_PARAMETERS", {})
        sol.morton_perm((2, 2, 3), "cpu", "2d_frame")
        sol._perm_for((2, 2, 3), "2d_frame", "cpu", 3)
        sol._sol_kernel_parameters(ck.sol_attn)
        actual = inspect_sol_caches(sol)
        assert actual["lookup_metadata_verified"] and actual["derived_permutation_content_verified"]
        assert not actual["full_position_coordinates_verified"]
        pair = sol._DEVICE_CACHE[((2, 2, 3), "2d_frame", "cpu", 61)]
        original = pair[0].clone()
        pair[0][0] = -1
        with pytest.raises(UnverifiedModelStack, match="permutation content"):
            inspect_sol_caches(sol)
        pair[0].copy_(original)
        sol._SOL_KERNEL_PARAMETERS[id(ck.sol_attn)] = frozenset({"foreign"})
        with pytest.raises(UnverifiedModelStack, match="signature cache"):
            inspect_sol_caches(sol)
        sol._SOL_KERNEL_PARAMETERS[id(ck.sol_attn)] = frozenset(inspect.signature(ck.sol_attn).parameters)
        actual_entry = sol._SPANS[span_key]
        with monkeypatch.context() as changed:
            changed.setitem(sol._SPANS, span_key, (layout, (0, 1), *actual_entry[2:]))
            with pytest.raises(UnverifiedModelStack, match="bounds"):
                inspect_sol_caches(sol)
        assert inspect_sol_caches(sol) == actual
    assert sol._SPANS == before_spans and not torch.cuda.is_initialized()


def test_unavailable_dependencies_and_malformed_caches_are_unverified_not_executed(monkeypatch, tmp_path):
    with pytest.raises(UnverifiedModelStack, match="another kitchen"):
        inspect_sol_kitchen(None)
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    with monkeypatch.context() as changed:
        changed.setattr(cuda, "_C", None)
        with pytest.raises(UnverifiedModelStack, match="native binary"):
            inspect_sol_kitchen(ck)
        assert cuda._C is None
    with monkeypatch.context() as changed:
        missing = str(tmp_path / "missing.py")
        changed.setattr(ck, "__file__", missing)
        with pytest.raises(UnverifiedModelStack, match="identity unavailable"):
            inspect_sol_kitchen(ck)
        assert ck.__file__ == missing
    with monkeypatch.context() as changed:
        malformed = {0: None}
        changed.setattr(sol, "_SPANS", malformed)
        with pytest.raises(UnverifiedModelStack, match="malformed"):
            inspect_sol_caches(sol)
        assert sol._SPANS is malformed and malformed == {0: None}
    assert not torch.cuda.is_initialized()
