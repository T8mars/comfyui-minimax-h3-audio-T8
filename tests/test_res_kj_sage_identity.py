"""Installed source/dispatcher identity, not fake CPU math or CUDA execution."""
import importlib
from pathlib import Path
import sys
from types import FunctionType, MethodType, ModuleType

import pytest
import torch
import triton
import triton.language as tl
import comfy.model_management as mm
import comfy.quant_ops

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.relay_kj_backend import _codes
from h3_audio_t8_pkg.res_kj_sage_identity import inspect_raw_kj_sage
from h3_audio_t8_pkg.res_rope_identity import inspect_rope_kitchen
from test_relay_kj_memory import memory_nodes  # noqa: F401 -- source-backed Sol fixture
from test_relay_kj_backend import kj  # noqa: F401 -- dependent KJ fixture


@pytest.fixture
def actual_source_chain(monkeypatch):
    # Importing all of KJ would query devices/install other nodes. Extract the
    # actual source-backed functions only; branch selection is explicitly a
    # SM89 CPU fixture, not a claim about this process's available hardware.
    path = Path(mm.__file__).resolve().parents[1] / "custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py"
    source = path.read_bytes()
    compiled = tuple(_codes(compile(source, str(path), "exec", dont_inherit=True)))
    module = ModuleType("test_res_actual_KJ_Sage_chain")
    module.__file__ = str(path)
    quant = importlib.import_module("sageattention.quant")
    module.__dict__.update(torch=torch, triton=triton, tl=tl, mm=mm, _ck=comfy.quant_ops.ck,
        _cuda_archs=["sm89"], HAS_TRITON=True, sageplus_sm89_available=True,
        per_channel_fp8=quant.per_channel_fp8, _qattn_sm89=torch.ops.sageattention_qattn_sm89)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    defaults = {"minimax_sageattn_forward": (None, {}), "_sageattn_int8_fp8_nhd": None,
                "_per_thread_int8_i64": (None, 128, 32, 64, 64, "NHD")}
    for name, values in defaults.items():
        code = next(item for item in compiled if item.co_name == name)
        setattr(module, name, FunctionType(code, vars(module), argdefs=values))
    # Use the actual full-module compiled code, not AST-only recompilation:
    # Python 3.12 call opcodes also depend on the surrounding symbol table.
    # Real triton.jit gets installed source/line information from those codes.
    for name in ("_quant_query_per_thread_int8_i64_kernel", "_quant_key_per_thread_int8_i64_kernel"):
        code = next(item for item in compiled if item.co_name == name)
        function = FunctionType(code, vars(module))
        function.__annotations__ = {"C": tl.constexpr, "BLK": tl.constexpr}
        setattr(module, name, triton.jit(function))
    owner = object()
    return module, owner, MethodType(module.minimax_sageattn_forward, owner)


def test_actual_RoPE_public_op_native_binary_partial_path_and_hot_CPU_identity():
    ck = comfy.quant_ops.ck
    contract = inspect_rope_kitchen(ck)
    assert contract == inspect_rope_kitchen(ck)
    assert contract["dispatch_content_verified"] and not contract["CUDA_execution_qualified"]
    assert contract["CUDA_binary"]["sha256"] == "d41adbb507c46868446beb1d756d3953a4fc5b9187970baf83f7abe5993c58b9"
    assert set(contract["candidates"]) == {"eager"}
    # One real eager op warms the actual mutating CustomOpDef wrapper; no
    # selected CUDA/Sage or substituted SDPA calculation is executed here.
    q = torch.arange(4 * 128, dtype=torch.float32).reshape(1, 4, 1, 128) / 512
    k = q.clone()
    original = q.clone()
    freqs = torch.eye(2, dtype=torch.float32).reshape(1, 1, 1, 1, 2, 2).expand(1, 4, 1, 32, 2, 2).clone()
    ck.rms_rope_split_half_(q, k, freqs, torch.ones(128), torch.ones(128), rot_dim=64)
    assert q.shape == original.shape and torch.isfinite(q).all() and not torch.equal(q, original)
    assert inspect_rope_kitchen(ck) == contract and not torch.cuda.is_initialized()


def test_actual_SM89_native_ops_binary_i64_quantizers_and_RoPE_content(actual_source_chain):
    module, owner, method = actual_source_chain
    kernels = [module._quant_query_per_thread_int8_i64_kernel, module._quant_key_per_thread_int8_i64_kernel]
    before = [(item.src, dict(item.device_caches), list(item.pre_run_hooks)) for item in kernels]
    contract = inspect_raw_kj_sage(method, owner)
    assert contract == inspect_raw_kj_sage(method, owner)
    assert contract["calculation_content_verified"] and not contract["portable_cache_reuse"]
    assert not contract["CUDA_execution_qualified"] and contract["value_scale_max"] == 2.25
    assert len(contract["native_operations"]) == 3 and len(contract["quantizers"]) == 2
    assert all(not item["CUDA_kernel_launched"] for item in contract["quantizers"].values())
    assert all(not item["compiled_device_cache_content_qualified"] for item in contract["quantizers"].values())
    assert [(item.src, dict(item.device_caches), list(item.pre_run_hooks)) for item in kernels] == before
    assert not torch.cuda.is_initialized()


def test_unknown_Sage_quantizer_operator_and_JIT_hooks_are_preserved_unverified(actual_source_chain, monkeypatch):
    module, owner, method = actual_source_chain
    assert inspect_raw_kj_sage(method, owner)["calculation_content_verified"]
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("identity must not execute unknown computation")
    quant = importlib.import_module("sageattention.quant")
    for target, name, reason in ((module, "_sageattn_int8_fp8_nhd", "live executable"),
            (module, "_per_thread_int8_i64", "live executable"),
            (module, "per_channel_fp8", "value quantizer source/owner"),
            (quant._fused, "scale_fuse_quant_cuda", "operator packet")):
        with monkeypatch.context() as changed:
            changed.setattr(target, name, foreign)
            with pytest.raises(UnverifiedModelStack, match=reason):
                inspect_raw_kj_sage(method, owner)
            assert getattr(target, name) is foreign
    kernel = module._quant_query_per_thread_int8_i64_kernel
    for name, value, reason in (("pre_run_hooks", [foreign], "another launch hook"),
            ("_src", "def foreign(): pass\n", "JIT source differs"),
            ("run", foreign, "execution method")):
        with monkeypatch.context() as changed:
            changed.setattr(kernel, name, value, raising=False)
            with pytest.raises(UnverifiedModelStack, match=reason):
                inspect_raw_kj_sage(method, owner)
            assert getattr(kernel, name) is value
    with monkeypatch.context() as changed:
        changed.setattr(module, "_cuda_archs", ["sm120"])
        with pytest.raises(UnverifiedModelStack, match="not audited SM89"):
            inspect_raw_kj_sage(method, owner)
        assert module._cuda_archs == ["sm120"]
    with monkeypatch.context() as changed:
        changed.setitem(sys.modules, "triton", None)
        with pytest.raises(UnverifiedModelStack, match="compiler is unavailable"):
            inspect_raw_kj_sage(method, owner)
        assert sys.modules["triton"] is None
    assert not calls and not torch.cuda.is_initialized()


def test_foreign_top_level_Sol_helper_is_not_omitted_from_identity(request, tmp_path, monkeypatch):
    from h3_audio_t8_pkg import sol_attn_minimax_v2 as sol
    from h3_audio_t8_pkg.res_sol_identity import inspect_original_sol
    from test_res_original_sol_identity import original_sol, snapshot, assert_untouched

    selected = original_sol(request, tmp_path)
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unknown Sol helper must not be invoked")
    monkeypatch.setattr(sol, "_video_span", foreign)
    before = snapshot(selected)
    with pytest.raises(UnverifiedModelStack, match="helper was replaced"):
        inspect_original_sol(selected)
    assert not calls and sol._video_span is foreign
    assert_untouched(selected, before)


def test_RoPE_unknown_helper_and_settings_are_content_bound_without_dispatch(actual_source_chain, monkeypatch):
    module, owner, method = actual_source_chain
    original = inspect_raw_kj_sage(method, owner)
    with monkeypatch.context() as changed:
        changed.setattr(module, "sageplus_sm89_available", False)
        fallback = inspect_raw_kj_sage(method, owner)
        assert fallback["sha256"] != original["sha256"] and fallback["value_scale_max"] == 448.0
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("identity must not execute foreign RoPE")
    with monkeypatch.context() as changed:
        changed.setattr(cuda, "_rms_rope_cuda", foreign)
        with pytest.raises(UnverifiedModelStack):
            inspect_rope_kitchen(module._ck)
        assert cuda._rms_rope_cuda is foreign
    with monkeypatch.context() as changed:
        changed.setattr(module, "_ck", None)
        with pytest.raises(UnverifiedModelStack):
            inspect_raw_kj_sage(method, owner)
        assert module._ck is None
    assert not calls and not torch.cuda.is_initialized()
