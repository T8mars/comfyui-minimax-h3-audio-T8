"""RES-only content identity of KJ's actual SM89 Sage calculation chain.

No kernel is launched, no device is queried, and no installed patch is changed.
This is source/dispatch evidence, not CUDA numerical or portable-cache approval.
Other hardware/unknown owners remain usable for ordinary complete sampling.
"""
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
import sys
from types import BuiltinFunctionType, FunctionType, MethodType

import torch
import comfy.model_management as mm
import comfy.quant_ops

from .patch_stack_policy import UnverifiedModelStack
from .res_kitchen_identity import _SourceAudit, _data, _digest, _require
from .res_rope_identity import inspect_rope_kitchen
from .res_triton_identity import inspect_native_jit


_KJ_SOURCE = "bcc2791dfc88620ee5147ccfd8d6ff708e78fa3c7c47f4fb737606dbc8673428"
_QUANT_SOURCE = "7a2565348f97ce9b797a5ae306c4fa4852a06d7ed3c38c541e98eaf4239add02"
_BINARIES = {
    "sageattention._fused": "75afc1fdd92d8caf2e1b2a8f394e4723e5383988bc3bc694e378c586864267c9",
    "sageattention._qattn_sm89": "2d926077a59dc84d5469672df809b06111e8c7db40f82e38de5ad3ebe0b359de",
}


def _native_binary(name):
    module = importlib.import_module(name)
    _require(module is sys.modules.get(name) and isinstance(module.__spec__.loader, ExtensionFileLoader),
             "Sage extension is not its native loaded owner")
    path = Path(module.__file__).resolve()
    package = importlib.import_module("sageattention")
    _require(path.parent == Path(package.__file__).resolve().parent,
             "Sage binary has another package owner")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
             and digest.hexdigest() == _BINARIES[name], "Sage binary requires a fresh content audit")
    return {"module": name, "bytes": before.st_size, "sha256": digest.hexdigest()}


def _native_op(namespace, name, cuda_registration):
    """Check actual C++ dispatcher registration, not nonexistent Python exports."""
    from torch._ops import _OpNamespace, OpOverloadPacket, OpOverload

    qualified = namespace.__name__.removeprefix("torch.ops.") + "::" + name
    _require(type(namespace) is _OpNamespace
             and namespace is getattr(torch.ops, namespace.name), "Sage namespace has another owner")
    packet = getattr(namespace, name)
    _require(type(packet) is OpOverloadPacket and packet._qualified_op_name == qualified
             and type(packet._op) is BuiltinFunctionType
             and packet._op.__name__ == name and packet.overloads() == ["default"],
             "Sage operator packet has another executable owner")
    overload = packet.default
    _require(type(overload) is OpOverload and overload._overloadpacket is packet
             and overload.name() == qualified and not overload._defined_in_python
             and type(overload._op) is BuiltinFunctionType and type(overload._op_dk) is BuiltinFunctionType,
             "Sage overload is not its native operator")
    _require(not overload.py_kernels and not overload.python_key_table and not overload.functorch_table,
             "Sage operator has an extra Python dispatch rule")
    _require(not (set(vars(packet)) & {"__call__", "overloads"})
             and not (set(vars(overload)) & {"__call__", "redispatch"}),
             "Sage operator invocation has an instance override")
    schema = str(overload._schema)
    _require(schema == str(overload._handle.schema()), "Sage operator schema/handle disagree")
    table = torch._C._dispatch_dump_table(qualified)
    row = next((line for line in table.splitlines() if line.startswith("CUDA:")), "")
    _require(cuda_registration in row and row.endswith("[kernel]"),
             "Sage CUDA dispatcher is not the audited native registration")
    dump = torch._C._dispatch_dump(qualified)
    _require("alias analysis kind: FROM_SCHEMA" in dump,
             "Sage operator schema has another aliasing contract")
    return {"qualified_name": qualified, "schema": schema, "CUDA_registration": row,
            "native_CPP_dispatch_verified": True, "CUDA_kernel_launched": False}


def _inspect(method, owner, observed_artifacts):
    _require(type(method) is MethodType and method.__self__ is owner,
             "raw KJ Sage delegate is not bound to its attention owner")
    function = method.__func__
    _require(type(function) is FunctionType, "raw KJ Sage delegate is opaque")
    module = sys.modules.get(function.__globals__.get("__name__"))
    _require(module is not None and function is module.minimax_sageattn_forward,
             "raw KJ Sage forward was replaced")
    path = Path(module.__file__).resolve()
    _require(path.name == "ltxv_nodes.py" and hashlib.sha256(path.read_bytes()).hexdigest() == _KJ_SOURCE,
             "KJ Sage source requires a fresh calculation-chain audit")
    audit = _SourceAudit()
    audit.function(function, module, dependencies=False)
    _require(module.torch is torch and module.mm is mm and module._ck is comfy.quant_ops.ck,
             "KJ Sage computation globals were replaced")
    audit.function(mm.cast_to, mm, dependencies=False)
    _require(type(module._cuda_archs) in (list, tuple) and bool(module._cuda_archs)
             and all(type(item) is str for item in module._cuda_archs)
             and module._cuda_archs[0] == "sm89", "raw KJ Sage hardware branch is not audited SM89")
    _require(type(module.sageplus_sm89_available) is bool and module.HAS_TRITON is True,
             "KJ Sage quantization branch has foreign settings")
    # Audited SM89 does not execute other architectures' quantizers/attention.
    # Their availability must not be mistaken for a hardware qualification.
    for name in ("_sageattn_int8_fp8_nhd", "_per_thread_int8_i64"):
        audit.function(getattr(module, name), module, dependencies=False)
    quant = importlib.import_module("sageattention.quant")
    _require(hashlib.sha256(Path(quant.__file__).read_bytes()).hexdigest() == _QUANT_SOURCE
             and module.per_channel_fp8 is quant.per_channel_fp8 and quant.torch is torch,
             "Sage value quantizer source/owner changed")
    audit.function(quant.per_channel_fp8, quant, dependencies=False)
    fused = torch.ops.sageattention_fused
    qattn = torch.ops.sageattention_qattn_sm89
    _require(quant._fused is fused and module._qattn_sm89 is qattn,
             "Sage quantizer or attention uses another dispatcher namespace")
    operations = {
        name: _native_op(fused, name, "csrc\\fused\\pybind.cpp:220")
        for name in ("transpose_pad_permute_cuda", "scale_fuse_quant_cuda")
    }
    attention_name = ("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf" if module.sageplus_sm89_available
                      else "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")
    operations[attention_name] = _native_op(qattn, attention_name, "csrc\\qattn\\pybind_sm89.cpp:301")
    quantizers = {name: inspect_native_jit(getattr(module, name), module, _KJ_SOURCE,
                                         observed_artifacts=observed_artifacts) for name in (
        "_quant_query_per_thread_int8_i64_kernel", "_quant_key_per_thread_int8_i64_kernel")}
    result = {"schema": "t8.minimax_h3.RES_raw_KJ_Sage_content.v1",
              "KJ_source_sha256": _KJ_SOURCE, "sources": audit.sources, "executables": audit.functions,
              "selected_branch": "sm89", "sageplus_sm89_available": module.sageplus_sm89_available,
              "arch_selection": _data(module._cuda_archs),
              "value_scale_max": 2.25 if module.sageplus_sm89_available else 448.0,
              "value_smoothing": False, "QK_layout": "NHD", "QK_row_offsets": "int64",
              "native_operations": operations, "quantizers": quantizers,
              "native_binaries": {name: _native_binary(name) for name in _BINARIES},
              "RoPE_execution_chain": inspect_rope_kitchen(module._ck, observed_artifacts=observed_artifacts),
              "calculation_content_verified": True, "CUDA_execution_qualified": False,
              "portable_cache_reuse": False,
              "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["sha256"] = _digest(result)
    return result


def inspect_raw_kj_sage(method, owner, *, observed_artifacts=None):
    try:
        return _inspect(method, owner, observed_artifacts)
    except UnverifiedModelStack:
        raise
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise UnverifiedModelStack("RES raw KJ Sage identity unavailable: " + str(error)) from error
