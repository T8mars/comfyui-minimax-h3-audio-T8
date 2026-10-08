"""RES-only standalone EAV routing for the authenticated original KJ/Sol stack.

No synthetic Relay plan or conditioning is installed. Disabled EAV never calls
this adapter. Original KJ projection and Sol gates remain unchanged; the already
source-bound RES delegate keeps its private-K, mask-preserving execution policy.
"""
import hashlib
from pathlib import Path

from .patch_stack_policy import UnverifiedModelStack, warn_patch_stack
from .relay_kj_memory import HeadGroupedBackend, _make_memory_forward, bind_memory_runtime
from .relay_sol_backend import UserSelectedBackend
from .vdn_attention_compat import _factory_closure
from . import res_memory_effects as memory

KEY = "t8_RES_standalone_EAV_memory_v1"


def _router(backend):
    def selected_attention(func, q, k, v, heads, *args, **kwargs):
        if args:
            raise RuntimeError("RES standalone EAV requires native keyword attention arguments")
        return backend.attention(q, k, v, heads, **kwargs)
    return selected_attention


def _owner(model):
    owner = model.get_attachment(KEY)
    if (type(owner) is not dict or set(owner) != {"selector", "original_override", "backend", "source_sha256"}
            or type(owner["backend"]) is not HeadGroupedBackend
            or type(owner["backend"].delegate) is not memory.RESMemoryDelegate
            or type(owner["backend"].delegate.original) is not UserSelectedBackend
            or owner["backend"].delegate.original.override is not owner["original_override"]
            or owner["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise UnverifiedModelStack("RES standalone EAV memory has no exact source-bound owner")
    state = _factory_closure(owner["selector"], _router, "selected_attention")
    if state is None or state != {"backend": owner["backend"]}:
        raise UnverifiedModelStack("RES standalone EAV selected delegate was replaced")
    return owner


def prepare(model):
    from . import prompt_relay_advanced as relay
    from .res_sol_identity import inspect_original_sol
    if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        return model
    if model.get_attachment(KEY) is not None:
        raise ValueError("RES standalone EAV memory was already bound")
    from . import sol_attn_minimax_v2 as sol
    original_override = model.model_options["transformer_options"].get("optimized_attention_override")
    if _factory_closure(original_override, sol.make_override, "override") is None:
        return model
    try:
        inspected = inspect_original_sol(model)
        if inspected is None or not inspected[0]["unwrapped_attention_delegates_authenticated"]:
            raise UnverifiedModelStack("Standalone EAV keeps unverified original KJ/Sol producers")
    except UnverifiedModelStack as error:
        warn_patch_stack(str(error))
        return model
    delegates = inspected[1]
    kernels = {method.__func__.__globals__["_sageattn_int8_fp8_nhd"] for method in delegates.values()}
    if len(kernels) != 1:
        warn_patch_stack("Standalone EAV retains unmatched KJ kernels; actual coverage remains unverified")
        return model
    groups = model.model_options["transformer_options"].get("minimax_head_chunks", 1)
    if type(groups) is not int or not 1 <= groups <= 56:
        raise ValueError("Standalone EAV KJ head chunks are outside the native supported range")
    source = next(iter(delegates.values())).__func__.__globals__["__file__"]
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    backend = HeadGroupedBackend(memory.RESMemoryDelegate(UserSelectedBackend(original_override), kernels.pop(), digest),
        groups, dict(kind="RES_KJ_Sol_standalone_EAV", head_chunks=groups, ffn_settings=None, source_sha256s=[digest]))
    branch = model.clone()
    original = {path: model.object_patches[path] for path in delegates}
    for path, delegate in delegates.items():
        branch.add_object_patch(path, sol._compose_module_patch(delegate.__self__, _make_memory_forward(delegate, backend)))
    branch.set_attachments(memory.KEY, dict(backend=backend, original_composers=original, source_sha256=memory._source_methods()))
    selector = _router(backend)
    branch.model_options["transformer_options"]["optimized_attention_override"] = selector
    branch.set_attachments(KEY, dict(selector=selector, original_override=original_override,
        backend=backend, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    _owner(branch)
    memory.inspect_owner(branch, backend)
    return branch


def bind_runtime(model, route):
    """Add only the actual EAV route token; never invent a Relay runtime."""
    if model.get_attachment(KEY) is not None:
        bind_memory_runtime(_owner(model)["backend"], route)


def project(model):
    """Authenticate the standalone selector; normalize only an inspection clone."""
    if model.get_attachment(KEY) is None:
        return model, None
    owner = _owner(model)
    if model.model_options["transformer_options"].get("optimized_attention_override") is not owner["selector"]:
        raise UnverifiedModelStack("RES standalone EAV selector was replaced by another execution owner")
    _, contract = memory.inspect_owner(model, owner["backend"])
    branch = model.clone()
    branch.model_options["transformer_options"]["optimized_attention_override"] = owner["original_override"]
    branch.remove_attachments(KEY)
    return branch, dict(source_sha256=owner["source_sha256"], original_KJ_Sol_memory=contract,
        synthetic_Relay_installed=False, inspecting_MODEL_mutated=False, CUDA_numerical_qualified=False)
