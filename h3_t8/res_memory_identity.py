"""RES-only inspection adapter for native T8 FFN plus audited KJ memory.

Never unpatch the executing MODEL. Sol's additional closures/hooks and unknown
owners are deliberately retained, and therefore still require their own adapter.
"""
from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

from comfy.ldm.minimax import model as core_h3

from . import h3_memory_advanced as t8
from .long_video_dual_identity import _audited_stage_model_identity, content_identity
from .modular_sampling.progressive_effect_identity import function_identity
from .patch_stack_policy import UnverifiedModelStack
from .relay_kj_memory import inspect_memory_composition
from .vdn_attention_compat import _factory_closure


def mixed_memory_model_identity(model):
    """Return an identity only when each removed owner is authenticated.

    Normalization is confined to an inspection-only shallow module tree. Every
    tensor, ordered weight patch, backend, callback, hook and unrecognized owner
    remains available to the existing strict identity validator.
    """
    receipt = model.get_attachment(t8.ATTACHMENT_KEY)
    if receipt is None or receipt.ffn is None or receipt.attention is not None:
        return None
    record = receipt.ffn
    extra = {key: value for key, value in model.object_patches.items() if key not in record.paths}
    if not extra:
        return None
    if type(receipt) is not t8._MemoryReceipt or type(record) is not t8._PatchRecord:
        raise UnverifiedModelStack("RES mixed memory requires the actual native T8 receipt")
    blocks = tuple(model.model.diffusion_model.blocks)
    expected_paths = tuple(f"diffusion_model.blocks.{index}.mlp.forward" for index in range(len(blocks)))
    if record.paths != expected_paths or record.owners != tuple(block.mlp for block in blocks):
        raise UnverifiedModelStack("RES T8 FFN receipt does not cover the actual complete block set")
    settings = dict(record.settings)
    if (set(settings) != {"chunks", "seq_threshold"}
            or any(type(value) is not int for value in settings.values())
            or not 2 <= settings["chunks"] <= 64 or not 256 <= settings["seq_threshold"] <= 262144):
        raise ValueError("RES T8 FFN settings are malformed")
    # Authenticate just the T8 projection. Unknown KJ/other methods have not
    # been authorized: they are verified independently below, never discarded.
    t8_view = model.clone()
    t8_view.object_patches = {path: model.object_patches[path] for path in record.paths}
    t8_contract = t8.inspect_t8_memory_composition(t8_view)
    if t8_contract is None:
        raise UnverifiedModelStack("RES T8 runtime receipt/wrapper is not portable")
    guard = _factory_closure(record.wrapper, t8._bind_runtime_guard, "guard")
    if guard is None or set(guard) != {"record"}:
        raise UnverifiedModelStack("RES T8 FFN guard is not its native factory")
    original_record = guard["record"]
    if (type(original_record) is not t8._PatchRecord or original_record.wrapper is not None
            or any(getattr(original_record, key) != getattr(record, key) for key in (
                "kind", "token", "paths", "owners", "methods", "settings", "wrapper_key"))):
        raise ValueError("RES T8 FFN guard and receipt differ")
    for owner, method in zip(record.owners, record.methods, strict=True):
        if type(owner) is not core_h3.MLP or not isinstance(method, MethodType) or method.__self__ is not owner:
            raise UnverifiedModelStack("RES FFN method has a foreign owner")
        state = _factory_closure(method.__func__, t8._make_ffn_forward, "forward")
        if (state is None or set(state) != {"chunks", "seq_threshold", "run"}
                or state["chunks"] != settings["chunks"] or state["seq_threshold"] != settings["seq_threshold"]
                or _factory_closure(state["run"], t8._make_ffn_forward, "run") != {"previous_forward": None}):
            raise UnverifiedModelStack("RES FFN requires the actual direct native factory, not an opaque delegate")
        current = vars(owner).get("forward")
        if current is not None and current is not method and not (
                isinstance(current, MethodType) and current.__self__ is owner
                and current.__func__ is type(owner).forward):
            raise UnverifiedModelStack("RES live FFN owner differs from its selected patch")
    kj_view = model.clone()
    kj_view.object_patches = extra
    kj_contract = inspect_memory_composition(kj_view)
    if (kj_contract is None or set(kj_contract["methods"]) != set(extra)
            or kj_contract["backend"] is not None):
        raise UnverifiedModelStack("RES additional memory forwards need their own portable adapter")
    # A shallow structural tree is needed only because Core may have installed
    # selected FFN methods on shared modules. Mutating those real modules would
    # corrupt the executing graph, so clone their containers, never their weights.
    normalized = model.clone()
    normalized.object_patches = extra
    normalized.remove_attachments(t8.ATTACHMENT_KEY)
    wrappers = normalized.wrappers.get("diffusion_model", {})
    wrappers.pop(record.wrapper_key)
    options = normalized.model_options["transformer_options"]
    tokens = options.pop(t8.RUNTIME_TOKEN_KEY)
    if set(tokens) != {"ffn"} or tokens["ffn"] is not record.token:
        raise ValueError("RES T8 FFN runtime token map changed")
    normalized.model = copy(model.model)
    normalized.model._modules = dict(model.model._modules)
    diffusion = copy(model.model.diffusion_model)
    diffusion._modules = dict(model.model.diffusion_model._modules)
    block_list = copy(model.model.diffusion_model.blocks)
    block_list._modules = dict(model.model.diffusion_model.blocks._modules)
    for index, block in enumerate(blocks):
        block_view = copy(block)
        block_view._modules = dict(block._modules)
        block_path = f"diffusion_model.blocks.{index}.forward"
        current = vars(block).get("forward")
        expected = extra.get(block_path)
        if current is not None:
            if not (isinstance(current, MethodType) and current.__self__ is block
                    and (current.__func__ is type(block).forward or
                         (expected is not None and current.__func__ is expected.__func__))):
                raise UnverifiedModelStack("RES live block owner differs from its selected KJ patch")
            block_view.forward = MethodType(current.__func__, block_view)
        if expected is not None:
            normalized.object_patches[block_path] = MethodType(expected.__func__, block_view)
        mlp_view = copy(block.mlp)
        mlp_view.__dict__.pop("forward", None)
        block_view._modules["mlp"] = mlp_view
        block_list._modules[str(index)] = block_view
    diffusion._modules["blocks"] = block_list
    normalized.model._modules["diffusion_model"] = diffusion
    # This strict call preserves all other gates. In particular a retained Sol
    # selector/composer/hook is NOT converted into a bare/pytorch execution.
    base = _audited_stage_model_identity(normalized, _omit_dormant_sampling=True)
    contract = dict(kind="native_T8_FFN_plus_audited_KJ_memory", paths=list(record.paths), settings=settings,
        factory=function_identity(t8._make_ffn_forward), guard=function_identity(t8._bind_runtime_guard),
        runtime_options=function_identity(t8._runtime_options),
        source_sha256=hashlib.sha256(Path(inspect.getsourcefile(t8._make_ffn_forward)).read_bytes()).hexdigest(),
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload = content_identity(dict(base=base, T8_FFN=contract))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**base, "schema": "t8.minimax_h3.RES_mixed_memory.v1", "sha256": digest,
            "mixed_memory": contract, "portable_cache_reuse": True}
