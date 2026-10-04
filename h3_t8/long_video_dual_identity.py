"""Content identity for native H3 dual-stage inputs, never a filename claim."""
from __future__ import annotations

from .patch_stack_policy import UnverifiedModelStack

import hashlib
import importlib.metadata
import inspect
from copy import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:
    # Pre-quantization Core can still import and run existing plain H3 nodes.
    # This sentinel grants no INT8 capability; that adapter needs actual Core.
    class QuantizedTensor:
        pass

from .h3_core_compat import plain_attention_backend
from .h3_memory_advanced import (
    ATTACHMENT_KEY as T8_MEMORY_ATTACHMENT_KEY,
    inspect_t8_memory_composition,
)
from .long_video_in_node_loop_advanced import _sha256_json, _check_interrupted
from .relay_kj_memory import inspect_memory_composition as inspect_kj_memory_composition
from .relay_sol_backend import capture_composed_backend
from .runtime_precision_identity import matmul_precision_identity
from .video_outpaint_identity import value_identity


def content_identity(value):
    if isinstance(value, QuantizedTensor):
        raise ValueError("QuantizedTensor identity requires an audited raw-storage adapter; no dequantization")
    if isinstance(value, torch.Tensor) and str(value.dtype).startswith("torch.float8_"):
        if value.device.type == "meta" or value.layout != torch.strided or not value.is_contiguous():
            raise ValueError("Float8 identity requires a materialized contiguous tensor")
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        # CPU isfinite does not implement all float8 formats. Widen only a
        # bounded1MiB chunk for the check, hash original bytes (no dequantize).
        for start in range(0, flat.numel(), 1024 * 1024):
            _check_interrupted()
            chunk = flat[start:start + 1024 * 1024].cpu()
            if not torch.isfinite(chunk.float()).all():
                raise ValueError("execution tensor has nonfinite values")
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return {"tensor_sha256": digest.hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}
    # New Core stores LoRA descriptors as adapter instances, older Core as
    # tuples. Inspect only this exact inert weight container, never arbitrary
    # object repr, callback code, subclass or adapter forward hooks.
    try:
        from comfy.weight_adapter.lora import LoRAAdapter
    except ImportError:
        LoRAAdapter = None
    if LoRAAdapter is not None and type(value) is LoRAAdapter:
        if not all(type(key) is str for key in value.loaded_keys):
            raise ValueError("LoRAAdapter loaded_keys must be a string set")
        if set(vars(value)) != {"loaded_keys", "weights"}:
            content_identity(value.weights)
            raise UnverifiedModelStack("LoRAAdapter has additional execution state outside its portable weight descriptor")
        return {"adapter": "comfy.LoRAAdapter", "implementation": _implementation(LoRAAdapter),
                "loaded_keys": sorted(value.loaded_keys), "weights": content_identity(value.weights)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [content_identity(item) for item in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: content_identity(item) for key, item in sorted(value.items())}
    try:
        return value_identity(value, interrupt_check=_check_interrupted)
    except ValueError as error:
        if "explicitly audited identity adapter" not in str(error):
            raise
        raise UnverifiedModelStack(str(error)) from error


def _implementation(function):
    path = inspect.getsourcefile(function)
    if path is None:
        raise ValueError("Stage implementation source is unavailable")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int8_backup_state(model, key, weight, state):
    """Serialize one original INT8 weight without changing its live module."""
    from comfy import ops, utils
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    if type(weight) is not QuantizedTensor or not key.endswith(".weight"):
        raise ValueError("Quantized backup requires an exact INT8 weight adapter")
    module, name = utils.resolve_attr(model.model, key)
    params = weight.params
    if (name != "weight" or weight.layout_cls is not TensorWiseINT8Layout
            or weight._layout_cls != "TensorWiseINT8Layout"
            or type(params) is not TensorWiseINT8Layout.Params
            or getattr(module, "quant_format", None) != "int8_tensorwise"
            or getattr(module, "layout_type", None) != weight._layout_cls
            or type(getattr(module, "_full_precision_mm_config", None)) is not bool):
        raise ValueError("Quantized backup layout or module configuration is outside the audited INT8 adapter")
    fields = {"scale", "orig_dtype", "orig_shape", "is_weight", "convrot", "convrot_groupsize", "transposed"}
    if (set(vars(params)) != fields or params.is_weight is not True or params.transposed is not False
            or type(params.convrot) is not bool or type(params.convrot_groupsize) is not int
            or params.convrot_groupsize <= 0 or params.convrot_groupsize & (params.convrot_groupsize - 1)
            or params.orig_dtype not in (torch.float32, torch.float16, torch.bfloat16)):
        raise ValueError("Quantized backup INT8 parameters are not the audited immutable weight schema")
    raw, scale = weight._qdata, params.scale
    if (type(raw) not in (torch.Tensor, torch.nn.Parameter) or raw.dtype != torch.int8
            or raw.ndim != 2 or raw.device.type == "meta" or raw.layout != torch.strided
            or not raw.is_contiguous() or tuple(raw.shape) != tuple(params.orig_shape)
            or type(scale) not in (torch.Tensor, torch.nn.Parameter) or not scale.is_floating_point()
            or scale.device.type == "meta" or scale.layout != torch.strided or not scale.is_contiguous()
            or (params.convrot and raw.shape[-1] % params.convrot_groupsize)):
        raise ValueError("Quantized backup requires materialized contiguous INT8 storage and scale")
    try:
        if torch.broadcast_shapes(raw.shape, scale.shape) != raw.shape:
            raise ValueError("Quantized backup scale does not broadcast to its original INT8 storage")
    except RuntimeError as error:
        raise ValueError("Quantized backup scale does not broadcast to its original INT8 storage") from error
    prefix = key[:-len("weight")]
    proxy = SimpleNamespace(weight=weight, bias=None, quant_format=module.quant_format,
                            _full_precision_mm_config=module._full_precision_mm_config)
    expected = {key, key + "_scale", prefix + "comfy_quant"}
    for extra in ("input_scale", "pre_quant_scale"):
        extra_key = prefix + extra
        if (getattr(module, extra, None) is not None) != (extra_key in state):
            raise ValueError("Quantized backup extra-scale state is missing or inconsistent")
        value = state.get(extra_key)
        if value is not None:
            if type(value) not in (torch.Tensor, torch.nn.Parameter):
                raise ValueError("Quantized backup extra scales require original plain tensor state")
            expected.add(extra_key)
        setattr(proxy, extra, value)
    # Reuse Core's serializer with an inert read-only proxy. In particular the
    # scale and ConvRot metadata come from the backup, not the applied LoRA.
    restored = ops._quantized_weight_state_dict(
        proxy, {}, prefix, extra_quant_params=("input_scale", "pre_quant_scale"))
    if (set(restored) != expected or not expected.issubset(state)
            or restored[key] is not raw or restored[key + "_scale"] is not scale):
        raise ValueError("Quantized backup serializer did not return its exact original raw-storage schema")
    metadata = restored[prefix + "comfy_quant"]
    if (type(metadata) is not torch.Tensor or metadata.dtype != torch.uint8 or metadata.ndim != 1
            or not 0 < metadata.numel() <= 65536):
        raise ValueError("Quantized backup metadata requires a bounded Core byte tensor")
    return restored


def _original_state(model, state):
    """Recover pre-LoRA storage from Core backups, never unpatch shared MODELs."""
    backups = getattr(model, "backup", {})
    buffers = getattr(model, "backup_buffers", {})
    if not backups and not buffers:
        return state
    original = dict(state)
    # Dynamic Core may cast buffers as well as weights. Restore them before
    # reading extra quantization scales; never mix original weights/live scales.
    for key, value in buffers.items():
        if (key not in original or not isinstance(value, torch.Tensor)
                or isinstance(value, QuantizedTensor)):
            raise ValueError("Cannot reconstruct original MODEL identity from this buffer backup format")
        original[key] = value
    quantized = []
    for key, backup in backups.items():
        weight = getattr(backup, "weight", None)
        if key not in original or not isinstance(weight, torch.Tensor):
            raise ValueError("Cannot reconstruct original MODEL identity from this backup format")
        if isinstance(weight, QuantizedTensor):
            quantized.append((key, weight))
        else:
            original[key] = weight
    # First finish all plain buffer/parameter restoration, then atomically merge
    # each complete raw-weight/scale/metadata group into this private dictionary.
    for key, weight in quantized:
        original.update(_int8_backup_state(model, key, weight, original))
    return original


CORE_SIGMA_SHIFT_SOURCE_SHA256 = "5999ed8c22b13c8251084c0629fd82137d2fb52b55106abfc73bf5cafc73841d"
CORE_SIGMA_SHIFT_EXECUTE_SHA256 = "ef0b67d716bab7cd65c83f86ac8067859d8f8712b64dba88a3d7b84356a710a3"


def _native_sigma_shift_factory(cls, core_sampling):
    """Recognize one audited empty Core carrier, never arbitrary subclasses.

    The local class has no executable members. Bind its exact actual Core
    factory source and bytecode too: filenames/qualnames alone are not proof.
    Existing instance, inherited-method, sigma-buffer and hook checks still run.
    This admits sampling identity, not arbitrary effect or video quality claims.
    """
    import json
    from types import CodeType

    native_av = getattr(core_sampling, "ModelSamplingAV", None)
    if (native_av is None or type(cls) is not type
            or cls.__bases__ != (native_av, core_sampling.CONST)
            or cls.__module__ != "comfy_extras.nodes_minimax_h3"
            or cls.__qualname__ != "MiniMaxH3SigmaShift.execute.<locals>.ModelSamplingAdvanced"
            or set(vars(cls)) - {"__module__", "__doc__"}):
        return None
    try:
        from comfy_extras import nodes_minimax_h3 as core_h3
    except ImportError as error:
        raise UnverifiedModelStack("Core SigmaShift factory is unavailable") from error
    expected_path = Path(core_sampling.__file__).resolve().parents[1] / "comfy_extras/nodes_minimax_h3.py"
    source_path = Path(core_h3.__file__).resolve()
    factory = inspect.unwrap(core_h3.MiniMaxH3SigmaShift.execute)
    if (source_path != expected_path or Path(inspect.getsourcefile(cls)).resolve() != source_path
            or factory.__module__ != core_h3.__name__
            or factory.__qualname__ != "MiniMaxH3SigmaShift.execute"
            or factory.__globals__ is not vars(core_h3)
            or factory.__closure__ is not None
            or Path(factory.__code__.co_filename).resolve() != source_path):
        raise UnverifiedModelStack("Core SigmaShift has a foreign executable factory")
    source = source_path.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    execute_sha = hashlib.sha256(inspect.getsource(factory).encode()).hexdigest()
    if source_sha != CORE_SIGMA_SHIFT_SOURCE_SHA256 or execute_sha != CORE_SIGMA_SHIFT_EXECUTE_SHA256:
        raise UnverifiedModelStack("Core SigmaShift source is outside the audited sampling factory")
    compiled = compile(source, factory.__code__.co_filename, "exec", dont_inherit=True)
    def code_named(parent, name):
        matches = [item for item in parent.co_consts if isinstance(item, CodeType) and item.co_name == name]
        if len(matches) != 1:
            raise UnverifiedModelStack("Core SigmaShift has an ambiguous factory implementation")
        return matches[0]
    expected = code_named(code_named(compiled, "MiniMaxH3SigmaShift"), "execute")
    def code_payload(code):
        def constant(item):
            if isinstance(item, CodeType):
                return {"type": "code", "value": code_payload(item)}
            if type(item) is bytes:
                return {"type": "bytes", "value": item.hex()}
            if type(item) is tuple:
                return {"type": "tuple", "value": [constant(value) for value in item]}
            if item is None or type(item) in (bool, int, float, str):
                return {"type": type(item).__name__, "value": item}
            raise UnverifiedModelStack("Core SigmaShift has an unknown executable constant")
        # marshal's intern/reference flags can differ for equal loaded/compiled
        # code objects. Compare exact typed executable fields instead. The
        # verified source path is deliberately not a portable content field.
        fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
                  "co_stacksize", "co_flags", "co_name", "co_qualname", "co_firstlineno",
                  "co_names", "co_varnames", "co_freevars", "co_cellvars")
        return {**{name: getattr(code, name, None) for name in fields},
                "co_code": code.co_code.hex(), "co_linetable": code.co_linetable.hex(),
                "co_exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
                "co_consts": [constant(value) for value in code.co_consts]}
    live_payload, expected_payload = code_payload(factory.__code__), code_payload(expected)
    if live_payload != expected_payload:
        raise UnverifiedModelStack("Core SigmaShift live factory bytecode differs from its source")
    return {"owner": factory.__qualname__, "source_sha256": source_sha,
            "execute_sha256": execute_sha,
            "bytecode_sha256": hashlib.sha256(json.dumps(live_payload, sort_keys=True).encode()).hexdigest()}


def _v2_sampling_identity(value):
    """Audited inert T8 and native Core AV sampling surfaces only."""
    import comfy.model_sampling as core_sampling
    from . import sampling

    cls = type(value)
    native_factory = None
    native_av = getattr(core_sampling, "ModelSamplingAV", None)
    if cls is sampling.MiniMaxH3FlowSampling:
        if set(vars(cls)) - {"__module__", "__doc__", "audio_scale"}:
            raise UnverifiedModelStack("V2 sampling class contains unknown execution members")
        protocol = "t8_raw_audio_flow"
    elif (native_av is not None
          and cls.__bases__ == (native_av, core_sampling.CONST)
          and cls.__module__ == sampling.__name__
          and cls.__qualname__ == "_make_sampling.<locals>.MiniMaxH3NativeAVSampling"
          and not (set(vars(cls)) - {"__module__", "__doc__"})):
        # _make_sampling creates a fresh empty subclass each time. Its class
        # identity is not stable, but its exact bases and empty executable
        # namespace are: no arbitrary subclass or repr-based exemption.
        protocol = "native_av_carrier"
    elif (native_factory := _native_sigma_shift_factory(cls, core_sampling)) is not None:
        protocol = "core_minimax_h3_sigma_shift"
    else:
        raise UnverifiedModelStack("V2 model_sampling lacks a portable native sampling schema")
    reference = cls()
    attributes = vars(value)
    if set(attributes) != set(vars(reference)):
        raise UnverifiedModelStack("V2 model_sampling contains unknown instance execution state")
    scalars = {"training", "noise_scale", "shift", "multiplier", "audio_shift"}
    configuration = {}
    for key, item in attributes.items():
        if key in scalars:
            configuration[key] = content_identity(item)
        elif key == "_buffers":
            if set(item) != {"sigmas"} or not isinstance(item["sigmas"], torch.Tensor):
                raise ValueError("V2 model_sampling requires its native sigma buffer")
            if item["sigmas"].ndim != 1 or not item["sigmas"].numel():
                raise ValueError("V2 model_sampling sigma buffer has an invalid shape")
            configuration[key] = content_identity(dict(item))
        elif key == "_is_full_backward_hook":
            if item is not None:
                raise UnverifiedModelStack("V2 model_sampling contains backward execution hooks")
        elif isinstance(item, (dict, set)):
            if item:
                raise UnverifiedModelStack("V2 model_sampling contains unknown live hooks or modules")
        else:
            raise UnverifiedModelStack(f"V2 model_sampling field {key} lacks a portable schema")
    if type(configuration["training"]) is not bool or any(
        type(configuration[key]) not in (int, float)
        for key in ("noise_scale", "shift", "multiplier")
    ):
        raise ValueError("V2 model_sampling has non-native scalar parameters")
    methods = {}
    for name in ("set_parameters", "set_noise_scale", "timestep", "sigma", "percent_to_sigma",
                 "calculate_input", "calculate_denoised", "noise_scaling", "inverse_noise_scaling"):
        function = getattr(cls, name)
        if inspect.getsourcefile(function) != inspect.getsourcefile(core_sampling.ModelSamplingDiscreteFlow):
            raise UnverifiedModelStack("V2 model_sampling inherited method has a foreign source owner")
        methods[name] = {"qualname": function.__qualname__, "source": _implementation(function)}
    audio_scale = inspect.getattr_static(cls, "audio_scale")
    if not isinstance(audio_scale, property):
        raise ValueError("V2 model_sampling audio carrier property was replaced")
    expected_source = sampling if protocol == "t8_raw_audio_flow" else core_sampling
    if inspect.getsourcefile(audio_scale.fget) != expected_source.__file__:
        raise ValueError("V2 model_sampling audio carrier has a foreign source owner")
    methods["audio_scale"] = {"source": _implementation(audio_scale.fget),
                               "value": content_identity(value.audio_scale)}
    identity = {"schema": "t8.fasth3_v2.native_sampling/v1", "protocol": protocol,
            "configuration": configuration, "implementation": _implementation(cls), "methods": methods}
    if native_factory is not None:
        identity["factory"] = native_factory
    return identity


def _v2_stage_adapter(model):
    """Normalize only authenticated V2 ownership on a read-only MODEL clone."""
    from . import fast_h3_v2_advanced as v2
    from .vdn_attention_compat import _factory_closure, native_sparse_state
    import comfy.patcher_extension as extension

    receipt = v2.capture_fast_h3_v2_owner(model)
    if receipt is None or type(receipt) is not v2._V2Receipt or type(receipt.runtime) is not v2._V2Runtime:
        raise ValueError("V2 stage identity needs the exact authenticated runtime receipt")
    runtime = receipt.runtime
    live = model.model_options["transformer_options"]
    dit = live.get("patches_replace", {}).get("dit", {})
    if (live.get("optimized_attention_override") is not runtime.override
            or set(dit) != set(runtime.dit)
            or any(dit[key] is not function for key, function in runtime.dit.items())):
        raise UnverifiedModelStack("V2 live user-selected owners need execution-local cache identity")
    if any(key in vars(runtime) for key in ("validate_options", "_config", "block_patch", "dense_sol_contract")):
        raise ValueError("V2 runtime methods were replaced outside their source owner")
    original_sampling = getattr(model, "object_patches_backup", {}).get("model_sampling")
    if original_sampling is not None:
        selected_sampling = model.object_patches.get("model_sampling")
        if selected_sampling is None or model.model.model_sampling is not selected_sampling:
            raise UnverifiedModelStack("V2 live model_sampling owner differs from selected patch")
    previous = runtime.override
    sparse_config = None
    kernel = None
    protected_sol = runtime.dense_sol_backend is not None
    if protected_sol:
        if receipt.profile != "dense_compat_exp":
            raise ValueError("Protected Sol cannot replace trained V2 VSA")
        state = _factory_closure(runtime.override, v2._install_runtime, "sol_override")
        if state is None or state.get("runtime") is not runtime:
            raise ValueError("Protected Sol has a foreign execution factory")
        previous = runtime.previous_override
    if receipt.profile != "dense_compat_exp":
        sparse_state = native_sparse_state(runtime.override, "override")
        if sparse_state is None or sparse_state["patch"] is not runtime.patch:
            raise ValueError("V2 stage identity needs the actual native sparse override closure")
        if runtime.sparse is not inspect.getmodule(type(runtime.patch)):
            raise ValueError("V2 runtime sparse module was replaced outside native Core")
        previous = sparse_state["previous"]
        for role, function, name in (
            ("guard", runtime.guard, "guard"), ("prepare", runtime.prepare, "prepare"),
            ("cleanup", runtime.cleanup, "cleanup"),
        ):
            state = _factory_closure(function, v2._install_runtime, name)
            if state is None or state.get("runtime") is not runtime:
                raise ValueError(f"V2 {role} has a foreign execution factory")
        blocks = v2._gates(model).blocks
        if set(runtime.dit) != {("double_block", index) for index in range(len(blocks))}:
            raise UnverifiedModelStack("V2 stage identity found unaudited block producer keys")
        for (_, index), hook in runtime.dit.items():
            state = _factory_closure(hook, v2._V2Runtime.block_patch, "patch")
            if (state is None or state.get("self") is not runtime
                    or state.get("block") is not blocks[index] or state.get("index") != index):
                raise UnverifiedModelStack("V2 block producer is composed with a foreign execution factory")
            attention = _factory_closure(state.get("attention"), v2._V2Runtime.block_patch, "attention")
            if attention is None or attention.get("self") is not runtime or attention.get("block") is not blocks[index]:
                raise ValueError("V2 gate-aware attention producer has a foreign factory")
        sparse_config = {key: content_identity(getattr(runtime.patch, key)) for key in
                         ("tau", "topk_ratio", "vsa", "sigma_start", "sigma_end", "min_tokens",
                          "sink_conditioning", "extra_tokens", "verbose")}
        sparse_config["dense_blocks"] = sorted(runtime.patch.dense_blocks)
        kernel = {"core_source": _implementation(runtime.sparse.SparseAttnPatch),
                  "kitchen_source": _implementation(runtime.sparse.ck.sol_attn_chunked),
                  "kitchen_version": importlib.metadata.version("comfy-kitchen"),
                  "block_size": runtime.sparse.BLOCK_SIZE, "head_dim": runtime.sparse.HEAD_DIM,
                  "producer_chunk": runtime.sparse.PRODUCER_CHUNK, "vsa_cube": list(runtime.sparse.VSA_CUBE)}
    elif runtime.dit:
        raise UnverifiedModelStack("Dense V2 does not authenticate another DiT producer")
    if receipt.profile != "dense_compat_exp" or protected_sol:
        for function, name in ((runtime.guard, "guard"), (runtime.prepare, "prepare"), (runtime.cleanup, "cleanup")):
            state = _factory_closure(function, v2._install_runtime, name)
            if state is None or state.get("runtime") is not runtime:
                raise ValueError(f"V2 {name} has a foreign execution factory")
    if previous is not None and plain_attention_backend(previous) is None and not protected_sol:
        raise UnverifiedModelStack("V2 stage identity cannot authenticate the selected attention backend")
    from comfy.ldm.modules import attention
    backend = (attention.optimized_attention if previous is None else previous)
    contract = {"schema": v2.SCHEMA, "profile": receipt.profile, "head_chunks": runtime.head_chunks,
                "sparse": sparse_config, "kernel": kernel, "implementation": _implementation(v2._V2Runtime),
                "provider": _implementation(_v2_stage_adapter), "rungs": list(v2.RUNG_STEPS),
                "video_shift": v2.VIDEO_SHIFT, "audio_shift": v2.AUDIO_SHIFT,
                "initialization": "raw_target_gaussian_first_window; native_AV_partial_restart_exp",
                "original_backend_source": _implementation(backend),
                "official_template_schedule": "Core simple8/res_multistep" if receipt.profile == "official_comfy_template_exp" else None,
                "video_sigmas": content_identity(v2.dmd_sigmas()),
                "audio_sigmas": content_identity(v2.dmd_sigmas(v2.AUDIO_SHIFT))}
    if protected_sol:
        contract["protected_sol"] = runtime.dense_sol_contract()
    cloned = model.clone()
    if original_sampling is not None:
        # A prior stage may leave Core's selected sampling object installed
        # while HIGH loads. Restore only the dormant module on a read-only
        # shallow network view; never unpatch or unload the executing MODEL.
        cloned.model = copy(model.model)
        cloned.model._modules = dict(model.model._modules)
        cloned.model._modules["model_sampling"] = original_sampling
        cloned.object_patches_backup = dict(model.object_patches_backup)
        cloned.object_patches_backup.pop("model_sampling")
    options = cloned.model_options["transformer_options"]
    options.pop(v2.RUN_KEY)
    if previous is None:
        options.pop("optimized_attention_override", None)
    else:
        options["optimized_attention_override"] = previous
    replaces = dict(options.get("patches_replace", {}))
    if runtime.dit:
        # Exact-key/function validation above proves this set belongs only to
        # V2. Never clear another replacement namespace or arbitrary callback.
        replaces.pop("dit")
        if replaces:
            options["patches_replace"] = replaces
        else:
            options.pop("patches_replace", None)
    wrappers = cloned.wrappers.get("diffusion_model", {})
    if receipt.profile != "dense_compat_exp" or protected_sol:
        wrappers.pop(v2.KEY)
        for role in (extension.CallbacksMP.ON_PREPARE_STATE, extension.CallbacksMP.ON_CLEANUP):
            groups = cloned.callbacks[role]
            groups.pop(v2.KEY)
            if not groups:
                cloned.callbacks.pop(role)
    cloned.remove_attachments(v2.KEY)
    sampling_object = cloned.object_patches.pop("model_sampling", None)
    contract["model_sampling"] = (None if sampling_object is None else _v2_sampling_identity(sampling_object))
    return cloned, contract


def stage_model_identity(model, *, _omit_dormant_sampling=False):
    from .taeh3_sampling_preview import cache_projection
    model = cache_projection(model)
    try:
        if model.get_injections("bypass_lora"):
            from .modular_sampling import core_bypass_identity
            view, contract = core_bypass_identity.project(model)
            base = _audited_stage_model_identity(view, _omit_dormant_sampling=_omit_dormant_sampling)
            return {**base, "schema": "t8.h3.dual_stage_model/core-bypass-v1", "core_bypass_lora": contract,
                    "sha256": _sha256_json({"base": base, "core_bypass_lora": contract})}
        return _audited_stage_model_identity(model, _omit_dormant_sampling=_omit_dormant_sampling)
    except UnverifiedModelStack as error:
        from .patch_stack_policy import nonportable_model_identity
        return nonportable_model_identity(model, str(error), schema="t8.h3.dual_stage_model/user_stack_v1")


def _audited_stage_model_identity(model, *, _omit_dormant_sampling=False):
    from comfy.model_base import MiniMaxH3
    if not isinstance(model.model, MiniMaxH3):
        raise ValueError("Dual-model native4+4 loop requires native H3 MODELs; VDN uses a separate8+4 contract")
    if "apply_model" in vars(model.model):
        selected_apply = model.model.apply_model
        if (getattr(selected_apply, "__self__", None) is not model.model
                or getattr(selected_apply, "__func__", None) is not type(model.model).apply_model):
            raise UnverifiedModelStack("Dual-stage base MODEL apply_model was replaced outside its class")
    if getattr(model, "attachments", {}).get("t8_fasth3_v2_owner_v1") is not None:
        normalized, v2_contract = _v2_stage_adapter(model)
        identity = stage_model_identity(normalized, _omit_dormant_sampling=True)
        return {**identity, "schema": "t8.h3.dual_stage_model/v2",
                "sha256": _sha256_json({"base": identity, "fast_h3_v2": v2_contract}),
                "fast_h3_v2": v2_contract}
    # New algorithms need a deliberate identity and owner adapter. A plausible
    # string representation or UUID does not prove byte-identical execution.
    for name in ("weight_wrapper_patches", "additional_models", "callbacks", "injections",
                 "hook_patches", "forced_hooks", "current_hooks"):
        if getattr(model, name, None):
            raise UnverifiedModelStack(f'Dual-stage content identity does not yet cover MODEL {name}')
    attachments = {key: value for key, value in getattr(model, "attachments", {}).items() if value is not None}
    lora_metadata = attachments.pop("t8_h3_lora_metadata", None)
    if lora_metadata is not None and (type(lora_metadata) is not dict or
            not all(type(key) is str and type(value) is str for key, value in lora_metadata.items())):
        raise ValueError("H3 LoRA metadata must be the loader's plain safetensors string map")
    # Current stock Core LoRA loaders use this separate attachment name for
    # the same inert safetensors string map. Bind its exact data; it does not
    # authenticate another hook, adapter subclass or executable attachment.
    core_lora_metadata = attachments.pop("lora_metadata", None)
    if core_lora_metadata is not None and (type(core_lora_metadata) is not dict or
            not all(type(key) is str and type(value) is str for key, value in core_lora_metadata.items())):
        raise ValueError("Core LoRA metadata must be the loader's plain safetensors string map")
    # The compatibility loader attaches an inert diagnostic report after
    # adding native Core weight patches. Preserve its exact contents in the
    # identity, but do not mistake that known JSON receipt for an executable
    # attachment. Actual patch tensor bytes/order are bound below as before.
    lora_report = attachments.pop("t8_h3_lora_compat_report", None)
    if lora_report is not None and (type(lora_report) is not dict or
            lora_report.get("schema") != "t8.minimax_h3.lora_compat.v1" or
            lora_report.get("status") not in ("applied", "no_compatible_patches") or
            lora_report.get("full_algorithm_verified") is not False or
            lora_report.get("model_class") != type(model.model).__name__ or
            type(lora_report.get("file")) is not dict or
            lora_report["file"].get("identity_policy") != "display_only_not_a_load_gate_no_hash_scan"):
        raise UnverifiedModelStack("H3 LoRA compatibility report is not the known inert loader receipt")
    lora_report_identity = None if lora_report is None else content_identity(lora_report)
    t8_memory_attachment = attachments.pop(T8_MEMORY_ATTACHMENT_KEY, None)
    t8_memory = inspect_t8_memory_composition(model)
    if (t8_memory_attachment is None) != (t8_memory is None):
        raise UnverifiedModelStack("Dual-stage T8 memory stack is not portable-cache authenticated")
    if attachments:
        raise UnverifiedModelStack('Dual-stage MODEL has unknown attachments; identity adapter required')
    # T8 and KJ both replace the same H3 forward methods.  Once the exact T8
    # receipt/token/wrapper owner has authenticated those replacements, do not
    # feed them to the unrelated KJ source-contract inspector.
    kj_memory = None if t8_memory is not None else inspect_kj_memory_composition(model)
    if t8_memory is not None and kj_memory is not None:
        raise UnverifiedModelStack('Dual-stage MODEL cannot combine T8 and KJ memory owners')
    memory = t8_memory if t8_memory is not None else kj_memory
    allowed = {} if memory is None else {key[:-8]: method for key, method in memory["methods"].items()}
    if set(model.object_patches) != (set() if memory is None else set(memory["methods"])):
        raise UnverifiedModelStack('Dual-stage MODEL has object patches outside the audited memory route')
    active_wrappers = {
        key
        for wrapper_type in model.wrappers.values()
        for key, values in wrapper_type.items()
        if values
    }
    allowed_wrappers = set(t8_memory.get("wrapper_keys", ())) if t8_memory is not None else set()
    if active_wrappers != allowed_wrappers:
        raise UnverifiedModelStack('Dual-stage MODEL has wrappers outside the audited memory route')
    options = dict(model.model_options.get("transformer_options", {}))
    override = options.pop("optimized_attention_override", None)
    backend = capture_composed_backend(override)
    if backend is not None:
        backend_contract = backend.report()
        if backend_contract.get("portable_cache_reuse") is False:
            raise UnverifiedModelStack("Unrecognized attention delegate needs execution-local cache identity")
        backend_contract.pop("completed_calls", None)
    elif override is not None:
        plain = plain_attention_backend(override)
        if plain is None:
            raise UnverifiedModelStack('Dual-stage MODEL contains an unrecognized attention owner')
        backend_contract = {"kind": "core_plain", "name": plain, "implementation": _implementation(override)}
    else:
        from comfy.ldm.modules import attention
        backend_contract = {"kind": "core_global", "name": attention.optimized_attention.__name__,
                            "implementation": _implementation(attention.optimized_attention)}
    if "sol_take_forward" in options:
        # The selected memory adapter already authenticates this exact callable.
        if memory is None:
            raise UnverifiedModelStack('Sol forward delegate without its verified memory owner')
        options.pop("sol_take_forward")
    if t8_memory is not None:
        for key in t8_memory["runtime_option_keys"]:
            options.pop(key, None)
    configuration = {key: value for key, value in model.model_options.items() if key != "transformer_options"}
    configuration["transformer_options"] = options
    implementations = {}
    classes = []
    for name, module in model.model.named_modules():
        if _omit_dormant_sampling and (name == "model_sampling" or name.startswith("model_sampling.")):
            continue
        if module._forward_pre_hooks or module._forward_hooks:
            raise UnverifiedModelStack('Dual-stage MODEL contains shared live hooks; no cross-branch identity proof')
        current = vars(module).get("forward")
        expected = allowed.get(name)
        if current is not None and not (
            isinstance(current, MethodType) and current.__self__ is module
            and (current.__func__ is type(module).forward or
                 (expected is not None and current.__func__ is expected.__func__))
        ):
            raise UnverifiedModelStack('Dual-stage model forward was replaced outside the selected MODEL')
        cls = type(module)
        if cls not in implementations:
            implementations[cls] = _implementation(cls)
        classes.append((name, cls.__module__ + "." + cls.__qualname__))
    state = model.model_state_dict()
    if not state:
        raise ValueError("Dual-stage MODEL has no loaded tensor state")
    state = _original_state(model, state)
    if _omit_dormant_sampling:
        # The dormant Core sampling module may be swapped on first load while
        # its selected V2 replacement is already fully authenticated above by
        # _v2_sampling_identity. It is not executed in this stage. Bind every
        # diffusion/LoRA tensor unchanged; omit only dormant sampling buffers.
        state = {key: value for key, value in state.items()
                 if not key.startswith("model_sampling.")}
    data = {"schema": "t8.h3.dual_stage_model/v1", "classes": classes,
            "implementations": sorted(implementations.values()), "state": content_identity(state),
            "patches": content_identity(model.patches), "model_options": content_identity(configuration),
            "lora_metadata": content_identity(lora_metadata),
            "backend": backend_contract,
            "memory": None if memory is None else {key: memory[key] for key in
                ("kind", "head_chunks", "ffn_settings", "source_sha256s")},
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "matmul_precision": matmul_precision_identity(),
                        "deterministic": torch.are_deterministic_algorithms_enabled()}}
    if _omit_dormant_sampling:
        data["v2_raw_noise_scale"] = content_identity(
            getattr(model.model.model_sampling, "noise_scale", None))
    if lora_report_identity is not None:
        data["lora_compat_report"] = lora_report_identity
    if core_lora_metadata is not None:
        data["core_lora_metadata"] = content_identity(core_lora_metadata)
    config = getattr(model.model, "model_config", None)
    data["unet_config"] = content_identity(getattr(config, "unet_config", None))
    data["manual_cast_dtype"] = str(getattr(model.model, "manual_cast_dtype", None))
    return {"sha256": _sha256_json(data), "schema": data["schema"], "backend": backend_contract,
            "memory": data["memory"], "model_filename_trusted": False,
            "tensor_count": len(state), "lora_target_count": len(model.patches)}
