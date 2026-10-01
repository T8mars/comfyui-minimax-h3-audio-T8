"""Authenticate HyperFlow's actual two-time closures without changing a MODEL.

Only an inspection clone loses the 52 proven object patches and scope wrapper.
All content LoRAs remain in the stage identity. Common-base and original-adapter
content are separate from those deliberately independent per-stage patches.
Opaque user owners are classified by the caller, never stripped or prohibited.
"""
from dataclasses import asdict
from copy import copy
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, MethodType

import comfy.lora
import comfy.patcher_extension
import torch

from .. import hyperflow_runtime_advanced as runtime
from .. import long_video
from ..hyperflow_weights_advanced import HyperFlowMetadata, HyperFlowWeights
from ..long_video_dual_identity import content_identity, _original_state, _audited_stage_model_identity
from ..vdn_attention_compat import _factory_closure
from .effect_identity import _require
from .progressive_effect_identity import function_identity, _code_identity
from .results import sha


def _class_method(kind, name):
    function = inspect.getattr_static(kind, name)
    declaring = next(owner for owner in kind.__mro__ if name in vars(owner))
    _require(type(function) is FunctionType
             and inspect.getsourcefile(function) == inspect.getsourcefile(declaring)
             and function.__globals__ is vars(sys.modules[declaring.__module__]),
             f"HyperFlow native class {name} has a foreign source owner")
    return function


def _closure(function, factory, name, fields, *, marked=False, kwdefaults=None):
    state = _factory_closure(function, factory, name)
    _require(type(function) is FunctionType and state is not None and set(state) == set(fields),
             f"HyperFlow {name} needs its exact source closure")
    _require(function.__defaults__ is None and function.__kwdefaults__ == kwdefaults,
             f"HyperFlow {name} has unknown default arguments")
    attributes = {"_t8_hyperflow_owner", "_t8_hyperflow_inner"} if marked else set()
    _require(set(vars(function)) == attributes, f"HyperFlow {name} has unknown execution attributes")
    return state


def _native(delegate, module):
    descriptor = _class_method(type(module), "forward")
    _require(type(delegate) is MethodType and delegate.__self__ is module
             and delegate.__func__ is descriptor, "HyperFlow native delegate has another user owner")
    return function_identity(descriptor)


def _marked(function, binding, delegate):
    _require(function._t8_hyperflow_owner == binding.owner
             and function._t8_hyperflow_inner is delegate,
             "HyperFlow restoration descriptor differs from its executing delegate")


def _endpoint(function, time_module, weights, original):
    names = {"wi", "bi", "wo", "bo", "ai", "ui", "ao", "uo",
             "alpha_i", "alpha_o", "freq_dim", "cache"}
    state = _closure(function, runtime._endpoint_forward, "forward", names)
    _require(type(state["freq_dim"]) is int and state["freq_dim"] == int(time_module.freq_dim),
             "HyperFlow endpoint frequency geometry changed")
    expected = [original[f"diffusion_model.time_embedder.{part}.{kind}"]
                for part in ("proj_in", "proj_out") for kind in ("weight", "bias")]
    expected += [weights.endpoint[part][index] for part in ("proj_in", "proj_out") for index in (0, 1)]
    actual = [state[name] for name in ("wi", "bi", "wo", "bo", "ai", "ui", "ao", "uo")]
    for index, (observed, value) in enumerate(zip(actual, expected)):
        _require(type(observed) is torch.Tensor and type(value) in (torch.Tensor, torch.nn.Parameter),
                 "HyperFlow endpoint requires materialized native tensors")
        # Base snapshot is explicitly FP32; adapter A/B closures keep original
        # safetensors dtype until the real endpoint device cache casts them.
        reference = value.detach().to(device="cpu", dtype=torch.float32) if index < 4 else value
        _require(content_identity(observed) == content_identity(reference),
                 "HyperFlow endpoint captured weights differ from original base/adapter")
    _require(state["alpha_i"] == weights.endpoint["proj_in"][2]
             and state["alpha_o"] == weights.endpoint["proj_out"][2],
             "HyperFlow endpoint alpha differs from its loaded adapter")
    cache = state["cache"]
    _require(type(cache) is dict, "HyperFlow endpoint device cache has another owner")
    for device, tensors in cache.items():
        _require(type(device) is torch.device and type(tensors) is tuple and len(tensors) == 8,
                 "HyperFlow endpoint device cache has an unknown layout")
        for observed, value in zip(tensors, actual):
            _require(type(observed) is torch.Tensor and observed.device == device
                     and observed.dtype == torch.float32
                     and content_identity(observed) == content_identity(value.detach().to(device="cpu", dtype=torch.float32)),
                     "HyperFlow warm endpoint cache differs from its source tensors")
    # Devices and populated-cache keys are not numerical content; every cached
    # value above must equal the actual canonical source before normalization.
    return {"source": content_identity(tuple(actual)), "alpha": [state["alpha_i"], state["alpha_o"]],
            "freq_dim": state["freq_dim"], "code": sha(_code_identity(function.__code__))}


def _project_live_forwards(model, inspection, paths):
    """Normalize only proven installed two-time methods, never the live network.

    Standard SamplerCustomAdvanced leaves object patches installed. Continuous
    stages restore them themselves. Both must describe the same native base;
    removing patch descriptors alone does not normalize installed forwards.
    Clone only module containers on affected paths, sharing all tensor storage.
    """
    copied = {}

    def container(path):
        if path in copied:
            return copied[path]
        original = model.model.get_submodule(path) if path else model.model
        projected = copy(original)
        projected._modules = dict(original._modules)
        current = vars(projected).get("forward")
        if (type(current) is MethodType and current.__self__ is original
                and current.__func__ is _class_method(type(original), "forward")):
            # Core's restored native method alias is not an external override.
            del projected.forward
        copied[path] = projected
        if path:
            parent, _, child = path.rpartition(".")
            container(parent)._modules[child] = projected
        else:
            inspection.model = projected
        return projected

    for key in paths:
        path = key.removesuffix(".forward")
        module = model.model.get_submodule(path)
        current = vars(module).get("forward")
        if current is model.object_patches[key]:
            del container(path).forward
        elif current is not None:
            if type(current) is MethodType:
                _native(current, module)
            else:
                # Another authentic LOW/HIGH branch can occupy shared storage
                # before Core installs the selected branch. It is dormant for
                # this MODEL, not part of this branch's numerical selection.
                # Never normalize arbitrary marker-bearing user functions.
                if key == "diffusion_model.time_embedder.forward":
                    fields = _closure(current, runtime._install_two_time, "time_forward",
                        {"owner", "source_forward", "endpoint_forward", "weights"}, marked=True)
                    delegate = fields["source_forward"]
                elif key == "diffusion_model.final_layer.forward":
                    fields = _closure(current, runtime._install_two_time, "final_forward",
                        {"owner", "final_original", "remap"}, marked=True)
                    delegate = fields["final_original"]
                else:
                    defaults = getattr(current, "__kwdefaults__", None)
                    _require(type(defaults) is dict and set(defaults) == {"_inner"},
                             "HyperFlow dormant block has another delegate")
                    fields = _closure(current, runtime._install_two_time, "block_forward",
                        {"owner", "remap"}, marked=True, kwdefaults=defaults)
                    delegate = defaults["_inner"]
                _require(type(fields["owner"]) is str and bool(fields["owner"])
                         and current._t8_hyperflow_owner == fields["owner"]
                         and current._t8_hyperflow_inner is delegate,
                         "HyperFlow dormant branch owner changed")
                _native(delegate, module)
                if "remap" in fields:
                    _closure(fields["remap"], runtime._install_two_time, "remap", set())
                del container(path).forward


def project(model):
    binding = model.get_attachment(runtime.ATTACHMENT_KEY)
    _require(type(binding) is runtime.HyperFlowBinding
             and set(vars(binding)) == set(runtime.HyperFlowBinding.__dataclass_fields__)
             and binding.model_identity == id(model.model) and type(binding.owner) is str and bool(binding.owner),
             "HyperFlow requires its actual dedicated loader binding")
    diffusion = model.get_model_object("diffusion_model")
    _require(len(diffusion.blocks) == 50 and len(diffusion.token_refiner.blocks) == 2,
             "HyperFlow portable adapter requires its full 50+2 structure")
    key = "diffusion_model.time_embedder.forward"
    time_forward = model.object_patches.get(key)
    time = _closure(time_forward, runtime._install_two_time, "time_forward",
                    {"owner", "source_forward", "endpoint_forward", "weights"}, marked=True)
    _require(time["owner"] == binding.owner, "HyperFlow time owner changed")
    _marked(time_forward, binding, time["source_forward"])
    delegates = {key: _native(time["source_forward"], diffusion.time_embedder)}
    weights = time["weights"]
    _require(type(weights) is HyperFlowWeights and set(vars(weights)) == set(HyperFlowWeights.__dataclass_fields__)
             and type(weights.metadata) is HyperFlowMetadata
             and set(vars(weights.metadata)) == set(HyperFlowMetadata.__dataclass_fields__),
             "HyperFlow captured adapter has unknown execution state")
    _require(binding.sha256 == weights.source_sha256
             and all(getattr(binding, field) == getattr(weights.metadata, field)
                     for field in ("version", "gate", "video_shift", "audio_shift", "raw_sigmas")),
             "HyperFlow binding and executing weights disagree")
    runtime._preflight_structure(model, weights)
    _require(set(weights.endpoint) == {"proj_in", "proj_out"}, "HyperFlow endpoint target inventory changed")
    # Prove the original HyperFlow adapter is still actually installed, without
    # removing, reordering or normalizing any preceding/following content LoRA.
    source, key_map = {}, {}
    for target, (a, b, alpha) in weights.patches.items():
        source[f"{target}.lora_A.weight"] = a
        source[f"{target}.lora_B.weight"] = b
        source[f"{target}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
        key_map[target] = f"diffusion_model.{target}.weight"
    parsed = comfy.lora.load_lora(source, key_map, log_missing=False)
    _require(set(parsed) == set(key_map.values()) and len(parsed) == 210,
             "HyperFlow backbone/time adapter target inventory changed")
    for target, patch in parsed.items():
        expected = content_identity((1.0, patch, 1.0, None, None))
        _require(any(content_identity(value) == expected for value in model.patches.get(target, ())),
                 "HyperFlow original adapter no longer occurs in the selected patch stack")
    remaps = []
    for index, block in enumerate(diffusion.blocks):
        key = f"diffusion_model.blocks.{index}.forward"
        forward = model.object_patches.get(key)
        kwargs = getattr(forward, "__kwdefaults__", None)
        _require(type(kwargs) is dict and set(kwargs) == {"_inner"}, "HyperFlow block delegate is missing")
        state = _closure(forward, runtime._install_two_time, "block_forward", {"owner", "remap"},
                         marked=True, kwdefaults=kwargs)
        _require(state["owner"] == binding.owner, "HyperFlow block owner changed")
        _marked(forward, binding, kwargs["_inner"])
        delegates[key] = _native(kwargs["_inner"], block)
        remaps.append(state["remap"])
    key = "diffusion_model.final_layer.forward"
    forward = model.object_patches.get(key)
    final = _closure(forward, runtime._install_two_time, "final_forward",
                     {"owner", "final_original", "remap"}, marked=True)
    _require(final["owner"] == binding.owner, "HyperFlow final owner changed")
    _marked(forward, binding, final["final_original"])
    delegates[key] = _native(final["final_original"], diffusion.final_layer)
    remaps.append(final["remap"])
    _require(all(remap is remaps[0] for remap in remaps), "HyperFlow remap delegates disagree")
    _closure(remaps[0], runtime._install_two_time, "remap", set())
    kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    wrapper_key = "t8_hyperflow_" + binding.owner
    wrappers = model.wrappers.get(kind, {}).get(wrapper_key, ())
    _require(len(wrappers) == 1, "HyperFlow scope wrapper inventory changed")
    scope = _closure(wrappers[0], runtime._install_two_time, "scope", {"owner", "diffusion"})
    _require(scope["owner"] == binding.owner and scope["diffusion"] is diffusion,
             "HyperFlow scope is bound to another diffusion model")
    original = _original_state(model, model.model_state_dict())
    endpoint = _endpoint(time["endpoint_forward"], diffusion.time_embedder, weights, original)
    inspection = model.clone()
    _project_live_forwards(model, inspection, delegates)
    for key in delegates:
        inspection.object_patches.pop(key)
    inspection.remove_wrappers_with_key(kind, wrapper_key)
    inspection.remove_attachments(runtime.ATTACHMENT_KEY)
    adapter = {"metadata": content_identity(asdict(weights.metadata)),
               "source_sha256": weights.source_sha256, "patches": content_identity(weights.patches),
               "endpoint_weights": content_identity(weights.endpoint), "endpoint_execution": endpoint,
               "native_delegates": delegates,
               "runtime_factory": function_identity(runtime._install_two_time),
               "endpoint_factory": function_identity(runtime._endpoint_forward),
               "runtime_helpers": {name: function_identity(getattr(runtime, name)) for name in
                   ("_forward_context", "_mask_time_endpoints", "_scale_shift", "_f32", "active_step", "push_step", "pop_step")}}
    return inspection, adapter, original


def _project_long_video_patch(inspection, live_model):
    """Factor only the project's exact extra_conds patch on an inspection clone."""
    selected = inspection.object_patches.get("extra_conds")
    if selected is None:
        return inspection, None, None
    _require(type(selected) is MethodType and selected.__self__ is live_model.model,
             "HyperFlow long-video patch has another MODEL owner")
    function = selected.__func__
    state = _factory_closure(function, long_video.patch_long_video_model, "_patched_extra_conds")
    _require(type(function) is FunctionType and state is not None and set(state) == {"original"}
             and function.__defaults__ is None and function.__kwdefaults__ is None
             and set(vars(function)) == {"_t8_long_video_patch_version",
                                        "_t8_long_video_original_extra_conds"}
             and function._t8_long_video_patch_version == long_video.LONG_VIDEO_PATCH_VERSION,
             "HyperFlow long-video extra_conds needs its exact T8 factory")
    original = state["original"]
    _require(type(original) is MethodType and original.__self__ is live_model.model
             and original.__func__ is _class_method(type(live_model.model), "extra_conds")
             and function._t8_long_video_original_extra_conds is original,
             "HyperFlow long-video patch does not restore native extra_conds")
    live = getattr(live_model.model, "extra_conds")
    if live != original and live is not selected:
        _require(type(live) is MethodType and live.__self__ is live_model.model,
                 "HyperFlow live extra_conds has another MODEL owner")
        sibling = live.__func__
        sibling_state = _factory_closure(sibling, long_video.patch_long_video_model,
                                         "_patched_extra_conds")
        _require(type(sibling) is FunctionType and sibling_state == {"original": original}
                 and sibling.__defaults__ is None and sibling.__kwdefaults__ is None
                 and set(vars(sibling)) == set(vars(function))
                 and sibling._t8_long_video_patch_version == long_video.LONG_VIDEO_PATCH_VERSION
                 and sibling._t8_long_video_original_extra_conds == original,
                 "HyperFlow live extra_conds is not a known sibling T8 patch")
    inspection.object_patches.pop("extra_conds")
    if "extra_conds" in vars(inspection.model):
        # Core may leave a sibling's method installed on shared network storage.
        # Remove it only from this shallow inspection view, never from live Core.
        base = copy(inspection.model)
        base._modules = dict(inspection.model._modules)
        del base.extra_conds
        inspection.model = base
    source = Path(long_video.__file__).resolve()
    descriptor = {"version": long_video.LONG_VIDEO_PATCH_VERSION,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "patch_factory": function_identity(long_video.patch_long_video_model),
                  "repair": function_identity(long_video.repair_long_video_payload),
                  "original": function_identity(original.__func__)}
    return inspection, descriptor, original


def model_identity(model):
    from .effect_identity import project_stage_effects
    from .hyperflow_effects import project_owner
    from . import hyperflow_fresh
    view, effects = project_stage_effects(model)
    view, effect_stage = project_owner(view)
    fresh_owner = hyperflow_fresh.capture_owner(view) if view.get_attachment(hyperflow_fresh.KEY) is not None else None
    inspection, adapter, original = project(view)
    fresh_stage = None
    if fresh_owner is not None:
        inspection, fresh_stage = hyperflow_fresh.project_owner(inspection, fresh_owner)
        # The original full base, not the selected per-phase AV sampling buffer,
        # remains common identity. Projection only: never mutate shared network.
        original = {key: value for key, value in original.items() if not key.startswith("model_sampling.")}
        original.update({"model_sampling." + key: value for key, value in inspection.model.model_sampling.state_dict().items()})
    inspection, long_video_patch, original_extra_conds = _project_long_video_patch(inspection, view)
    # Unknown remaining object patches, callbacks, wrappers, attachments and
    # attention backends must fail classification, not disappear on this clone.
    selected = _audited_stage_model_identity(inspection)
    dispatch, functions = {}, {}
    for name, module in inspection.model.named_modules():
        function = _class_method(type(module), "forward")
        if function not in functions:
            functions[function] = function_identity(function)
        dispatch[name] = functions[function]
    protocol = {}
    for name in ("audio_scale", "_scale_audio_slice", "process_latent_in", "process_latent_out",
                 "extra_conds", "apply_model", "_apply_model"):
        method = original_extra_conds if name == "extra_conds" and original_extra_conds is not None else getattr(model.model, name, None)
        if method is None and not hasattr(type(model.model), name):
            protocol[name] = {"absent": True}
            continue
        _require(type(method) is MethodType and method.__self__ is model.model
                 and method.__func__ is _class_method(type(model.model), name),
                 f"HyperFlow base {name} has an unauthenticated user owner")
        protocol[name] = function_identity(method.__func__)
    latent_format = model.model.latent_format
    for name in ("process_in", "process_out"):
        method = getattr(latent_format, name)
        _require(type(method) is MethodType and method.__self__ is latent_format
                 and method.__func__ is _class_method(type(latent_format), name),
                 "HyperFlow latent coordinate transform has an unknown owner")
        protocol["latent_" + name] = function_identity(method.__func__)
    protocol["latent_instance"] = content_identity(vars(latent_format))
    protocol["latent_scale_factor"] = content_identity(latent_format.scale_factor)
    base = {"raw_state": content_identity(original), "class_forward_dispatch": dispatch, "base_protocol": protocol,
            "unet_config": content_identity(getattr(model.model.model_config, "unet_config", None)),
            "manual_cast_dtype": str(getattr(model.model, "manual_cast_dtype", None))}
    # No UUID, object id or filename in the portable projection. The native
    # runtime still validates the live owner at each actual forward.
    contract = {"selected": selected, "adapter": adapter, "base": base,
                "stage_effects": effects, "effect_stage": effect_stage, "fresh_stage": fresh_stage}
    if long_video_patch is not None:
        contract["long_video_patch"] = long_video_patch
    binding = asdict(model.get_attachment(runtime.ATTACHMENT_KEY))
    binding.pop("owner")
    binding.pop("model_identity")
    result = {"schema": "t8.modular-sampling.hyperflow-portable-model.v1", "sha256": sha(contract),
            "portable_cache_reuse": True, "model_filename_trusted": False,
            "common_base_sha256": sha(base), "adapter_sha256": sha(adapter), "binding": binding,
            "selected": selected, "stage_effects": effects, "effect_stage": effect_stage, "fresh_stage": fresh_stage}
    if long_video_patch is not None:
        result["long_video_patch"] = long_video_patch
    return result
