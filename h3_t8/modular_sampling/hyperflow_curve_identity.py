"""Authenticate only the new curve owner; preserve foreign owners unverified.

No basename/model-file fiction: portable common-base identity uses actual raw
native storage, class/protocol dispatch and all residual LoRA/backend state.
Inspection containers are copied; the live MODEL is never cleared/unpatched.
"""
from copy import copy
import hashlib
from types import FunctionType, MethodType

import comfy.lora
import comfy.patcher_extension
import torch

from .. import hyperflow_curve_fit_exp as fitting
from .. import hyperflow_curve_runtime_exp as runtime
from ..long_video_dual_identity import content_identity, _original_state, _audited_stage_model_identity
from ..hyperflow_weights_advanced import HyperFlowWeights
from ..vdn_attention_compat import _factory_closure
from .effect_identity import _require
from .hyperflow_identity import _native, _class_method
from .progressive_effect_identity import function_identity
from .results import sha


def _closure(function, name, fields, *, marked=False, defaults=None):
    state = _factory_closure(function, runtime._install, name)
    _require(type(function) is FunctionType and state is not None and set(state) == set(fields),
             "Curve owner requires its exact native runtime closure: " + name)
    _require(function.__defaults__ is None and function.__kwdefaults__ == defaults,
             "Curve runtime closure defaults changed")
    _require(set(vars(function)) == ({"_t8_curve_owner", "_t8_curve_inner"} if marked else set()),
             "Curve runtime has extra executable attributes")
    return state


def _marked(function, binding, delegate):
    _require(function._t8_curve_owner == binding.owner and function._t8_curve_inner is delegate,
             "Curve installed delegate/owner changed")


def _project_live(model, inspection, paths):
    """Normalize only source-authenticated selected/dormant curve methods.

    Branches share native module storage. A sibling's installed curve is not
    executed by this selected MODEL, but its source owner must still be proven
    before removing it from a separate inspection view. Never clear live Core.
    """
    diffusion = model.model.diffusion_model
    current_time = vars(diffusion).get("time_embedder")
    live_owner = None
    if current_time is not None:
        state = _closure(current_time, "time_embedder", {"audit", "fit", "owner"})
        _require(type(state["owner"]) is str and bool(state["owner"])
                 and type(state["fit"]) is fitting.CurveFit and type(state["audit"]) is dict,
                 "Dormant curve time has an unknown captured owner")
        state["fit"].verify()
        live_owner = state["owner"]
    copied = {}
    def container(path):
        if path not in copied:
            original = model.model.get_submodule(path) if path else model.model
            view = copy(original)
            view._modules = dict(original._modules)
            current = vars(view).get("forward")
            if type(current) is MethodType and current.__self__ is original:
                _native(current, original)
                del view.forward
            copied[path] = view
            if path:
                parent, _, child = path.rpartition(".")
                container(parent)._modules[child] = view
            else:
                inspection.model = view
        return copied[path]
    for key in paths:
        path = key.removesuffix(".forward")
        module = model.model.get_submodule(path)
        current = vars(module).get("forward")
        if current is None:
            continue
        if type(current) is MethodType:
            _native(current, module)
            container(path)
            continue
        if key == "diffusion_model.final_layer.forward":
            state = _closure(current, "final_forward", {"diffusion", "owner", "original_final", "remap"}, marked=True)
            delegate = state["original_final"]
        else:
            defaults = getattr(current, "__kwdefaults__", None)
            index = int(path.rsplit(".", 1)[1])
            _require(type(defaults) is dict and set(defaults) == {"_inner", "_index"}
                     and defaults["_index"] == index, "Dormant curve block index/delegate changed")
            state = _closure(current, "block_forward", {"diffusion", "owner", "remap"}, marked=True, defaults=defaults)
            delegate = defaults["_inner"]
        _require(state["owner"] == live_owner and current._t8_curve_owner == live_owner
                 and current._t8_curve_inner is delegate and state["diffusion"] is diffusion,
                 "Live curve methods belong to mismatched owners")
        _native(delegate, module)
        _closure(state["remap"], "remap", set())
        view = container(path)
        del view.forward
    view = container("diffusion_model")
    _require(diffusion.use_adaln_curves is True or (diffusion.use_adaln_curves is False and live_owner is not None),
             "Native curve dispatch lacks its authenticated installed owner")
    view.use_adaln_curves = True
    if current_time is not None:
        del view.time_embedder


def project(model):
    binding, details = model.get_attachment(runtime.KEY), model.get_attachment(runtime.DETAILS_KEY)
    _require(type(binding) is runtime.CurveBinding and type(details) is runtime.CurveDetails,
             "Curve identity requires actual dedicated attachments")
    _require(set(vars(binding)) == set(runtime.CurveBinding.__dataclass_fields__)
             and set(vars(details)) == set(runtime.CurveDetails.__dataclass_fields__)
             and binding.model_identity == id(model.model), "Curve live binding fields/instance changed")
    weights, fit = details.weights, details.fit
    _require(type(weights) is HyperFlowWeights and type(fit) is fitting.CurveFit,
             "Curve captured weight/fit has an unknown subclass owner")
    runtime._recipe(weights, fit)
    _require(runtime._curve_basis_identity(model, fit) == binding.basis_content_sha256,
             "Actual curve basis changed after its fitted installation")
    _require((weights.source_sha256, fit.sha256, fit.metadata["base"]["sha256"], fit.metadata["teacher"]["sha256"]) ==
             (binding.adapter_sha256, binding.fit_sha256, binding.base_sha256, binding.teacher_sha256),
             "Curve executing fit/source differs from binding")
    content = hashlib.sha256(fitting.canonical(content_identity({"patches": weights.patches,
        "endpoint": weights.endpoint, "metadata": vars(weights.metadata)})).encode()).hexdigest()
    _require(content == binding.adapter_content_sha256, "Actual converted original adapter tensors changed")
    _require(fitting._sha256(weights.path) == binding.adapter_sha256,
             "Actual original adapter file bytes changed")
    identity = sha({"fit": fit.sha256, "base": binding.base_sha256,
                    "teacher": binding.teacher_sha256, "adapter": binding.adapter_sha256})
    _require(binding.identity == identity and (binding.raw_sigmas, binding.gate) ==
             (weights.metadata.raw_sigmas, weights.metadata.gate), "Curve bound recipe identity changed")
    diffusion = model.get_model_object("diffusion_model")
    _require(len(diffusion.blocks) == 50 and len(diffusion.token_refiner.blocks) == 2,
             "Curve base must retain its actual fifty-block native structure")
    time = model.object_patches.get("diffusion_model.time_embedder")
    state = _closure(time, "time_embedder", {"audit", "fit", "owner"})
    _require(state["fit"] is fit and state["audit"] is details.audit and state["owner"] == binding.owner,
             "Curve time function has another actual fit/audit owner")
    _require(model.object_patches.get("diffusion_model.use_adaln_curves") is False,
             "Curve typed dispatch patch changed")
    _require(diffusion.use_adaln_curves in (False, True), "Native curve dispatch changed to an unknown value")
    delegates, remaps = {}, []
    for index, block in enumerate(diffusion.blocks):
        key = f"diffusion_model.blocks.{index}.forward"
        function = model.object_patches.get(key)
        kwargs = getattr(function, "__kwdefaults__", None)
        _require(type(kwargs) is dict and set(kwargs) == {"_inner", "_index"} and kwargs["_index"] == index,
                 "Curve projection index/delegate changed")
        state = _closure(function, "block_forward", {"diffusion", "owner", "remap"}, marked=True, defaults=kwargs)
        _require(state["owner"] == binding.owner and state["diffusion"] is diffusion, "Curve block belongs to another fit")
        _marked(function, binding, kwargs["_inner"])
        delegates[key] = _native(kwargs["_inner"], block)
        remaps.append(state["remap"])
    key = "diffusion_model.final_layer.forward"
    final = model.object_patches.get(key)
    state = _closure(final, "final_forward", {"diffusion", "owner", "original_final", "remap"}, marked=True)
    _require(state["owner"] == binding.owner and state["diffusion"] is diffusion, "Curve final belongs to another fit")
    _marked(final, binding, state["original_final"])
    delegates[key] = _native(state["original_final"], diffusion.final_layer)
    remaps.append(state["remap"])
    _require(all(remap is remaps[0] for remap in remaps), "Curve role remappers disagree")
    _closure(remaps[0], "remap", set())
    kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    wrapper_key = "t8_curve_" + binding.owner
    wrappers = model.wrappers.get(kind, {}).get(wrapper_key, ())
    _require(len(wrappers) == 1, "Curve interval wrapper inventory changed")
    state = _closure(wrappers[0], "scope", {"audit", "basis_content", "diffusion", "fit", "immutable_metadata", "owner", "patched"})
    _require(state["owner"] == binding.owner and state["diffusion"] is diffusion and state["fit"] is fit
             and state["audit"] is details.audit and state["immutable_metadata"] == fitting.canonical(fit.metadata),
             "Curve interval scope changed its exact inputs")
    _require(state["basis_content"] == binding.basis_content_sha256
             and state["patched"].model is model.model,
             "Curve interval scope lost its verified initial basis")
    expected_audit = {"forwards", "pairs", "portable_cache_reuse", "full_backbone_file_identity_certified", "human_quality_accepted"}
    _require(type(details.audit) is dict and set(details.audit) == expected_audit
             and all(details.audit[k] is False for k in expected_audit - {"forwards", "pairs"})
             and type(details.audit["forwards"]) is list and type(details.audit["pairs"]) is list,
             "Curve observation state is not its known non-certifying counter")
    source, mapping = {}, {}
    for target, (a, b, alpha) in weights.patches.items():
        if target.startswith("time_embedder."):
            continue
        source[target + ".lora_A.weight"] = a
        source[target + ".lora_B.weight"] = b
        source[target + ".alpha"] = torch.tensor(alpha)
        mapping[target] = "diffusion_model." + target + ".weight"
    parsed = comfy.lora.load_lora(source, mapping, log_missing=False)
    _require(len(parsed) == 208 and set(parsed) == set(mapping.values()), "Curve original target inventory changed")
    for key, value in parsed.items():
        expected = content_identity((1., value, 1., None, None))
        _require(any(content_identity(item) == expected for item in model.patches.get(key, ())),
                 "Curve original adapter no longer occurs in the selected patch stack")
    inspection = model.clone()
    _project_live(model, inspection, delegates)
    for key in delegates:
        inspection.object_patches.pop(key)
    inspection.object_patches.pop("diffusion_model.time_embedder")
    inspection.object_patches.pop("diffusion_model.use_adaln_curves")
    inspection.remove_wrappers_with_key(kind, wrapper_key)
    inspection.remove_attachments(runtime.KEY)
    inspection.remove_attachments(runtime.DETAILS_KEY)
    actual = _original_state(model, model.model_state_dict())
    adapter = {"fit_file_sha256": fit.sha256, "fit_metadata": fit.metadata,
               "tensor_content": content_identity((fit.generated, fit.pinned, fit.t, fit.r)),
               "adapter_content_sha256": content, "native_delegates": delegates,
               "runtime_factory": function_identity(runtime._install),
               "runtime_helpers": {name: function_identity(getattr(runtime, name)) for name in
                   ("_coordinates", "_recipe", "_curve_context", "_curve_basis_identity", "_native_projection", "_unwrap", "build_plan", "setup_sampler")},
               "shared_clock_helpers": {name: function_identity(getattr(runtime.full_runtime, name)) for name in
                   ("_forward_context", "_mask_time_endpoints", "_scale_shift", "_f32", "active_step", "push_step", "pop_step")}}
    return inspection, adapter, actual


def model_identity(model):
    from ..hyperflow_curve_loader_exp import precision_identity
    from .effect_identity import project_stage_effects
    from .hyperflow_curve_effects import project_owner
    view, effects = project_stage_effects(model)
    view, effect_stage = project_owner(view)
    inspection, adapter, original = project(view)
    selected = _audited_stage_model_identity(inspection)
    dispatch = {name: function_identity(_class_method(type(module), "forward"))
                for name, module in inspection.model.named_modules()}
    protocol = {}
    for name in ("audio_scale", "_scale_audio_slice", "process_latent_in", "process_latent_out",
                 "extra_conds", "apply_model", "_apply_model"):
        method = getattr(model.model, name, None)
        if method is None and not hasattr(type(model.model), name):
            protocol[name] = {"absent": True}
            continue
        _require(type(method) is MethodType and method.__self__ is model.model
                 and method.__func__ is _class_method(type(model.model), name),
                 "Curve base protocol has an unauthenticated user owner: " + name)
        protocol[name] = function_identity(method.__func__)
    latent = model.model.latent_format
    for name in ("process_in", "process_out"):
        method = getattr(latent, name)
        _require(type(method) is MethodType and method.__self__ is latent
                 and method.__func__ is _class_method(type(latent), name),
                 "Curve latent transform has an unknown owner")
        protocol["latent_" + name] = function_identity(method.__func__)
    protocol["latent_instance"] = content_identity(vars(latent))
    protocol["latent_scale_factor"] = content_identity(latent.scale_factor)
    base = {"raw_state": content_identity(original), "class_dispatch": dispatch,
            "native_precision_islands": precision_identity(model),
            "base_protocol": protocol,
            "unet_config": content_identity(getattr(getattr(model.model, "model_config", None), "unet_config", None)),
            "manual_cast_dtype": str(getattr(model.model, "manual_cast_dtype", None))}
    contract = {"selected": selected, "base": base, "adapter": adapter,
                "stage_effects": effects, "effect_stage": effect_stage}
    return {"schema": "t8.hyperflow.curve_portable_model.v1", "sha256": sha(contract),
            "portable_cache_reuse": True, "common_base_sha256": sha(base),
            "adapter_sha256": sha(adapter), "model_filename_trusted": False,
            "selected": selected}
