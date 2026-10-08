"""Actual RES MODEL identity using existing raw-weight/patch adapters.

Inspection clones never change the executing MODEL. Unknown owners remain usable
for full sampling, but cannot claim portable history recovery.
"""
import hashlib
import inspect
import json
from pathlib import Path

from .long_video_dual_identity import content_identity, stage_model_identity, _v2_sampling_identity
from .modular_sampling.progressive_effect_identity import function_identity
from .patch_stack_policy import UnverifiedModelStack


def _portable(value):
    if isinstance(value, dict):
        return value.get("portable_cache_reuse") is not False and all(_portable(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_portable(item) for item in value)
    return True


def loaded_model_identity(model, *, observed_artifacts=None):
    """Bind original storage + ordered native LoRA + selected AV coordinates.

    Existing INT8/ConvRot identity hashes serialized raw storage/scales, never a
    dequantized substitute. An unknown callable's nonce is not a crypto proof.
    """
    inspection = model.clone()
    from . import res_effects_exp
    effects = None
    if model.get_attachment(res_effects_exp.KEY) is not None:
        try:
            inspection, effects = res_effects_exp.project_identity(inspection)
        except UnverifiedModelStack as error:
            effects = {"portable_cache_reuse": False, "reason": str(error)}
    sampling = model.get_model_object("model_sampling")
    inspection.object_patches.pop("model_sampling", None)
    from .res_sol_normalization import normalized_original_sol_identity
    sol_structure = sol_normalization = weights = None
    try:
        sol = normalized_original_sol_identity(inspection, observed_artifacts=observed_artifacts)
        if sol is not None:
            weights, sol_structure, sol_normalization = sol
    except UnverifiedModelStack as error:
        sol_structure = {"portable_cache_reuse": False, "python_composition_verified": False,
                         "reason": str(error)}
    from .res_memory_identity import mixed_memory_model_identity
    if weights is None:
        try:
            weights = mixed_memory_model_identity(inspection)
        except UnverifiedModelStack:
            weights = None
    if weights is None:
        weights = stage_model_identity(inspection, _omit_dormant_sampling=True)
    dispatch = {}
    portable = _portable(weights) and (sol_structure is None or _portable(sol_structure))
    try:
        coordinates = _v2_sampling_identity(sampling)
        coordinates["actual_method_execution"] = {
            name: function_identity(getattr(type(sampling), name)) for name in (
                "set_parameters", "set_noise_scale", "timestep", "sigma", "percent_to_sigma",
                "calculate_input", "calculate_denoised", "noise_scaling", "inverse_noise_scaling")}
        coordinates["actual_audio_scale_execution"] = function_identity(
            inspect.getattr_static(type(sampling), "audio_scale").fget)
        functions = {}
        for name, module in model.model.named_modules():
            if name == "model_sampling" or name.startswith("model_sampling."):
                continue
            function = inspect.getattr_static(type(module), "forward")
            if function not in functions:
                functions[function] = function_identity(function)
            dispatch[name] = functions[function]
    except UnverifiedModelStack as error:
        coordinates = {"portable_cache_reuse": False, "reason": str(error)}
        portable = False
    try:
        latent = model.get_model_object("latent_format")
        latent_identity = content_identity({"class": type(latent).__module__ + "." + type(latent).__qualname__,
            "state": vars(latent), "process_in": function_identity(type(latent).process_in),
            "process_out": function_identity(type(latent).process_out)})
    except UnverifiedModelStack as error:
        latent_identity = {"portable_cache_reuse": False, "reason": str(error)}
        portable = False
    value = dict(schema="t8.minimax_h3.RES_loaded_model.v1", weights=weights,
        coordinates=coordinates, actual_class_forward=dispatch, latent_format=latent_identity,
        placement=content_identity(dict(load_device=model.load_device, offload_device=model.offload_device,
            force_cast_weights=getattr(model, "force_cast_weights", None))),
        portable_cache_reuse=portable, automatic_loaded_weights_verified=_portable(weights),
        model_filename_trusted=False,
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if effects is not None:
        value["RES_external_effects"] = effects
        value["portable_cache_reuse"] = value["portable_cache_reuse"] and _portable(effects)
    if sol_structure is not None:
        value["original_sol_structure"] = sol_structure
    if sol_normalization is not None:
        value["original_sol_normalization"] = sol_normalization
    if sol_structure is not None:
        from .res_recovery_identity import scoped_history_identity
        value["scoped_RES_history_content"] = scoped_history_identity(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
