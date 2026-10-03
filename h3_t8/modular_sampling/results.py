"""One native Core sampler invocation plus source-bound, typed result evidence.

No loop, retry, automatic cache lookup, or second stage runs here. Unknown
samplers/guiders still execute, but do not acquire a false verified recipe label.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType
import uuid

import comfy.conds
import comfy.samplers
from comfy_extras.nodes_custom_sampler import Guider_Basic, SamplerCustomAdvanced

from .. import fast_h3_v2_advanced as v2
from ..long_video_dual_identity import stage_model_identity, _original_state, _v2_sampling_identity
from ..patch_stack_policy import (UnverifiedModelStack, model_identity_matches,
                                  nonportable_model_identity, _execution_selection)
from ..progressive_continuation_runtime import _input_identity as _native_input_identity
from .contracts import StageContext
from .eav import validate_stage


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _identity_difference_path(expected, current, path="$"):
    """Report only a mismatch location, never tensor bytes or prompt content."""
    if type(expected) is not type(current):
        return path + ".type"
    if isinstance(expected, dict):
        if expected.keys() != current.keys():
            return path + ".keys"
        ignored = {"sha256"} if (expected.get("portable_cache_reuse") is False
                                  and current.get("portable_cache_reuse") is False) else set()
        for key in sorted(expected.keys() - ignored):
            found = _identity_difference_path(expected[key], current[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(current):
            return path + ".length"
        for index, (left, right) in enumerate(zip(expected, current, strict=True)):
            found = _identity_difference_path(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None if expected == current else path


def _input_identity(value):
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _input_identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [_input_identity(item) for item in value]}
    return _native_input_identity(value)


def _conditions(value):
    # Native cond wrappers are inert only for these exact Core classes/fields.
    # Never use repr or class names as a portable identity for foreign callables.
    known = (comfy.conds.CONDRegular, comfy.conds.CONDNoiseShape, comfy.conds.CONDCrossAttn, comfy.conds.CONDConstant)
    if type(value) is uuid.UUID:
        return {"uuid": value.hex}
    if type(value) in known:
        if set(vars(value)) != {"cond"}:
            raise UnverifiedModelStack("Native condition has additional unbound state")
        return {"native_condition": type(value).__name__, "cond": _conditions(value.cond)}
    if isinstance(value, dict):
        return {key: _conditions(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [_conditions(item) for item in value]}
    return _input_identity(value)


def request_conditions(conditions):
    # Only Core's top-level converted condition UUIDs are allocation labels.
    # Other UUID metadata remains literal. Unknown executable stacks are never
    # made portable by this projection, even if they consume these labels.
    normalized = {}
    for branch, entries in conditions.items():
        if not isinstance(entries, list):
            raise UnverifiedModelStack("Unknown converted condition structure")
        normalized[branch] = []
        for index, entry in enumerate(entries):
            item = dict(entry)
            if type(item.get("uuid")) is uuid.UUID:
                item["uuid"] = {"core_condition_slot": [branch, index]}
            normalized[branch].append(item)
    return _conditions(normalized)


def _selected_base_identity(model):
    from . import hyperflow_fresh, hyperflow, speed_effects
    if model.get_attachment(hyperflow_fresh.KEY) is not None:
        return hyperflow.model_selection(model)
    if model.get_attachment(speed_effects.KEY) is not None:
        try:
            view, contract = speed_effects.project_identity(model)
        except UnverifiedModelStack as error:
            return nonportable_model_identity(model, str(error), schema="t8.modular-sampling.unverified-speed.v1")
        base = stage_model_identity(view)
        return {**base, "schema": "t8.modular-sampling.speed-stage-model.v1", "speed_stage": contract,
                "sha256": sha({"base": base, "speed_stage": contract})}
    from . import native_dual, manual_pass, rf_stages, native_explicit, pdd_stages, vdn_stages
    for adapter, field in ((vdn_stages, "vdn_stage"), (pdd_stages, "pdd_stage"), (native_explicit, "native_explicit_stage"), (rf_stages, "rf_stage"),
                           (native_dual, "native_dual_stage"), (manual_pass, "manual_pass_stage")):
        if model.get_attachment(adapter.KEY) is not None:
            try:
                view, contract = adapter.project_identity(model)
            except UnverifiedModelStack as error:
                return nonportable_model_identity(model, str(error), schema="t8.modular-sampling.unverified-native.v1")
            base = stage_model_identity(view)
            return {**base, "schema": "t8.modular-sampling.native-stage-model.v1", field: contract,
                    "sha256": sha({"base": base, field: contract})}
    # Existing identity deliberately refuses a live model_sampling backup.
    # Give it a read-only shallow network view with the exact dormant original
    # module restored. Shared diffusion weights, hooks and LoRA backups remain
    # visible. Never unpatch/unload a user's live network for an identity check.
    previous = getattr(model, "object_patches_backup", {}).get("model_sampling")
    if previous is None:
        return stage_model_identity(model)
    selected = model.object_patches.get("model_sampling")
    if model.model.model_sampling is not selected:
        return nonportable_model_identity(model, "Live sampling ownership is outside the native load projection",
                                          schema="t8.modular-sampling.unverified-model.v1")
    try:
        _v2_sampling_identity(selected)
    except UnverifiedModelStack:
        return stage_model_identity(model)
    view = model.clone()
    view.model = copy(model.model)
    view.model._modules = dict(model.model._modules)
    view.model._modules["model_sampling"] = previous
    view.object_patches_backup = dict(model.object_patches_backup)
    view.object_patches_backup.pop("model_sampling")
    return stage_model_identity(view)


def _project_selected_effects(model):
    """Use one exact operator projection before and after native sampling."""
    from .. import veda_identity
    from .effect_identity import project_stage_effects
    from .continuation_identity import project as project_continuation
    from . import hyperflow_fresh
    # Peel only the exact new Core wrapper from an inspection clone first;
    # existing Relay/EAV owners and prior block delegates remain untouched.
    from ..h3_fun_union2 import ATTACHMENT_KEY as union_key
    union = None
    if model.get_attachment(union_key) is not None:
        from .. import h3_fun_union2_identity
        model, union = h3_fun_union2_identity.project(model)
    view, effects = project_stage_effects(model)
    # HyperFlow authenticates its live selected owner and verified sibling.
    if view.get_attachment(hyperflow_fresh.KEY) is not None:
        continuation = None
    else:
        view, continuation = project_continuation(view)
    view, veda = veda_identity.project(view)
    if continuation is not None:
        effects = {"effects": effects, "continuation": continuation}
    if veda is not None:
        effects = {"effects": effects, "veda_operator": veda}
    if union is not None:
        effects = {"effects": effects, "fun_union2_operator": union}
    return view, effects


def selected_model_identity(model):
    try:
        view, effects = _project_selected_effects(model)
    except UnverifiedModelStack as error:
        return nonportable_model_identity(model, str(error), schema="t8.modular-sampling.unverified-effects.v1")
    identity = _selected_base_identity(view)
    if effects is None:
        return identity
    return {**identity, "schema": "t8.modular-sampling.model-effects.v1", "stage_effects": effects,
            "sha256": sha({"base": identity, "stage_effects": effects})}


def implementation_identity():
    # Numerical dependencies only; a Director UI edit must not invalidate LOW.
    from .. import sampling, h3_core_compat
    import comfy.model_base
    import comfy.sample
    import comfy.sampler_helpers
    import comfy.model_sampling
    import comfy.model_patcher
    from . import contracts, eav, sparse_eav, effect_identity, native_dual, manual_pass, noise, rf_stages, rf_restart, detail_effects, native_explicit
    from .. import freenoise_advanced, long_video_sampling_plan_advanced, detail_sampling_advanced, pdd_advanced
    from . import pdd_stages, pdd_dynamic, vdn_stages, vdn_identity, vdn_effects, vdn_relay, vdn_relay_math
    from .. import vdn_h3_advanced, vdn_two_pass, vdn_sdpa_backend, vdn_attention_compat
    from comfy.ldm.minimax import model as minimax_model
    from comfy.weight_adapter import bypass as bypass_lora, lora as native_lora
    import comfy.k_diffusion.sampling
    from .. import long_video_dual_model_runner, learned_latent_upscale_advanced
    from .. import enhance_a_video_advanced, prompt_relay_advanced, progressive_eav_masks, progressive_masking, tst_runtime
    from comfy.ldm.modules import attention
    from comfy_extras import nodes_sparse_attention
    from . import hyperflow_fresh, hyperflow_identity
    from .. import hyperflow_runtime_advanced, hyperflow_sampling_advanced
    selected = (v2, sampling, h3_core_compat, contracts, eav, comfy.model_base, comfy.sample,
                comfy.samplers, comfy.conds, comfy.sampler_helpers, comfy.model_sampling,
                comfy.model_patcher, inspect.getmodule(SamplerCustomAdvanced), sparse_eav,
                enhance_a_video_advanced, prompt_relay_advanced, nodes_sparse_attention, effect_identity,
                progressive_eav_masks, progressive_masking, tst_runtime, attention,
                native_dual, long_video_dual_model_runner, learned_latent_upscale_advanced,
                manual_pass, noise, freenoise_advanced, long_video_sampling_plan_advanced, comfy.k_diffusion.sampling,
                rf_stages, rf_restart, detail_effects, detail_sampling_advanced, native_explicit,
                pdd_stages, pdd_dynamic, pdd_advanced, minimax_model, bypass_lora, native_lora,
                vdn_stages, vdn_identity, vdn_effects, vdn_relay, vdn_relay_math,
                vdn_h3_advanced, vdn_two_pass, vdn_sdpa_backend, vdn_attention_compat,
                hyperflow_fresh, hyperflow_identity, hyperflow_runtime_advanced, hyperflow_sampling_advanced)
    paths = {Path(module.__file__).resolve() for module in selected}
    paths.add(Path(__file__).resolve())
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def execution_identity(model, initial_sampling):
    """Read-only projection of the selected sampling object across Core loading.

    Core patch_model installs object_patches['model_sampling'] on first load.
    Hash that explicitly selected object's complete native configuration and
    buffers both before and after, rather than the dormant base sigma buffer.
    This does not unload, unpatch, mutate, or ignore diffusion/LoRA weights.
    """
    from . import pdd_stages, pdd_dynamic, hyperflow_fresh, hyperflow
    if model.get_attachment(hyperflow_fresh.KEY) is not None:
        selected = model.object_patches.get("model_sampling")
        if model.model.model_sampling is not selected and model.model.model_sampling is not initial_sampling:
            raise ValueError("Stage live model_sampling owner changed during sampling")
        return hyperflow.model_selection(model)
    dynamic_contract = None
    owner = model.get_attachment(pdd_stages.KEY)
    if owner is not None and json.loads(owner.context.profile)["pdd"]["mode"] == pdd_stages.DYNAMIC:
        try:
            model, dynamic_contract = pdd_dynamic.project(model)
        except UnverifiedModelStack:
            pass  # Unknown executable selection remains unverified, not erased.
    selected = model.object_patches.get("model_sampling")
    try:
        sampling_contract = _v2_sampling_identity(selected)
    except UnverifiedModelStack:
        sampling_contract = {"unverified_selection": _execution_selection(selected)}
    if model.model.model_sampling is not selected and model.model.model_sampling is not initial_sampling:
        raise ValueError("Stage live model_sampling owner changed during sampling")
    state = dict(_original_state(model, model.model_state_dict()))
    state = {key: value for key, value in state.items() if not key.startswith("model_sampling.")}
    state.update({"model_sampling." + key: value for key, value in selected.state_dict().items()})

    class ReadOnlySelection:
        backup = {}
        backup_buffers = {}

        def __getattr__(self, name):
            return getattr(model, name)

        def model_state_dict(self):
            return state

    identity = nonportable_model_identity(ReadOnlySelection(), "Stage within-run integrity snapshot",
                                          schema="t8.modular-sampling.execution-selection.v1")
    # Core installs selected T8 memory forwards only when the network first
    # loads. They are not a new user patch. Suppress only methods authenticated
    # by this MODEL's own memory receipt, after checking the exact live bound
    # method; every unknown forward or hook remains in the mutation snapshot.
    from ..h3_memory_advanced import inspect_t8_memory_composition
    active_wrapper_keys = tuple(key for group in model.wrappers.values()
                                for key, values in group.items() if values)
    memory = inspect_t8_memory_composition(model, allowed_wrapper_keys=active_wrapper_keys)
    if memory is not None:
        modules = dict(model.model.named_modules())
        network = identity["execution_selection"]["network_execution"]
        for patch_path, expected_method in memory["methods"].items():
            module_path = patch_path.removesuffix(".forward")
            if module_path not in network:
                continue
            module = modules.get(module_path)
            current = None if module is None else vars(module).get("forward")
            if (not isinstance(current, MethodType) or current.__self__ is not module
                    or current.__func__ is not expected_method.__func__):
                raise ValueError("Authenticated T8 memory forward changed during sampling")
            network.pop(module_path)
    identity["selected_native_sampling"] = sampling_contract
    if dynamic_contract is not None:
        identity["pdd_dynamic"] = dynamic_contract
    return identity


@dataclass(frozen=True)
class StageResult:
    """The two distinct Core outputs and immutable JSON evidence of this run."""
    output: dict
    denoised_output: dict
    receipt_json: str

    def verify(self):
        receipt = json.loads(self.receipt_json)
        if receipt.get("schema") != "t8.modular-sampling.result.v1":
            raise ValueError("Unknown stage result schema")
        if sha(receipt["request"]) != receipt["request_sha256"]:
            raise ValueError("Stage request integrity failed")
        context = StageContext.from_dict(receipt["request"]["stage_context"])
        from ..sampling import nested_av_parts
        for value in (self.output, self.denoised_output):
            video, audio = nested_av_parts(value)
            if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
                raise ValueError("Stage result shape differs from its bound context")
        if _input_identity(self.output) != receipt["outputs"]["output"]:
            raise ValueError("Stage output was mutated after sampling")
        if _input_identity(self.denoised_output) != receipt["outputs"]["denoised_output"]:
            raise ValueError("Stage denoised_output was mutated after sampling")
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256")
        if sha(unsigned) != claimed:
            raise ValueError("Stage result receipt integrity failed")
        from .hyperflow_fresh import verify_receipt
        verify_receipt(self, receipt)
        return receipt


class _DenoiserObserver:
    def __init__(self, original, observations, before=None):
        self.original, self.observations, self.before = original, observations, before

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        if self.before is not None:
            self.before()
        value = self.original(*args, **kwargs)
        self.observations["denoiser_evaluations"] += 1
        return value


def sample_stage(noise, guider, sampler, sigmas, latent_image, stage_context):
    from .noise import operator_identity
    model = guider.model_patcher
    validate_stage(model, sigmas, latent_image, stage_context)
    observations = {"callbacks": [], "denoiser_evaluations": 0, "noise_identity": None}
    source_identity = _input_identity(latent_image)
    model_identity = selected_model_identity(model)
    noise_operator = operator_identity(noise)
    initial_sampling = model.model.model_sampling
    execution_before = execution_identity(model, initial_sampling)
    portable = model_identity.get("portable_cache_reuse", True) and noise_operator["portable"]
    try:
        condition_identity = _conditions(guider.original_conds)
        bound_conditions = request_conditions(guider.original_conds)
    except UnverifiedModelStack as error:
        condition_identity = {"execution_only": uuid.uuid4().hex, "reason": str(error)}
        bound_conditions = condition_identity
        portable = False
    sampler_known = (type(sampler) is comfy.samplers.KSAMPLER and
                     sampler.sampler_function is v2.sample_v2_euler and
                     sampler.extra_options.get("stage_start", 0) == stage_context.start and
                     sampler.extra_options.get("stage_end", 8) == stage_context.end)
    from . import native_dual, manual_pass, rf_stages, native_explicit, pdd_stages, vdn_stages, hyperflow_fresh
    if stage_context.recipe == native_dual.RECIPE:
        sampler_known = native_dual.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == manual_pass.RECIPE:
        sampler_known = manual_pass.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == rf_stages.RECIPE:
        sampler_known = rf_stages.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == native_explicit.RECIPE:
        sampler_known = native_explicit.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == pdd_stages.RECIPE:
        sampler_known = pdd_stages.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == vdn_stages.RECIPE:
        sampler_known = vdn_stages.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe in hyperflow_fresh.RECIPES:
        sampler_known = hyperflow_fresh.sampler_is_known(model, sampler, stage_context)
    # Core outer_sample deletes inner_model only on success. After a failed
    # run this exact selected MODEL may remain; prepare_sampling overwrites it
    # on retry. Authenticate that one derived slot, not arbitrary callables.
    from .core_sampler_identity import native_basic_guider_class
    guider_known = (type(guider) in (comfy.samplers.CFGGuider, Guider_Basic, native_basic_guider_class())
                    and not any(callable(value) and not (key == "inner_model" and value is model.model)
                                for key, value in vars(guider).items()))
    if not guider_known or not sampler_known:
        portable = False
    implementation = implementation_identity()
    vdn_execution = None
    before_denoiser = None
    if stage_context.recipe == vdn_stages.RECIPE:
        from .vdn_identity import weight_execution
        vdn_execution = {"verified": True, "contracts": [], "call_indices": []}

        def before_denoiser():
            try:
                contract = weight_execution(model)
            except UnverifiedModelStack as error:
                # Unknown user stacks still run, but cannot inherit the
                # certified native side-patch execution/cache identity.
                contract = {"unverified": str(error)}
                vdn_execution["verified"] = False
            if contract not in vdn_execution["contracts"]:
                vdn_execution["contracts"].append(contract)
            vdn_execution["call_indices"].append(vdn_execution["contracts"].index(contract))

    controls_before = _execution_selection({"sampler": vars(sampler), "cfg": getattr(guider, "cfg", None),
                                             "sigmas": sigmas, "seed": noise.seed})
    observed_sampler = copy(sampler)
    original_function = getattr(sampler, "sampler_function", None)
    if callable(original_function):
        def observed_function(denoiser, *args, **kwargs):
            return original_function(_DenoiserObserver(denoiser, observations, before_denoiser), *args, **kwargs)
        observed_sampler.sampler_function = observed_function

    class ObservedGuider:
        model_patcher = model

        def sample(self, actual_noise, *args, callback=None, **kwargs):
            observations["noise_identity"] = _input_identity(actual_noise)

            def record(*items, **options):
                index = items[0] if items else options.get("step")
                observations["callbacks"].append(int(index))
                if callback is not None:
                    return callback(*items, **options)

            return guider.sample(actual_noise, *args, callback=record, **kwargs)

    endpoint_capture = None
    if stage_context.recipe == rf_stages.RECIPE and sampler_known:
        endpoint_capture = rf_stages.rf_restart.EndpointCapture(observed_sampler)
        observed_sampler = endpoint_capture
    with hyperflow_fresh.execution(model, guider, stage_context) as fresh_execution:
        output, denoised = SamplerCustomAdvanced.execute(noise, ObservedGuider(), observed_sampler, sigmas, latent_image).result
    if fresh_execution is not None:
        fresh_execution.update(sampler_verified=sampler_known, guider_verified=guider_known)
        observations["hyperflow_fresh"] = fresh_execution
    # Sampling exceptions propagate before any StageResult exists. Reject live
    # input changes instead of certifying the pre-run state of a changed model.
    if _input_identity(latent_image) != source_identity:
        raise ValueError("Stage source latent changed during sampling")
    execution_after = execution_identity(model, initial_sampling)
    if not model_identity_matches(execution_before, execution_after):
        # HyperFlow deliberately compresses its full execution selection to a
        # digest. A mismatch must still reject the stage, even when there is
        # no expanded network map available solely for this diagnostic.
        before_network = execution_before["execution_selection"].get("network_execution", {})
        after_network = execution_after["execution_selection"].get("network_execution", {})
        added = sorted(after_network.keys() - before_network.keys())
        raise ValueError("Stage MODEL identity changed during sampling at " +
                         str(_identity_difference_path(execution_before, execution_after)) +
                         f"; extra live methods={len(added)} {added[:8]}")
    if stage_context.recipe == native_dual.RECIPE:
        native_dual.validate_stage(model, sigmas, latent_image, stage_context)
    elif stage_context.recipe == manual_pass.RECIPE:
        manual_pass.validate_stage(model, sigmas, latent_image, stage_context)
    elif stage_context.recipe == rf_stages.RECIPE:
        rf_stages.validate_stage(model, sigmas, latent_image, stage_context)
        if sampler_known:
            rf_stages.sampler_is_known(model, sampler, stage_context)
    elif stage_context.recipe == native_explicit.RECIPE:
        native_explicit.validate_stage(model, sigmas, latent_image, stage_context)
        if sampler_known and not native_explicit.sampler_is_known(model, sampler, stage_context):
            raise ValueError("Native explicit sampler ownership changed during execution")
    elif stage_context.recipe == pdd_stages.RECIPE:
        pdd_stages.validate_stage(model, sigmas, latent_image, stage_context)
    elif stage_context.recipe == vdn_stages.RECIPE:
        vdn_stages.validate_stage(model, sigmas, latent_image, stage_context)
    elif stage_context.recipe in hyperflow_fresh.RECIPES:
        hyperflow_fresh.validate_stage(model, sigmas, latent_image, stage_context)
        if sampler_known:
            if not hyperflow_fresh.sampler_is_known(model, sampler, stage_context):
                raise ValueError("HyperFlow fresh sampler ownership changed during execution")
    if "stage_effects" in model_identity:
        try:
            _, after_effects = _project_selected_effects(model)
        except UnverifiedModelStack as error:
            raise ValueError("Stage effect identity changed during sampling") from error
        if after_effects != model_identity["stage_effects"]:
            raise ValueError("Stage effect identity changed during sampling at " +
                             str(_identity_difference_path(model_identity["stage_effects"], after_effects)))
    if implementation_identity() != implementation:
        raise ValueError("Stage implementation changed during sampling")
    controls_after = _execution_selection({"sampler": vars(sampler), "cfg": getattr(guider, "cfg", None),
                                            "sigmas": sigmas, "seed": noise.seed})
    if controls_after != controls_before:
        raise ValueError("Stage sampling controls changed during sampling")
    if operator_identity(noise) != noise_operator:
        raise ValueError("Stage noise provider changed during sampling")
    if "execution_only" not in condition_identity and _conditions(guider.original_conds) != condition_identity:
        raise ValueError("Stage conditions changed during sampling")
    completed = (sampler_known and guider_known and observations["callbacks"] == list(range(stage_context.steps))
                 and observations["denoiser_evaluations"] == stage_context.steps)
    if fresh_execution is not None:
        completed &= set(fresh_execution["absolute_apply_intervals"]) == set(range(stage_context.start, stage_context.end))
        portable &= fresh_execution["composition_verified"] is True
    if stage_context.recipe == rf_stages.RECIPE:
        output, denoised = rf_stages.retain_anchor(output, denoised, latent_image, sampler, stage_context,
            raw_endpoint=endpoint_capture.endpoint if endpoint_capture is not None else None)
    request = {"stage_context": stage_context.to_dict(), "source": source_identity, "model": model_identity,
               "conditions": bound_conditions, "noise": observations["noise_identity"], "noise_operator": noise_operator,
               "seed": int(noise.seed), "cfg": getattr(guider, "cfg", None),
               "sigmas": _input_identity(sigmas), "implementation": implementation}
    if vdn_execution is not None:
        request["vdn_weight_execution"] = vdn_execution
        portable = portable and vdn_execution["verified"] and bool(vdn_execution["call_indices"])
    receipt = {"schema": "t8.modular-sampling.result.v1", "request": request, "request_sha256": sha(request),
               "outputs": {"output": _input_identity(output), "denoised_output": _input_identity(denoised)},
               "execution": observations, "verified_recipe_completion": completed,
               "portable_identity": bool(portable and completed), "cache_reuse_authorized": False,
               "boundary": "One actual native Core sampler return. Denoiser evaluations are not a claim about "
                           "CFG/internal network forwards. Persisted cache/consumer identity must still be validated."}
    receipt["receipt_sha256"] = sha(receipt)
    result = StageResult(output, denoised, canonical(receipt))
    result.verify()
    return output, denoised, result, json.dumps(receipt, ensure_ascii=False, indent=2)
