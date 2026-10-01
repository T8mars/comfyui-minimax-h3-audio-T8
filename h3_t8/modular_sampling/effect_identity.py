"""Read-only identity projection for source-authenticated external stage effects.

This never unpatches the live MODEL or executes a reconstructed sampler. Unknown
owners remain usable for fresh sampling, but cannot authorize persisted reuse.
"""
from copy import copy
from dataclasses import asdict
import hashlib
from pathlib import Path

import comfy.model_patcher
import comfy.patcher_extension as extension

from .. import enhance_a_video_advanced as feta
from .. import fast_h3_v2_advanced as v2
from .. import prompt_relay_advanced as relay
from ..h3_core_compat import _set_legacy_attention_backend, plain_attention_backend
from ..long_video_dual_identity import content_identity
from ..patch_stack_policy import UnverifiedModelStack
from ..progressive_eav_masks import NativeProgressiveMaskContract
from ..vdn_attention_compat import _factory_closure
from . import eav, sparse_eav, native_dual, manual_pass, rf_stages, native_explicit, pdd_stages, vdn_stages
from .contracts import StageContext


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack(message)


def _mask_identity(mask, *, accepted_source=None):
    if mask is None:
        return None
    _require(type(mask) is NativeProgressiveMaskContract and set(vars(mask)) == {
        "_shapes", "_long_video", "_expected", "_description", "_device_masks"
    }, "Stage EAV mask has an unknown executable owner")
    if mask._long_video is not None:
        from ..progressive_continuation import ProgressiveContinuationSource
        _require(type(accepted_source) is ProgressiveContinuationSource,
                 "Stage EAV continuation mask needs its authenticated accepted source")
        binding = accepted_source.revalidate()
        _require(mask.report().get("accepted_source_sha256") == accepted_source.sha256
                 and mask._long_video["context_frames"] == binding["request"]["context_frames"]
                 and mask._long_video["segment_index"] == binding["request"]["segment_index"],
                 "Continuation EAV mask and accepted source differ")
    else:
        _require(accepted_source is None, "Accepted-source EAV lost its long-video scope")
    # _device_masks is a derived validation cache, not an independent mask or
    # generation input. Both the immutable source masks and description bind.
    return content_identity({"shapes": mask._shapes, "expected": mask._expected,
                             "description": mask._description})


def _project_plain_relay(model, selector, binding, query_rows):
    """Authenticate plain Relay plus inert Dense V2; normalize only a clone.

    The shadow V2 owner represents the numerical base after the separately bound
    Relay operator is factored out. No new owner is installed on an execution
    MODEL and no actual callback, weight or selected sampling object is changed.
    Sparse Relay and third-party Relay backends need their own adapters.
    """
    from . import hyperflow_effects, hyperflow_fresh, speed_effects
    native = native_dual.capture_owner(model) if model.get_attachment(native_dual.KEY) is not None else None
    if native is None and model.get_attachment(speed_effects.KEY) is not None:
        native = speed_effects.capture_owner(model)
    if native is None and model.get_attachment(hyperflow_fresh.KEY) is not None:
        native = hyperflow_fresh.capture_owner(model)
    if native is None and model.get_attachment(hyperflow_effects.KEY) is not None:
        native = hyperflow_effects.capture_owner(model)
    if native is None and model.get_attachment(manual_pass.KEY) is not None:
        native = manual_pass.capture_owner(model)
    if native is None and model.get_attachment(rf_stages.KEY) is not None:
        native = rf_stages.capture_owner(model)
    if native is None and model.get_attachment(native_explicit.KEY) is not None:
        native = native_explicit.capture_owner(model)
    if native is None and model.get_attachment(pdd_stages.KEY) is not None:
        native = pdd_stages.capture_owner(model)
    if native is None and model.get_attachment(vdn_stages.KEY) is not None:
        native = vdn_stages.capture_owner(model)
    owner = None if native is not None else v2.capture_fast_h3_v2_owner(model)
    if native is None:
        _require(owner is not None and type(owner.runtime) is v2._V2Runtime
                 and owner.profile == "dense_compat_exp", "Sparse Relay persistent identity is not yet adapted")
        runtime = owner.runtime
        reference = v2._V2Runtime("dense_compat_exp", None, None, runtime.head_chunks)
        _require(set(vars(runtime)) == set(vars(reference)), "Relay V2 runtime contains unknown execution state")
        _require(runtime.override is selector and not runtime.dit and not runtime.previous_dit,
                 "Relay V2 selector/producer has another execution owner")
        for name in ("sparse", "patch", "guard", "prepare", "cleanup", "previous_override", "dense_sol_backend",
                     "_dense_sol_frozen", "_dense_sol_kernel", "_dense_sol_fallback", "_dense_sol_attention",
                     "_dense_sol_type", "_dense_sol_report", "_dense_sol_fallback_execution"):
            _require(getattr(runtime, name) is None, "Relay Dense V2 has another executable adapter")
        _require(runtime.frozen_config == reference.frozen_config, "Relay V2 numerical configuration changed")
    selected = None
    for factory in (getattr(comfy.model_patcher.ModelPatcher, "set_model_optimized_attention", None),
                    _set_legacy_attention_backend):
        state = _factory_closure(selector, factory, "optimized_attention_override")
        if state is not None:
            selected = state.get("optimized_attention")
            break
    router = _factory_closure(selected, relay._install_prompt_relay_model, "_attention_router")
    _require(router is not None and router.get("relay_backend") is None
             and router.get("execution_observer") is None,
             "Relay attention backend/observer lacks its persistent owner adapter")
    _require(getattr(selector, "container_function", None) is None
             and type(router.get("query_chunk_rows")) is int and router["query_chunk_rows"] == query_rows
             and 32 <= query_rows <= 2048, "Relay actual query router differs from its descriptor")
    _require(type(binding) is dict, "Relay binding is not a plain immutable-content descriptor")
    unsigned = dict(binding)
    claimed = unsigned.pop("binding_hash", None)
    if claimed != relay._sha256_json(unsigned) or getattr(selector, "_t8_prompt_relay_binding_hash", None) != claimed:
        raise ValueError("Stage Relay binding hash changed")
    contract = {"binding": content_identity(binding), "query_chunk_rows": query_rows,
                "backend": "native_global_unbiased__pytorch_temporal_bias"}
    cloned = model.clone()
    options = cloned.model_options["transformer_options"]
    options.pop("optimized_attention_override", None)
    if native is None:
        shadow = copy(runtime)
        shadow.override = None
        options[v2.RUN_KEY] = shadow
        cloned.set_attachments(v2.KEY, v2._V2Receipt(owner.profile, shadow))
    return cloned, contract


def _project_long_video_relay_attachment(cloned, attachment, binding):
    from .. import prompt_relay_long_video_advanced as long_video_relay
    from ..prompt_relay_long_video_advanced import (
        PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
        PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
    )
    if attachment is None:
        return None
    fields = {"schema", "global_plan_hash", "projected_plan_hash", "binding_hash", "segment_index"}
    _require(type(attachment) is dict and set(attachment) == fields
             and type(attachment["schema"]) is int
             and attachment["schema"] == PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA
             and type(attachment["segment_index"]) is int
             and attachment["segment_index"] >= 0
             and all(type(attachment[key]) is str
                     and len(attachment[key]) == 64
                     and all(char in "0123456789abcdef" for char in attachment[key])
                     for key in ("global_plan_hash", "projected_plan_hash", "binding_hash"))
             and type(binding) is dict
             and attachment["binding_hash"] == binding["binding_hash"],
             "Long Video Relay attachment differs from its actual bound owner")
    _require(attachment["projected_plan_hash"] == binding["plan_hash"],
             "Long Video Relay projected Plan differs from the installed binding")
    cloned.remove_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    return {"long_video_projection": content_identity(attachment),
            "long_video_builder_sha256": hashlib.sha256(
                Path(long_video_relay.__file__).read_bytes()).hexdigest()}


def _project_standalone_relay(model):
    from ..prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
    wrappers = model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    long_video_attachment = model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    if not wrappers and model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is None:
        _require(long_video_attachment is None,
                 "Long Video Relay projection lacks its actual bound Relay owner")
        return model, None
    _require(len(wrappers) == 1, "Relay wrapper count is not its single native owner")
    contract = relay.prompt_relay_model_contract(model)
    wrapper = _factory_closure(wrappers[0], relay._install_prompt_relay_model, "_diffusion_wrapper")
    _require(wrapper is not None and wrapper.get("relay_backend") is None
             and wrapper.get("execution_observer") is None and contract["attention_owner_verified"],
             "Relay wrapper/backend ownership is unverified")
    source = wrapper.get("source_plain_override")
    _require(source is None or plain_attention_backend(source) is not None,
             "Relay source backend no longer has its native owner")
    _require(wrapper.get("binding") == contract["binding"]
             and wrapper.get("expected_hash") == contract["binding_hash"], "Relay bound wrapper changed")
    cloned, result = _project_plain_relay(model, wrapper["installed_override"],
                                         contract["binding"], contract["query_chunk_rows"])
    cloned.remove_wrappers_with_key("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    cloned.remove_attachments(relay.PROMPT_RELAY_WRAPPER_KEY)
    long_video = _project_long_video_relay_attachment(cloned, long_video_attachment, contract["binding"])
    if long_video is not None:
        result = {**result, **long_video}
    return cloned, {"schema": "t8.modular-sampling.effect-identity.v1", "relay": result}


def project_stage_effects(model):
    from . import vdn_relay
    model, vdn_contract = vdn_relay.project(model)
    model, contract = _project_stage_effects(model)
    if vdn_contract is not None:
        contract = {**(contract or {}), "vdn_relay": vdn_contract}
    return model, contract


def _project_stage_effects(model):
    """Return an inspection-only base view and the actual bound effect contract."""
    from . import hyperflow_effects, hyperflow_fresh, speed_effects
    runtime = model.get_attachment(eav.KEY)
    wrappers = model.get_wrappers("diffusion_model", eav.KEY)
    if runtime is None and not wrappers:
        return _project_standalone_relay(model)
    _require(type(runtime) is eav.StageEAVRuntime and len(wrappers) == 1,
             "Stage EAV wrapper/runtime ownership is unverified")
    expected_fields = {"config", "context", "blocks", "closed", "completed_forwards", "selector_calls",
                       "sparse_calls", "relay_calls", "observed_sigmas", "relay_required", "v2_runtime", "telemetry"}
    _require(set(vars(runtime)) == expected_fields, "Stage EAV runtime has unknown execution state")
    _require(type(runtime.config) is eav.EAVConfig and type(runtime.context) is StageContext,
             "Stage EAV configuration/context is not the native contract")
    _require(type(runtime.telemetry) is feta.EAVRuntime and set(vars(runtime.telemetry)) == {
        "config", "_lock", "_run_index", "_consumed", "_aborted", "_forwards"
    }, "Stage EAV telemetry has unknown executable state")
    wrapper = _factory_closure(wrappers[0], eav.apply_stage_eav, "stage_wrapper")
    _require(wrapper is not None and wrapper.get("runtime") is runtime
             and wrapper.get("config") is runtime.config
             and wrapper.get("stage_context") is runtime.context,
             "Stage EAV wrapper is not paired with its exact source runtime")
    options = model.model_options["transformer_options"]
    selector = options.get("optimized_attention_override")
    state = _factory_closure(selector, eav.apply_stage_eav, "stage_attention")
    _require(state is not None and selector is wrapper.get("stage_attention")
             and state.get("runtime") is runtime and state.get("binding") is wrapper.get("binding"),
             "Stage EAV selected attention owner was replaced")
    binding, relay_contract = wrapper.get("binding"), state.get("relay_contract")
    _require((binding is not None) == runtime.relay_required
             and (relay_contract is not None) == runtime.relay_required,
             "Stage EAV Relay runtime pairing changed")
    if binding is not None:
        _require(relay_contract.get("binding") == binding and relay_contract.get("attention_owner_verified") is True
                 and relay_contract.get("attention_backend") is None,
                 "Composed Relay backend/binding lacks its persistent owner adapter")
    owner = (speed_effects.capture_owner(model) if runtime.context.recipe == speed_effects.RECIPE
             else hyperflow_fresh.capture_owner(model) if runtime.context.recipe in hyperflow_fresh.RECIPES
             else hyperflow_effects.capture_owner(model) if runtime.context.recipe == hyperflow_effects.RECIPE
             else vdn_stages.capture_owner(model) if runtime.context.recipe == vdn_stages.RECIPE
             else pdd_stages.capture_owner(model) if runtime.context.recipe == pdd_stages.RECIPE
             else native_explicit.capture_owner(model) if runtime.context.recipe == native_explicit.RECIPE
             else rf_stages.capture_owner(model) if runtime.context.recipe == rf_stages.RECIPE
             else native_dual.capture_owner(model) if runtime.context.recipe == native_dual.RECIPE
             else manual_pass.capture_owner(model) if runtime.context.recipe == manual_pass.RECIPE
             else v2.capture_fast_h3_v2_owner(model))
    _require(owner is not None and runtime.v2_runtime is owner.runtime,
             "Stage EAV and recipe have different runtime owners")
    if owner.runtime is None:
        _require(owner.context == runtime.context, "Native Dual and EAV have different stage contexts")
    blocks = model.get_model_object("diffusion_model").blocks
    _require(runtime.blocks == len(blocks), "Stage EAV expected block count changed")
    for role, name in ((extension.CallbacksMP.ON_PREPARE_STATE, "prepare"),
                       (extension.CallbacksMP.ON_CLEANUP, "cleanup")):
        callbacks = model.callbacks.get(role, {}).get(eav.KEY, [])
        _require(len(callbacks) == 1 and getattr(callbacks[0], "__self__", None) is runtime
                 and getattr(callbacks[0], "__func__", None) is getattr(eav.StageEAVRuntime, name),
                 "Stage EAV lifecycle callback was replaced")
    cloned = model.clone()
    if runtime.context.recipe == vdn_stages.RECIPE:
        from .vdn_effects import project
        cloned = project(cloned, runtime)
    cloned_options = cloned.model_options["transformer_options"]
    previous = state.get("previous")
    if previous is None:
        cloned_options.pop("optimized_attention_override", None)
    else:
        cloned_options["optimized_attention_override"] = previous
    if owner.runtime is not None and owner.profile != "dense_compat_exp":
        dit = options.get("patches_replace", {}).get("dit", {})
        for index, block in enumerate(blocks):
            key = ("double_block", index)
            wrapped = _factory_closure(dit.get(key), sparse_eav.wrap_producer, "wrapped")
            _require(wrapped is not None and wrapped.get("original") is owner.runtime.dit.get(key),
                     "Stage sparse EAV producer has an unknown wrapper")
            attention = _factory_closure(wrapped.get("sparse_attention"), sparse_eav.wrap_producer, "sparse_attention")
            _require(attention is not None and attention.get("runtime") is runtime
                     and attention.get("owner") is owner.runtime and attention.get("block") is block
                     and attention.get("index") == index,
                     "Stage sparse EAV attention is paired with another producer")
            original = _factory_closure(wrapped["original"], v2._V2Runtime.block_patch, "patch")
            _require(original is not None and attention.get("original_attention") is original.get("attention")
                     and wrapped.get("original_attention") is original.get("attention"),
                     "Stage sparse EAV native delegate changed")
            cloned.set_model_patch_replace(wrapped["original"], "dit", "double_block", index)
    cloned.remove_wrappers_with_key("diffusion_model", eav.KEY)
    cloned.remove_attachments(eav.KEY)
    for role in (extension.CallbacksMP.ON_PREPARE_STATE, extension.CallbacksMP.ON_CLEANUP):
        cloned.callbacks[role].pop(eav.KEY)
        if not cloned.callbacks[role]:
            cloned.callbacks.pop(role)
    contract = {"schema": "t8.modular-sampling.effect-identity.v1", "eav": asdict(runtime.config),
                "stage_context": runtime.context.to_dict(), "blocks": runtime.blocks,
                "mask": _mask_identity(wrapper.get("mask_contract"))}
    if binding is not None:
        cloned, contract["relay"] = _project_plain_relay(cloned, previous, binding, relay_contract["query_chunk_rows"])
    from ..prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
    attachment = model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    _require(attachment is None or binding is not None,
             "Long Video Relay projection lacks its actual bound Relay owner")
    long_video = _project_long_video_relay_attachment(
        cloned, attachment, binding)
    if long_video is not None:
        contract["relay"] = {**contract["relay"], **long_video}
    return cloned, contract
