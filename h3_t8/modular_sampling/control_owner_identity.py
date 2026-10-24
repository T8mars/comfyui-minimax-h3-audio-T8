"""Exact regional-control owners, inspection-only projection and run integrity.

Unknown executable stacks still run. A portable identity is granted only to
the actual source closures and native producers described here, never labels.
Runtime counters, streams and validated derived caches are not new inputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
from pathlib import Path
from types import CodeType, FunctionType, MethodType

import torch
from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
from comfy.patcher_extension import CallbacksMP, WrappersMP

from .. import control_region_runtime as region
from .. import control_regions as geometry
from .. import h3_fun_control_advanced as fun
from ..long_video_dual_identity import (
    content_identity, _original_state, _original_buffers, _check_interrupted,
    UnknownBufferBackupSchema,
)
from ..patch_stack_policy import UnverifiedModelStack, _execution_selection, warn_patch_stack
from ..progressive_producers import (
    _configuration,
    _dynamic_runtime_fields,
    _legacy_runtime_fields,
    native_producer_identity,
)
from .effect_identity import _require
from .progressive_effect_identity import _code_identity, function_identity


KEY = "t8_h3_regional_control_owners_v1"


def prepared_image_identity(value):
    """Logical IMAGE bytes after native NCHW resize -> strided NHWC.

    This adapter is only for ordinary, already materialized RGB images. It
    neither widens the MODEL/VAE weight gate nor makes quantized/foreign
    tensor storage portable. Individual copies stay at most 4 MiB even if
    a single source frame is larger, without changing live storage/strides.
    """
    if (type(value) not in (torch.Tensor, torch.nn.Parameter) or value.ndim != 4
            or value.shape[-1] < 3 or value.device.type == "meta"
            or value.layout != torch.strided or value.is_quantized
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        return content_identity(value)
    if value.is_contiguous():
        return content_identity(value)

    limit = 4 * 1024 * 1024 // value.element_size()

    def chunks(view):
        if view.numel() <= limit:
            yield view
            return
        row = math.prod(view.shape[1:])
        if row <= limit:
            count = max(1, limit // max(1, row))
            for start in range(0, view.shape[0], count):
                yield view[start:start + count]
        else:
            for index in range(view.shape[0]):
                yield from chunks(view[index])

    digest = hashlib.sha256()
    for part in chunks(value.detach()):
        _check_interrupted()
        chunk = part.cpu().contiguous()
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("execution tensor has nonfinite values")
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return {"tensor_sha256": digest.hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}


@dataclass(frozen=True)
class ControlOwner:
    backend: str
    runtime: object
    patcher: object
    hooks: tuple
    wrappers: tuple
    rows: object = None
    patch: object = None
    prepared: object = None
    geometry_json: str | None = None
    rows_sha256: str | None = None
    previous_fun: object = None
    previous_region: object = None
    expected_fun: object = None


def capture_compat(runtime, patcher, rows, hooks, wrappers):
    runtime.owner = ControlOwner(
        "compatibility", runtime, patcher, tuple(hooks), tuple(wrappers), rows=rows
    )


def capture_official(runtime, patch, hooks, wrapper):
    runtime.owner = ControlOwner(
        "official_model_patch",
        runtime,
        patch.model_patch,
        tuple(hooks),
        (wrapper,),
        patch=patch,
    )


def bind(model, upstream, runtime, prepared):
    owner = runtime.owner
    if type(owner) is not ControlOwner:
        return  # Other real APIs retain ordinary execution, without this adapter.
    previous = upstream.get_attachment(KEY)
    if previous is None:
        previous = ()
    if type(previous) is not tuple or any(
        type(value) is not ControlOwner for value in previous
    ):
        runtime.owner = None
        runtime.retain("control_owner_stack_unknown_portable_identity_unavailable")
        return
    owner = replace(
        owner,
        prepared=prepared,
        geometry_json=geometry.canonical(runtime.geometry),
        rows_sha256=None if owner.rows is None else geometry.tensor_sha(owner.rows),
        previous_fun=deepcopy(upstream.get_attachment(fun.FUN_CONTROL_ATTACHMENT_KEY)),
        previous_region=upstream.get_attachment(region.REGION_ATTACHMENT),
        expected_fun=deepcopy(model.get_attachment(fun.FUN_CONTROL_ATTACHMENT_KEY)),
    )
    runtime.owner = owner
    model.set_attachments(KEY, (*previous, owner))


def _geometry(owner):
    state = owner.runtime
    current = state.geometry
    if geometry.canonical(current) != owner.geometry_json:
        raise ValueError(
            "Control owner geometry changed after actual source preparation"
        )
    unsigned = {
        key: value for key, value in current.items() if key != "geometry_sha256"
    }
    if (
        hashlib.sha256(geometry.canonical(unsigned).encode()).hexdigest()
        != current["geometry_sha256"]
    ):
        raise ValueError(
            "Control owner geometry SHA does not match its actual contents"
        )
    if geometry.tensor_sha(owner.prepared) != current["prepared_sha256"]:
        raise ValueError("Control owner prepared RGB changed")
    if owner.rows is not None and geometry.tensor_sha(owner.rows) != owner.rows_sha256:
        raise ValueError("Control owner encoded rows changed")
    gate = state.gate
    if (None if gate is None else gate.report()) != current["region"]:
        raise ValueError("Control owner region mask or native geometry changed")
    if gate is not None:
        # Validate every populated derived entry before omitting cache residency.
        fresh = geometry.RegionGate(
            gate.mask, gate.frame_indices, gate.width, gate.height, gate.length
        )
        for key, value in gate._cache.items():
            if (
                type(key) is not tuple
                or len(key) != 3
                or any(type(number) is not int for number in key)
                or content_identity(value) != content_identity(fresh.rows(*key))
            ):
                raise ValueError("Control owner derived region-row cache changed")
    if owner.patch is not None:
        owner.patch.verify_derived()
    return current


def _function_live(function):
    function = getattr(function, "__func__", function)
    if type(function) is not FunctionType:
        return _execution_selection(function)
    return {
        "object": id(function),
        "code": _code_identity(function.__code__),
        "defaults": _execution_selection(function.__defaults__),
        "kwdefaults": _execution_selection(function.__kwdefaults__),
    }


def _captured(function):
    return dict(
        zip(
            getattr(getattr(function, "__code__", None), "co_freevars", ()),
            [cell.cell_contents for cell in getattr(function, "__closure__", ()) or ()],
            strict=True,
        )
    )


def _execution_storage(patcher):
    """Known original contents; opaque backup formats never gain portability."""
    try:
        state = _original_state(patcher, patcher.model_state_dict())
        buffers = _original_buffers(patcher)
        unknown = None
    except UnknownBufferBackupSchema as error:
        warn_patch_stack(str(error))
        # Do not catch real malformed registration, bad shape or nonfinite data.
        content_identity(patcher.model_state_dict())
        state = _original_state(patcher, patcher.model_state_dict(), allow_unknown_buffers=True)
        buffers = _original_buffers(patcher, allow_unknown=True)
        unknown = _execution_selection(getattr(patcher, "backup_buffers", None))
    return content_identity(state), content_identity(buffers), unknown


def integrity_snapshot(owner):
    """Within-request content integrity, including an otherwise opaque user stack.

    This is not a portable identity or opaque callable-internals certificate.
    Plain tensor data, selected control weights and actual known source code
    remain checkable even when an unrelated foreign wrapper is executable.
    """
    actual_geometry = _geometry(owner)
    patcher = owner.patcher
    network = patcher.model
    methods = {
        name: {
            method: _function_live(getattr(module, method, None))
            for method in ("forward", "step", "init_stream")
        }
        for name, module in network.named_modules()
    }
    # Live identities bind helper code too, without requiring that an unknown
    # user replacement acquire a native portable factory qualification.
    helpers = {
        name: _function_live(getattr(fun, name))
        for name in (
            "_enter_control_stream",
            "_dense_control_options",
            "_video_span",
            "_patchify_control",
        )
    }
    helpers.update(
        {
            "runtime_" + name: _function_live(
                getattr(region.ControlRegionRuntime, name)
            )
            for name in ("pre_run", "cleanup", "consume")
        }
    )
    helpers.update(
        {
            "gate_rows": _function_live(geometry.RegionGate.rows),
            "target_span": _function_live(region.target_span),
            "selected_control_block": _function_live(fun.DiTBlock.forward),
        }
    )
    selected = []
    for _, hook, _ in owner.hooks:
        if type(hook) is FunctionType:
            captured = _captured(hook)
            selected.append(
                {
                    key: _execution_selection(value)
                    for key, value in captured.items()
                    if key not in {"state", "region_runtime", "active"}
                }
            )
            selected[-1]["active"] = {
                "code": _function_live(captured.get("active")),
                "fields": _execution_selection(_captured(captured.get("active"))),
            }
        else:
            selected.append(
                {
                    "call": _function_live(type(hook).__call__),
                    "previous": _execution_selection(getattr(hook, "previous", None)),
                    "control_patch": _execution_selection(
                        getattr(hook, "control_patch", None)
                    ),
                }
            )
    lazy = None
    if owner.patch is not None:
        patch = owner.patch
        lazy = {
            "RGB": content_identity(patch.control_video.contiguous()),
            "strength": patch.strength,
            "sigmas": [patch.sigma_start, patch.sigma_end],
            "methods": {
                name: _function_live(getattr(type(patch), name))
                for name in (
                    "diffusion_model_wrapper",
                    "after_block",
                    "before_block",
                    "prepare_control_latent",
                    "verify_derived",
                    "cleanup",
                    "_encode",
                    "_fit_frames",
                )
            },
            "vae_methods": {
                name: _function_live(getattr(patch.vae, name, None))
                for name in ("encode", "process_input", "spacial_compression_encode")
            },
        }
        vae_patcher = getattr(patch.vae, "patcher", None)
        if callable(getattr(vae_patcher, "model_state_dict", None)):
            lazy["vae_state"], lazy["vae_buffers"], lazy["unverified_buffer_backups"] = _execution_storage(vae_patcher)
    state, buffers, unknown = _execution_storage(patcher)
    return {
        "weights": state,
        "buffers": buffers,
        "unverified_buffer_backups": unknown,
        "patches": _execution_selection(patcher.patches),
        "geometry": deepcopy(actual_geometry),
        "prepared": geometry.tensor_sha(owner.prepared),
        "rows": None if owner.rows is None else geometry.tensor_sha(owner.rows),
        "methods": methods,
        "helpers": helpers,
        "selected_control_delegates": selected,
        "lazy_encoder": lazy,
        "hooks": [_function_live(hook) for _, hook, _ in owner.hooks],
        "wrappers": [_function_live(wrapper) for wrapper in owner.wrappers],
        "source": {
            Path(module.__file__).name: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (fun, region, geometry)
        },
    }


def _nested(function, factory, *path, fields=None, attributes=()):
    expected = factory.__code__
    for name in path:
        matches = [
            code
            for code in expected.co_consts
            if type(code) is CodeType and code.co_name == name
        ]
        _require(len(matches) == 1, "Control owner factory code is ambiguous")
        expected = matches[0]
    _require(
        type(function) is FunctionType
        and function.__code__ is expected
        and function.__globals__ is factory.__globals__
        and function.__defaults__ is None
        and function.__kwdefaults__ is None
        and set(vars(function)) == set(attributes),
        "Control owner has another executable closure",
    )
    captured = dict(
        zip(
            function.__code__.co_freevars,
            [cell.cell_contents for cell in function.__closure__ or ()],
            strict=True,
        )
    )
    _require(
        fields is None or set(captured) == set(fields),
        "Control owner closure fields changed",
    )
    return captured


def _network(owner):
    patcher, network = owner.patcher, owner.patcher.model
    from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl

    expected = (
        MiniMaxH3FunControl
        if owner.backend == "official_model_patch"
        else fun._MiniMaxH3FunControl
    )
    _require(
        type(patcher) in (ModelPatcher, ModelPatcherDynamic)
        and type(network) is expected,
        "Control network has an unknown portable executable owner",
    )
    for name in (
        "weight_wrapper_patches",
        "additional_models",
        "attachments",
        "wrappers",
        "callbacks",
        "injections",
        "hook_patches",
        "forced_hooks",
        "current_hooks",
    ):
        _require(
            not getattr(patcher, name, None),
            "Control patcher has another executable stack: " + name,
        )
    # Weight-only LoRAs remain selected content, never silently removed.
    objects = dict(patcher.object_patches)
    cast = objects.pop("manual_cast_dtype", None)
    _require(
        not objects and (cast is None or isinstance(cast, torch.dtype)),
        "Control patcher has unknown executable object patches",
    )
    modules = list(network.named_modules())
    refs = {id(module): "control." + name for name, module in modules}
    internals = set(vars(torch.nn.Module())) - {"training"}
    classes, configurations = {}, {}
    for name, module in modules:
        _require(
            not module._forward_hooks
            and not module._forward_pre_hooks
            and not any(
                key in vars(module) for key in ("forward", "step", "init_stream")
            ),
            "Control module has unknown live hooks or method overrides",
        )
        cls = type(module)
        _require(
            cls.__module__.startswith(("comfy.", "torch.nn."))
            or cls in (fun._MiniMaxH3FunControl, fun._ControlDiTBlock),
            "Control submodule requires a separate portable execution adapter",
        )
        label = cls.__module__ + "." + cls.__qualname__
        ignored = (
            internals
            | _dynamic_runtime_fields(patcher, module, name)
            | _legacy_runtime_fields(patcher, module)
        )
        ignored |= {"comfy_force_cast_weights", "comfy_patched_weights"}
        if module is network:
            ignored |= {
                "device",
                "model_loaded_weight_memory",
                "lowvram_patch_counter",
                "model_lowvram",
                "current_weight_patches_uuid",
                "model_offload_buffer_memory",
                "current_patcher",
                "dynamic_vbars",
                "dynamic_pins",
                "dynamic_patchers",
            }
            if cast is not None:
                ignored.add("manual_cast_dtype")
        from .hyperflow_identity import _class_method

        classes[label] = {
            method: {
                "code": function_identity(_class_method(cls, method)),
                "source": hashlib.sha256(
                    Path(inspect.getsourcefile(getattr(cls, method))).read_bytes()
                ).hexdigest(),
            }
            for method in ("forward", "step", "init_stream")
            if type(getattr(cls, method, None)) is FunctionType
        }
        configurations[name] = {
            "class": label,
            "fields": _configuration(
                {
                    key: value
                    for key, value in vars(module).items()
                    if key not in ignored
                },
                refs,
            ),
        }
    return {
        "classes": classes,
        "configuration": configurations,
        "state": content_identity(_original_state(patcher, patcher.model_state_dict())),
        "patches": content_identity(patcher.patches),
        "buffers": content_identity(_original_buffers(patcher)),
        "options": _configuration(patcher.model_options, refs),
        "cast": content_identity(cast),
    }


def _proof(owner):
    from .hyperflow_identity import _source_functions

    source = _source_functions(
        fun,
        (
            "_apply_compatibility",
            "_apply_official_model_patch",
            "_enter_control_stream",
            "_dense_control_options",
            "_video_span",
            "_patchify_control",
        ),
    )
    region_source = _source_functions(
        region, ("official_region_patch", "target_span", "register_runtime")
    )
    _require(
        type(owner.runtime) is region.ControlRegionRuntime
        and owner.runtime.owner is owner,
        "Control runtime is not bound to its actual selected owner",
    )
    network = owner.patcher.model
    from comfy.ldm.minimax import model as native_model
    from comfy.ldm.minimax import controlnet as native_control

    _require(
        fun.DiTBlock is native_model.DiTBlock
        and native_control.DiTBlock is native_model.DiTBlock,
        "Control block global delegate has another execution owner",
    )
    _require(
        tuple(layer for layer, _, _ in owner.hooks) == tuple(network.injection_layers),
        "Control hook inventory differs from actual injection layers",
    )
    if owner.backend == "compatibility":
        for layer, hook, previous in owner.hooks:
            state = _nested(
                hook,
                fun._apply_compatibility,
                "make_hook",
                "hook",
                fields={
                    "active",
                    "control",
                    "inner",
                    "layer",
                    "region_runtime",
                    "rows",
                    "state",
                    "strength",
                },
                attributes={"_t8_h3_fun_control", "_t8_h3_fun_control_layer"},
            )
            _require(
                state["control"] is network
                and state["rows"] is owner.rows
                and state["region_runtime"] is owner.runtime
                and state["layer"] == layer
                and state["inner"] is previous
                and hook._t8_h3_fun_control is True
                and hook._t8_h3_fun_control_layer == layer,
                "Control hook captured another tower/rows/runtime/delegate",
            )
            _nested(
                state["active"],
                fun._apply_compatibility,
                "active",
                fields={"end_sigma", "start_sigma"},
            )
            _require(
                type(state["state"]) is dict
                and set(state["state"]) == {"stream"}
                and state["state"]["stream"] is None,
                "Control stream is in flight or has another owner",
            )
        _require(len(owner.wrappers) == 1, "Control forward scope inventory changed")
        scope = _nested(
            owner.wrappers[0],
            fun._apply_compatibility,
            "isolated_control_forward",
            fields={"clear_stream", "region_runtime"},
        )
        clear = _nested(
            scope["clear_stream"],
            fun._apply_compatibility,
            "clear_stream",
            fields={"state"},
        )
        _require(
            scope["region_runtime"] is owner.runtime
            and clear["state"] is state["state"]
            and owner.runtime.cleanup_callbacks == [scope["clear_stream"]],
            "Control cleanup scope has another stream owner",
        )
        execution = {
            "strength": state["strength"],
            "sigmas": _nested(state["active"], fun._apply_compatibility, "active"),
        }
    else:
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3FunControlPatch,
            MiniMaxH3FunControlBlockPatch,
        )

        patch = owner.patch
        _require(
            type(patch).__bases__ == (MiniMaxH3FunControlPatch,),
            "Official region patch has another base owner",
        )
        method = type(patch).after_block
        captured = _nested(
            method,
            region.official_region_patch,
            "RegionalPatch",
            "after_block",
            fields={"runtime"},
        )
        _require(
            captured["runtime"] is owner.runtime and patch.model_patch is owner.patcher,
            "Official region patch captured another runtime or tower",
        )
        for name in (
            "__init__",
            "diffusion_model_wrapper",
            "prepare_control_latent",
            "verify_derived",
            "cleanup",
        ):
            function = getattr(type(patch), name)
            _nested(function, region.official_region_patch, "RegionalPatch", name)
        _require(
            len(owner.wrappers) == 1
            and type(owner.wrappers[0]) is MethodType
            and owner.wrappers[0].__self__ is patch
            and owner.wrappers[0].__func__ is type(patch).diffusion_model_wrapper,
            "Official wrapper belongs to another control patch",
        )
        for layer, hook, previous in owner.hooks:
            _require(
                type(hook) is MiniMaxH3FunControlBlockPatch
                and set(vars(hook)) == {"control_patch", "block_index", "previous"}
                and hook.control_patch is patch
                and hook.block_index == layer
                and hook.previous is previous,
                "Official control block has another predecessor or patch owner",
            )
        _require(
            patch.mask is None
            and patch.source_video is None
            and patch.control_stream is None
            and patch.pristine_stream is None,
            "Official control has another source or in-flight state",
        )
        try:
            vae = native_producer_identity(patch.vae, "video_vae")
        except (ValueError, AttributeError) as error:
            raise UnverifiedModelStack(
                "Control lazy VAE remains executable without a known portable producer"
            ) from error
        _require(
            vae.get("portable_cache_reuse") is not False,
            "Control lazy VAE has an unknown execution owner",
        )
        execution = {
            "strength": patch.strength,
            "sigmas": [patch.sigma_start, patch.sigma_end],
            "RGB": content_identity(patch.control_video.contiguous()),
            "vae": vae,
            "core_patch_methods": {
                name: function_identity(value)
                for cls in (MiniMaxH3FunControlPatch, MiniMaxH3FunControlBlockPatch)
                for name, value in vars(cls).items()
                if type(value) is FunctionType
            },
        }
    return {
        "factory": source,
        "region_factory": region_source,
        "execution": execution,
        "runtime_methods": {
            name: function_identity(getattr(region.ControlRegionRuntime, name))
            for name in ("pre_run", "cleanup", "consume")
        },
        "geometry_rows": function_identity(geometry.RegionGate.rows),
    }


def _remove_callback(view, owner, role, name):
    groups = view.callbacks.get(role, {})
    selected = [
        (key, index)
        for key, values in groups.items()
        for index, method in enumerate(values)
        if type(method) is MethodType
        and method.__self__ is owner.runtime
        and method.__func__ is getattr(region.ControlRegionRuntime, name)
    ]
    _require(len(selected) == 1, "Control lifecycle callback inventory changed")
    key, index = selected[0]
    groups[key].pop(index)
    if not groups[key]:
        groups.pop(key)
    if not groups:
        view.callbacks.pop(role, None)


def project(model):
    owners = model.get_attachment(KEY)
    if owners is None:
        return model, None
    _require(
        type(owners) is tuple
        and owners
        and all(type(owner) is ControlOwner for owner in owners),
        "Control owner stack has an unknown schema",
    )
    view = model.clone()
    contracts = []
    for owner in reversed(owners):
        _require(
            view.get_attachment(region.REGION_ATTACHMENT) is owner.runtime,
            "Control runtime attachment does not match the actual hook stack",
        )
        _require(
            view.get_attachment(fun.FUN_CONTROL_ATTACHMENT_KEY) == owner.expected_fun,
            "Control diagnostic attachment differs from its selected owner stack",
        )
        actual_geometry = _geometry(owner)
        proof = _proof(owner)
        replaces = view.model_options["transformer_options"]["patches_replace"]
        dit = replaces["dit"]
        for layer, hook, previous in owner.hooks:
            _require(
                dit.get(("double_block", layer)) is hook,
                "Control selected block delegate was replaced",
            )
            if previous is None:
                dit.pop(("double_block", layer))
            else:
                dit[("double_block", layer)] = previous
        if not dit:
            replaces.pop("dit")
        if not replaces:
            view.model_options["transformer_options"].pop("patches_replace")
        for wrapper in owner.wrappers:
            groups = view.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
            selected = [
                (key, index)
                for key, values in groups.items()
                for index, value in enumerate(values)
                if value is wrapper
            ]
            _require(len(selected) == 1, "Control selected forward scope was replaced")
            key, index = selected[0]
            groups[key].pop(index)
            if not groups[key]:
                groups.pop(key)
            if not groups:
                view.wrappers.pop(WrappersMP.DIFFUSION_MODEL, None)
        _remove_callback(view, owner, CallbacksMP.ON_PRE_RUN, "pre_run")
        _remove_callback(view, owner, CallbacksMP.ON_CLEANUP, "cleanup")
        if owner.backend == "compatibility":
            additional = view.additional_models.get(
                fun.FUN_CONTROL_ADDITIONAL_MODEL_KEY, []
            )
            # Core.clone legitimately clones additional patcher containers,
            # sharing the actual tower. Prove selected contents, not UUID/id.
            _require(
                additional
                and additional[-1].model is owner.patcher.model
                and _network(replace(owner, patcher=additional[-1])) == _network(owner),
                "Control additional model order or selected contents changed",
            )
            additional.pop()
            if not additional:
                view.additional_models.pop(fun.FUN_CONTROL_ADDITIONAL_MODEL_KEY, None)
        for key, value in (
            (fun.FUN_CONTROL_ATTACHMENT_KEY, owner.previous_fun),
            (region.REGION_ATTACHMENT, owner.previous_region),
        ):
            if value is None:
                view.remove_attachments(key)
            else:
                view.set_attachments(key, value)
        contracts.append(
            {
                "backend": owner.backend,
                "geometry": deepcopy(actual_geometry),
                "prepared_RGB": prepared_image_identity(owner.prepared),
                "encoded_rows": None
                if owner.rows is None
                else content_identity(owner.rows),
                "control": _network(owner),
                "proof": proof,
            }
        )
    view.remove_attachments(KEY)
    return view, {
        "schema": "t8.h3.control-owner-content.v1",
        "controls": list(reversed(contracts)),
        "projection_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
