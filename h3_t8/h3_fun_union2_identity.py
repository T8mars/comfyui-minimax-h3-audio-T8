"""Read-only content projection of an explicit native Union2 stage operator.

Unknown owners still execute; they do not acquire portable Save/Load identity.
"""

from types import FunctionType, MethodType

import torch
from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
from comfy.patcher_extension import WrappersMP
from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3FunControlPatch,
    MiniMaxH3FunControlBlockPatch,
)
from comfy.ldm.minimax.vae import IMAGENET_MEAN

from .h3_fun_union2 import ATTACHMENT_KEY, LAYERS, PROFILE
from .h3_fun_union2_runtime import AlignedUnion2Patch
from .long_video_dual_identity import content_identity, _implementation, _original_state
from .patch_stack_policy import UnverifiedModelStack
from .progressive_producers import (
    _configuration,
    _dynamic_runtime_fields,
    native_producer_identity,
)
from .modular_sampling.progressive_effect_identity import function_identity


def require(condition, message):
    if not condition:
        raise UnverifiedModelStack(message)


def control_description(patcher):
    require(
        type(patcher) in (ModelPatcher, ModelPatcherDynamic),
        "Union2 control patcher owner is unknown",
    )
    network = patcher.model
    require(
        type(network) is MiniMaxH3FunControl
        and network.injection_layers == LAYERS
        and network.control_in_dim == 49
        and network.inpaint_post_norm is True,
        "Union2 live control architecture differs from the explicit profile",
    )
    for name in (
        "patches",
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
        require(
            not getattr(patcher, name, None),
            "Union2 control has an unknown executable stack: " + name,
        )
    objects = dict(patcher.object_patches)
    cast = objects.pop("manual_cast_dtype", None)
    require(
        not objects and (cast is None or isinstance(cast, torch.dtype)),
        "Union2 control object patches are unverified",
    )
    modules = list(network.named_modules())
    references = {id(module): "control." + name for name, module in modules}
    internals = set(vars(torch.nn.Module())) - {"training"}
    classes, configuration = {}, {}
    for name, module in modules:
        require(
            not any(
                getattr(module, key, None)
                for key in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                )
            ),
            "Union2 control contains a live hook",
        )
        require(
            not any(key in vars(module) for key in ("forward", "step", "init_stream")),
            "Union2 control execution method was replaced",
        )
        cls = type(module)
        require(
            cls.__module__.startswith(("comfy.", "torch.nn.")),
            "Union2 control has a foreign module class",
        )
        ignored = internals | _dynamic_runtime_fields(patcher, module, name)
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
        label = cls.__module__ + "." + cls.__qualname__
        if label not in classes:
            classes[label] = {
                "source": _implementation(cls),
                "methods": {
                    method: function_identity(getattr(cls, method))
                    for method in ("forward", "step", "init_stream")
                    if type(getattr(cls, method, None)) is FunctionType
                },
            }
        configuration[name] = {
            "class": label,
            "settings": _configuration(
                {
                    key: value
                    for key, value in vars(module).items()
                    if key not in ignored
                },
                references,
            ),
        }
    state = _original_state(patcher, patcher.model_state_dict())
    require(bool(state), "Union2 control has no actual loaded tensor state")
    return {
        "classes": classes,
        "configuration": configuration,
        "state": content_identity(state),
        "buffers": content_identity(dict(network.named_buffers())),
        "model_options": _configuration(patcher.model_options, references),
        "placement": content_identity(
            {
                "load": patcher.load_device,
                "offload": patcher.offload_device,
                "force_cast_weights": patcher.force_cast_weights,
                "manual_cast": cast,
            }
        ),
    }


def project(model):
    receipt = model.get_attachment(ATTACHMENT_KEY)
    if receipt is None:
        return model, None
    # Do not import this adapter for an old Core unless this explicit attachment exists.
    require(
        type(receipt) is dict
        and set(receipt)
        == {
            "profile",
            "filename",
            "width",
            "height",
            "length",
            "strength",
            "start_percent",
            "end_percent",
            "control_present",
            "broadcast_single_mask",
        }
        and receipt["profile"] == PROFILE,
        "Union2 attachment is not the explicit inert apply receipt",
    )
    groups = model.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
    selected = [
        (group, index, method)
        for group, methods in groups.items()
        for index, method in enumerate(methods)
        if type(method) is MethodType and type(method.__self__) is AlignedUnion2Patch
    ]
    require(
        len(selected) == 1,
        "Union2 requires one exact live owner per stage for portable projection",
    )
    group, index, method = selected[0]
    owner = method.__self__
    require(
        method.__func__ is MiniMaxH3FunControlPatch.diffusion_model_wrapper,
        "Union2 diffusion wrapper was replaced",
    )
    expected_fields = {
        "model_patch",
        "vae",
        "control_video",
        "mask",
        "source_video",
        "strength",
        "sigma_start",
        "sigma_end",
        "control_latent",
        "control_latent_shape",
        "control_stream",
        "pristine_stream",
        "active",
        "expected_shape",
        "derived_content",
    }
    require(
        set(vars(owner)) == expected_fields
        and owner.control_stream is None
        and owner.pristine_stream is None,
        "Union2 owner contains foreign or in-flight execution state",
    )
    require(
        owner.strength == receipt["strength"] and type(owner.active) is bool,
        "Union2 captured strength changed",
    )
    from comfy_extras.nodes_minimax_h3 import video_latent_t

    expected = (
        1,
        24,
        video_latent_t(receipt["length"]),
        receipt["height"] // 16,
        receipt["width"] // 16,
    )
    if type(owner.expected_shape) is not tuple or owner.expected_shape != expected:
        raise ValueError("Union2 declared grid and actual owner shape changed")
    if (owner.control_video is not None) != receipt["control_present"]:
        raise ValueError("Union2 control presence changed")
    # Native sigma values, not assumed percent equivalence after another stage setup.
    require(
        type(owner.sigma_start) is float and type(owner.sigma_end) is float,
        "Union2 native sigma range has another owner",
    )
    owner.verify_derived()
    controls = (
        model.model_options.get("transformer_options", {})
        .get("patches_replace", {})
        .get("dit", {})
    )
    for layer in LAYERS:
        entry = controls.get(("double_block", layer))
        require(
            type(entry) is MiniMaxH3FunControlBlockPatch
            and set(vars(entry)) == {"control_patch", "block_index", "previous"}
            and entry.control_patch is owner
            and entry.block_index == layer,
            "Union2 block wrapper is overwritten or has another owner",
        )
        require(
            entry.__call__.__func__ is MiniMaxH3FunControlBlockPatch.__call__,
            "Union2 block delegate execution was replaced",
        )
    try:
        vae = native_producer_identity(owner.vae, "video_vae")
    except ValueError as error:
        raise UnverifiedModelStack(
            "Union2 VAE is not a known native H3 encoding producer"
        ) from error
    require(
        vae.get("portable_cache_reuse") is not False,
        "Union2 VAE remains executable but nonportable",
    )
    contract = {
        "schema": "t8.fun_union2.selected-operator-content.v1",
        "profile": PROFILE,
        "settings": content_identity(
            {key: value for key, value in receipt.items() if key != "filename"}
        ),
        "native_sigmas": [owner.sigma_start, owner.sigma_end],
        "source_video": content_identity(owner.source_video.contiguous()),
        "mask": content_identity(owner.mask.contiguous()),
        "control_video": None
        if owner.control_video is None
        else content_identity(owner.control_video.contiguous()),
        "vae": vae,
        "control": control_description(owner.model_patch),
        "post_norm_mean": content_identity(IMAGENET_MEAN),
        "implementation": {
            cls.__name__: {
                "source": _implementation(cls),
                "methods": {
                    name: function_identity(value)
                    for name, value in vars(cls).items()
                    if type(value) is FunctionType
                },
            }
            for cls in (
                AlignedUnion2Patch,
                MiniMaxH3FunControlPatch,
                MiniMaxH3FunControlBlockPatch,
            )
        },
        "projection_source": _implementation(project),
        "boundary": "Content only; no completion, quality, provenance or original-audio guarantee.",
    }
    view = model.clone()
    view.remove_attachments(ATTACHMENT_KEY)
    wrappers = view.wrappers[WrappersMP.DIFFUSION_MODEL][group]
    wrappers.pop(index)
    if not wrappers:
        view.wrappers[WrappersMP.DIFFUSION_MODEL].pop(group)
    if not view.wrappers[WrappersMP.DIFFUSION_MODEL]:
        view.wrappers.pop(WrappersMP.DIFFUSION_MODEL)
    replaces = dict(view.model_options["transformer_options"]["patches_replace"])
    dit = dict(replaces["dit"])
    for layer in LAYERS:
        entry = controls[("double_block", layer)]
        if entry.previous is None:
            dit.pop(("double_block", layer))
        else:
            dit[("double_block", layer)] = entry.previous
    if dit:
        replaces["dit"] = dit
    else:
        replaces.pop("dit")
    if replaces:
        view.model_options["transformer_options"]["patches_replace"] = replaces
    else:
        view.model_options["transformer_options"].pop("patches_replace")
    return view, contract
