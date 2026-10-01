"""Inspection-only identity for the original OpenVDN Composer and side model."""
import torch
from torch import nn
import comfy.model_patcher
import comfy.ops

from .. import vdn_h3_advanced as vdn
from ..long_video_dual_identity import content_identity, _original_state
from ..patch_stack_policy import UnverifiedModelStack
from ..vdn_attention_compat import _factory_closure


def require(condition, message):
    if not condition:
        raise UnverifiedModelStack("VDN " + message)


def original_hook(hook, block, branch):
    state = _factory_closure(hook, vdn.compose_vdn_model, "hook")
    return (state == {} and hook.__defaults__ == (block, branch)
            and hook.__kwdefaults__ is None and not vars(hook))


def _cast_configuration(patcher, module, path, configuration, checked_patches, patch_content):
    """Authenticate native delayed weight application, without mutating it.

    Core clones share modules/backups but copy the patch dictionary. Compare
    exact patch contents, not dictionary identity. Prepared copies must still
    carry the same bytes/order; a name or UUID alone is never sufficient.
    """
    manual = type(module) in (comfy.ops.manual_cast.Linear, comfy.ops.manual_cast.Conv1d,
                             comfy.ops.manual_cast.Conv2d)
    if not manual:
        return
    default = type(module).comfy_cast_weights
    previous = configuration.pop("prev_comfy_cast_weights", None)
    current = configuration.pop("comfy_cast_weights", default)
    require(type(default) is bool and current is default and (previous is None or previous is default),
            "branch cast flags differ from the native manual-cast operator")
    for parameter in ("weight", "bias"):
        functions = configuration.pop(parameter + "_function", [])
        require(type(functions) is list and len(functions) <= 1,
                "branch has an unadapted weight callback")
        key = path + "." + parameter
        if not functions:
            require(not (previous is not None and patcher.model.model_lowvram
                         and key in patcher.patches and key not in patcher.backup),
                    "native delayed patch disappeared: " + key)
            continue
        function = functions[0]
        require(type(function) is comfy.model_patcher.LowVramPatch and set(vars(function)) == {
            "key", "patches", "convert_func", "set_func", "prepared_patches"},
            "branch delayed patch implementation/state is unknown")
        require(patcher.model.model_lowvram is True and previous is not None
                and key not in patcher.backup and function.key == key and key in patcher.patches,
                "branch delayed patch target/loading owner differs")
        require(function.convert_func is None and function.set_func is None,
                "branch delayed patch has unadapted conversion callbacks")
        require(type(function.patches) is dict, "branch delayed patches are not a native dictionary")
        if id(function.patches) not in checked_patches:
            require(content_identity(function.patches) == patch_content,
                    "branch delayed patch contents differ from selected side model")
            checked_patches.add(id(function.patches))
        prepared = function.prepared_patches
        require(prepared is None or (type(prepared) is list
                and content_identity(prepared) == content_identity(patcher.patches[key])),
                "branch prepared delayed patches differ from selected contents")


def branch_identity(patcher):
    require(type(patcher) is comfy.model_patcher.ModelPatcher and type(patcher.model) is vdn.VDNBranchModel,
            "additional model has another patcher/architecture")
    for name in ("object_patches", "weight_wrapper_patches", "callbacks", "injections", "wrappers",
                 "hook_patches", "forced_hooks", "current_hooks", "additional_models", "attachments"):
        require(not getattr(patcher, name, None), "additional model has unknown " + name)
    allowed = (vdn.VDNBranchModel, vdn.VDNBlockBranch, vdn.BidirectionalLinearBranch,
        vdn.LinearAttentionSepConv, vdn.FrameKDAAlpha, vdn.OutputGate, vdn.BranchRMSNorm,
        nn.ModuleList, comfy.ops.manual_cast.Linear, comfy.ops.manual_cast.Conv1d, comfy.ops.manual_cast.Conv2d)
    standard = set(vars(nn.Module()))
    structure = []
    patch_content = content_identity(patcher.patches)
    checked_patches = set()
    for name, module in patcher.model.named_modules():
        require(type(module) in allowed, "branch module implementation is unknown: " + name)
        require(not any(value for key, value in vars(module).items() if "hook" in key),
                "branch module has executable hooks: " + name)
        configuration = {key: value for key, value in vars(module).items() if key not in standard}
        # Core ModelPatcher.load bookkeeping is derived, not another branch
        # operator. Authenticate exact empty/typed values, never ignore live
        # weight functions or foreign UUID ownership to fabricate portability.
        if module is patcher.model:
            for key in ("model_loaded_weight_memory", "lowvram_patch_counter", "model_offload_buffer_memory"):
                value = configuration.pop(key, 0)
                require(type(value) is int and value >= 0, "branch load bookkeeping changed")
            require(type(configuration.pop("model_lowvram", False)) is bool, "branch load mode is invalid")
            token = configuration.pop("current_weight_patches_uuid", None)
            require(token is None or token == patcher.patches_uuid, "side-model loaded patch owner differs")
            device = configuration.pop("device", None)
            require(device is None or type(device) is torch.device, "branch device bookkeeping changed")
        _cast_configuration(patcher, module, name, configuration, checked_patches, patch_content)
        for key in ("weight_function", "bias_function"):
            if key in configuration:
                require(configuration.pop(key) == [], "branch has an unadapted weight callback")
        if "comfy_patched_weights" in configuration:
            require(type(configuration.pop("comfy_patched_weights")) is bool, "branch patched flag is invalid")
        if "comfy_force_cast_weights" in configuration:
            require(configuration.pop("comfy_force_cast_weights") is patcher.force_cast_weights,
                    "branch force-cast owner differs")
        structure.append({"path": name, "class": type(module).__qualname__, "configuration": content_identity(configuration)})
    require(vdn.LinearAttentionSepConv.KERNEL == 5 and vdn.BidirectionalLinearBranch.TEXT_STATE_SCALE == .5,
            "branch class-level numerical settings changed")
    return {"structure": structure, "state": content_identity(_original_state(patcher, patcher.model_state_dict())),
            "patches": patch_content, "options": content_identity(patcher.model_options),
            "force_cast_weights": patcher.force_cast_weights}


def weight_execution(model):
    """Observe selected native side-patch math just before a denoiser call.

    Logical raw weights/LoRA identity is residency independent, but materialized
    low-precision patches round at a different point from delayed patches. The
    stage receipt must retain that actual path, not merge both request hashes.
    Full content/owner authentication remains in the pre/post identity checks.
    """
    branches = model.additional_models.get(vdn.ADDITIONAL_MODEL_KEY)
    require(type(branches) is list and len(branches) == 1, "missing side-model execution owner")
    patcher = branches[0]
    require(type(patcher) is comfy.model_patcher.ModelPatcher and type(patcher.model) is vdn.VDNBranchModel,
            "unadapted side-model execution owner")
    require(patcher.model.current_weight_patches_uuid == patcher.patches_uuid,
            "side-model weights are not loaded for the selected patches")
    records = []
    for key in sorted(patcher.patches):
        path, parameter = key.rsplit(".", 1)
        module = patcher.model.get_submodule(path)
        weight = getattr(module, parameter)
        require(type(weight) in (torch.Tensor, nn.Parameter), "unadapted side-weight storage")
        functions = getattr(module, parameter + "_function", [])
        if functions:
            require(type(functions) is list and len(functions) == 1
                    and type(functions[0]) is comfy.model_patcher.LowVramPatch
                    and functions[0].key == key and key not in patcher.backup,
                    "unadapted delayed side-weight execution")
            mode = "core_delayed_compute_dtype"
        else:
            require(key in patcher.backup, "side-weight patch was not materialized or delayed")
            mode = "core_materialized_storage_dtype"
        records.append({"key": key, "application": mode, "storage_dtype": str(weight.dtype)})
    return {"schema": "t8.modular-sampling.vdn-weight-execution.v1", "patches": records,
            "force_cast_weights": patcher.force_cast_weights}


def inspect(model):
    branches = model.additional_models.get(vdn.ADDITIONAL_MODEL_KEY)
    require(type(branches) is list and len(branches) == 1, "original additional branch is missing")
    patcher = branches[0]
    contract = branch_identity(patcher)
    blocks = model.get_model_object("diffusion_model").blocks
    require(len(blocks) == len(patcher.model.blocks), "main/side block counts differ")
    options = model.model_options["transformer_options"]
    hooks = options.get(vdn.OWNER_HOOKS_KEY)
    require(type(hooks) is tuple and len(hooks) == len(blocks), "Composer hook owner tuple is invalid")
    active = options.get("patches_replace", {}).get("dit", {})
    from .vdn_effects import unwrap
    for index, (block, branch, hook) in enumerate(zip(blocks, patcher.model.blocks, hooks)):
        require(active.get(("double_block", index)) is hook and original_hook(unwrap(hook), block, branch),
                "block has another executable owner")
    require(model.get_wrappers("diffusion_model", vdn.WRAPPER_KEY) == [vdn._layout_wrapper],
            "layout wrapper source/order changed")
    if vdn.LAYOUT_KEY in options:
        require(type(options[vdn.LAYOUT_KEY]) is vdn.VDNSequenceLayout, "derived layout has another owner")
    return {"schema": "t8.modular-sampling.vdn-identity.v1", "branch": contract,
            "receipt": content_identity(model.get_attachment(vdn.ATTACHMENT_KEY))}


def project(model):
    contract = inspect(model)
    view = model.clone()
    view.additional_models = dict(view.additional_models)
    view.additional_models.pop(vdn.ADDITIONAL_MODEL_KEY)
    view.remove_attachments(vdn.ATTACHMENT_KEY)
    view.remove_wrappers_with_key("diffusion_model", vdn.WRAPPER_KEY)
    options = view.model_options["transformer_options"]
    hooks = options.pop(vdn.OWNER_HOOKS_KEY)
    options.pop(vdn.LAYOUT_KEY, None)
    replacements = dict(options.get("patches_replace", {}))
    dit = dict(replacements.get("dit", {}))
    for index in range(len(hooks)):
        dit.pop(("double_block", index))
    if dit:
        replacements["dit"] = dit
    else:
        replacements.pop("dit", None)
    if replacements:
        options["patches_replace"] = replacements
    else:
        options.pop("patches_replace", None)
    return view, contract
