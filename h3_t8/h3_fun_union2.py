"""Explicit Union2 profile; convert raw names, delegate the running Core.

No automatic training identity, legacy defaults, shared Core or audio changes.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import re
import torch
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from .h3_fun_control_advanced import (
    H3FunControlBundle,
    _official_model_patch_fun_control_modules,
    _resolve_fun_control_path,
    _convert_diffusers_state_dict,
    _normalize_fun_quantization,
    _checkpoint_structure,
    _assert_compatible_adaln_pair,
)

CONTROL_TYPE = "H3_T8_FUN_UNION2"
PROFILE = "union2_10x5_post_norm_exp"
LAYERS = tuple(range(0, 50, 5))
ATTACHMENT_KEY = "t8_minimax_h3_fun_union2_v1"
SOURCE = "https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/tree/0cb77d08bd6902d54eefede0d74b3450ae4840eb"


def union_structure(state, metadata):
    raw_names = "control_blocks.0.attn.to_q.weight" in state
    if raw_names:
        state = _convert_diffusers_state_dict(state)
    state, metadata = _normalize_fun_quantization(state, metadata)
    metadata = metadata or {}
    structure = _checkpoint_structure(state)
    ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"control_blocks\.(\d+)\.", key))
    }
    if (
        ids != set(range(10))
        or structure["block_count"] != 10
        or structure["control_in_dim"] != 49
    ):
        raise ValueError(
            "显式Union2需要连续10块和49通道，不能将旧5块/其它结构当成此配置"
        )
    if state["control_proj_in.weight"].shape[1] != 196:
        raise ValueError("Union2 projection必须为24控制+1 keep+24源潜变量的196列")
    if "control_blocks_places" in metadata:
        places = json.loads(metadata["control_blocks_places"])
        if (
            not isinstance(places, list)
            or any(type(value) is not int for value in places)
            or tuple(places) != LAYERS
        ):
            raise ValueError("权重placement元数据与显式Union2每5层配置冲突")
    if metadata.get("inpaint_masked_pixel_mode", "post_norm") != "post_norm":
        raise ValueError("权重遮罩模式不是post_norm，不能静默改成Union2")
    audio = metadata.get("control_apply_audio", "false")
    if not (audio is False or type(audio) is str and audio in ("false", "False")):
        raise ValueError("Union2控制skip不应用到音频，所选元数据冲突")
    curves = metadata.get("minimax_h3_fun_controlnet") == "adaln_basis"
    if structure["time_embed_dim"] != (8 if curves else 2688):
        raise ValueError("Union AdaLN实物宽度与full/curve元数据不符；不补零或猜转换")
    structure.update(
        raw_names_converted=raw_names,
        injection_layers=LAYERS,
        inpaint_post_norm=True,
        adaln_curves=curves,
    )
    return state, structure


def load_union2(name):
    modules = _official_model_patch_fun_control_modules()
    if (
        modules is None
        or "inpaint_post_norm" not in inspect.signature(modules[1]).parameters
    ):
        raise RuntimeError("当前运行Core没有Union post_norm接口；不自动升级或回退旧5块")
    path, folder = _resolve_fun_control_path(name)
    state, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )
    state, structure = union_structure(state, metadata)
    if not modules[0].is_minimax_h3_fun_state_dict(state):
        raise ValueError("转换后的权重不满足当前Core H3 Fun控制合同")
    from comfy_extras.nodes_model_patch import dit_patch_operations

    dtype, operations = dit_patch_operations(state)
    control = modules[1](
        control_in_dim=49,
        injection_layers=LAYERS,
        inpaint_post_norm=True,
        hidden_size=structure["hidden_size"],
        num_attention_heads=structure["num_heads"],
        attention_head_dim=structure["head_dim"],
        ffn_hidden_size=structure["ffn_hidden_size"],
        time_embed_dim=structure["time_embed_dim"],
        use_adaln_curves=structure["adaln_curves"],
        operations=operations,
        dtype=dtype,
        device=comfy.model_management.unet_offload_device(),
    )
    control.requires_grad_(False)
    patcher_class = getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )
    patcher = patcher_class(
        control,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    control.load_state_dict(state, strict=True, assign=patcher.is_dynamic())
    report = {
        "schema": "t8.fun_union2.loader.v1",
        "profile": PROFILE,
        "status": "loaded_native_core",
        "source_folder": folder,
        "filename": name,
        "file_bytes_diagnostic": Path(path).stat().st_size,
        "structure": {
            key: value for key, value in structure.items() if key != "state_dict"
        },
        "compute_dtype": str(dtype),
        "source": SOURCE,
        "training_identity_inferred": False,
        "boundary": "Explicit layout/strict weight load, not trained quality or universal provenance.",
    }
    bundle = H3FunControlBundle(
        "official_model_patch", patcher, name, str(path), report
    )
    return bundle, json.dumps(report, ensure_ascii=False, sort_keys=True)


def _frames(value, label, length, height, width):
    if (
        type(value) is not torch.Tensor
        or value.ndim != 4
        or tuple(value.shape[:3]) != (length, height, width)
        or value.shape[-1] not in (3, 4)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or value.min() < 0
        or value.max() > 1
    ):
        raise ValueError(
            label
            + "必须是同一明确帧格/画布的有限0–1 RGB IMAGE；先在外部统一resize/crop/trim"
        )
    return value[..., :3]


def apply_union2(
    model,
    positive,
    bundle,
    vae,
    source_video,
    regen_mask,
    width,
    height,
    length,
    strength=0.7,
    start_percent=0.0,
    end_percent=0.75,
    control_video=None,
    broadcast_single_mask=False,
):
    for value in (strength, start_percent, end_percent):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Union强度/范围必须是有限数字")
    if not 0 <= strength <= 2 or not 0 <= start_percent <= end_percent <= 1:
        raise ValueError("Union强度需0–2，作用范围需0≤start≤end≤1")
    if strength == 0:
        return (
            model,
            positive,
            json.dumps(
                {
                    "schema": "t8.fun_union2.apply.v1",
                    "status": "bypass",
                    "model_identity_preserved": True,
                    "conditioning_identity_preserved": True,
                    "vae_encoded": False,
                },
                sort_keys=True,
            ),
        )
    if (
        not isinstance(bundle, H3FunControlBundle)
        or bundle.report.get("profile") != PROFILE
    ):
        raise ValueError("请连接显式Union2加载器，不能将旧ControlNet误当post_norm配置")
    control = getattr(bundle.control, "model", None)
    if (
        getattr(control, "injection_layers", None) != LAYERS
        or getattr(control, "inpaint_post_norm", None) is not True
        or getattr(control, "control_in_dim", None) != 49
    ):
        raise ValueError("实际控制模块布局/post_norm/通道与加载回执不符")
    if (
        any(type(v) is not int for v in (width, height, length))
        or width < 32
        or height < 32
        or width % 32
        or height % 32
        or length < 5
        or (length - 5) % 17
    ):
        raise ValueError("Union画布需32对齐，帧数需17n+5，不自动pad或改时钟")
    source = _frames(source_video, "source_video", length, height, width)
    hint = (
        _frames(control_video, "control_video", length, height, width)
        if control_video is not None
        else None
    )
    if type(broadcast_single_mask) is not bool:
        raise ValueError("单帧遮罩广播须显式布尔选择")
    if (
        type(regen_mask) is not torch.Tensor
        or regen_mask.ndim != 3
        or tuple(regen_mask.shape[1:]) != (height, width)
        or not regen_mask.is_floating_point()
        or not torch.isfinite(regen_mask).all()
        or regen_mask.min() < 0
        or regen_mask.max() > 1
    ):
        raise ValueError("regen_mask必须为同画布的有限0–1 MASK；白=重绘，黑=保留")
    mask = regen_mask
    if mask.shape[0] == 1 and broadcast_single_mask:
        mask = mask.expand(length, -1, -1)
    if mask.shape[0] != length:
        raise ValueError("遮罩必须逐帧对齐；静态1帧MASK需明确开启广播")
    base = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(base, "blocks", None)
    if blocks is not None and len(blocks) <= max(LAYERS):
        raise ValueError("实际底模层数不足以注入Union2的0–45层")
    adaln = _assert_compatible_adaln_pair(model, bundle)
    modules = _official_model_patch_fun_control_modules()
    if modules is None:
        raise RuntimeError("当前运行Core缺少Fun patch；不复制控制算法或静默fallback")
    sampling = model.get_model_object("model_sampling")
    from .h3_fun_union2_runtime import AlignedUnion2Patch
    from comfy_extras.nodes_minimax_h3 import video_latent_t

    patch = AlignedUnion2Patch(
        bundle.control,
        vae,
        hint.movedim(-1, 1) if hint is not None else None,
        mask,
        source.movedim(-1, 1),
        float(strength),
        float(sampling.percent_to_sigma(start_percent)),
        float(sampling.percent_to_sigma(end_percent)),
        expected_shape=(1, 24, video_latent_t(length), height // 16, width // 16),
    )
    patched = model.clone()
    patch.register(patched)
    patched.set_attachments(
        ATTACHMENT_KEY,
        {
            "profile": PROFILE,
            "filename": bundle.filename,
            "width": width,
            "height": height,
            "length": length,
            "strength": float(strength),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "control_present": hint is not None,
            "broadcast_single_mask": broadcast_single_mask,
        },
    )
    report = {
        "schema": "t8.fun_union2.apply.v1",
        "status": "registered_native_core",
        "profile": PROFILE,
        "injection_layers": LAYERS,
        "source_shape": list(source.shape),
        "mask_shape": list(mask.shape),
        "control_present": hint is not None,
        "channel_order": ["control24", "keep1", "masked_source24"],
        "mask_policy": "white_regenerate_Core_computes_keep_1_minus_regen_threshold_0.5",
        "hole_policy": "Core.IMAGENET_MEAN_post_norm_not_black_or_fixed_0.5",
        "audio_skip": "Core_zero_only_not_a_joint_audio_identity_guarantee",
        "adaln_pair": adaln,
        "sampling_math_changed": False,
        "automatic_resize_or_temporal_padding": False,
        "portable_cache_certified": False,
        "boundary": "Index/shape alignment, not semantic/VFR clock provenance. Original-audio delivery is separate; masking does not guarantee exact outside RGB.",
    }
    return patched, positive, json.dumps(report, ensure_ascii=False, sort_keys=True)
