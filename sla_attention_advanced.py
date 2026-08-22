from __future__ import annotations

import hashlib
import inspect
import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path

import torch

import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.minimax.model import (
    FRAME_PER_TOKEN,
    Attention,
    MiniMaxH3Model,
    PackedLayout,
    patchify_video,
)
from comfy.ldm.modules import attention as attention_module
from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel

from .sampling import native_flow_sigmas


# LightX2V dynamic sparse attention parity adapter for the released MiniMax H3
# Turbo-SLA LoRA. This is intentionally named after the executable LightX2V
# path, not the broader paper SLA family: the pinned H3 release uses a learned
# block router plus Sage2 block-sparse attention and does not expose a separate
# sparse+linear projection branch in ComfyUI.
SLA_RUNTIME_TYPE = "H3_T8_LIGHTX2V_SLA_RUNTIME"
SLA_PATCH_VERSION = 1
SLA_RUNTIME_KEY = "t8_h3_lightx2v_sla_runtime_v1"
SLA_WRAPPER_KEY = "t8_h3_lightx2v_sla_v1"
SLA_MODES = (
    "apply_lightx2v_sla",
    "dense_lora_control",
    "disabled_identity",
)
SLA_BASE_POLICIES = ("auto_detect_exp", "official_bf16_only")
SLA_EXTERNAL_ATTENTION_POLICIES = ("reject", "compose_kj_sage")
SLA_LORA_FILENAME = (
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)
SLA_LORA_SHA256 = "5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac"
SLA_LORA_BYTES = 1_956_192_992
SLA_LORA_TENSORS = 624
SLA_LORA_PATCHES = 208
SLA_SPARSITY_RATIO = 0.85
SLA_KEEP_RATIO = 1.0 - SLA_SPARSITY_RATIO
SLA_Q_BLOCK = 128
SLA_K_BLOCK = 64
SLA_HEADS = 56
SLA_HEAD_DIM = 128
SLA_EXPECTED_BLOCKS = 50
SLA_EXPECTED_NFE = 4
SLA_SHIFT_VIDEO = 6.0
SLA_SHIFT_AUDIO = 3.0
SLA_LIGHTX2V_REVISION = "f8aee98b5462cca8d7288888146ebd95592bf266"
SLA_MODEL_REVISION = "10ade67cd15ff7a135fa35c2a0673ea96c839247"

ATTENTION_FORWARD_SHA256S = {
    "4e8888f72ea5ccf68fb5ce5b1178ab0ddea66ca61137fcf01df2308ef27bf0be",
}
PACKED_LAYOUT_SHA256S = {
    "1124904e8835c6db068e61e304490d93784e6a8da6ca6b38afd93975611b3af4",
}
MODEL_FORWARD_SHA256S = {
    "14bdfccd6860f252005b8d43ab446aa9a938a13dc819061724b8f914218f5fd1",
}
PATCHIFY_VIDEO_SHA256S = {
    "b53a83b308cd69152a27f79a9e36f296f74f9f9a8ba8889319f8d32609cda645",
}

_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_LOCK = threading.Lock()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _source_sha256(function) -> str:
    function = getattr(function, "__func__", function)
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE.clear()
        _HASH_CACHE[key] = value
    return value


def _validate_sigmas(sigmas: torch.Tensor) -> dict:
    values = torch.as_tensor(sigmas).detach().float().cpu().flatten()
    if values.numel() != SLA_EXPECTED_NFE + 1:
        raise ValueError(
            "H3 LightX2V SLA requires exactly 4 NFE / 5 sigma entries including zero"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("H3 LightX2V SLA received NaN/Inf sigmas")
    if not bool((values[:-1] > 0).all()) or abs(float(values[-1])) > 1.0e-7:
        raise ValueError("H3 LightX2V SLA requires four positive sigmas then zero")
    if not bool((values[:-1] >= values[1:]).all()):
        raise ValueError("H3 LightX2V SLA requires a monotonic sigma schedule")
    expected = native_flow_sigmas(SLA_EXPECTED_NFE, SLA_SHIFT_VIDEO).float()
    if not bool(torch.allclose(values, expected, rtol=0.0, atol=1.0e-6)):
        raise ValueError(
            "H3 LightX2V SLA requires T8 native_flow with 4 steps and video shift 6.0"
        )
    return {
        "nfe": SLA_EXPECTED_NFE,
        "entries": int(values.numel()),
        "scheduler": "native_flow",
        "shift_video": SLA_SHIFT_VIDEO,
        "sigma_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


def _validate_lora_header(path: str | Path) -> dict:
    from safetensors import safe_open

    resolved = Path(path).resolve()
    stat = resolved.stat()
    if int(stat.st_size) != SLA_LORA_BYTES:
        raise RuntimeError(
            "H3 SLA LoRA size does not match the pinned LightX2V ComfyUI artifact: "
            f"observed={stat.st_size}, expected={SLA_LORA_BYTES}"
        )
    digest = _file_sha256(resolved)
    if digest != SLA_LORA_SHA256:
        raise RuntimeError(
            "H3 SLA LoRA SHA-256 does not match the pinned LightX2V artifact: "
            f"observed={digest}"
        )
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})
    if len(keys) != SLA_LORA_TENSORS:
        raise RuntimeError(
            f"H3 SLA LoRA expected {SLA_LORA_TENSORS} tensors, observed {len(keys)}"
        )
    suffix_counts = {
        suffix: sum(key.endswith(suffix) for key in keys)
        for suffix in (".alpha", ".lora_A.weight", ".lora_B.weight")
    }
    if suffix_counts != {
        ".alpha": SLA_LORA_PATCHES,
        ".lora_A.weight": SLA_LORA_PATCHES,
        ".lora_B.weight": SLA_LORA_PATCHES,
    }:
        raise RuntimeError(f"H3 SLA LoRA tensor structure changed: {suffix_counts}")
    expected_metadata = {
        "base_model": "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors",
        "training_rank": "128",
        "training_alpha": "128.0",
        "training_scale": "1.0",
        "target_format": "ComfyUI generic LoRA",
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"H3 SLA LoRA metadata changed: {mismatches}")
    return {
        "filename": resolved.name,
        "bytes": int(stat.st_size),
        "sha256": digest,
        "tensor_count": len(keys),
        "patch_count": SLA_LORA_PATCHES,
        "metadata": metadata,
        "model_revision": SLA_MODEL_REVISION,
    }


def _model_dtype_contract(model, base_policy: str) -> dict:
    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    try:
        weight = diffusion.blocks[0].attn.qkv_proj.weight
        dtype = str(weight.dtype)
        module = diffusion.blocks[0].attn.qkv_proj
        module_type = type(module).__name__
        quant_format = getattr(module, "quant_format", None)
        weight_type = type(weight).__name__
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("H3 SLA could not inspect the native H3 base weights") from exc
    quantized = quant_format is not None or weight_type == "QuantizedTensor"
    official_bf16 = dtype == "torch.bfloat16" and not quantized
    if base_policy == "official_bf16_only" and not official_bf16:
        raise RuntimeError(
            "official_bf16_only requires the upstream BF16 FL2VA base; "
            f"observed {dtype} in {module_type}"
        )
    return {
        "policy": base_policy,
        "observed_qkv_dtype": dtype,
        "observed_qkv_module": module_type,
        "observed_weight_type": weight_type,
        "observed_quant_format": (
            None if quant_format is None else str(quant_format)
        ),
        "model_storage_bytes": int(model.model_size()),
        "quantized_base_observed": quantized,
        "official_bf16_base_observed": official_bf16,
        "compatibility_status": (
            "official_base_dtype" if official_bf16 else "quantized_base_experimental"
        ),
    }


def _kj_sage_patch_key(index: int) -> str:
    return f"diffusion_model.blocks.{int(index)}.attn.forward"


def _inspect_kj_sage_contract(model) -> dict:
    """Authenticate the exact MiniMax H3 KJ Sage object-patch surface.

    KJNodes replaces each H3 attention module's whole ``forward`` method.  SLA
    can compose with that patch only when all 50 native blocks are covered by
    the same known implementation and every bound method targets the matching
    native attention module.  Unknown or partial patch sets stay fail-closed.
    """
    patches = dict(getattr(model, "object_patches", {}) or {})
    observed_keys = sorted(key for key in patches if key != "model_sampling")
    expected_keys = [_kj_sage_patch_key(index) for index in range(SLA_EXPECTED_BLOCKS)]
    if set(observed_keys) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(observed_keys))
        extra = sorted(set(observed_keys) - set(expected_keys))
        raise RuntimeError(
            "H3 SLA + KJ Sage Composer requires exactly 50 MiniMax H3 Sage "
            f"forward patches; missing={missing[:4]}, extra={extra[:4]}"
        )

    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = list(getattr(diffusion, "blocks", ()) or ())
    if len(blocks) != SLA_EXPECTED_BLOCKS:
        raise RuntimeError(
            "H3 SLA + KJ Sage Composer requires the native 50-block H3 model; "
            f"observed {len(blocks)} blocks"
        )

    source_hashes: dict[str, str] = {}
    modules: set[str] = set()
    for index, block in enumerate(blocks):
        key = _kj_sage_patch_key(index)
        patch = patches[key]
        function = getattr(patch, "__func__", patch)
        module = getattr(block, "attn", None)
        if getattr(patch, "__self__", None) is not module:
            raise RuntimeError(f"H3 SLA + KJ Sage patch is bound to the wrong module: {key}")
        code = getattr(function, "__code__", None)
        names = set(getattr(code, "co_names", ()) or ())
        constants = {
            value for value in (getattr(code, "co_consts", ()) or ()) if isinstance(value, str)
        }
        if (
            getattr(function, "__name__", "") != "minimax_sageattn_forward"
            or not {"_sageattn_int8_fp8_nhd", "qkv_proj", "out_proj"}.issubset(names)
            or "minimax_head_chunks" not in constants
        ):
            raise RuntimeError(
                "H3 SLA + KJ Sage Composer found an unrecognized attention forward: "
                f"{key} -> {getattr(function, '__module__', '?')}."
                f"{getattr(function, '__name__', type(function).__name__)}"
            )
        try:
            source_hash = _source_sha256(function)
        except (OSError, TypeError):
            source_hash = hashlib.sha256(code.co_code).hexdigest()
        source_hashes[key] = source_hash
        modules.add(str(getattr(function, "__module__", "unknown")))

    unique_hashes = sorted(set(source_hashes.values()))
    if len(unique_hashes) != 1:
        raise RuntimeError(
            "H3 SLA + KJ Sage Composer requires one consistent Sage forward "
            f"implementation, observed {len(unique_hashes)} source hashes"
        )
    return {
        "backend": "KJNodes MiniMax H3 memory-efficient SageAttention",
        "patch_count": len(source_hashes),
        "patch_keys_sha256": hashlib.sha256(
            "\n".join(expected_keys).encode("utf-8")
        ).hexdigest(),
        "source_sha256": unique_hashes[0],
        "modules": sorted(modules),
        "dispatch_contract": (
            "SLA-routed apply calls use stock H3 forward plus block-sparse Sage2; "
            "dense control and calls outside the SLA route retain KJ Sage"
        ),
    }


def _assert_core_contract(
    model,
    *,
    base_policy: str,
    external_attention_policy: str = "reject",
) -> dict:
    if base_policy not in SLA_BASE_POLICIES:
        raise ValueError(f"Unknown H3 SLA base policy {base_policy!r}")
    if external_attention_policy not in SLA_EXTERNAL_ATTENTION_POLICIES:
        raise ValueError(
            f"Unknown H3 SLA external attention policy {external_attention_policy!r}"
        )
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise ValueError("H3 SLA requires a ComfyUI MODEL patcher")
    base = getattr(model, "model", None)
    if not isinstance(base, MiniMaxH3BaseModel):
        if type(getattr(base, "diffusion_model", None)).__name__ != "MiniMaxH3Model":
            raise ValueError("H3 SLA currently requires the native ComfyUI MiniMax H3 model")

    hashes = {
        "attention_forward": _source_sha256(Attention.forward),
        "packed_layout": _source_sha256(PackedLayout.__init__),
        "model_forward": _source_sha256(MiniMaxH3Model._forward),
        "patchify_video": _source_sha256(patchify_video),
    }
    expected = {
        "attention_forward": ATTENTION_FORWARD_SHA256S,
        "packed_layout": PACKED_LAYOUT_SHA256S,
        "model_forward": MODEL_FORWARD_SHA256S,
        "patchify_video": PATCHIFY_VIDEO_SHA256S,
    }
    mismatches = [key for key, value in hashes.items() if value not in expected[key]]
    if mismatches:
        raise RuntimeError(
            "H3 SLA has not validated this ComfyUI H3 core contract: "
            + ", ".join(f"{key}={hashes[key]}" for key in mismatches)
        )

    options = getattr(model, "model_options", {})
    conflict_keys = (
        "sampler_cfg_function",
        "sampler_pre_cfg_function",
        "sampler_post_cfg_function",
        "sampler_calc_cond_batch_function",
        "model_function_wrapper",
    )
    conflicts = [key for key in conflict_keys if bool(options.get(key))]
    if conflicts:
        raise RuntimeError("H3 SLA refuses guidance/model hooks: " + ", ".join(conflicts))
    transformer = options.get("transformer_options", {})
    if "optimized_attention_override" in transformer:
        raise RuntimeError(
            "H3 SLA owns attention and cannot stack with SageAttention, Sol-Attn, "
            "FETA, Prompt Relay, or another attention override"
        )
    replacements = transformer.get("patches_replace", {})
    if isinstance(replacements, Mapping) and any(bool(value) for value in replacements.values()):
        raise RuntimeError("H3 SLA cannot stack with BlockCache/STG/block replacements yet")
    wrappers = getattr(model, "wrappers", {})
    diffusion_wrappers = wrappers.get(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}
    )
    if any(bool(value) for value in diffusion_wrappers.values()):
        raise RuntimeError("H3 SLA cannot stack with an existing diffusion wrapper")
    if bool(getattr(model, "patches", {})):
        raise RuntimeError(
            "H3 SLA must load its authenticated LoRA itself; remove external LoRA loaders"
        )
    if any(bool(value) for value in getattr(model, "injections", {}).values()):
        raise RuntimeError("H3 SLA rejects bypass-LoRA and other model injections")

    shifts = {
        "video": float(transformer.get("minimax_h3_sigma_shift_video", float("nan"))),
        "audio": float(transformer.get("minimax_h3_sigma_shift_audio", float("nan"))),
    }
    if not math.isclose(shifts["video"], SLA_SHIFT_VIDEO, abs_tol=1.0e-7):
        raise RuntimeError("H3 SLA requires MiniMax H3 Dual-Clock video shift 6.0")
    if not math.isclose(shifts["audio"], SLA_SHIFT_AUDIO, abs_tol=1.0e-7):
        raise RuntimeError("H3 SLA requires MiniMax H3 Dual-Clock audio shift 3.0")

    if external_attention_policy == "compose_kj_sage":
        external_attention = _inspect_kj_sage_contract(model)
    else:
        object_conflicts = sorted(
            key
            for key in getattr(model, "object_patches", {})
            if key != "model_sampling"
        )
        if object_conflicts:
            raise RuntimeError(
                "H3 SLA refuses existing model object patches: "
                + ", ".join(object_conflicts)
            )
        external_attention = None
    return {
        "core_hashes": hashes,
        "base": _model_dtype_contract(model, base_policy),
        "dual_clock": shifts,
        "external_attention_policy": external_attention_policy,
        "external_attention": external_attention,
    }


def _apply_authenticated_lora(model, path: str | Path) -> tuple[object, dict]:
    contract = _validate_lora_header(path)
    state, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    state = comfy.lora_convert.convert_lora(state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)
    if len(loaded) != SLA_LORA_PATCHES:
        raise RuntimeError(
            "H3 SLA LoRA did not map completely to this H3 base: "
            f"mapped={len(loaded)}, expected={SLA_LORA_PATCHES}"
        )
    patched = model.clone()
    applied = set(patched.add_patches(loaded, 1.0))
    if applied != set(loaded):
        missing = sorted(set(loaded) - applied)
        raise RuntimeError(
            "H3 SLA LoRA patch application was incomplete: "
            f"applied={len(applied)}, missing={missing[:8]}"
        )
    if metadata and hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_sla_lora_metadata", dict(metadata))
    contract["mapped_patch_count"] = len(loaded)
    contract["applied_patch_count"] = len(applied)
    contract["strength_model"] = 1.0
    return patched, contract


def _sla_route_from_call(args, kwargs) -> tuple[dict | None, object]:
    options = kwargs.get("transformer_options")
    if not isinstance(options, dict):
        options = next(
            (
                value
                for value in args
                if isinstance(value, dict) and SLA_RUNTIME_KEY in value
            ),
            {},
        )
    route = options.get(SLA_RUNTIME_KEY)
    x = args[0] if args else kwargs.get("x")
    tensor = (
        x[0]
        if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0])
        else x
    )
    if not isinstance(route, dict) or not torch.is_tensor(tensor):
        return None, tensor
    tokens = int(tensor.shape[0]) if tensor.ndim == 2 else -1
    if tokens != int(route.get("seq_len", -2)):
        return None, tensor
    return route, tensor


def _compose_kj_sage_forward(module, patched_forward, *, source_sha256: str):
    """Give SLA and a KJ whole-forward patch one deterministic owner.

    The released SLA path already computes selected blocks with Sage2.  For an
    SLA-routed apply call we therefore run the native H3 forward so it reaches
    ``route_sla_attention`` exactly once.  Dense-control and non-SLA calls keep
    the original KJ Sage forward.  No branch evaluates both attention kernels.
    """
    stock_forward = type(module).forward

    def forward(*args, **kwargs):
        route, tensor = _sla_route_from_call(args, kwargs)
        if route is None:
            return patched_forward(*args, **kwargs)
        if route["mode"] == "apply_lightx2v_sla":
            x = args[0] if args else kwargs.get("x")
            if tensor is not x and isinstance(x, list):
                x.clear()
                if args:
                    args = (tensor,) + args[1:]
                else:
                    kwargs = dict(kwargs)
                    kwargs["x"] = tensor
            return stock_forward(module, *args, **kwargs)
        if route["mode"] != "dense_lora_control":
            return patched_forward(*args, **kwargs)
        output = patched_forward(*args, **kwargs)
        runtime: SLARuntime = route["runtime"]
        runtime.record_attention(
            int(route["forward_index"]),
            sparse=False,
            external_backend="kj_sage",
        )
        return output

    forward._t8_h3_sla_kj_sage_composed = True
    forward._t8_h3_sla_kj_sage_source_sha256 = str(source_sha256)
    return forward


def _compose_kj_sage_object_patches(model, contract: Mapping) -> None:
    diffusion = model.model.diffusion_model
    source_sha256 = str(contract["source_sha256"])
    for index, block in enumerate(diffusion.blocks):
        key = _kj_sage_patch_key(index)
        patched_forward = model.object_patches[key]
        model.add_object_patch(
            key,
            _compose_kj_sage_forward(
                block.attn,
                patched_forward,
                source_sha256=source_sha256,
            ),
        )


def _verify_kj_sage_runtime(model, contract: Mapping) -> None:
    expected_hash = str(contract["source_sha256"])
    for index, block in enumerate(model.model.diffusion_model.blocks):
        installed = block.attn.__dict__.get("forward")
        if not getattr(installed, "_t8_h3_sla_kj_sage_composed", False):
            raise RuntimeError(
                "H3 SLA + KJ Sage composed forward was replaced after binding: "
                + _kj_sage_patch_key(index)
            )
        if getattr(installed, "_t8_h3_sla_kj_sage_source_sha256", None) != expected_hash:
            raise RuntimeError(
                "H3 SLA + KJ Sage source fingerprint changed after binding: "
                + _kj_sage_patch_key(index)
            )


def mean_pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """LightX2V router pooling, including exact non-padded tail divisors."""
    if x.ndim != 4:
        raise ValueError("H3 SLA router requires a [B,H,L,D] tensor")
    length = int(x.shape[-2])
    if length <= 0 or int(block_size) <= 0:
        raise ValueError("H3 SLA router requires positive sequence and block sizes")
    blocks = math.ceil(length / int(block_size))
    padded_length = blocks * int(block_size)
    if padded_length != length:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_length - length))
    pooled = x.reshape(*x.shape[:-2], blocks, int(block_size), x.shape[-1])
    pooled = pooled.float().sum(dim=-2)
    counts = torch.full(
        (blocks,), float(block_size), device=x.device, dtype=torch.float32
    )
    counts[-1] = float(length - (blocks - 1) * int(block_size))
    return (pooled / counts.view(1, 1, blocks, 1)).to(x.dtype)


def lightx2v_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    keep_ratio: float = SLA_KEEP_RATIO,
    q_block: int = SLA_Q_BLOCK,
    k_block: int = SLA_K_BLOCK,
) -> tuple[torch.Tensor, int]:
    """Reproduce LightX2V get_block_map for its H3 Sage2 configuration."""
    if q.shape[:2] != k.shape[:2] or q.shape[-2:] != k.shape[-2:]:
        raise ValueError("H3 SLA currently requires self-attention with matching Q/K")
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("H3 SLA keep_ratio must be in (0,1]")
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)
    pooled_q = mean_pool_blocks(q.contiguous(), int(q_block))
    pooled_k = mean_pool_blocks(arg_k.contiguous(), int(k_block))
    scores = pooled_q @ pooled_k.transpose(-1, -2)
    key_blocks = int(scores.shape[-1])
    topk = max(1, min(key_blocks, int(float(keep_ratio) * key_blocks)))
    indices = torch.topk(scores, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(scores, dtype=torch.int8)
    sparse_map.scatter_(-1, indices, 1)
    return sparse_map, topk


def _router_workspace_bytes(q: torch.Tensor) -> int:
    batch, heads, seq_len, dim = (int(value) for value in q.shape)
    q_blocks = math.ceil(seq_len / SLA_Q_BLOCK)
    k_blocks = math.ceil(seq_len / SLA_K_BLOCK)
    pooled = batch * heads * (q_blocks + k_blocks) * dim * q.element_size()
    scores = batch * heads * q_blocks * k_blocks * q.element_size()
    mask = batch * heads * q_blocks * k_blocks
    indices = batch * heads * q_blocks * max(1, int(SLA_KEEP_RATIO * k_blocks)) * 8
    return pooled + scores + mask + indices


def _pixel_frame_count(latent_frames: int) -> int:
    return sum(
        FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
        for index in range(int(latent_frames))
    )


def _runtime_route(*, x, context, payload, denoise_mask, audio_denoise_mask) -> dict:
    if denoise_mask is not None or audio_denoise_mask is not None:
        raise RuntimeError("H3 SLA currently rejects video/audio denoise masks")
    if not isinstance(payload, Mapping):
        raise RuntimeError("H3 SLA could not find native minimax_payload")
    refs = list(payload.get("refs") or ())
    if refs:
        raise RuntimeError("The released H3 Turbo-SLA checkpoint supports FL2VA only")
    keyframes = list(payload.get("keyframes") or ())
    if len(keyframes) != 2:
        raise RuntimeError("H3 Turbo-SLA requires exactly first and last visual frames")
    video, audio = x
    text_len = int(context.shape[1])
    latent_t, latent_h, latent_w = (int(value) for value in video.shape[2:])
    audio_t = int(audio.shape[-1])
    layout = payload.get("layout")
    if not isinstance(layout, PackedLayout) or layout.signature != (
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
    ):
        layout = PackedLayout(
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=keyframes,
            refs=None,
        )
    final_frame = _pixel_frame_count(latent_t) - 1
    positions = []
    for keyframe in keyframes:
        if not isinstance(keyframe, Mapping):
            raise RuntimeError("H3 SLA keyframe payload is invalid")
        if keyframe.get("latent") is None or keyframe.get("audio_latent") is not None:
            raise RuntimeError("H3 SLA accepts visual first/last frames only")
        positions.append(int(keyframe.get("resolved_frame_index", -1)))
    if positions != [0, final_frame]:
        raise RuntimeError(
            f"H3 SLA requires FL2VA positions [0,{final_frame}], observed {positions}"
        )
    kinds = [kind for _start, _end, kind in layout.segments]
    if kinds[-2:] != ["audio", "video"] or any(kind.startswith("ref_") for kind in kinds):
        raise RuntimeError(f"H3 SLA packed layout changed: {kinds}")
    return {
        "task": "FL2VA",
        "seq_len": int(layout.seq_len),
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "pixel_frames": final_frame + 1,
        "text_len": text_len,
        "layout_segments": [list(segment) for segment in layout.segments],
    }


class SLARuntime:
    def __init__(self, config: Mapping):
        self.config = dict(config)
        self._lock = threading.RLock()
        self._forwards: list[dict] = []
        self._aborted: str | None = None
        self._consumed = False

    def begin_forward(self, route: Mapping) -> int:
        with self._lock:
            if self._consumed:
                raise RuntimeError("H3 SLA runtime token was already consumed")
            index = len(self._forwards)
            self._forwards.append(
                {
                    "index": index,
                    "task": route["task"],
                    "seq_len": int(route["seq_len"]),
                    "pixel_frames": int(route["pixel_frames"]),
                    "latent_shape": [
                        int(route["latent_t"]),
                        int(route["latent_h"]),
                        int(route["latent_w"]),
                    ],
                    "main_attention_calls": 0,
                    "sparse_kernel_calls": 0,
                    "dense_control_calls": 0,
                    "external_sage_calls": 0,
                    "kernel_failures": [],
                    "router_workspace_peak_bytes": 0,
                    "key_blocks_min": None,
                    "key_blocks_max": None,
                    "retained_key_blocks_min": None,
                    "retained_key_blocks_max": None,
                    "retained_ratio_min": None,
                    "retained_ratio_max": None,
                    "retained_ratio_sum": 0.0,
                    "retained_ratio_count": 0,
                }
            )
            return index

    def record_attention(
        self,
        forward_index: int,
        *,
        sparse: bool,
        workspace_bytes: int = 0,
        key_blocks: int = 0,
        retained_key_blocks: int = 0,
        external_backend: str | None = None,
    ) -> None:
        with self._lock:
            forward = self._forwards[int(forward_index)]
            forward["main_attention_calls"] += 1
            if sparse:
                forward["sparse_kernel_calls"] += 1
                forward["router_workspace_peak_bytes"] = max(
                    int(forward["router_workspace_peak_bytes"]), int(workspace_bytes)
                )
                ratio = float(retained_key_blocks) / float(key_blocks)
                for prefix, value in (
                    ("key_blocks", int(key_blocks)),
                    ("retained_key_blocks", int(retained_key_blocks)),
                    ("retained_ratio", ratio),
                ):
                    minimum = f"{prefix}_min"
                    maximum = f"{prefix}_max"
                    forward[minimum] = (
                        value if forward[minimum] is None else min(forward[minimum], value)
                    )
                    forward[maximum] = (
                        value if forward[maximum] is None else max(forward[maximum], value)
                    )
                forward["retained_ratio_sum"] += ratio
                forward["retained_ratio_count"] += 1
            else:
                forward["dense_control_calls"] += 1
                if external_backend == "kj_sage":
                    forward["external_sage_calls"] += 1

    def record_failure(self, forward_index: int, exc: BaseException) -> None:
        with self._lock:
            self._forwards[int(forward_index)]["kernel_failures"].append(
                f"{type(exc).__name__}: {exc}"
            )

    def abort(self, exc: BaseException) -> None:
        with self._lock:
            self._aborted = f"{type(exc).__name__}: {exc}"

    def snapshot(self, *, consume: bool) -> dict:
        with self._lock:
            if self._consumed:
                raise RuntimeError("H3 SLA runtime token was already consumed")
            forwards = [dict(value) for value in self._forwards]
            ratio_count = sum(
                int(forward["retained_ratio_count"]) for forward in forwards
            )
            ratio_sum = sum(float(forward["retained_ratio_sum"]) for forward in forwards)
            ratio_mins = [
                float(forward["retained_ratio_min"])
                for forward in forwards
                if forward["retained_ratio_min"] is not None
            ]
            ratio_maxs = [
                float(forward["retained_ratio_max"])
                for forward in forwards
                if forward["retained_ratio_max"] is not None
            ]
            report = {
                "config": dict(self.config),
                "aborted": self._aborted,
                "model_forward_count": len(forwards),
                "main_attention_calls_per_forward": [
                    value["main_attention_calls"] for value in forwards
                ],
                "sparse_kernel_calls_per_forward": [
                    value["sparse_kernel_calls"] for value in forwards
                ],
                "dense_control_calls_per_forward": [
                    value["dense_control_calls"] for value in forwards
                ],
                "external_sage_calls_per_forward": [
                    value["external_sage_calls"] for value in forwards
                ],
                "kernel_failure_count": sum(
                    len(value["kernel_failures"]) for value in forwards
                ),
                "retained_ratio_min": min(ratio_mins) if ratio_mins else None,
                "retained_ratio_mean": ratio_sum / ratio_count if ratio_count else None,
                "retained_ratio_max": max(ratio_maxs) if ratio_maxs else None,
                "router_workspace_peak_mib": max(
                    (forward["router_workspace_peak_bytes"] for forward in forwards),
                    default=0,
                )
                / (1024 * 1024),
                "forwards": forwards,
            }
            if consume:
                self._consumed = True
            return report


def _dense_delegate(q, k, v, heads, *, transformer_options, kwargs):
    delegate_kwargs = dict(kwargs)
    delegate_kwargs["_inside_attn_wrapper"] = True
    return attention_module.optimized_attention(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=True,
        skip_output_reshape=False,
        transformer_options=transformer_options,
        **delegate_kwargs,
    )


def route_sla_attention(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    transformer_options=None,
    **kwargs,
):
    transformer_options = transformer_options or {}
    route = transformer_options.get(SLA_RUNTIME_KEY)
    if route is None or int(q.shape[-2]) != int(route["seq_len"]):
        return _dense_delegate(
            q, k, v, heads, transformer_options=transformer_options, kwargs=kwargs
        )
    if mask is not None or attn_precision is not None:
        raise RuntimeError("H3 SLA requires the native unmasked H3 attention call")
    if not skip_reshape or skip_output_reshape:
        raise RuntimeError("H3 SLA received an unsupported attention layout")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("H3 SLA requires matching [B,H,S,D] Q/K/V tensors")
    if q.shape[0] != 1 or int(heads) != SLA_HEADS or q.shape[1] != SLA_HEADS:
        raise RuntimeError("H3 SLA requires batch-1 native 56-head H3 attention")
    if q.shape[-1] != SLA_HEAD_DIM:
        raise RuntimeError("H3 SLA requires the native H3 head dimension 128")
    runtime: SLARuntime = route["runtime"]
    forward_index = int(route["forward_index"])
    if route["mode"] == "dense_lora_control":
        output = _dense_delegate(
            q, k, v, heads, transformer_options=transformer_options, kwargs=kwargs
        )
        runtime.record_attention(forward_index, sparse=False)
        return output

    workspace = _router_workspace_bytes(q)
    if workspace > int(route["max_router_workspace_mib"]) * 1024 * 1024:
        raise RuntimeError(
            "H3 SLA router workspace estimate exceeds the configured limit: "
            f"{workspace / (1024 * 1024):.1f} MiB > "
            f"{route['max_router_workspace_mib']} MiB"
        )
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        sparse_map, topk = lightx2v_block_map(q, k)
        key_blocks = int(sparse_map.shape[-1])
        output = block_sparse_sage2_attn_cuda(
            q,
            k,
            v,
            mask_id=sparse_map,
            dropout_p=0.0,
            scale=None,
            smooth_k=True,
            pvthreshd=1.0e6,
            attention_sink=False,
            tensor_layout="HND",
            output_dtype=q.dtype,
            return_sparsity=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        if output.shape != q.shape:
            raise RuntimeError(
                f"H3 SLA kernel returned invalid output {tuple(output.shape)} {output.dtype}"
            )
    except BaseException as exc:
        runtime.record_failure(forward_index, exc)
        raise
    runtime.record_attention(
        forward_index,
        sparse=True,
        workspace_bytes=workspace,
        key_blocks=key_blocks,
        retained_key_blocks=topk,
    )
    batch, _heads, seq_len, head_dim = output.shape
    return output.transpose(1, 2).reshape(batch, seq_len, int(heads) * head_dim)


def _kernel_contract() -> dict:
    try:
        import importlib.metadata

        package_version = importlib.metadata.version("spas-sage-attn")
        from spas_sage_attn import block_sparse_sage2_attn_cuda
    except Exception as exc:
        raise RuntimeError(
            "H3 SLA requires the optional spas-sage-attn CUDA package. "
            "Install the wheel matching this ComfyUI Torch/CUDA runtime."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("H3 SLA requires an NVIDIA CUDA GPU")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (8, 9):
        raise RuntimeError(
            "This first audited H3 SLA backend is pinned to Ada sm89 / RTX 40-series; "
            f"observed sm{capability[0]}{capability[1]}"
        )
    return {
        "package": "spas-sage-attn",
        "version": package_version,
        "callable": block_sparse_sage2_attn_cuda.__name__,
        "cuda_capability": f"sm{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "operator": "sage2_block_sparse",
        "q_block": SLA_Q_BLOCK,
        "k_block": SLA_K_BLOCK,
        "smooth_k": True,
        "pv_threshold": 1.0e6,
    }


def build_sla_model(
    model,
    sigmas: torch.Tensor,
    *,
    lora_path: str,
    mode: str,
    base_policy: str,
    max_router_workspace_mib: int,
    external_attention_policy: str = "reject",
):
    mode = str(mode)
    base_policy = str(base_policy)
    if mode not in SLA_MODES:
        raise ValueError(f"Unknown H3 SLA mode {mode!r}")
    if base_policy not in SLA_BASE_POLICIES:
        raise ValueError(f"Unknown H3 SLA base policy {base_policy!r}")
    external_attention_policy = str(external_attention_policy)
    if external_attention_policy not in SLA_EXTERNAL_ATTENTION_POLICIES:
        raise ValueError(
            f"Unknown H3 SLA external attention policy {external_attention_policy!r}"
        )
    if not 32 <= int(max_router_workspace_mib) <= 2048:
        raise ValueError("H3 SLA max_router_workspace_mib must be 32..2048")

    config = {
        "schema": SLA_PATCH_VERSION,
        "mode": mode,
        "implementation": (
            "LightX2V dynamic_sparse_attn routing-math and Sage2 block-kernel parity adapter"
        ),
        "lightx2v_revision": SLA_LIGHTX2V_REVISION,
        "model_revision": SLA_MODEL_REVISION,
        "scope": "native MiniMax H3 FL2VA main 50-block packed attention",
        "sparsity_ratio_requested": SLA_SPARSITY_RATIO,
        "keep_ratio_requested": SLA_KEEP_RATIO,
        "q_block": SLA_Q_BLOCK,
        "k_block": SLA_K_BLOCK,
        "expected_main_blocks": SLA_EXPECTED_BLOCKS,
        "base_policy": base_policy,
        "external_attention_policy": external_attention_policy,
        "max_router_workspace_mib": int(max_router_workspace_mib),
        "scientific_boundary": (
            "This node reproduces the released LightX2V H3 dynamic block-routing math "
            "and Sage2 block-sparse execution path inside ComfyUI. It does not reproduce "
            "LightX2V's entire inference runtime and is not a claim that ComfyUI implements "
            "every sparse+linear branch described by the general SLA paper."
        ),
    }
    runtime = SLARuntime(config)
    if mode == "disabled_identity":
        config.update(
            {
                "sigma_contract": None,
                "core_contract": None,
                "lora_contract": None,
                "kernel_contract": None,
                "status": "disabled_identity",
            }
        )
        runtime.config = dict(config)
        return model, runtime, _json(config)

    config["sigma_contract"] = _validate_sigmas(sigmas)
    config["core_contract"] = _assert_core_contract(
        model,
        base_policy=base_policy,
        external_attention_policy=external_attention_policy,
    )
    if mode == "apply_lightx2v_sla":
        config["kernel_contract"] = _kernel_contract()
    else:
        config["kernel_contract"] = {
            "operator": "ComfyUI dense optimized attention control",
            "sparse_kernel_loaded": False,
        }
    patched, lora_contract = _apply_authenticated_lora(model, lora_path)
    config["lora_contract"] = lora_contract
    external_attention_contract = config["core_contract"]["external_attention"]
    if external_attention_policy == "compose_kj_sage":
        _compose_kj_sage_object_patches(patched, external_attention_contract)

    def _diffusion_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError("H3 SLA detected another diffusion wrapper after binding")
        installed = transformer_options.get("optimized_attention_override")
        if getattr(installed, "_t8_h3_sla_patch_version", None) != SLA_PATCH_VERSION:
            raise RuntimeError("H3 SLA attention override was replaced after binding")
        if external_attention_policy == "compose_kj_sage":
            _verify_kj_sage_runtime(patched, external_attention_contract)
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(bool(value) for value in replacements.values()):
            raise RuntimeError("H3 SLA detected a runtime block replacement")
        if SLA_RUNTIME_KEY in transformer_options:
            raise RuntimeError("Nested H3 SLA runtime state was refused")
        try:
            route = _runtime_route(
                x=x,
                context=context,
                payload=kwargs.get("minimax_payload"),
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
            )
            forward_index = runtime.begin_forward(route)
            route.update(
                {
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "mode": mode,
                    "max_router_workspace_mib": int(max_router_workspace_mib),
                }
            )
            transformer_options[SLA_RUNTIME_KEY] = route
            return executor(x, timestep, context, transformer_options, **kwargs)
        except BaseException as exc:
            runtime.abort(exc)
            raise
        finally:
            transformer_options.pop(SLA_RUNTIME_KEY, None)

    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        SLA_WRAPPER_KEY,
        _diffusion_wrapper,
    )
    patched.set_model_optimized_attention(route_sla_attention)
    installed = patched.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    installed._t8_h3_sla_patch_version = SLA_PATCH_VERSION
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(SLA_WRAPPER_KEY, dict(config))
    config["status"] = "ready_for_runtime_audit"
    runtime.config = dict(config)
    return patched, runtime, _json(config)


def finalize_sla_runtime(av_latent, runtime: SLARuntime):
    if not isinstance(runtime, SLARuntime):
        raise TypeError("H3 SLA Audit requires the matching SLA runtime token")
    report = runtime.snapshot(consume=True)
    if report["aborted"]:
        raise RuntimeError("H3 SLA sampling aborted: " + report["aborted"])
    mode = report["config"]["mode"]
    if mode == "disabled_identity":
        if report["model_forward_count"] != 0:
            raise RuntimeError("Disabled H3 SLA unexpectedly observed model execution")
        report["status"] = "disabled_identity_verified"
    else:
        if report["model_forward_count"] != SLA_EXPECTED_NFE:
            raise RuntimeError(
                f"H3 SLA expected {SLA_EXPECTED_NFE} model forwards, observed "
                f"{report['model_forward_count']}"
            )
        expected = [SLA_EXPECTED_BLOCKS] * SLA_EXPECTED_NFE
        if report["main_attention_calls_per_forward"] != expected:
            raise RuntimeError(
                "H3 SLA expected 50 main attention calls per forward, observed "
                f"{report['main_attention_calls_per_forward']}"
            )
        if report["kernel_failure_count"]:
            raise RuntimeError("H3 SLA runtime recorded a sparse-kernel failure")
        external_policy = report["config"].get(
            "external_attention_policy", "reject"
        )
        if mode == "apply_lightx2v_sla":
            if report["sparse_kernel_calls_per_forward"] != expected:
                raise RuntimeError(
                    "H3 SLA sparse calls do not cover all 50 blocks: "
                    f"{report['sparse_kernel_calls_per_forward']}"
                )
            if report["dense_control_calls_per_forward"] != [0] * SLA_EXPECTED_NFE:
                raise RuntimeError("H3 SLA silently fell back to dense main-block attention")
            if report["external_sage_calls_per_forward"] != [0] * SLA_EXPECTED_NFE:
                raise RuntimeError(
                    "H3 SLA apply path was incorrectly bypassed by external KJ Sage"
                )
            report["status"] = "lightx2v_dynamic_sparse_verified"
        else:
            if report["dense_control_calls_per_forward"] != expected:
                raise RuntimeError("H3 SLA dense control did not cover all 50 blocks")
            if report["sparse_kernel_calls_per_forward"] != [0] * SLA_EXPECTED_NFE:
                raise RuntimeError("H3 SLA dense control unexpectedly used sparse kernels")
            if external_policy == "compose_kj_sage":
                if report["external_sage_calls_per_forward"] != expected:
                    raise RuntimeError(
                        "H3 SLA dense control did not use KJ Sage for all 50 blocks"
                    )
                report["status"] = "dense_lora_kj_sage_control_verified"
            else:
                if report["external_sage_calls_per_forward"] != [0] * SLA_EXPECTED_NFE:
                    raise RuntimeError("H3 SLA strict control observed external Sage calls")
                report["status"] = "dense_lora_control_verified"
    report["quality_claim"] = (
        "mechanically audited only; visual quality, speedup and INT8-base parity require "
        "a same-input/same-seed dense-control A/B review"
    )
    return av_latent, _json(report)
