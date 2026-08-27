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
import comfy.weight_adapter
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
SLA_PATCH_VERSION = 2
SLA_RUNTIME_KEY = "t8_h3_lightx2v_sla_runtime_v1"
SLA_WRAPPER_KEY = "t8_h3_lightx2v_sla_v1"
SLA_CONSUMER_TURBO_MODE = "consumer_turbo8_recommended"
SLA_MODES = (
    "apply_lightx2v_sla",
    "dense_lora_control",
    "disabled_identity",
    "apply_lightx2v_sla_upstream_exact_exp",
)
SLA_BASE_POLICIES = ("auto_detect_exp", "official_bf16_only")
SLA_EXTERNAL_ATTENTION_POLICIES = ("reject", "compose_kj_sage")
SLA_LORA_APPLICATION_POLICIES = ("standard_patch", "bypass_model_only")
SLA_LORA_FILENAME = (
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)
SLA_LORA_SHA256 = "5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac"
SLA_LORA_BYTES = 1_956_192_992
SLA_LORA_TENSORS = 624
SLA_LORA_PATCHES = 208
SLA_SPARSITY_RATIO = 0.85
SLA_KEEP_RATIO = 1.0 - SLA_SPARSITY_RATIO
SLA_AUTO_SAFE_MIN_SPARSE_SEQUENCE = 50_000
SLA_AUTO_SAFE_DENSE_EDGE_FORWARDS = 1
SLA_Q_BLOCK = 128
SLA_K_BLOCK = 64
SLA_HEADS = 56
SLA_HEAD_DIM = 128
SLA_EXPECTED_BLOCKS = 50
SLA_EXPECTED_NFE = 4
SLA_MAX_NFE = 64
SLA_SHIFT_VIDEO = 6.0
SLA_SHIFT_AUDIO = 3.0
SLA_FULL_RANGE_START_PERCENT = 0.0
SLA_FULL_RANGE_END_PERCENT = 1.0
SLA_LIGHTX2V_REVISION = "f8aee98b5462cca8d7288888146ebd95592bf266"
SLA_MODEL_REVISION = "10ade67cd15ff7a135fa35c2a0673ea96c839247"

SLA_SPARSE_MODES = {
    "apply_lightx2v_sla",
    "apply_lightx2v_sla_upstream_exact_exp",
}
SLA_EXECUTION_DENSE = "dense"
SLA_EXECUTION_SPARSE = "sparse"


def _validate_sla_percent_window(start_percent: float, end_percent: float) -> dict:
    start = float(start_percent)
    end = float(end_percent)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("H3 SLA start/end percent must be finite")
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(
            "H3 SLA percent window requires 0 <= start_percent < end_percent <= 1"
        )
    return {
        "start_percent": start,
        "end_percent": end,
        "full_range": math.isclose(start, 0.0, abs_tol=1.0e-9)
        and math.isclose(end, 1.0, abs_tol=1.0e-9),
        "boundary_semantics": "inclusive_current_model_forward_sigma",
    }


def _sampling_percent_from_video_sigma(video_sigma: float, shift_video: float) -> float:
    """Invert the native-flow shift and return ComfyUI denoising progress.

    ComfyUI defines percent 0 as the first/high-sigma model call and percent 1
    as the terminal zero sigma. MiniMax H3 receives the shifted video sigma, so
    the shift must be inverted before converting it to progress.
    """
    sigma = float(video_sigma)
    shift = float(shift_video)
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0 + 1.0e-6:
        raise RuntimeError(f"H3 SLA observed invalid video sigma {sigma!r}")
    if not math.isfinite(shift) or shift <= 0.0:
        raise RuntimeError(f"H3 SLA observed invalid video shift {shift!r}")
    sigma = min(max(sigma, 0.0), 1.0)
    denominator = shift + sigma * (1.0 - shift)
    if denominator <= 0.0:
        raise RuntimeError("H3 SLA could not invert the shifted video sigma")
    base_sigma = sigma / denominator
    return min(max(1.0 - base_sigma, 0.0), 1.0)


def _video_sigma_from_timestep(timestep) -> float:
    if not torch.is_tensor(timestep) or int(timestep.numel()) < 1:
        raise RuntimeError("H3 SLA requires the MiniMax H3 tensor timestep")
    # MiniMaxH3Model receives model_sampling.timestep(sigma) = sigma * 1000.
    return float(timestep.detach().float().flatten()[0].item()) / 1000.0


def _forward_attention_policy(
    *,
    mode: str,
    seq_len: int,
    forward_index: int,
    expected_nfe: int,
    sampling_percent: float | None = None,
    sparse_start_percent: float = SLA_FULL_RANGE_START_PERCENT,
    sparse_end_percent: float = SLA_FULL_RANGE_END_PERCENT,
) -> dict:
    """Choose one attention owner for a complete H3 model forward.

    The published H3 adapter was validated around a roughly 111K-token 768p
    sequence.  Sparse-only SLA ablations are not a scientifically safe default
    for the much shorter practical ComfyUI sequences.  The legacy public mode
    therefore becomes a quality-first policy without changing its saved widget
    value: short sequences stay dense, while eligible sequences keep dense
    boundary denoising forwards and protect the packed condition prefix during
    sparse middle forwards.  The exact released all-sparse route remains an
    explicitly named experimental mode.
    """
    mode = str(mode)
    seq_len = int(seq_len)
    forward_index = int(forward_index)
    expected_nfe = int(expected_nfe)
    if mode == "dense_lora_control":
        return {
            "execution": SLA_EXECUTION_DENSE,
            "reason": "explicit_dense_lora_control",
            "protect_condition_prefix": False,
        }
    if mode == SLA_CONSUMER_TURBO_MODE:
        return {
            "execution": SLA_EXECUTION_DENSE,
            "reason": "consumer_turbo_attention_owned_outside_sla",
            "protect_condition_prefix": False,
        }
    if mode == "apply_lightx2v_sla_upstream_exact_exp":
        window = _validate_sla_percent_window(
            sparse_start_percent, sparse_end_percent
        )
        if not window["full_range"]:
            if sampling_percent is None or not math.isfinite(float(sampling_percent)):
                raise RuntimeError(
                    "H3 SLA percent-window routing requires the current sampling percent"
                )
            current = float(sampling_percent)
            epsilon = 1.0e-6
            if (
                current < float(window["start_percent"]) - epsilon
                or current > float(window["end_percent"]) + epsilon
            ):
                return {
                    "execution": SLA_EXECUTION_DENSE,
                    "reason": "sla_percent_window_dense_boundary",
                    "protect_condition_prefix": False,
                }
            return {
                "execution": SLA_EXECUTION_SPARSE,
                "reason": "sla_percent_window_active_85pct_sparse",
                "protect_condition_prefix": False,
            }
        return {
            "execution": SLA_EXECUTION_SPARSE,
            "reason": "explicit_upstream_exact_85pct_sparse_experiment",
            "protect_condition_prefix": False,
        }
    if mode != "apply_lightx2v_sla":
        raise RuntimeError(f"H3 SLA cannot plan attention for mode {mode!r}")
    if seq_len < SLA_AUTO_SAFE_MIN_SPARSE_SEQUENCE:
        return {
            "execution": SLA_EXECUTION_DENSE,
            "reason": "auto_safe_short_sequence_dense_fallback",
            "protect_condition_prefix": False,
        }
    edge = SLA_AUTO_SAFE_DENSE_EDGE_FORWARDS
    if expected_nfe <= edge * 2 or forward_index < edge or forward_index >= expected_nfe - edge:
        return {
            "execution": SLA_EXECUTION_DENSE,
            "reason": "auto_safe_dense_boundary_forward",
            "protect_condition_prefix": False,
        }
    return {
        "execution": SLA_EXECUTION_SPARSE,
        "reason": "auto_safe_sparse_middle_forward",
        "protect_condition_prefix": True,
    }


def _route_attention_execution(route: Mapping) -> str:
    execution = route.get("attention_execution")
    if execution in {SLA_EXECUTION_DENSE, SLA_EXECUTION_SPARSE}:
        return str(execution)
    # Compatibility for isolated callers created before auto-safe v1. Missing
    # planning state must fail toward dense diagnostics, never toward the old
    # all-sparse route. Neither route is a quality claim; only the explicit
    # upstream-exact experiment can opt into sparse.
    return (
        SLA_EXECUTION_SPARSE
        if route.get("mode") == "apply_lightx2v_sla_upstream_exact_exp"
        else SLA_EXECUTION_DENSE
    )

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


def _validate_sigmas(
    sigmas: torch.Tensor,
    shift_video: float = SLA_SHIFT_VIDEO,
) -> dict:
    values = torch.as_tensor(sigmas).detach().float().cpu().flatten()
    nfe = int(values.numel()) - 1
    if not 1 <= nfe <= SLA_MAX_NFE:
        raise ValueError(
            f"H3 LightX2V SLA requires 1..{SLA_MAX_NFE} NFE plus the final zero sigma"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("H3 LightX2V SLA received NaN/Inf sigmas")
    if not bool((values[:-1] > 0).all()) or abs(float(values[-1])) > 1.0e-7:
        raise ValueError("H3 LightX2V SLA requires positive sigmas then a final zero")
    if not bool((values[:-1] >= values[1:]).all()):
        raise ValueError("H3 LightX2V SLA requires a monotonic sigma schedule")
    shift_video = float(shift_video)
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("H3 LightX2V SLA requires a finite positive video shift")
    expected = native_flow_sigmas(nfe, shift_video).float()
    if not bool(torch.allclose(values, expected, rtol=0.0, atol=1.0e-6)):
        raise ValueError(
            f"H3 LightX2V SLA requires T8 native_flow with {nfe} steps and "
            f"video shift {shift_video}"
        )
    return {
        "nfe": nfe,
        "entries": int(values.numel()),
        "video_sigmas": [float(value) for value in values.tolist()],
        "scheduler": "native_flow",
        "shift_video": shift_video,
        "official_checkpoint_nfe": SLA_EXPECTED_NFE,
        "schedule_status": (
            "official_4step_contract"
            if nfe == SLA_EXPECTED_NFE
            and math.isclose(shift_video, SLA_SHIFT_VIDEO, abs_tol=1.0e-7)
            else "experimental_user_selected_nfe"
        ),
        "sigma_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


def _validate_lora_header(path: str | Path) -> dict:
    from safetensors import safe_open

    resolved = Path(path).resolve()
    stat = resolved.stat()
    keys = []
    metadata = {}
    header_error = None
    try:
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = dict(handle.metadata() or {})
    except Exception as error:
        header_error = f"{type(error).__name__}: {error}"
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    suffix_alpha = ".alpha"
    prefixes_a = {key[: -len(suffix_a)] for key in keys if key.endswith(suffix_a)}
    prefixes_b = {key[: -len(suffix_b)] for key in keys if key.endswith(suffix_b)}
    prefixes_alpha = {
        key[: -len(suffix_alpha)] for key in keys if key.endswith(suffix_alpha)
    }
    unsupported = [
        key
        for key in keys
        if not key.endswith((suffix_a, suffix_b, suffix_alpha))
    ]
    foreign_targets = sorted(
        prefix for prefix in prefixes_a if not prefix.startswith("diffusion_model.")
    )
    base_model = str(metadata.get("base_model", "")).strip()
    header_fingerprint = hashlib.sha256(
        json.dumps(
            {"keys": sorted(keys), "metadata": metadata},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "filename": resolved.name,
        "bytes": int(stat.st_size),
        "sha256": None,
        "file_sha256_enforced": False,
        "identity_policy": "diagnostic_only_not_a_load_gate",
        "header_error": header_error,
        "header_fingerprint_sha256": header_fingerprint,
        "tensor_count": len(keys),
        "patch_count": len(prefixes_a),
        "alpha_count": len(prefixes_alpha),
        "unsupported_keys": unsupported,
        "missing_a_prefixes": sorted(prefixes_b - prefixes_a),
        "missing_b_prefixes": sorted(prefixes_a - prefixes_b),
        "unpaired_alpha_prefixes": sorted(prefixes_alpha - prefixes_a),
        "foreign_targets": foreign_targets,
        "metadata_base_model_reference_match": (
            not base_model
            or "minimax" in base_model.lower()
            or "h3" in base_model.lower()
        ),
        "metadata": metadata,
        "reference_artifact": {
            "filename": SLA_LORA_FILENAME,
            "bytes": SLA_LORA_BYTES,
            "sha256": SLA_LORA_SHA256,
            "matches_filename": resolved.name == SLA_LORA_FILENAME,
            "matches_size": int(stat.st_size) == SLA_LORA_BYTES,
            "not_enforced": True,
        },
    }


def _required_parameters(function, required: set[str], label: str) -> list[str]:
    parameters = list(inspect.signature(function).parameters)
    missing = sorted(required - set(parameters))
    if missing:
        raise RuntimeError(f"H3 SLA core semantic contract lost {label}: {missing}")
    return parameters


def _core_semantic_contract() -> dict:
    # Some older Painter MiniMax nodes install a process-global PackedLayout
    # wrapper that forwards the now-removed ``frame_count`` keyword.  Reuse the
    # conditioning path's executable compatibility probe: it unwraps only that
    # specifically marked patch, and only when the enclosed native constructor
    # independently passes the current keyframe+reference ordering contract.
    from .conditioning import assert_hybrid_layout_contract

    packed_layout_compatibility = assert_hybrid_layout_contract()
    source_hashes = {
        "attention_forward": _source_sha256(Attention.forward),
        "packed_layout": _source_sha256(PackedLayout.__init__),
        "model_forward": _source_sha256(MiniMaxH3Model._forward),
        "patchify_video": _source_sha256(patchify_video),
    }
    signatures = {
        "attention_forward": _required_parameters(
            Attention.forward,
            {"self", "x", "rope_freqs", "transformer_options"},
            "Attention.forward parameters",
        ),
        "packed_layout": _required_parameters(
            PackedLayout.__init__,
            {
                "self",
                "text_len",
                "latent_t",
                "latent_h",
                "latent_w",
                "audio_t",
                "keyframes",
                "refs",
            },
            "PackedLayout parameters",
        ),
        "model_forward": _required_parameters(
            MiniMaxH3Model._forward,
            {
                "self",
                "x",
                "timestep",
                "context",
                "transformer_options",
                "minimax_payload",
            },
            "MiniMaxH3Model._forward parameters",
        ),
    }

    patch_probe = torch.arange(64, dtype=torch.float32).reshape(1, 2, 2, 4, 4)
    patch_rows = patchify_video(patch_probe, (1, 2, 2))
    expected_first = torch.tensor(
        [0.0, 1.0, 4.0, 5.0, 32.0, 33.0, 36.0, 37.0]
    )
    if tuple(patch_rows.shape) != (8, 8) or not torch.equal(
        patch_rows[0].cpu(), expected_first
    ):
        raise RuntimeError(
            "H3 SLA core semantic contract changed native video patch ordering"
        )

    keyframe = torch.zeros(1, 24, 1, 4, 4)
    layout = PackedLayout(
        3,
        2,
        4,
        4,
        5,
        keyframes=[
            {"resolved_frame_index": 0, "latent": keyframe},
            {"resolved_frame_index": 4, "latent": keyframe},
        ],
        refs=None,
    )
    kinds = [kind for _start, _end, kind in layout.segments]
    if layout.signature != (3, 2, 4, 4, 5) or kinds[-2:] != ["audio", "video"]:
        raise RuntimeError(
            "H3 SLA core semantic contract changed FL2VA packed target ordering"
        )
    if any(kind.startswith("ref_") for kind in kinds):
        raise RuntimeError("H3 SLA core semantic probe unexpectedly packed references")
    if int(layout.segments[-2][1] - layout.segments[-2][0]) != 10:
        raise RuntimeError("H3 SLA core semantic contract changed target audio rows")
    if int(layout.segments[-1][1] - layout.segments[-1][0]) != 8:
        raise RuntimeError("H3 SLA core semantic contract changed target video rows")
    return {
        "status": "semantic_contract_validated",
        "source_hashes": source_hashes,
        "source_hash_policy": "diagnostic_only_not_a_compatibility_gate",
        "packed_layout_compatibility": packed_layout_compatibility,
        "signatures": signatures,
        "patchify_video": {"shape": list(patch_rows.shape), "ordering": "validated"},
        "packed_layout": {
            "signature": list(layout.signature),
            "segment_kinds": kinds,
            "target_tail": kinds[-2:],
            "seq_len": int(layout.seq_len),
        },
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
        quant_params = getattr(weight, "_params", None)
        convrot = getattr(quant_params, "convrot", None)
    except (AttributeError, IndexError) as exc:
        return {
            "policy": base_policy,
            "inspection_error": f"{type(exc).__name__}: {exc}",
            "model_identity_policy": "diagnostic_only_not_a_load_gate",
            "compatibility_status": "uninspected_user_selected_base",
        }
    quantized = quant_format is not None or weight_type == "QuantizedTensor"
    official_bf16 = dtype == "torch.bfloat16" and not quantized

    def _target_modules(blocks):
        modules = []
        for block in blocks:
            attention = getattr(block, "attn", None)
            mlp = getattr(block, "mlp", None)
            modules.extend(
                value
                for value in (
                    getattr(attention, "qkv_proj", None),
                    getattr(attention, "out_proj", None),
                    getattr(mlp, "fc1", None),
                    getattr(mlp, "fc2", None),
                )
                if value is not None
            )
        return modules

    main_targets = _target_modules(list(getattr(diffusion, "blocks", ()) or ()))
    token_refiner = getattr(diffusion, "token_refiner", None)
    refiner_targets = _target_modules(
        list(getattr(token_refiner, "blocks", ()) or ())
    )

    def _is_int8_convrot(target) -> bool:
        target_weight = getattr(target, "weight", None)
        params = getattr(target_weight, "_params", None)
        return (
            str(getattr(target, "quant_format", "")) == "int8_tensorwise"
            and type(target_weight).__name__ == "QuantizedTensor"
            and bool(getattr(params, "convrot", False))
        )

    def _is_unquantized(target) -> bool:
        target_weight = getattr(target, "weight", None)
        return (
            getattr(target, "quant_format", None) is None
            and type(target_weight).__name__ != "QuantizedTensor"
        )

    target_quantization = {
        "main_target_count": len(main_targets),
        "main_int8_convrot_count": sum(_is_int8_convrot(value) for value in main_targets),
        "main_unquantized_count": sum(_is_unquantized(value) for value in main_targets),
        "token_refiner_target_count": len(refiner_targets),
        "token_refiner_int8_convrot_count": sum(
            _is_int8_convrot(value) for value in refiner_targets
        ),
        "token_refiner_unquantized_count": sum(
            _is_unquantized(value) for value in refiner_targets
        ),
    }
    return {
        "policy": base_policy,
        "observed_qkv_dtype": dtype,
        "observed_qkv_module": module_type,
        "observed_weight_type": weight_type,
        "observed_quant_format": (
            None if quant_format is None else str(quant_format)
        ),
        "observed_convrot": bool(convrot),
        "lora_target_quantization": target_quantization,
        "model_storage_bytes": int(model.model_size()),
        "quantized_base_observed": quantized,
        "official_bf16_base_observed": official_bf16,
        "requested_policy_match": (
            base_policy != "official_bf16_only" or official_bf16
        ),
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
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


def _existing_attention_contract(transformer_options: Mapping) -> dict:
    installed = transformer_options.get("optimized_attention_override")
    if installed is None:
        return {"status": "none", "backend": None}
    get_attention = getattr(attention_module, "get_attention_function", None)
    if callable(get_attention):
        for backend in ("pytorch", "comfy_kitchen_int8"):
            try:
                candidate = get_attention(backend)
            except Exception:
                candidate = None
            if candidate is not None and installed is candidate:
                return {
                    "status": "recognized_builtin_override_replaced",
                    "backend": backend,
                    "module": str(getattr(installed, "__module__", "unknown")),
                    "name": str(getattr(installed, "__name__", type(installed).__name__)),
                }
    raise RuntimeError(
        "H3 SLA owns attention and cannot stack with an unrecognized attention "
        "override such as SageAttention, Sol-Attn, FETA or Prompt Relay"
    )


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
    native_h3_model_observed = isinstance(base, MiniMaxH3BaseModel) or (
        type(getattr(base, "diffusion_model", None)).__name__ == "MiniMaxH3Model"
    )

    semantic_core = _core_semantic_contract()

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
    existing_attention = _existing_attention_contract(transformer)
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
    if any(not math.isfinite(value) or value <= 0.0 for value in shifts.values()):
        raise RuntimeError("H3 SLA requires finite positive MiniMax H3 Dual-Clock shifts")

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
        "semantic_core": semantic_core,
        "core_hashes": semantic_core["source_hashes"],
        "base": _model_dtype_contract(model, base_policy),
        "native_h3_model_observed": native_h3_model_observed,
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
        "dual_clock": {
            **shifts,
            "official_4step_shift_match": (
                math.isclose(shifts["video"], SLA_SHIFT_VIDEO, abs_tol=1.0e-7)
                and math.isclose(shifts["audio"], SLA_SHIFT_AUDIO, abs_tol=1.0e-7)
            ),
        },
        "preexisting_attention": existing_attention,
        "external_attention_policy": external_attention_policy,
        "external_attention": external_attention,
    }


def _apply_authenticated_lora(
    model,
    path: str | Path,
    *,
    application_policy: str = "standard_patch",
) -> tuple[object, dict]:
    application_policy = str(application_policy)
    if application_policy not in SLA_LORA_APPLICATION_POLICIES:
        raise ValueError(
            f"Unknown H3 SLA LoRA application policy {application_policy!r}"
        )
    contract = _validate_lora_header(path)
    state, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    state = comfy.lora_convert.convert_lora(state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)
    patched = model.clone()
    if application_policy == "bypass_model_only":
        manager = comfy.weight_adapter.BypassInjectionManager()
        for key, adapter in loaded.items():
            manager.add_adapter(key, adapter, strength=1.0)
        injections = manager.create_injections(patched.model)
        hook_count = int(manager.get_hook_count())
        patched.set_injections("bypass_lora", injections)
        contract["bypass_hook_count"] = hook_count
        contract["applied_patch_count"] = hook_count
        contract["application_mode"] = "comfyui_bypass_model_only"
        contract["base_weight_mutation"] = False
    else:
        applied = set(patched.add_patches(loaded, 1.0))
        contract["applied_patch_count"] = len(applied)
        contract["unapplied_patch_keys"] = sorted(set(loaded) - applied)
        contract["application_mode"] = "comfyui_standard_weight_patch"
        contract["base_weight_mutation"] = True
    if metadata and hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_sla_lora_metadata", dict(metadata))
    contract["mapped_patch_count"] = len(loaded)
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
        if _route_attention_execution(route) == SLA_EXECUTION_SPARSE:
            x = args[0] if args else kwargs.get("x")
            if tensor is not x and isinstance(x, list):
                x.clear()
                if args:
                    args = (tensor,) + args[1:]
                else:
                    kwargs = dict(kwargs)
                    kwargs["x"] = tensor
            return stock_forward(module, *args, **kwargs)
        if route["mode"] not in SLA_SPARSE_MODES and route["mode"] != "dense_lora_control":
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


def _overlapping_block_range(
    start: int,
    end: int,
    block_size: int,
    total_blocks: int,
) -> tuple[int, int]:
    first = max(0, min(int(total_blocks), int(start) // int(block_size)))
    last = max(
        first,
        min(
            int(total_blocks),
            math.ceil(int(end) / int(block_size)),
        ),
    )
    return first, last


def protect_condition_prefix_blocks(
    sparse_map: torch.Tensor,
    layout_segments,
    *,
    k_block: int = SLA_K_BLOCK,
) -> int:
    """Pin text/keyframe/audio keys without displacing learned video top-k.

    FL2VA packs every non-target-video segment before the final target video.
    Pinning those key blocks on top of the learned router preserves access to
    text, both keyframes and joint audio while leaving the video top-k intact.
    This does not freeze audio and does not alter query rows.
    """
    if sparse_map.ndim != 4:
        raise ValueError("H3 SLA prefix protection requires [B,H,QB,KB] map")
    video_segments = [
        (int(start), int(end))
        for start, end, kind in layout_segments
        if str(kind) == "video"
    ]
    if len(video_segments) != 1:
        raise RuntimeError("H3 SLA prefix protection requires one target video segment")
    video_start, _video_end = video_segments[0]
    protected = min(int(sparse_map.shape[-1]), math.ceil(video_start / int(k_block)))
    if protected > 0:
        sparse_map[..., :protected] = 1
    return protected


def _selection_stats(values: torch.Tensor) -> dict:
    values = values.detach().float()
    if values.numel() == 0:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    minimum, mean, maximum = torch.stack(
        (values.min(), values.mean(), values.max())
    ).cpu().tolist()
    return {
        "minimum": float(minimum),
        "mean": float(mean),
        "maximum": float(maximum),
    }


def sparse_route_coverage(
    sparse_map: torch.Tensor,
    layout_segments,
    *,
    q_block: int = SLA_Q_BLOCK,
    k_block: int = SLA_K_BLOCK,
) -> dict:
    """Summarize selected keys by packed segment and video-time quadrant."""
    if sparse_map.ndim != 4:
        raise ValueError("H3 SLA coverage requires [B,H,QB,KB] map")
    q_blocks = int(sparse_map.shape[-2])
    k_blocks = int(sparse_map.shape[-1])
    segments = [
        (int(start), int(end), str(kind)) for start, end, kind in layout_segments
    ]
    by_kind: dict[str, list[tuple[int, int]]] = {}
    for start, end, kind in segments:
        by_kind.setdefault(kind, []).append((start, end))

    coverage: dict[str, dict] = {}
    for kind, ranges in by_kind.items():
        key_indices: set[int] = set()
        for start, end in ranges:
            first, last = _overlapping_block_range(start, end, k_block, k_blocks)
            key_indices.update(range(first, last))
        if key_indices:
            index = torch.tensor(
                sorted(key_indices), device=sparse_map.device, dtype=torch.long
            )
            selected = sparse_map.index_select(-1, index).sum(dim=-1)
        else:
            selected = sparse_map[..., :0].sum(dim=-1)
        coverage[f"keys_{kind}"] = {
            "key_block_count": len(key_indices),
            **_selection_stats(selected),
        }

    video = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    if video is not None:
        video_start, video_end = video
        video_length = video_end - video_start
        for q_index in range(4):
            q_start = video_start + (video_length * q_index) // 4
            q_end = video_start + (video_length * (q_index + 1)) // 4
            q_first, q_last = _overlapping_block_range(
                q_start, q_end, q_block, q_blocks
            )
            for k_index in range(4):
                k_start = video_start + (video_length * k_index) // 4
                k_end = video_start + (video_length * (k_index + 1)) // 4
                k_first, k_last = _overlapping_block_range(
                    k_start, k_end, k_block, k_blocks
                )
                values = sparse_map[..., q_first:q_last, k_first:k_last].sum(dim=-1)
                coverage[f"video_q{q_index + 1}_to_k{k_index + 1}"] = {
                    "query_block_count": max(0, q_last - q_first),
                    "key_block_count": max(0, k_last - k_first),
                    **_selection_stats(values),
                }
    return coverage


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

    def begin_forward(self, route: Mapping, *, video_sigma: float | None = None) -> int:
        with self._lock:
            if self._consumed:
                raise RuntimeError("H3 SLA runtime token was already consumed")
            index = len(self._forwards)
            sigma_contract = self.config.get("sigma_contract") or {}
            expected_nfe = int(sigma_contract.get("nfe", SLA_EXPECTED_NFE))
            if index >= expected_nfe:
                raise RuntimeError(
                    f"H3 SLA observed more than the expected {expected_nfe} model forwards"
                )
            configured_sigmas = list(sigma_contract.get("video_sigmas") or [])
            if video_sigma is not None:
                video_sigma = float(video_sigma)
                if configured_sigmas and not math.isclose(
                    video_sigma,
                    float(configured_sigmas[index]),
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                ):
                    raise RuntimeError(
                        "H3 SLA runtime sigma does not match the validated schedule: "
                        f"forward={index}, observed={video_sigma}, "
                        f"expected={configured_sigmas[index]}"
                    )
            window = dict(self.config.get("sparse_percent_window") or {})
            start_percent = float(
                window.get("start_percent", SLA_FULL_RANGE_START_PERCENT)
            )
            end_percent = float(
                window.get("end_percent", SLA_FULL_RANGE_END_PERCENT)
            )
            window_contract = _validate_sla_percent_window(
                start_percent, end_percent
            )
            sampling_percent = None
            if video_sigma is not None:
                sampling_percent = _sampling_percent_from_video_sigma(
                    video_sigma,
                    float(sigma_contract.get("shift_video", SLA_SHIFT_VIDEO)),
                )
            elif not window_contract["full_range"]:
                raise RuntimeError(
                    "H3 SLA percent-window routing requires the current video sigma"
                )
            policy = _forward_attention_policy(
                mode=str(self.config.get("mode", "dense_lora_control")),
                seq_len=int(route["seq_len"]),
                forward_index=index,
                expected_nfe=expected_nfe,
                sampling_percent=sampling_percent,
                sparse_start_percent=start_percent,
                sparse_end_percent=end_percent,
            )
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
                    "attention_execution": policy["execution"],
                    "attention_policy_reason": policy["reason"],
                    "video_sigma": video_sigma,
                    "sampling_percent": sampling_percent,
                    "protect_condition_prefix": bool(
                        policy["protect_condition_prefix"]
                    ),
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
                    "retained_key_blocks_sum": 0.0,
                    "retained_key_blocks_count": 0,
                    "router_topk_min": None,
                    "router_topk_max": None,
                    "protected_prefix_blocks_min": None,
                    "protected_prefix_blocks_max": None,
                    "retained_ratio_min": None,
                    "retained_ratio_max": None,
                    "retained_ratio_sum": 0.0,
                    "retained_ratio_count": 0,
                    "coverage": {},
                }
            )
            return index

    def forward_policy(self, forward_index: int) -> dict:
        with self._lock:
            forward = self._forwards[int(forward_index)]
            return {
                "execution": forward["attention_execution"],
                "reason": forward["attention_policy_reason"],
                "protect_condition_prefix": forward["protect_condition_prefix"],
            }

    def should_capture_route_diagnostics(self, forward_index: int) -> bool:
        with self._lock:
            calls = int(self._forwards[int(forward_index)]["main_attention_calls"])
            return calls in {0, SLA_EXPECTED_BLOCKS // 2, SLA_EXPECTED_BLOCKS - 1}

    def should_capture_coverage(self, forward_index: int) -> bool:
        with self._lock:
            return int(self._forwards[int(forward_index)]["main_attention_calls"]) == 0

    def record_attention(
        self,
        forward_index: int,
        *,
        sparse: bool,
        workspace_bytes: int = 0,
        key_blocks: int = 0,
        retained_key_blocks: int = 0,
        retained_key_blocks_min: int | None = None,
        retained_key_blocks_mean: float | None = None,
        retained_key_blocks_max: int | None = None,
        router_topk: int | None = None,
        protected_prefix_blocks: int = 0,
        coverage: Mapping | None = None,
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
                if int(key_blocks) <= 0:
                    return
                retained_min = int(
                    retained_key_blocks
                    if retained_key_blocks_min is None
                    else retained_key_blocks_min
                )
                retained_max = int(
                    retained_key_blocks
                    if retained_key_blocks_max is None
                    else retained_key_blocks_max
                )
                retained_mean = float(
                    retained_key_blocks
                    if retained_key_blocks_mean is None
                    else retained_key_blocks_mean
                )
                ratio_min = float(retained_min) / float(key_blocks)
                ratio_max = float(retained_max) / float(key_blocks)
                ratio_mean = retained_mean / float(key_blocks)
                for prefix, minimum_value, maximum_value in (
                    ("key_blocks", int(key_blocks), int(key_blocks)),
                    ("retained_key_blocks", retained_min, retained_max),
                    ("retained_ratio", ratio_min, ratio_max),
                ):
                    minimum = f"{prefix}_min"
                    maximum = f"{prefix}_max"
                    forward[minimum] = (
                        minimum_value
                        if forward[minimum] is None
                        else min(forward[minimum], minimum_value)
                    )
                    forward[maximum] = (
                        maximum_value
                        if forward[maximum] is None
                        else max(forward[maximum], maximum_value)
                    )
                forward["retained_key_blocks_sum"] += retained_mean
                forward["retained_key_blocks_count"] += 1
                if router_topk is not None:
                    for suffix, function in (("min", min), ("max", max)):
                        key = f"router_topk_{suffix}"
                        value = int(router_topk)
                        forward[key] = (
                            value if forward[key] is None else function(forward[key], value)
                        )
                for suffix, function in (("min", min), ("max", max)):
                    key = f"protected_prefix_blocks_{suffix}"
                    value = int(protected_prefix_blocks)
                    forward[key] = (
                        value if forward[key] is None else function(forward[key], value)
                    )
                forward["retained_ratio_sum"] += ratio_mean
                forward["retained_ratio_count"] += 1
                for label, values in dict(coverage or {}).items():
                    aggregate = forward["coverage"].setdefault(
                        str(label),
                        {
                            "minimum": None,
                            "maximum": None,
                            "mean_sum": 0.0,
                            "count": 0,
                            "query_block_count": values.get("query_block_count"),
                            "key_block_count": values.get("key_block_count"),
                        },
                    )
                    minimum = float(values["minimum"])
                    maximum = float(values["maximum"])
                    aggregate["minimum"] = (
                        minimum
                        if aggregate["minimum"] is None
                        else min(aggregate["minimum"], minimum)
                    )
                    aggregate["maximum"] = (
                        maximum
                        if aggregate["maximum"] is None
                        else max(aggregate["maximum"], maximum)
                    )
                    aggregate["mean_sum"] += float(values["mean"])
                    aggregate["count"] += 1
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
            forwards = []
            for source in self._forwards:
                forward = dict(source)
                retained_count = int(forward.pop("retained_key_blocks_count", 0))
                retained_sum = float(forward.pop("retained_key_blocks_sum", 0.0))
                forward["retained_key_blocks_mean"] = (
                    retained_sum / retained_count if retained_count else None
                )
                coverage_report = {}
                for label, values in dict(forward.get("coverage") or {}).items():
                    values = dict(values)
                    count = int(values.pop("count", 0))
                    mean_sum = float(values.pop("mean_sum", 0.0))
                    values["mean"] = mean_sum / count if count else None
                    coverage_report[label] = values
                forward["coverage"] = coverage_report
                forwards.append(forward)
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
    if _route_attention_execution(route) == SLA_EXECUTION_DENSE:
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
        protected_prefix_blocks = 0
        if bool(route.get("protect_condition_prefix")):
            protected_prefix_blocks = protect_condition_prefix_blocks(
                sparse_map,
                route.get("layout_segments") or (),
            )
        key_blocks = int(sparse_map.shape[-1])
        capture_diagnostics = runtime.should_capture_route_diagnostics(forward_index)
        if capture_diagnostics:
            selected_counts = sparse_map.sum(dim=-1)
            retained_min = int(selected_counts.min().item())
            retained_mean = float(selected_counts.float().mean().item())
            retained_max = int(selected_counts.max().item())
            coverage = (
                sparse_route_coverage(
                    sparse_map,
                    route.get("layout_segments") or (),
                )
                if runtime.should_capture_coverage(forward_index)
                else None
            )
        else:
            retained_min = retained_mean = retained_max = None
            coverage = None
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
        key_blocks=key_blocks if capture_diagnostics else 0,
        retained_key_blocks=topk,
        retained_key_blocks_min=retained_min,
        retained_key_blocks_mean=retained_mean,
        retained_key_blocks_max=retained_max,
        router_topk=topk,
        protected_prefix_blocks=protected_prefix_blocks,
        coverage=coverage,
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
    lora_application_policy: str = "standard_patch",
    sla_start_percent: float = SLA_FULL_RANGE_START_PERCENT,
    sla_end_percent: float = SLA_FULL_RANGE_END_PERCENT,
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
    lora_application_policy = str(lora_application_policy)
    if lora_application_policy not in SLA_LORA_APPLICATION_POLICIES:
        raise ValueError(
            f"Unknown H3 SLA LoRA application policy {lora_application_policy!r}"
        )
    if not 32 <= int(max_router_workspace_mib) <= 2048:
        raise ValueError("H3 SLA max_router_workspace_mib must be 32..2048")
    sparse_percent_window = _validate_sla_percent_window(
        sla_start_percent, sla_end_percent
    )

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
        "lora_application_policy": lora_application_policy,
        "sparse_percent_window": sparse_percent_window,
        "max_router_workspace_mib": int(max_router_workspace_mib),
        "quality_safety_policy": {
            "legacy_apply_mode": "auto_safe_v1",
            "minimum_sparse_sequence_tokens": SLA_AUTO_SAFE_MIN_SPARSE_SEQUENCE,
            "dense_boundary_forwards_each_side": SLA_AUTO_SAFE_DENSE_EDGE_FORWARDS,
            "sparse_middle_condition_prefix_protected": True,
            "upstream_exact_mode": "apply_lightx2v_sla_upstream_exact_exp",
            "hard_size_rejection": False,
            "diagnostic_layers": [0, SLA_EXPECTED_BLOCKS // 2, SLA_EXPECTED_BLOCKS - 1],
            "coverage_layer": 0,
        },
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

    config["core_contract"] = _assert_core_contract(
        model,
        base_policy=base_policy,
        external_attention_policy=external_attention_policy,
    )
    config["sigma_contract"] = _validate_sigmas(
        sigmas,
        shift_video=config["core_contract"]["dual_clock"]["video"],
    )
    if mode in SLA_SPARSE_MODES:
        config["kernel_contract"] = _kernel_contract()
    else:
        config["kernel_contract"] = {
            "operator": "ComfyUI dense optimized attention control",
            "sparse_kernel_loaded": False,
        }
    patched, lora_contract = _apply_authenticated_lora(
        model,
        lora_path,
        application_policy=lora_application_policy,
    )
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
            forward_index = runtime.begin_forward(
                route,
                video_sigma=_video_sigma_from_timestep(timestep),
            )
            attention_policy = runtime.forward_policy(forward_index)
            route.update(
                {
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "mode": mode,
                    "max_router_workspace_mib": int(max_router_workspace_mib),
                    "attention_execution": attention_policy["execution"],
                    "attention_policy_reason": attention_policy["reason"],
                    "protect_condition_prefix": attention_policy[
                        "protect_condition_prefix"
                    ],
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
    elif mode == SLA_CONSUMER_TURBO_MODE:
        sigma_contract = report["config"].get("sigma_contract") or {}
        expected_nfe = int(sigma_contract.get("nfe", 8))
        if expected_nfe != 8:
            raise RuntimeError(
                "H3 consumer Turbo profile lost its validated 8-NFE contract"
            )
        if report["model_forward_count"] != expected_nfe:
            raise RuntimeError(
                f"H3 consumer Turbo expected {expected_nfe} model forwards, observed "
                f"{report['model_forward_count']}"
            )
        zeros = [0] * expected_nfe
        for key in (
            "main_attention_calls_per_forward",
            "sparse_kernel_calls_per_forward",
            "dense_control_calls_per_forward",
            "external_sage_calls_per_forward",
        ):
            if report[key] != zeros:
                raise RuntimeError(
                    "H3 consumer Turbo profile unexpectedly entered the SLA attention "
                    f"runtime: {key}={report[key]}"
                )
        if report["kernel_failure_count"]:
            raise RuntimeError("H3 consumer Turbo profile recorded a runtime failure")
        report["expected_nfe"] = expected_nfe
        report["attention_execution_plan"] = ["outside_sla_owner"] * expected_nfe
        report["status"] = "consumer_turbo8_profile_mechanically_verified"
    else:
        sigma_contract = report["config"].get("sigma_contract") or {}
        expected_nfe = int(sigma_contract.get("nfe", SLA_EXPECTED_NFE))
        if not 1 <= expected_nfe <= SLA_MAX_NFE:
            raise RuntimeError(f"H3 SLA runtime has invalid expected NFE {expected_nfe}")
        report["expected_nfe"] = expected_nfe
        if report["model_forward_count"] != expected_nfe:
            raise RuntimeError(
                f"H3 SLA expected {expected_nfe} model forwards, observed "
                f"{report['model_forward_count']}"
            )
        expected = [SLA_EXPECTED_BLOCKS] * expected_nfe
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
        executions = [
            str(value.get("attention_execution")) for value in report["forwards"]
        ]
        expected_sparse = [
            SLA_EXPECTED_BLOCKS if value == SLA_EXECUTION_SPARSE else 0
            for value in executions
        ]
        expected_dense = [
            SLA_EXPECTED_BLOCKS if value == SLA_EXECUTION_DENSE else 0
            for value in executions
        ]
        report["attention_execution_plan"] = executions
        if mode in SLA_SPARSE_MODES:
            if report["sparse_kernel_calls_per_forward"] != expected_sparse:
                raise RuntimeError(
                    "H3 SLA sparse calls do not match the planned forwards: "
                    f"{report['sparse_kernel_calls_per_forward']}"
                )
            if report["dense_control_calls_per_forward"] != expected_dense:
                raise RuntimeError(
                    "H3 SLA dense calls do not match the auto-safe execution plan: "
                    f"{report['dense_control_calls_per_forward']}"
                )
            expected_external = (
                expected_dense
                if external_policy == "compose_kj_sage"
                else [0] * expected_nfe
            )
            if report["external_sage_calls_per_forward"] != expected_external:
                raise RuntimeError(
                    "H3 SLA external KJ Sage calls do not match the planned dense forwards"
                )
            if mode == "apply_lightx2v_sla_upstream_exact_exp":
                percent_window = dict(
                    report["config"].get("sparse_percent_window") or {}
                )
                if bool(percent_window.get("full_range", True)):
                    report["status"] = "lightx2v_upstream_exact_sparse_exp_verified"
                elif (
                    report["config"].get("lora_application_policy")
                    == "bypass_model_only"
                ):
                    report["status"] = (
                        "lightx2v_int8_bypass_percent_window_exp_verified"
                    )
                else:
                    report["status"] = "lightx2v_sparse_percent_window_exp_verified"
            elif all(value == SLA_EXECUTION_DENSE for value in executions):
                reasons = {
                    str(value.get("attention_policy_reason"))
                    for value in report["forwards"]
                }
                report["status"] = (
                    "auto_safe_short_sequence_dense_fallback_verified"
                    if reasons == {"auto_safe_short_sequence_dense_fallback"}
                    else "auto_safe_all_dense_boundary_verified"
                )
            else:
                report["status"] = "auto_safe_dense_edge_sparse_middle_verified"
        else:
            if report["dense_control_calls_per_forward"] != expected:
                raise RuntimeError("H3 SLA dense control did not cover all 50 blocks")
            if report["sparse_kernel_calls_per_forward"] != [0] * expected_nfe:
                raise RuntimeError("H3 SLA dense control unexpectedly used sparse kernels")
            if external_policy == "compose_kj_sage":
                if report["external_sage_calls_per_forward"] != expected:
                    raise RuntimeError(
                        "H3 SLA dense control did not use KJ Sage for all 50 blocks"
                    )
                report["status"] = "dense_lora_kj_sage_control_verified"
            else:
                if report["external_sage_calls_per_forward"] != [0] * expected_nfe:
                    raise RuntimeError("H3 SLA strict control observed external Sage calls")
                report["status"] = "dense_lora_control_verified"
    report["effective_sparse_forward_indices"] = [
        int(index)
        for index, execution in enumerate(
            report.get("attention_execution_plan") or []
        )
        if execution == SLA_EXECUTION_SPARSE
    ]
    report["sampling_percents"] = [
        value.get("sampling_percent") for value in report.get("forwards") or []
    ]
    if mode == SLA_CONSUMER_TURBO_MODE:
        report["quality_claim"] = (
            "mechanically audited corrected-Alpha8 Turbo8 route only; visual quality, "
            "audio and INT8-base behavior still require full human review"
        )
    else:
        report["quality_claim"] = (
            "mechanically audited only; visual quality, speedup and INT8-base parity "
            "require a same-input/same-seed ordinary-Turbo A/B review"
        )
    return av_latent, _json(report)
