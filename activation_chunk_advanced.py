from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import math
from typing import Any


ACTIVATION_CHUNK_SCHEMA = "t8.minimax_h3.activation_chunk.v1"
ATTACHMENT_KEY = "t8_minimax_h3_activation_chunk"
SUPPORTED_SOURCE_CONTRACTS = {
    (
        "ec62dafa65d6eaf36c670b926a05b42503702cbd6e1e4bb9db279c0db2b4a3c5",
        "a117b068b48abfc4e3b6e0a92fdf6b964043028f9846212562ec192a7a9136e5",
        "abfe720a5640daf374201a9600432618bdefc4df0a337e51d7127c745d87b4f0",
    )
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _source_hash(value: Any) -> str | None:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def core_contract() -> dict[str, Any]:
    try:
        from comfy.ldm.minimax.model import DiTBlock, MLP, MiniMaxH3Model
        from .sla_attention_advanced import _core_semantic_contract
    except Exception as error:
        return {
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
            "hashes": None,
        }
    hashes = (
        _source_hash(MiniMaxH3Model._forward),
        _source_hash(DiTBlock.forward),
        _source_hash(MLP.forward),
    )
    try:
        semantic_contract = _core_semantic_contract()
        required_signatures = {
            "dit_block_forward": {
                "self",
                "x",
                "t_emb",
                "mod_segments",
                "rope_freqs",
                "transformer_options",
            },
            "mlp_forward": {"self", "x"},
        }
        functions = {
            "dit_block_forward": DiTBlock.forward,
            "mlp_forward": MLP.forward,
        }
        signatures = {}
        for name, required in required_signatures.items():
            parameters = list(inspect.signature(functions[name]).parameters)
            missing = sorted(required - set(parameters))
            if missing:
                raise RuntimeError(
                    f"Activation Chunk semantic contract lost {name} parameters: {missing}"
                )
            signatures[name] = parameters
    except Exception as error:
        return {
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
            "hashes": {
                "minimax_h3_forward": hashes[0],
                "dit_block_forward": hashes[1],
                "mlp_forward": hashes[2],
            },
            "source_hash_policy": "diagnostic_only_not_a_compatibility_gate",
        }
    return {
        "supported": True,
        "hashes": {
            "minimax_h3_forward": hashes[0],
            "dit_block_forward": hashes[1],
            "mlp_forward": hashes[2],
        },
        "source_hash_policy": "diagnostic_only_not_a_compatibility_gate",
        "reference_source_match": hashes in SUPPORTED_SOURCE_CONTRACTS,
        "semantic_contract": semantic_contract,
        "signatures": signatures,
        "contract": "current ComfyUI H3 dit/double_block callback with native block closure",
    }


def _extract_native_block(original_block: Any) -> Any:
    code = getattr(original_block, "__code__", None)
    closure = getattr(original_block, "__closure__", None)
    if code is None or closure is None:
        raise RuntimeError("H3 original_block does not expose the expected native closure")
    values = {
        name: cell.cell_contents
        for name, cell in zip(code.co_freevars, closure, strict=True)
    }
    block = values.get("block")
    required = ("adaln_proj", "norm1", "attn", "norm2", "mlp")
    if block is None or any(not hasattr(block, name) for name in required):
        raise RuntimeError("H3 original_block closure does not contain a compatible native block")
    return block


def _validate_segments(segments: Sequence[Sequence[int]], rows: int) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    cursor = 0
    for raw in segments:
        if len(raw) != 3:
            raise ValueError("H3 modulation segments must contain start, stop and row")
        start, stop, row = (int(value) for value in raw)
        if start != cursor or stop <= start or stop > rows or row < 0:
            raise ValueError("H3 modulation segments must be positive, contiguous and cover the stream")
        result.append((start, stop, row))
        cursor = stop
    if cursor != rows:
        raise ValueError("H3 modulation segments do not cover the complete packed stream")
    return result


def _mod_scale_shift(h, shift, scale, segments: Sequence[tuple[int, int, int]]):
    for start, stop, row in segments:
        h[start:stop].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments: Sequence[tuple[int, int, int]]):
    for start, stop, row in segments:
        x[start:stop].addcmul_(other[start:stop], gate[row].to(x.dtype))
    return x


class H3MLPActivationChunkPatch:
    """Clone-local H3 block replacement that chunks only the token-local MLP path."""

    def __init__(self, block_index: int, chunk_rows: int, preserve_short_path: bool):
        self.block_index = int(block_index)
        self.chunk_rows = int(chunk_rows)
        self.preserve_short_path = bool(preserve_short_path)
        if self.block_index < 0 or self.chunk_rows < 1:
            raise ValueError("block_index must be non-negative and chunk_rows must be positive")

    def __call__(self, args: Mapping[str, Any], extra_options: Mapping[str, Any]):
        original_block = extra_options.get("original_block")
        if not callable(original_block):
            raise RuntimeError("H3 Activation Chunk requires the native original_block callback")
        x = args.get("img")
        if x is None or getattr(x, "ndim", None) != 2:
            raise ValueError("H3 Activation Chunk expects a packed [rows, hidden] tensor")
        if self.preserve_short_path and x.shape[0] <= self.chunk_rows:
            return original_block(dict(args))

        block = _extract_native_block(original_block)
        t_emb = args["t_emb"]
        rope_freqs = args["rope_freqs"]
        transformer_options = args.get("transformer_options", {})
        segments = _validate_segments(args["mod_segments"], int(x.shape[0]))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, segments)
        attention = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        x = _mod_gate(x, gate_msa, attention, segments)
        del attention, h

        # The MLP is token-local. Chunk inside each modulation segment so its row-specific
        # scale/gate remains identical while the large SwiGLU projection is bounded.
        for start, stop, row in segments:
            row_shift = shift_mlp[row]
            row_scale = scale_mlp[row]
            row_gate = gate_mlp[row]
            for chunk_start in range(start, stop, self.chunk_rows):
                chunk_stop = min(stop, chunk_start + self.chunk_rows)
                target = x[chunk_start:chunk_stop]
                h_chunk = block.norm2(target)
                h_chunk.mul_(1.0 + row_scale.to(h_chunk.dtype)).add_(
                    row_shift.to(h_chunk.dtype)
                )
                mlp_chunk = block.mlp(h_chunk)
                target.addcmul_(mlp_chunk, row_gate.to(target.dtype))
                del h_chunk, mlp_chunk
        return {"img": x}


def _diffusion_model(model: Any) -> Any:
    getter = getattr(model, "get_model_object", None)
    try:
        return getter("diffusion_model") if callable(getter) else model.model.diffusion_model
    except Exception as error:
        raise ValueError(f"MODEL does not expose diffusion_model: {type(error).__name__}: {error}") from error


def _existing_double_block_replacements(model: Any) -> dict[Any, Any]:
    options = getattr(model, "model_options", {})
    transformer = options.get("transformer_options", {}) if isinstance(options, Mapping) else {}
    patches = transformer.get("patches_replace", {}) if isinstance(transformer, Mapping) else {}
    dit = patches.get("dit", {}) if isinstance(patches, Mapping) else {}
    return dict(dit) if isinstance(dit, Mapping) else {}


def _estimate_rows(width: int, height: int, length: int, reference_images: int) -> dict[str, int]:
    latent_t = max(1, ((length - 5) // 17) * 5 + 2)
    frame_rows = math.ceil(width / 32) * math.ceil(height / 32)
    target_rows = frame_rows * latent_t
    audio_rows = max(1, math.ceil(length / 24.0 * 40.0)) * 2
    estimated = target_rows + audio_rows + frame_rows * reference_images
    return {
        "frame_rows": frame_rows,
        "latent_video_t": latent_t,
        "target_video_rows": target_rows,
        "target_audio_rows": audio_rows,
        "estimated_total_rows_without_text_or_video_refs": estimated,
    }


def _mlp_backend_profile(block: Any) -> dict[str, Any]:
    mlp = getattr(block, "mlp", None)
    fc2 = getattr(mlp, "fc2", None)
    weight = getattr(fc2, "weight", None)
    layout = getattr(weight, "_layout_cls", None)
    layout_name = getattr(layout, "__name__", None) or (
        type(layout).__name__ if layout is not None else None
    )
    weight_functions = getattr(fc2, "weight_function", ())
    weight_function_count = len(weight_functions) if isinstance(weight_functions, Sequence) else None
    fused_int8 = layout_name == "TensorWiseINT8Layout" and weight_function_count == 0
    return {
        "kind": (
            "tensorwise_int8_fused_swiglu"
            if fused_int8
            else "eager_or_runtime_dependent"
        ),
        "fc2_weight_layout": layout_name,
        "fc2_weight_function_count": weight_function_count,
        "native_full_fc1_intermediate_expected": not fused_int8,
        "reason": (
            "Current ComfyUI folds SwiGLU into the TensorWise INT8 down-projection input "
            "quantizer, so the full BF16 fc1 activation proxy does not apply."
            if fused_int8
            else "The current static model state does not prove that the fused INT8 path applies."
        ),
    }


def _activation_proxy(
    rows: int,
    chunk_rows: int,
    ffn: int,
    backend: Mapping[str, Any],
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    original = rows * ffn * 2 * dtype_bytes
    chunked = min(rows, chunk_rows) * ffn * 2 * dtype_bytes
    theoretical_reduction = max(0, original - chunked) / 1024**2
    proxy_applies = bool(backend.get("native_full_fc1_intermediate_expected"))
    return {
        "proxy_basis": "theoretical unfused fc1 output only; allocator/kernel/attention/final-layer memory excluded",
        "dtype_bytes": dtype_bytes,
        "ffn_width": ffn,
        "theoretical_unchunked_fc1_output_mib": original / 1024**2,
        "theoretical_chunked_fc1_output_mib": chunked / 1024**2,
        "theoretical_proxy_reduction_mib": theoretical_reduction,
        "proxy_applies_to_detected_backend": proxy_applies,
        "expected_proxy_reduction_mib": theoretical_reduction if proxy_applies else 0.0,
        "not_a_vram_prediction": True,
    }


def configure_activation_chunk(
    model: Any,
    mode: str,
    chunk_rows: int,
    block_start: int,
    block_end: int,
    preserve_short_path: bool,
    expected_width: int,
    expected_height: int,
    expected_length: int,
    expected_single_image_references: int,
) -> tuple[Any, dict[str, Any]]:
    if mode not in {"report_only", "apply_exp"}:
        raise ValueError(f"unsupported Activation Chunk mode: {mode!r}")
    if chunk_rows < 16 or chunk_rows > 65536:
        raise ValueError("chunk_rows must be between 16 and 65536")
    if expected_width <= 0 or expected_height <= 0 or expected_width % 32 or expected_height % 32:
        raise ValueError("expected width/height must be positive multiples of 32")
    if expected_length < 5 or (expected_length - 5) % 17:
        raise ValueError("expected_length must satisfy the H3 17n+5 grid")
    if expected_single_image_references < 0:
        raise ValueError("expected_single_image_references must be non-negative")

    contract = core_contract()
    diffusion = _diffusion_model(model)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        raise ValueError("diffusion_model does not expose H3 blocks")
    block_count = len(blocks)
    if block_start < 0 or block_end < block_start or block_end >= block_count:
        raise ValueError(
            f"block range must be within 0..{block_count - 1}; got {block_start}..{block_end}"
        )

    existing = _existing_double_block_replacements(model)
    conflicts = [
        key
        for key in existing
        if isinstance(key, Sequence)
        and len(key) >= 2
        and key[0] == "double_block"
        and block_start <= int(key[1]) <= block_end
    ]
    rows = _estimate_rows(
        expected_width,
        expected_height,
        expected_length,
        expected_single_image_references,
    )
    selected_backend_profiles = [
        _mlp_backend_profile(blocks[index])
        for index in range(block_start, block_end + 1)
    ]
    backend_kinds = sorted({item["kind"] for item in selected_backend_profiles})
    backend = dict(selected_backend_profiles[0])
    if len(backend_kinds) > 1:
        backend.update({
            "kind": "heterogeneous",
            "detected_kinds": backend_kinds,
            "native_full_fc1_intermediate_expected": None,
            "reason": "Selected blocks expose heterogeneous MLP execution backends.",
        })
    ffn = int(getattr(getattr(blocks[0], "mlp", None), "fc1", object()).out_features // 2)
    report = {
        "schema": ACTIVATION_CHUNK_SCHEMA,
        "mode": mode,
        "applied": False,
        "core_contract": contract,
        "block_count": block_count,
        "selected_blocks": [int(block_start), int(block_end)],
        "selected_block_count": int(block_end - block_start + 1),
        "chunk_rows": int(chunk_rows),
        "preserve_short_path": bool(preserve_short_path),
        "existing_double_block_conflicts": [list(key) for key in conflicts],
        "estimated_rows": rows,
        "mlp_backend": backend,
        "expected_memory_benefit": (
            "low_or_none_on_detected_int8_fused_path"
            if backend["kind"] == "tensorwise_int8_fused_swiglu"
            else "conditional_unverified"
        ),
        "activation_proxy": _activation_proxy(
            rows["estimated_total_rows_without_text_or_video_refs"],
            int(chunk_rows),
            ffn,
            backend,
        ),
        "memory_safe_claim": False,
        "bit_exact_claim": False,
        "scientific_boundaries": [
            "Only the token-local DiT MLP path is chunked; global attention is unchanged.",
            "The proxy excludes attention workspaces, QKV, final projection, VAE, CLIP, weights, allocator fragmentation and other processes.",
            "Current TensorWise INT8 H3 folds SwiGLU into the down-projection quantizer; on that path the theoretical full-fc1 proxy is inapplicable and material savings may be zero.",
            "Different GEMM row shapes can change floating-point rounding even though the token-local formula is unchanged.",
            "Existing dit/double_block replacements are rejected rather than overwritten or silently reordered.",
        ],
    }
    if not contract.get("supported"):
        report["status"] = "unsupported_core"
        if mode == "apply_exp":
            raise RuntimeError(
                "MiniMax H3 Activation Chunk refuses this unknown ComfyUI H3 source contract"
            )
        return model, report
    if conflicts:
        report["status"] = "block_replace_conflict"
        if mode == "apply_exp":
            raise RuntimeError(
                "MiniMax H3 Activation Chunk conflicts with existing dit/double_block replacements: "
                + repr(conflicts)
            )
        return model, report
    if mode == "report_only":
        report["status"] = "report_only"
        return model, report

    cloned = model.clone()
    for block_index in range(block_start, block_end + 1):
        cloned.set_model_patch_replace(
            H3MLPActivationChunkPatch(block_index, chunk_rows, preserve_short_path),
            "dit",
            "double_block",
            block_index,
        )
    attachment = {
        "schema": ACTIVATION_CHUNK_SCHEMA,
        "chunk_rows": int(chunk_rows),
        "block_start": int(block_start),
        "block_end": int(block_end),
        "core_hashes": contract["hashes"],
    }
    setter = getattr(cloned, "set_attachments", None)
    if callable(setter):
        setter(ATTACHMENT_KEY, attachment)
    else:
        attachments = getattr(cloned, "attachments", None)
        if not isinstance(attachments, dict):
            raise RuntimeError("MODEL clone does not expose attachment storage")
        attachments[ATTACHMENT_KEY] = attachment
    report["status"] = "applied_exp"
    report["applied"] = True
    report["attachment"] = attachment
    return cloned, report
