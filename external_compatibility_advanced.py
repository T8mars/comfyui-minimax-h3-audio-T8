from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import struct
import tomllib
from typing import Any

import torch


CLIPPROJ_MIN_VERSION = (0, 1, 13)
SOL_ATTN_MIN_VERSION = (0, 6, 2)
SOL_ARCHES = {(8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)}
EXPECTED_H3_BLOCKS = 50


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    match = re.search(r"(\d+(?:\.\d+)+)", str(value))
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _custom_nodes_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_plugin_version(path: Path) -> str | None:
    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            value = data.get("project", {}).get("version")
            if value:
                return str(value)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            pass
    for name in ("README.md", "README_ZH.md"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            head = candidate.read_text(encoding="utf-8")[:12000]
        except (OSError, UnicodeDecodeError):
            continue
        match = re.search(
            r"(?im)^\s*(?:\*{1,2}|_{1,2})?\s*"
            r"(?:version\s*[:：]?|版本\s*[:：])\s*v?(\d+(?:\.\d+)+)",
            head,
        )
        if match:
            return match.group(1)
    return None


def _discover_plugin(kind: str, root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else _custom_nodes_root()
    signatures = {
        "clipproj": ("clipproj_nodes.py", "ClipProjApply"),
        "sol_attn": ("minimax.py", "MiniMaxH3ScheduledSolAttentionPatch"),
    }
    source_name, required_symbol = signatures[kind]
    candidates: list[Path] = []
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda value: value.name.lower()):
            if child.is_dir() and (child / source_name).is_file():
                candidates.append(child)
    matches = []
    for candidate in candidates:
        source = candidate / source_name
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        if required_symbol not in text:
            continue
        matches.append(
            {
                "path": str(candidate),
                "version": _read_plugin_version(candidate),
                "required_symbol_present": True,
            }
        )
    return {
        "kind": kind,
        "custom_nodes_root": str(root),
        "installed": len(matches) == 1,
        "ambiguous": len(matches) > 1,
        "matches": matches,
    }


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("projection file is shorter than a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length <= 2 or length > 128 * 1024 * 1024:
            raise ValueError("projection safetensors header length is invalid")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("projection safetensors header is truncated")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("projection safetensors header is not an object")
    return result


def _named_shape(header: Mapping[str, Any], suffix: str) -> list[int] | None:
    for key, value in header.items():
        if key == "__metadata__" or not (key == suffix or key.endswith("." + suffix)):
            continue
        shape = value.get("shape") if isinstance(value, Mapping) else None
        if isinstance(shape, list) and all(isinstance(item, int) for item in shape):
            return shape
    return None


def _projection_contract(path: Path) -> dict[str, Any]:
    header = _safetensors_header(path)
    mean_in = _named_shape(header, "mean_in")
    mean_out = _named_shape(header, "mean_out")
    matrix = _named_shape(header, "W")
    source_dim = mean_in[0] if mean_in else matrix[0] if matrix and len(matrix) == 2 else None
    output_dim = mean_out[0] if mean_out else matrix[1] if matrix and len(matrix) == 2 else None
    metadata = header.get("__metadata__", {})
    return {
        "path": str(path),
        "source_dim": source_dim,
        "output_dim": output_dim,
        "has_linear_matrix": matrix is not None,
        "has_residual_weights": any(
            key != "__metadata__" and ("mlp" in key.lower() or "residual" in key.lower())
            for key in header
        ),
        "tensor_count": sum(key != "__metadata__" for key in header),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _runtime_projection_contract(clip: Any) -> dict[str, Any]:
    namespace = getattr(clip, "__dict__", {})
    projection = namespace.get("_proj") if isinstance(namespace, Mapping) else None
    source_dim = output_dim = None
    if isinstance(projection, Mapping):
        mean_in = projection.get("mean_in")
        mean_out = projection.get("mean_out")
        matrix = projection.get("W")
        source_dim = int(mean_in.shape[0]) if torch.is_tensor(mean_in) else None
        output_dim = int(mean_out.shape[0]) if torch.is_tensor(mean_out) else None
        if torch.is_tensor(matrix) and matrix.ndim == 2:
            source_dim = source_dim or int(matrix.shape[0])
            output_dim = output_dim or int(matrix.shape[1])
    cls = type(clip)
    return {
        "class": cls.__name__,
        "module": cls.__module__,
        "projected_wrapper": cls.__name__ == "ProjectedCLIP" or "clipproj" in cls.__module__.lower(),
        "projection_name": namespace.get("_proj_name") if isinstance(namespace, Mapping) else None,
        "source_dim": source_dim,
        "output_dim": output_dim,
    }


def audit_clipproj_compatibility(
    clip: Any,
    encoder_family: str,
    encoder_architecture: str,
    encoder_quantization: str,
    load_mode: str,
    projection_path: str,
    has_reference_images: bool,
    has_reference_videos: bool,
    custom_nodes_root: Path | None = None,
) -> tuple[Any, bool, str, str, int, int, str]:
    hard: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def finding(target, code, message, **evidence):
        target.append({"code": code, "message": message, "evidence": evidence})

    plugin = _discover_plugin("clipproj", custom_nodes_root)
    if not plugin["installed"]:
        finding(
            hard,
            "clipproj_plugin_missing_or_ambiguous",
            "Exactly one compatible ComfyUI-ClipProj installation was not found.",
            matches=plugin["matches"],
        )
        plugin_version = None
    else:
        plugin_version = plugin["matches"][0]["version"]
        if _version_tuple(plugin_version) < CLIPPROJ_MIN_VERSION:
            finding(
                hard,
                "clipproj_version_too_old",
                "ClipProj 0.1.13 or newer is required for v3 matrices and the reviewed visual/int8 path.",
                version=plugin_version,
            )

    runtime = _runtime_projection_contract(clip)
    if not runtime["projected_wrapper"]:
        finding(
            hard,
            "clipproj_wrapper_not_connected",
            "The connected CLIP is not a recognized ClipProj ProjectedCLIP wrapper.",
            runtime_class=runtime["class"],
            runtime_module=runtime["module"],
        )
    expected_dims = {"auto": None, "4B": 2560, "8B": 4096, "32B": 5120}
    if encoder_family not in expected_dims:
        raise ValueError(f"unsupported ClipProj encoder_family: {encoder_family!r}")
    expected_source = expected_dims[encoder_family]
    if encoder_architecture == "text_only_qwen3":
        finding(
            hard,
            "text_only_encoder_rejected",
            "A text-only Qwen3 can match the hidden width but has no valid MiniMax H3 vision path.",
        )
    elif encoder_architecture == "unknown":
        finding(
            warnings,
            "encoder_architecture_unverified",
            "The encoder was not positively identified as Qwen3-VL.",
        )
    elif encoder_architecture != "qwen3_vl":
        raise ValueError(f"unsupported encoder_architecture: {encoder_architecture!r}")

    projection = None
    resolved_projection = None
    if projection_path.strip():
        requested = Path(projection_path.strip())
        discovery_root = Path(custom_nodes_root) if custom_nodes_root is not None else _custom_nodes_root()
        resolved_projection = requested if requested.is_absolute() else discovery_root.parent / "models" / "clip_projections" / requested
        if not resolved_projection.is_file():
            finding(
                hard,
                "projection_file_missing",
                "The declared projection file does not exist.",
                path=str(resolved_projection),
            )
        else:
            try:
                projection = _projection_contract(resolved_projection)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                finding(
                    hard,
                    "projection_header_invalid",
                    "The projection safetensors header could not be validated.",
                    path=str(resolved_projection),
                    error=f"{type(error).__name__}: {error}",
                )
    elif runtime["projection_name"] is None:
        finding(
            warnings,
            "projection_file_not_declared",
            "No projection filename was supplied; only the connected runtime wrapper can be inspected.",
        )

    source_dim = (
        projection.get("source_dim") if projection else None
    ) or runtime["source_dim"] or 0
    output_dim = (
        projection.get("output_dim") if projection else None
    ) or runtime["output_dim"] or 0
    if expected_source is not None and source_dim and source_dim != expected_source:
        finding(
            hard,
            "encoder_projection_dimension_mismatch",
            "The selected encoder family and projection input dimensions do not match.",
            expected=expected_source,
            observed=source_dim,
        )
    if output_dim and output_dim != 5120:
        finding(
            hard,
            "projection_output_dimension_mismatch",
            "MiniMax H3 conditioning requires a 5120-dimensional projection output.",
            observed=output_dim,
        )
    if source_dim == 0 or output_dim == 0:
        finding(
            warnings,
            "projection_dimensions_unverified",
            "Projection input/output dimensions could not be proven from the wrapper or file header.",
        )

    if load_mode == "clipproj_dynamic" and encoder_quantization == "int8_convrot":
        finding(
            hard,
            "int8_dynamic_loader_unsupported",
            "ClipProj's own dynamic loader path is not the reviewed int8 route; use stock Load CLIP + ClipProj Apply or resident mode.",
        )
    if load_mode == "clipproj_resident":
        finding(
            warnings,
            "resident_encoder_retains_vram",
            "Resident mode pins the encoder and is normally harmful on a single tight GPU.",
        )
    if load_mode not in {"stock_pageable", "clipproj_dynamic", "clipproj_resident"}:
        raise ValueError(f"unsupported ClipProj load_mode: {load_mode!r}")
    if encoder_quantization in {"gguf", "int4"}:
        finding(
            warnings,
            "community_quantization_unverified",
            "GGUF/int4 is community-reported only; reference-video and vision-tower behavior remains unverified upstream.",
        )
    if has_reference_videos:
        finding(
            warnings,
            "ref2va_projection_compounds_error",
            "Ref2VA reuses projected reference tokens at every sampling step and remains experimental upstream.",
        )
    if (has_reference_images or has_reference_videos) and encoder_architecture != "qwen3_vl":
        finding(
            hard,
            "visual_reference_requires_qwen3_vl",
            "Reference visuals require a positively identified Qwen3-VL encoder.",
        )

    report = {
        "schema": "t8.minimax_h3.clipproj_compatibility_audit.v1",
        "compatible": not hard,
        "decision": "PASS" if not hard else "ABSTAIN",
        "clip_passthrough_identity": True,
        "plugin": plugin,
        "minimum_plugin_version": ".".join(map(str, CLIPPROJ_MIN_VERSION)),
        "runtime": runtime,
        "declared": {
            "encoder_family": encoder_family,
            "encoder_architecture": encoder_architecture,
            "encoder_quantization": encoder_quantization,
            "load_mode": load_mode,
            "projection_path": str(resolved_projection) if resolved_projection else "",
            "has_reference_images": bool(has_reference_images),
            "has_reference_videos": bool(has_reference_videos),
        },
        "projection": projection,
        "source_dim": int(source_dim),
        "output_dim": int(output_dim),
        "hard_findings": hard,
        "warnings": warnings,
        "scientific_boundary": (
            "This bridge audits a separately installed ClipProj contract. It does not load, "
            "project, benchmark or claim equivalence to the native 32B encoder."
        ),
    }
    return (
        clip,
        not hard,
        report["decision"],
        str(plugin_version or "not_found"),
        int(source_dim),
        int(output_dim),
        _canonical_json(report),
    )


def _callable_identity(value: Any) -> dict[str, str]:
    function = getattr(value, "__func__", value)
    return {
        "name": str(getattr(function, "__name__", type(function).__name__)),
        "module": str(getattr(function, "__module__", type(function).__module__)),
    }


def _has_marker(value: Any, marker: str) -> bool:
    return hasattr(getattr(value, "__func__", value), marker)


def _hardware_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "compute_capability": None, "bf16_supported": False}
    index = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _model_total_blocks(model: Any) -> int | None:
    getter = getattr(model, "get_model_object", None)
    try:
        diffusion = getter("diffusion_model") if callable(getter) else model.model.diffusion_model
        blocks = getattr(diffusion, "blocks", None)
        return len(blocks) if blocks is not None else None
    except Exception:
        return None


def _parse_dense_blocks(spec: str, count: int) -> set[int]:
    result: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(
                f"cannot parse expected_dense_blocks entry {part!r}; use indices/ranges like '0-2,-1'"
            )
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        result.update(range(max(first, 0), min(last, count - 1) + 1))
    return result


def audit_sol_attn_compatibility(
    model: Any,
    intended_route: str,
    expected_dense_blocks: str,
    allow_unreviewed_composition: bool,
    custom_nodes_root: Path | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> tuple[Any, bool, str, str, int, str]:
    if intended_route not in {"h3_memory_efficient", "h3_scheduled", "generic_sol", "audit_only"}:
        raise ValueError(f"unsupported Sol-Attn intended_route: {intended_route!r}")
    hard: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def finding(target, code, message, **evidence):
        target.append({"code": code, "message": message, "evidence": evidence})

    plugin = _discover_plugin("sol_attn", custom_nodes_root)
    plugin_version = plugin["matches"][0]["version"] if plugin["installed"] else None
    if not plugin["installed"]:
        finding(
            hard,
            "sol_attn_plugin_missing_or_ambiguous",
            "Exactly one compatible ComfyUI-sol-attn installation was not found.",
            matches=plugin["matches"],
        )
    elif _version_tuple(plugin_version) < SOL_ATTN_MIN_VERSION:
        finding(
            hard,
            "sol_attn_version_too_old",
            "Sol-Attn 0.6.2 or newer is required for the reviewed SM86 and KJ handoff contracts.",
            version=plugin_version,
        )

    hardware_report = dict(hardware) if hardware is not None else _hardware_snapshot()
    raw_capability = hardware_report.get("compute_capability")
    capability = tuple(raw_capability) if isinstance(raw_capability, (list, tuple)) else None
    if not hardware_report.get("cuda_available"):
        finding(hard, "cuda_required", "Sol-Attn requires a supported NVIDIA CUDA GPU.")
    elif capability not in SOL_ARCHES:
        finding(
            hard,
            "compute_capability_unsupported",
            "The detected compute capability is outside Sol-Attn's declared H3 dispatch set.",
            capability=raw_capability,
            supported=sorted([list(value) for value in SOL_ARCHES]),
        )
    if not hardware_report.get("bf16_supported"):
        finding(hard, "bf16_required", "The H3 Sol kernel requires CUDA BF16 attention tensors.")

    object_patches = getattr(model, "object_patches", {})
    object_patches = object_patches if isinstance(object_patches, Mapping) else {}
    total_blocks = _model_total_blocks(model)
    attention: dict[int, Any] = {}
    full_blocks: dict[int, Any] = {}
    ffn = []
    for key, value in object_patches.items():
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r"diffusion_model\.blocks\.(\d+)\.attn\.forward", key)
        if match:
            attention[int(match.group(1))] = value
            continue
        match = re.fullmatch(r"diffusion_model\.blocks\.(\d+)\.forward", key)
        if match:
            full_blocks[int(match.group(1))] = value
            continue
        if key.endswith(".mlp.forward") and _has_marker(value, "_minimax_h3_ffn_fallback"):
            ffn.append(key)

    sol_blocks = sorted(
        index for index, value in attention.items() if _has_marker(value, "_minimax_h3_sol_fallback")
    )
    shadowing_attention = {
        index: _callable_identity(value)
        for index, value in attention.items()
        if not _has_marker(value, "_minimax_h3_sol_fallback")
    }
    fused_blocks = sorted(
        index for index, value in full_blocks.items() if _has_marker(value, "_minimax_h3_fusion_fallback")
    )
    unknown_full_blocks = {
        index: _callable_identity(value)
        for index, value in full_blocks.items()
        if not _has_marker(value, "_minimax_h3_fusion_fallback")
    }
    if intended_route in {"h3_memory_efficient", "h3_scheduled"}:
        expected = total_blocks or EXPECTED_H3_BLOCKS
        dense = _parse_dense_blocks(expected_dense_blocks, expected)
        expected_sol = sorted(set(range(expected)) - dense)
        if sol_blocks != expected_sol:
            finding(
                hard,
                "h3_sol_patch_incomplete_or_shadowed",
                "The outer MiniMax H3 Sol attention patch must own every non-dense H3 block.",
                total_blocks=total_blocks,
                expected_dense_blocks=sorted(dense),
                expected_sol_blocks=expected_sol,
                sol_blocks=sol_blocks,
                shadowing_attention=shadowing_attention,
            )
    elif sol_blocks:
        finding(
            warnings,
            "h3_sol_patch_present_outside_declared_route",
            "Direct H3 Sol patches are present although the declared route is not an H3 Sol route.",
            sol_blocks=sol_blocks,
        )

    model_options = getattr(model, "model_options", {})
    model_options = model_options if isinstance(model_options, Mapping) else {}
    transformer = model_options.get("transformer_options", {})
    transformer = transformer if isinstance(transformer, Mapping) else {}
    override = transformer.get("optimized_attention_override")
    override_identity = _callable_identity(override) if override is not None else None
    if intended_route == "generic_sol":
        text = _canonical_json(override_identity).lower() if override_identity else ""
        if "sol" not in text:
            finding(
                hard,
                "generic_sol_override_not_detected",
                "The connected MODEL does not expose a recognizable generic Sol attention override.",
                override=override_identity,
            )
    elif override is not None and sol_blocks:
        finding(
            warnings,
            "global_attention_used_as_fallback",
            "A global attention override is present under the complete direct H3 Sol patch; it is a fallback owner, not a second kernel on the same call.",
            override=override_identity,
        )

    patches_replace = transformer.get("patches_replace", {})
    dit_replacements = patches_replace.get("dit", {}) if isinstance(patches_replace, Mapping) else {}
    model_wrapper = model_options.get("model_function_wrapper")
    unreviewed = {
        "dit_replacement_count": len(dit_replacements) if isinstance(dit_replacements, Mapping) else -1,
        "model_function_wrapper": _callable_identity(model_wrapper) if model_wrapper is not None else None,
        "unknown_full_block_patches": unknown_full_blocks,
    }
    if any(
        [
            unreviewed["dit_replacement_count"] not in {0},
            model_wrapper is not None,
            bool(unknown_full_blocks),
        ]
    ):
        target = warnings if allow_unreviewed_composition else hard
        finding(
            target,
            "unreviewed_model_composition",
            "DiT replacement, whole-model wrapper or unknown full-block patches require a dedicated same-input combination audit.",
            **unreviewed,
        )

    if fused_blocks and len(fused_blocks) not in {total_blocks, EXPECTED_H3_BLOCKS}:
        finding(
            hard,
            "fused_modulation_patch_incomplete",
            "Fused modulation must cover the complete H3 block set or remain absent.",
            fused_blocks=fused_blocks,
            total_blocks=total_blocks,
        )
    if ffn:
        finding(
            warnings,
            "chunked_ffn_present",
            "Chunked FFN is token-local and independently composable, but its chunk count and memory benefit are not inferred by this audit.",
            patched_mlp_count=len(ffn),
        )

    report = {
        "schema": "t8.minimax_h3.sol_attn_compatibility_audit.v1",
        "compatible": not hard,
        "decision": "PASS" if not hard else "ABSTAIN",
        "model_passthrough_identity": True,
        "plugin": plugin,
        "minimum_plugin_version": ".".join(map(str, SOL_ATTN_MIN_VERSION)),
        "hardware": hardware_report,
        "intended_route": intended_route,
        "expected_dense_blocks": expected_dense_blocks,
        "total_blocks": total_blocks,
        "direct_h3_sol_blocks": sol_blocks,
        "shadowing_attention": shadowing_attention,
        "global_attention_override": override_identity,
        "fused_modulation_blocks": fused_blocks,
        "chunked_ffn_patch_count": len(ffn),
        "unreviewed_composition": unreviewed,
        "allow_unreviewed_composition": bool(allow_unreviewed_composition),
        "hard_findings": hard,
        "warnings": warnings,
        "required_order": (
            "UNET/LoRA -> optional global Sage -> optional KJ H3 Sage fallback -> "
            "one H3 Sol node -> optional Fused Modulation -> optional Chunk FFN -> cache -> sampler"
        ),
        "scientific_boundary": (
            "This bridge does not import, install or execute the Sol kernel and does not validate "
            "quality, speed, VRAM, audio or 16GB safety. It audits the connected patch ownership."
        ),
    }
    return (
        model,
        not hard,
        report["decision"],
        str(plugin_version or "not_found"),
        len(sol_blocks),
        _canonical_json(report),
    )
