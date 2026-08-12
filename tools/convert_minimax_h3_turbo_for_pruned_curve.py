#!/usr/bin/env python3
"""Adapt a LarryVrh MiniMax-H3 Turbo LoRA to an H3 curve-pruned checkpoint.

The curve-pruned H3 checkpoints replace the 2688-wide time embedding consumed by
51 AdaLN projections with an 8-wide sampled curve basis.  This tool keeps the
208 attention/MLP adapters bit-identical and converts each AdaLN adapter with a
1025-point affine least-squares projection::

    S @ A.T ~= C @ A8.T + c
    B @ (S @ A.T) ~= B @ (C @ A8.T) + diff_b

where ``S`` is the full checkpoint's post-SiLU time-embedding curve, ``C`` is
the pruned checkpoint's ``adaln_t_table``, ``A8`` is stored as BF16, and
``diff_b = B @ c`` is stored as FP32.  The intercept is mandatory; omitting it
destroys most of the AdaLN response.

The source LoRA, target checkpoint, and full reference checkpoint are read-only.
Final outputs are never overwritten and are published from same-directory
``.partial`` files only after validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


COMFY_PREFIX = "diffusion_model."
CURVE_KEY = "adaln_t_table"
GRID_SIZE = 1025
CURVE_WIDTH = 8
FULL_TIME_WIDTH = 2688
FREQ_WIDTH = 256
TIME_HIDDEN_WIDTH = 5376
SOURCE_TENSOR_COUNT = 518
SOURCE_ADAPTER_COUNT = 259
DIRECT_ADAPTER_COUNT = 208
PROJECTED_ADALN_COUNT = 51
CURVE_TENSOR_COUNT = 569
CORE_TENSOR_COUNT = 416

TIME_KEYS = {
    "proj_in_weight": "time_embedder.proj_in.weight",
    "proj_in_bias": "time_embedder.proj_in.bias",
    "proj_out_weight": "time_embedder.proj_out.weight",
    "proj_out_bias": "time_embedder.proj_out.bias",
}

KEY_RE = re.compile(
    r"^diffusion_model\."
    r"(?P<module>(?:blocks\.\d+|token_refiner\.blocks\.\d+)\."
    r"(?:adaln_proj\.linear|attn\.(?:qkv_proj|out_proj)|mlp\.fc[12])|"
    r"final_layer\.adaln_proj\.linear)\.lora_(?P<side>[AB])\.weight$"
)

MAIN_SHAPES = {
    "adaln_proj.linear": ((16, 2688), (96768, 16)),
    "attn.qkv_proj": ((64, 5376), (21504, 64)),
    "attn.out_proj": ((64, 7168), (5376, 64)),
    "mlp.fc1": ((64, 5376), (28672, 64)),
    "mlp.fc2": ((64, 14336), (5376, 64)),
}
REFINER_SHAPES = {
    key: value for key, value in MAIN_SHAPES.items() if key != "adaln_proj.linear"
}
FINAL_SHAPES = ((16, 2688), (10752, 16))


@dataclass(frozen=True)
class ProjectionMetric:
    module: str
    fp64_dense_relative_error: float
    stored_dense_relative_error: float
    stored_native4_relative_error: float


@dataclass(frozen=True)
class ProjectionSummary:
    fp64_dense_min: float
    fp64_dense_median: float
    fp64_dense_max: float
    stored_dense_min: float
    stored_dense_median: float
    stored_dense_max: float
    stored_dense_aggregate: float
    stored_native4_aggregate: float
    stored_native4_module_max: float


def expected_modules() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    modules: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for block in range(50):
        for name, shapes in MAIN_SHAPES.items():
            modules[f"blocks.{block}.{name}"] = shapes
    for block in range(2):
        for name, shapes in REFINER_SHAPES.items():
            modules[f"token_refiner.blocks.{block}.{name}"] = shapes
    modules["final_layer.adaln_proj.linear"] = FINAL_SHAPES
    return modules


EXPECTED_MODULES = expected_modules()
ADALN_MODULES = tuple(
    sorted(
        module for module in EXPECTED_MODULES if module.endswith("adaln_proj.linear")
    )
)
DIRECT_MODULES = tuple(sorted(set(EXPECTED_MODULES) - set(ADALN_MODULES)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_raw_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def require_hash(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def module_key(module: str, suffix: str) -> str:
    return f"{COMFY_PREFIX}{module}.{suffix}"


def validate_source_lora(state: dict[str, torch.Tensor]) -> None:
    if len(state) != SOURCE_TENSOR_COUNT:
        raise ValueError(
            f"source LoRA: expected {SOURCE_TENSOR_COUNT} tensors, found {len(state)}"
        )
    found: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        match = KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"source LoRA: unexpected key {key}")
        module = match.group("module")
        side = match.group("side")
        if module not in EXPECTED_MODULES:
            raise ValueError(f"source LoRA: unexpected H3 module {module}")
        if side in found.setdefault(module, {}):
            raise ValueError(f"source LoRA: duplicate side {side} for {module}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"source LoRA: {key} must be BF16, found {tensor.dtype}")
        found[module][side] = tensor
    if set(found) != set(EXPECTED_MODULES):
        missing = sorted(set(EXPECTED_MODULES) - set(found))
        extra = sorted(set(found) - set(EXPECTED_MODULES))
        raise ValueError(
            f"source LoRA module mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for module, pair in found.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"source LoRA: {module} is not a complete A/B pair")
        expected_a, expected_b = EXPECTED_MODULES[module]
        if tuple(pair["A"].shape) != expected_a or tuple(pair["B"].shape) != expected_b:
            raise ValueError(
                f"source LoRA: {module} expected A{expected_a}/B{expected_b}, "
                f"found A{tuple(pair['A'].shape)}/B{tuple(pair['B'].shape)}"
            )


def read_source_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    required = {
        "base_model": "MiniMax-H3",
        "sampler_steps": "4",
        "application": "W_eff = W + lora_B @ lora_A",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"source LoRA metadata {key!r}: expected {value!r}, found {metadata.get(key)!r}"
            )
    return metadata


def slice_shape(handle, key: str) -> tuple[int, ...]:
    return tuple(handle.get_slice(key).get_shape())


def validate_pruned_model(path: Path) -> tuple[torch.Tensor, dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if CURVE_KEY not in keys:
            raise ValueError(
                f"target model does not contain {CURVE_KEY}; it is not an H3 curve-pruned checkpoint"
            )
        if slice_shape(handle, CURVE_KEY) != (GRID_SIZE, CURVE_WIDTH):
            raise ValueError(
                f"target {CURVE_KEY}: expected {(GRID_SIZE, CURVE_WIDTH)}, found {slice_shape(handle, CURVE_KEY)}"
            )
        curve = handle.get_tensor(CURVE_KEY).float().cpu().contiguous()
        for module, (source_a_shape, source_b_shape) in EXPECTED_MODULES.items():
            weight_key = f"{module}.weight"
            bias_key = f"{module}.bias"
            if weight_key not in keys:
                raise ValueError(f"target model is missing {weight_key}")
            target_shape = slice_shape(handle, weight_key)
            expected_shape = (
                source_b_shape[0],
                CURVE_WIDTH if module in ADALN_MODULES else source_a_shape[1],
            )
            if target_shape != expected_shape:
                raise ValueError(
                    f"target {weight_key}: expected {expected_shape}, found {target_shape}"
                )
            if module in ADALN_MODULES:
                if bias_key not in keys:
                    raise ValueError(f"target model is missing AdaLN bias {bias_key}")
                if slice_shape(handle, bias_key) != (source_b_shape[0],):
                    raise ValueError(
                        f"target {bias_key}: expected {(source_b_shape[0],)}, found {slice_shape(handle, bias_key)}"
                    )
        metadata = dict(handle.metadata() or {})
    return curve, metadata


def load_time_reference(path: Path) -> dict[str, torch.Tensor]:
    expected_shapes = {
        "proj_in_weight": (TIME_HIDDEN_WIDTH, FREQ_WIDTH),
        "proj_in_bias": (TIME_HIDDEN_WIDTH,),
        "proj_out_weight": (FULL_TIME_WIDTH, TIME_HIDDEN_WIDTH),
        "proj_out_bias": (FULL_TIME_WIDTH,),
    }
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name, key in TIME_KEYS.items():
            if key not in keys:
                raise ValueError(f"time reference is missing {key}")
            if slice_shape(handle, key) != expected_shapes[name]:
                raise ValueError(
                    f"time reference {key}: expected {expected_shapes[name]}, found {slice_shape(handle, key)}"
                )
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"time reference {key}: expected FP32, found {tensor.dtype}"
                )
            tensors[name] = tensor.cpu().contiguous()
    return tensors


def full_adaln_inputs(
    t: torch.Tensor, reference: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Reproduce ComfyUI H3 TimeEmbedder plus AdalnProj's second SiLU."""
    t32 = t.to(dtype=torch.float32, device="cpu").reshape(-1)
    half = FREQ_WIDTH // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    arguments = t32[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    hidden = F.silu(
        F.linear(embedding, reference["proj_in_weight"], reference["proj_in_bias"])
    )
    time_embedding = F.linear(
        hidden, reference["proj_out_weight"], reference["proj_out_bias"]
    )
    return F.silu(time_embedding).contiguous()


def interpolate_curve(curve: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t32 = t.to(dtype=torch.float32, device="cpu").reshape(-1)
    position = t32.clamp(0.0, 1.0) * (curve.shape[0] - 1)
    lower = position.floor().long().clamp(max=curve.shape[0] - 2)
    return torch.lerp(curve[lower], curve[lower + 1], (position - lower).unsqueeze(1))


def native_four_step_times() -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, 5, dtype=torch.float32)[:-1]

    def shifted_sigma(shift: float) -> torch.Tensor:
        return shift * base / (1.0 + (shift - 1.0) * base)

    values = torch.cat(
        [
            1.0 - shifted_sigma(12.0),
            1.0 - shifted_sigma(3.0),
            torch.tensor([0.999, 1.0], dtype=torch.float32),
        ]
    )
    return torch.unique(values, sorted=True)


def output_error_squared(
    target_hidden: torch.Tensor,
    predicted_without_bias: torch.Tensor,
    b_matrix: torch.Tensor,
    diff_bias: torch.Tensor,
) -> tuple[float, float]:
    """Return exact output-space squared error and target energy without huge outputs."""
    y = target_hidden.to(torch.float64)
    h = y - predicted_without_bias.to(torch.float64)
    b = b_matrix.to(torch.float64)
    d = diff_bias.to(torch.float64)
    gram = b.T @ b
    error = torch.sum((h.T @ h) * gram.T)
    cross = torch.sum(h, dim=0) @ (b.T @ d)
    error = error - 2.0 * cross + h.shape[0] * torch.dot(d, d)
    energy = torch.sum((y.T @ y) * gram.T)
    return max(float(error), 0.0), float(energy)


def weighted_hidden_error_squared(
    error_hidden: torch.Tensor, target_hidden: torch.Tensor, b: torch.Tensor
) -> tuple[float, float]:
    b64 = b.to(torch.float64)
    gram = b64.T @ b64
    error = error_hidden.to(torch.float64)
    target = target_hidden.to(torch.float64)
    numerator = torch.sum((error.T @ error) * gram.T)
    denominator = torch.sum((target.T @ target) * gram.T)
    return max(float(numerator), 0.0), float(denominator)


def relative_error(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("projection target has zero output energy")
    return math.sqrt(numerator / denominator)


def summarize_metrics(
    metrics: Iterable[ProjectionMetric],
    dense_totals: tuple[float, float],
    native_totals: tuple[float, float],
) -> ProjectionSummary:
    rows = list(metrics)
    if len(rows) != PROJECTED_ADALN_COUNT:
        raise ValueError(
            f"expected {PROJECTED_ADALN_COUNT} projection metrics, found {len(rows)}"
        )

    def triplet(values: list[float]) -> tuple[float, float, float]:
        ordered = sorted(values)
        return ordered[0], ordered[len(ordered) // 2], ordered[-1]

    fp64 = triplet([row.fp64_dense_relative_error for row in rows])
    stored = triplet([row.stored_dense_relative_error for row in rows])
    return ProjectionSummary(
        fp64_dense_min=fp64[0],
        fp64_dense_median=fp64[1],
        fp64_dense_max=fp64[2],
        stored_dense_min=stored[0],
        stored_dense_median=stored[1],
        stored_dense_max=stored[2],
        stored_dense_aggregate=relative_error(*dense_totals),
        stored_native4_aggregate=relative_error(*native_totals),
        stored_native4_module_max=max(
            row.stored_native4_relative_error for row in rows
        ),
    )


def project_lora(
    source: dict[str, torch.Tensor],
    curve: torch.Tensor,
    reference: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[ProjectionMetric], ProjectionSummary]:
    grid_t = torch.arange(GRID_SIZE, dtype=torch.float32) / float(GRID_SIZE - 1)
    full_dense = full_adaln_inputs(grid_t, reference).to(torch.float64)
    curve_dense = curve.to(torch.float64)
    design = torch.cat(
        [curve_dense, torch.ones(GRID_SIZE, 1, dtype=torch.float64)], dim=1
    )
    projection = torch.linalg.pinv(design) @ full_dense

    native_t = native_four_step_times()
    full_native = full_adaln_inputs(native_t, reference).to(torch.float64)
    curve_native = interpolate_curve(curve, native_t).to(torch.float64)

    output = dict(source)
    metrics: list[ProjectionMetric] = []
    dense_num_total = dense_den_total = 0.0
    native_num_total = native_den_total = 0.0

    for module in ADALN_MODULES:
        a_key = module_key(module, "lora_A.weight")
        b_key = module_key(module, "lora_B.weight")
        diff_key = module_key(module, "diff_b")
        source_a = source[a_key]
        source_b = source[b_key]
        a64 = source_a.to(torch.float64)
        b64 = source_b.to(torch.float64)

        coefficients = projection @ a64.T
        a8 = coefficients[:CURVE_WIDTH].T.contiguous()
        intercept = coefficients[CURVE_WIDTH].contiguous()
        stored_a8 = a8.to(torch.bfloat16).contiguous()
        stored_diff = (b64 @ intercept).to(torch.float32).contiguous()
        output[a_key] = stored_a8
        output[diff_key] = stored_diff

        dense_target = full_dense @ a64.T
        fp64_error = dense_target - (curve_dense @ a8.T + intercept)
        fp64_num, fp64_den = weighted_hidden_error_squared(
            fp64_error, dense_target, source_b
        )

        dense_prediction = curve_dense @ stored_a8.to(torch.float64).T
        dense_num, dense_den = output_error_squared(
            dense_target, dense_prediction, source_b, stored_diff
        )
        native_target = full_native @ a64.T
        native_prediction = curve_native @ stored_a8.to(torch.float64).T
        native_num, native_den = output_error_squared(
            native_target, native_prediction, source_b, stored_diff
        )

        dense_num_total += dense_num
        dense_den_total += dense_den
        native_num_total += native_num
        native_den_total += native_den
        metrics.append(
            ProjectionMetric(
                module=module,
                fp64_dense_relative_error=relative_error(fp64_num, fp64_den),
                stored_dense_relative_error=relative_error(dense_num, dense_den),
                stored_native4_relative_error=relative_error(native_num, native_den),
            )
        )

    if len(output) != CURVE_TENSOR_COUNT:
        raise ValueError(
            f"curve output: expected {CURVE_TENSOR_COUNT} tensors, found {len(output)}"
        )
    summary = summarize_metrics(
        metrics,
        (dense_num_total, dense_den_total),
        (native_num_total, native_den_total),
    )
    return output, metrics, summary


def core208_state(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {
        key: value
        for key, value in source.items()
        if KEY_RE.fullmatch(key) is not None
        and KEY_RE.fullmatch(key).group("module") in DIRECT_MODULES
    }
    if len(output) != CORE_TENSOR_COUNT:
        raise ValueError(
            f"core208 output: expected {CORE_TENSOR_COUNT} tensors, found {len(output)}"
        )
    return output


def validate_curve_output(
    output: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
) -> None:
    if len(output) != CURVE_TENSOR_COUNT:
        raise ValueError(
            f"curve output: expected {CURVE_TENSOR_COUNT} tensors, found {len(output)}"
        )
    for module in DIRECT_MODULES:
        for side in ("A", "B"):
            key = module_key(module, f"lora_{side}.weight")
            if not torch.equal(output[key], source[key]):
                raise ValueError(f"direct adapter changed: {key}")
    for module in ADALN_MODULES:
        a_key = module_key(module, "lora_A.weight")
        b_key = module_key(module, "lora_B.weight")
        diff_key = module_key(module, "diff_b")
        expected_out = EXPECTED_MODULES[module][1][0]
        if (
            tuple(output[a_key].shape) != (16, CURVE_WIDTH)
            or output[a_key].dtype != torch.bfloat16
        ):
            raise ValueError(f"projected A contract failed: {a_key}")
        if not torch.equal(output[b_key], source[b_key]):
            raise ValueError(f"AdaLN B changed: {b_key}")
        if (
            tuple(output[diff_key].shape) != (expected_out,)
            or output[diff_key].dtype != torch.float32
        ):
            raise ValueError(f"projected diff_b contract failed: {diff_key}")


def validate_core_output(
    output: dict[str, torch.Tensor], source: dict[str, torch.Tensor]
) -> None:
    if len(output) != CORE_TENSOR_COUNT:
        raise ValueError(
            f"core208 output: expected {CORE_TENSOR_COUNT} tensors, found {len(output)}"
        )
    for key, tensor in output.items():
        if not torch.equal(tensor, source[key]):
            raise ValueError(f"core208 tensor changed: {key}")


def metadata_for_curve(
    source_metadata: dict[str, str],
    args: argparse.Namespace,
    hashes: dict[str, str],
    table_hash: str,
    summary: ProjectionSummary,
    comfy_commit: str,
) -> dict[str, str]:
    metadata = dict(source_metadata)
    # The source Turbo LoRA correctly rejects pruned bases because its AdaLN A
    # tensors consume 2688 inputs.  This converter replaces those tensors with
    # exact-target 8-wide projections, so inheriting that source warning would
    # contradict the compatibility fields written below.
    metadata.pop("incompatible_base", None)
    metadata.update(
        {
            "base_model": "MiniMax-H3",
            "sampler_steps": "4",
            "application": "W_eff = W + lora_B @ lora_A; bias_eff = bias + diff_b",
            "comfyui_key_prefix": COMFY_PREFIX,
            "comfyui_loader": "Load LoRA (Bypass, Model Only) (for debugging)",
            "compatible_base": f"exact curve-pruned checkpoint SHA256 {hashes['pruned_model']}",
            "compatibility_scope": "exact_checkpoint_sha256_only",
            "conversion_tool": Path(__file__).name,
            "conversion_algorithm": "adaln_curve_affine_lstsq_pinv1025_with_diff_b",
            "conversion_source_file": args.lora.name,
            "conversion_source_sha256": hashes["lora"],
            "compatible_main_file": args.pruned_model.name,
            "compatible_main_sha256": hashes["pruned_model"],
            "time_embedder_reference_file": args.time_embedder_reference.name,
            "time_embedder_reference_sha256": hashes["time_reference"],
            "adaln_t_table_raw_sha256": table_hash,
            "curve_grid_size": str(GRID_SIZE),
            "curve_width": str(CURVE_WIDTH),
            "adapter_count": str(SOURCE_ADAPTER_COUNT),
            "direct_adapter_count": str(DIRECT_ADAPTER_COUNT),
            "projected_adaln_count": str(PROJECTED_ADALN_COUNT),
            "bias_diff_count": str(PROJECTED_ADALN_COUNT),
            "tensor_count": str(CURVE_TENSOR_COUNT),
            "projected_A_dtype": "bfloat16",
            "diff_b_dtype": "float32",
            "stored_dense_aggregate_relative_error": f"{summary.stored_dense_aggregate:.12g}",
            "stored_native4_aggregate_relative_error": f"{summary.stored_native4_aggregate:.12g}",
            "stored_native4_module_max_relative_error": f"{summary.stored_native4_module_max:.12g}",
            "comfyui_commit": comfy_commit,
            "validation_status": "static_projection_validated; perceptual_render_pending",
        }
    )
    return metadata


def metadata_for_core(
    source_metadata: dict[str, str],
    args: argparse.Namespace,
    hashes: dict[str, str],
    table_hash: str,
    comfy_commit: str,
) -> dict[str, str]:
    metadata = dict(source_metadata)
    # The 51 incompatible full-width AdaLN adapters are absent from core208.
    # Do not carry their source-file incompatibility warning into this output.
    metadata.pop("incompatible_base", None)
    metadata.update(
        {
            "base_model": "MiniMax-H3",
            "sampler_steps": "4",
            "application": "W_eff = W + lora_B @ lora_A",
            "comfyui_key_prefix": COMFY_PREFIX,
            "comfyui_loader": "Load LoRA (Bypass, Model Only) (for debugging)",
            "compatible_base": f"exact curve-pruned checkpoint SHA256 {hashes['pruned_model']}",
            "compatibility_scope": "exact_checkpoint_sha256_only",
            "conversion_tool": Path(__file__).name,
            "conversion_algorithm": "core208_adaln_ablation",
            "conversion_source_file": args.lora.name,
            "conversion_source_sha256": hashes["lora"],
            "compatible_main_file": args.pruned_model.name,
            "compatible_main_sha256": hashes["pruned_model"],
            "adaln_t_table_raw_sha256": table_hash,
            "adapter_count": str(DIRECT_ADAPTER_COUNT),
            "direct_adapter_count": str(DIRECT_ADAPTER_COUNT),
            "removed_adaln_count": str(PROJECTED_ADALN_COUNT),
            "tensor_count": str(CORE_TENSOR_COUNT),
            "comfyui_commit": comfy_commit,
            "validation_status": "static_ablation_validated; perceptual_render_pending",
        }
    )
    return metadata


def temporary_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.{uuid.uuid4().hex}.partial")


def publish_safetensors_no_overwrite(
    final: Path,
    state: dict[str, torch.Tensor],
    metadata: dict[str, str],
    validator,
) -> str:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = temporary_path(final)
    try:
        save_file(state, partial, metadata=metadata)
        loaded = load_file(partial, device="cpu")
        validator(loaded)
        with safe_open(partial, framework="pt", device="cpu") as handle:
            actual_metadata = dict(handle.metadata() or {})
        if actual_metadata != metadata:
            raise ValueError(f"metadata read-back mismatch for {final.name}")
        output_hash = sha256_file(partial)
        if final.exists():
            raise FileExistsError(
                f"output appeared during conversion; refusing publish: {final}"
            )
        os.rename(partial, final)
        return output_hash
    finally:
        if partial.exists():
            partial.unlink()


def publish_json_no_overwrite(final: Path, payload: dict) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = temporary_path(final)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(partial.read_text(encoding="utf-8"))
        if final.exists():
            raise FileExistsError(
                f"manifest appeared during conversion; refusing publish: {final}"
            )
        os.rename(partial, final)
    finally:
        if partial.exists():
            partial.unlink()


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def check_output_paths(paths: Iterable[Path]) -> None:
    resolved: set[Path] = set()
    for path in paths:
        normalized = path.resolve()
        if normalized in resolved:
            raise ValueError(f"duplicate output path: {path}")
        resolved.add(normalized)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def run(args: argparse.Namespace) -> dict:
    inputs = [args.lora, args.pruned_model, args.time_embedder_reference]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = [args.output, manifest_path(args.output)]
    if args.core208_output is not None:
        outputs.extend([args.core208_output, manifest_path(args.core208_output)])
    check_output_paths(outputs)

    expected_hashes = {
        "lora": args.expected_lora_sha256,
        "pruned_model": args.expected_pruned_model_sha256,
        "time_reference": args.expected_time_reference_sha256,
    }
    hashes = {
        "lora": require_hash(args.lora, expected_hashes["lora"], "source LoRA"),
        "pruned_model": require_hash(
            args.pruned_model, expected_hashes["pruned_model"], "pruned model"
        ),
        "time_reference": require_hash(
            args.time_embedder_reference,
            expected_hashes["time_reference"],
            "time-embedder reference",
        ),
    }

    source_metadata = read_source_metadata(args.lora)
    source = load_file(args.lora, device="cpu")
    validate_source_lora(source)
    curve, target_metadata = validate_pruned_model(args.pruned_model)
    table_hash = tensor_raw_sha256(curve)
    if (
        args.expected_table_sha256 is not None
        and table_hash.lower() != args.expected_table_sha256.lower()
    ):
        raise ValueError(
            f"{CURVE_KEY} raw SHA-256 mismatch: expected {args.expected_table_sha256}, found {table_hash}"
        )
    reference = load_time_reference(args.time_embedder_reference)
    curve_state, metrics, projection_summary = project_lora(source, curve, reference)
    validate_curve_output(curve_state, source)

    comfy_commit = args.comfyui_commit or "unknown"
    curve_metadata = metadata_for_curve(
        source_metadata,
        args,
        hashes,
        table_hash,
        projection_summary,
        comfy_commit,
    )
    curve_hash = publish_safetensors_no_overwrite(
        args.output,
        curve_state,
        curve_metadata,
        lambda loaded: validate_curve_output(loaded, source),
    )
    curve_manifest = {
        "schema": "minimax_h3_pruned_curve_lora_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "curveproj1025_exp",
        "status": "static_projection_validated; perceptual_render_pending",
        "inputs": {
            "lora": {"path": str(args.lora.resolve()), "sha256": hashes["lora"]},
            "pruned_model": {
                "path": str(args.pruned_model.resolve()),
                "sha256": hashes["pruned_model"],
            },
            "time_embedder_reference": {
                "path": str(args.time_embedder_reference.resolve()),
                "sha256": hashes["time_reference"],
                "used_tensors": list(TIME_KEYS.values()),
            },
        },
        "target_metadata": target_metadata,
        "adaln_t_table": {"shape": [GRID_SIZE, CURVE_WIDTH], "raw_sha256": table_hash},
        "output": {
            "path": str(args.output.resolve()),
            "sha256": curve_hash,
            "tensor_count": CURVE_TENSOR_COUNT,
            "adapter_count": SOURCE_ADAPTER_COUNT,
            "direct_adapter_count": DIRECT_ADAPTER_COUNT,
            "projected_adaln_count": PROJECTED_ADALN_COUNT,
            "diff_b_count": PROJECTED_ADALN_COUNT,
        },
        "algorithm": {
            "name": "adaln_curve_affine_lstsq_pinv1025_with_diff_b",
            "grid": "t_j=j/1024, j=0..1024",
            "full_input": "SiLU(TimeEmbedder(t))",
            "solution": "pinv([adaln_t_table, 1]) @ full_input @ A.T",
            "stored_A_dtype": "bfloat16",
            "stored_diff_b_dtype": "float32",
        },
        "projection_summary": asdict(projection_summary),
        "module_metrics": [asdict(metric) for metric in metrics],
        "comfyui_commit": comfy_commit,
    }
    publish_json_no_overwrite(manifest_path(args.output), curve_manifest)

    core_result = None
    if args.core208_output is not None:
        core_state = core208_state(source)
        validate_core_output(core_state, source)
        core_metadata = metadata_for_core(
            source_metadata,
            args,
            hashes,
            table_hash,
            comfy_commit,
        )
        core_hash = publish_safetensors_no_overwrite(
            args.core208_output,
            core_state,
            core_metadata,
            lambda loaded: validate_core_output(loaded, source),
        )
        core_manifest = {
            "schema": "minimax_h3_pruned_curve_lora_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "core208_ablation",
            "status": "static_ablation_validated; perceptual_render_pending",
            "inputs": curve_manifest["inputs"],
            "adaln_t_table": curve_manifest["adaln_t_table"],
            "output": {
                "path": str(args.core208_output.resolve()),
                "sha256": core_hash,
                "tensor_count": CORE_TENSOR_COUNT,
                "adapter_count": DIRECT_ADAPTER_COUNT,
                "removed_adaln_count": PROJECTED_ADALN_COUNT,
            },
            "comfyui_commit": comfy_commit,
        }
        publish_json_no_overwrite(manifest_path(args.core208_output), core_manifest)
        core_result = core_manifest["output"]

    final_hashes = {
        "lora": require_hash(args.lora, hashes["lora"], "source LoRA after conversion"),
        "pruned_model": require_hash(
            args.pruned_model, hashes["pruned_model"], "pruned model after conversion"
        ),
        "time_reference": require_hash(
            args.time_embedder_reference,
            hashes["time_reference"],
            "time reference after conversion",
        ),
    }
    if final_hashes != hashes:
        raise ValueError("an input changed during conversion")
    return {
        "curve": curve_manifest["output"],
        "core208": core_result,
        "projection_summary": curve_manifest["projection_summary"],
        "input_hashes_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="existing 259-adapter ComfyUI Turbo LoRA",
    )
    parser.add_argument(
        "--pruned-model",
        type=Path,
        required=True,
        help="exact curve-pruned target checkpoint",
    )
    parser.add_argument(
        "--time-embedder-reference",
        type=Path,
        required=True,
        help="full non-pruned H3 checkpoint; only four FP32 time_embedder tensors are read",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new curve-projected acceleration LoRA",
    )
    parser.add_argument(
        "--core208-output",
        type=Path,
        help="optional 208-adapter AdaLN-free ablation LoRA",
    )
    parser.add_argument("--expected-lora-sha256")
    parser.add_argument("--expected-pruned-model-sha256")
    parser.add_argument("--expected-time-reference-sha256")
    parser.add_argument("--expected-table-sha256")
    parser.add_argument("--comfyui-commit")
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
