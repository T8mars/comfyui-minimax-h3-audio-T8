#!/usr/bin/env python3
"""Bind a converted LightX2V MiniMax-H3 LoRA to an exact pruned checkpoint.

LightX2V's corrected ComfyUI LoRA contains only fused attention and MLP
adapters.  It has no full-width AdaLN adapter, so a curve-pruned H3 checkpoint
does not require a numerical projection.  This tool validates the exact
LightX2V alpha/rank contract and all 208 target shapes, then writes a new
safetensors header while copying the complete tensor data section byte for
byte.  The source LoRA and target checkpoint are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


COMFY_PREFIX = "diffusion_model."
CURVE_KEY = "adaln_t_table"
MAIN_BLOCKS = 50
REFINER_BLOCKS = 2
ADAPTER_COUNT = 208
TENSOR_COUNT = 624
WEIGHT_TENSOR_COUNT = 416
ALPHA_TENSOR_COUNT = 208
DEFAULT_COMFY_ROOT = Path(__file__).resolve().parents[3]

MODULE_SPECS = {
    "attn.qkv_proj": ((384, 5376), (21504, 384)),
    "attn.out_proj": ((128, 7168), (5376, 128)),
    "mlp.fc1": ((128, 5376), (28672, 128)),
    "mlp.fc2": ((128, 14336), (5376, 128)),
}


@dataclass(frozen=True)
class SourceProfile:
    name: str
    inference_steps: int
    regular_alpha: float
    qkv_alpha: float
    effective_alpha_over_rank: float
    required_metadata: dict[str, str]


LEGACY_ALPHA8_PROFILE = SourceProfile(
    name="lightx2v_v0.1_corrected_alpha8",
    inference_steps=4,
    regular_alpha=8.0,
    qkv_alpha=24.0,
    effective_alpha_over_rank=0.0625,
    required_metadata={
        "base_model": "MiniMax-H3",
        "sampler_steps": "4",
        "conversion_family": "LightX2V Diffusers/PEFT split-QKV",
        "effective_lora_scale": "0.0625",
        "peft_lora_alpha": "8",
        "source_lora_rank": "128",
        "fused_qkv_rank": "384",
        "fused_qkv_alpha": "24",
    },
)

V1_COMMON_METADATA = {
    "base_model": "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors",
    "format": "pt",
    "qkv_fusion": "block diagonal B; concat A; alpha multiplied by 3",
    "source_format": "Diffusers PEFT LoRA",
    "swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]",
    "target_format": "ComfyUI generic LoRA",
    "training_rank": "128",
}

V1_4STEP_768P_PROFILE = SourceProfile(
    name="lightx2v_v1.0_4step_768p_official_comfyui_bf16",
    inference_steps=4,
    regular_alpha=128.0,
    qkv_alpha=384.0,
    effective_alpha_over_rank=1.0,
    required_metadata={
        **V1_COMMON_METADATA,
        "training_alpha": "128.0",
        "training_scale": "1.0",
    },
)

V1_8STEP_PROFILE = SourceProfile(
    name="lightx2v_v1.0_8step_official_comfyui_bf16",
    inference_steps=8,
    regular_alpha=8.0,
    qkv_alpha=24.0,
    effective_alpha_over_rank=0.0625,
    required_metadata={
        **V1_COMMON_METADATA,
        "training_alpha": "8.0",
        "training_scale": "0.0625",
    },
)

SOURCE_PROFILES = (
    LEGACY_ALPHA8_PROFILE,
    V1_4STEP_768P_PROFILE,
    V1_8STEP_PROFILE,
)


def expected_modules() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    layers = [f"blocks.{index}" for index in range(MAIN_BLOCKS)]
    layers.extend(
        f"token_refiner.blocks.{index}" for index in range(REFINER_BLOCKS)
    )
    return {
        f"{layer}.{name}": spec
        for layer in layers
        for name, spec in MODULE_SPECS.items()
    }


EXPECTED_MODULES = expected_modules()


def identify_source_profile(
    metadata: dict[str, str], inference_steps: int
) -> SourceProfile:
    matches = [
        profile
        for profile in SOURCE_PROFILES
        if all(
            metadata.get(key) == expected
            for key, expected in profile.required_metadata.items()
        )
    ]
    if len(matches) != 1:
        names = ", ".join(profile.name for profile in SOURCE_PROFILES)
        raise ValueError(
            "source metadata does not match exactly one supported LightX2V "
            f"profile ({names})"
        )
    profile = matches[0]
    if inference_steps != profile.inference_steps:
        raise ValueError(
            f"source profile {profile.name} requires {profile.inference_steps} "
            f"inference steps, found {inference_steps}"
        )
    return profile


def expected_alpha(module: str, profile: SourceProfile) -> float:
    if module.endswith(".attn.qkv_proj"):
        return profile.qkv_alpha
    return profile.regular_alpha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def read_header(
    path: Path,
) -> tuple[int, dict[str, dict], dict[str, str]]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"invalid safetensors prefix: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {header_length}")
        encoded = handle.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    parsed = json.loads(encoded)
    metadata = parsed.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"invalid safetensors metadata: {path}")
    return header_length, parsed, metadata


def tensor_key(module: str, suffix: str) -> str:
    return f"{COMFY_PREFIX}{module}.{suffix}"


def validate_source_lora(
    path: Path,
    inference_steps: int,
) -> tuple[int, dict[str, dict], dict[str, str], SourceProfile]:
    header_length, tensors, metadata = read_header(path)
    if len(tensors) != TENSOR_COUNT:
        raise ValueError(
            f"source LoRA: expected {TENSOR_COUNT} tensors, found {len(tensors)}"
        )
    profile = identify_source_profile(metadata, inference_steps)

    expected_keys: set[str] = set()
    for module, (a_shape, b_shape) in EXPECTED_MODULES.items():
        expected = {
            tensor_key(module, "lora_A.weight"): ("BF16", list(a_shape)),
            tensor_key(module, "lora_B.weight"): ("BF16", list(b_shape)),
            tensor_key(module, "alpha"): ("F32", []),
        }
        expected_keys.update(expected)
        for key, (dtype, shape) in expected.items():
            descriptor = tensors.get(key)
            if descriptor is None:
                raise ValueError(f"source LoRA is missing {key}")
            if descriptor.get("dtype") != dtype or descriptor.get("shape") != shape:
                raise ValueError(
                    f"{key}: expected {dtype} {shape}, found "
                    f"{descriptor.get('dtype')} {descriptor.get('shape')}"
                )
    if set(tensors) != expected_keys:
        missing = sorted(expected_keys - set(tensors))
        extra = sorted(set(tensors) - expected_keys)
        raise ValueError(
            f"source key set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for module in EXPECTED_MODULES:
            key = tensor_key(module, "alpha")
            alpha = handle.get_tensor(key)
            if alpha.dtype.is_floating_point is not True or alpha.ndim != 0:
                raise ValueError(f"{key}: expected a floating scalar")
            alpha_value = expected_alpha(module, profile)
            if alpha.item() != alpha_value:
                raise ValueError(
                    f"{key}: expected alpha {alpha_value}, found {alpha.item()}"
                )
    return header_length, tensors, metadata, profile


def raw_tensor_sha256(path: Path, key: str) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key).detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_pruned_target(path: Path) -> tuple[dict[str, str], str, str]:
    _, tensors, metadata = read_header(path)
    curve = tensors.get(CURVE_KEY)
    curve_dtype = curve.get("dtype") if curve is not None else None
    if (
        curve is None
        or curve_dtype not in {"F32", "BF16"}
        or curve.get("shape") != [1025, 8]
    ):
        raise ValueError(
            f"target must contain F32 or BF16 {CURVE_KEY} with shape [1025, 8]"
        )
    for module, (a_shape, b_shape) in EXPECTED_MODULES.items():
        key = f"{module}.weight"
        descriptor = tensors.get(key)
        expected_shape = [b_shape[0], a_shape[1]]
        if descriptor is None:
            raise ValueError(f"target model is missing {key}")
        if descriptor.get("shape") != expected_shape:
            raise ValueError(
                f"target {key}: expected {expected_shape}, "
                f"found {descriptor.get('shape')}"
            )
    return metadata, raw_tensor_sha256(path, CURVE_KEY), curve_dtype


def data_payload_sha256(path: Path, header_length: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        handle.seek(8 + header_length)
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def encoded_header(
    tensors: dict[str, dict], metadata: dict[str, str]
) -> bytes:
    payload: dict[str, object] = {"__metadata__": metadata}
    payload.update(tensors)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + (b" " * padding)


def temporary_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.{uuid.uuid4().hex}.partial")


def create_repacked_partial(
    source: Path,
    final: Path,
    tensors: dict[str, dict],
    metadata: dict[str, str],
    source_header_length: int,
) -> tuple[Path, str, int, str]:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = temporary_path(final)
    header = encoded_header(tensors, metadata)
    try:
        with source.open("rb") as source_handle, partial.open("xb") as output_handle:
            source_handle.seek(8 + source_header_length)
            output_handle.write(struct.pack("<Q", len(header)))
            output_handle.write(header)
            shutil.copyfileobj(source_handle, output_handle, 16 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        output_header_length, output_tensors, output_metadata = read_header(partial)
        if output_tensors != tensors:
            raise ValueError("output tensor descriptors changed during metadata binding")
        if output_metadata != metadata:
            raise ValueError("output metadata read-back mismatch")
        source_payload_hash, source_payload_bytes = data_payload_sha256(
            source, source_header_length
        )
        output_payload_hash, output_payload_bytes = data_payload_sha256(
            partial, output_header_length
        )
        if (
            output_payload_hash != source_payload_hash
            or output_payload_bytes != source_payload_bytes
        ):
            raise ValueError("tensor data payload changed during metadata binding")
        return (
            partial,
            source_payload_hash,
            source_payload_bytes,
            sha256_file(partial),
        )
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def publish_partial_no_overwrite(partial: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {final}")
    os.rename(partial, final)


def publish_json_no_overwrite(final: Path, payload: dict) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {final}")
    partial = temporary_path(final)
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(partial.read_text(encoding="utf-8"))
        publish_partial_no_overwrite(partial, final)
    finally:
        if partial.exists():
            partial.unlink()


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def metadata_for_binding(
    source_metadata: dict[str, str],
    source: Path,
    target: Path,
    source_hash: str,
    target_hash: str,
    table_hash: str,
    payload_hash: str,
    comfy_commit: str,
    profile: SourceProfile,
) -> dict[str, str]:
    metadata = dict(source_metadata)
    metadata.update(
        {
            "compatible_base": f"exact pruned checkpoint SHA256 {target_hash}",
            "compatibility_scope": "exact_checkpoint_sha256_validated",
            "compatible_main_file": target.name,
            "compatible_main_sha256": target_hash,
            "binding_tool": Path(__file__).name,
            "binding_algorithm": "identity_tensor_payload_metadata_only",
            "binding_source_file": source.name,
            "binding_source_sha256": source_hash,
            "binding_source_profile": profile.name,
            "source_tensor_data_payload_sha256": payload_hash,
            "adaln_projection": "not_applicable; LightX2V has no AdaLN adapters",
            "adaln_t_table_raw_sha256": table_hash,
            "adapter_count": str(ADAPTER_COUNT),
            "tensor_count": str(TENSOR_COUNT),
            "target_shape_match_count": str(ADAPTER_COUNT),
            "recommended_inference_steps": str(profile.inference_steps),
            "effective_lora_scale": str(profile.effective_alpha_over_rank),
            "comfyui_commit": comfy_commit,
            "validation_status": (
                "static_identity_binding_validated; perceptual_render_pending"
            ),
        }
    )
    return metadata


def comfyui_parser_check(
    output: Path, comfy_root: Path, profile: SourceProfile
) -> tuple[int, int, int, str]:
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"ComfyUI root does not exist: {comfy_root}")
    root_string = str(comfy_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    import comfy.lora  # type: ignore[import-not-found]
    from comfy.weight_adapter.lora import LoRAAdapter  # type: ignore[import-not-found]

    state = load_file(output, device="cpu")
    to_load = {
        f"{COMFY_PREFIX}{module}": f"{COMFY_PREFIX}{module}.weight"
        for module in EXPECTED_MODULES
    }
    patches = comfy.lora.load_lora(state, to_load, log_missing=False)
    if len(patches) != ADAPTER_COUNT or not all(
        isinstance(adapter, LoRAAdapter) for adapter in patches.values()
    ):
        raise ValueError(
            f"ComfyUI parsed {len(patches)} adapters; expected {ADAPTER_COUNT} LoRAAdapter"
        )
    for module in EXPECTED_MODULES:
        adapter = patches[f"{COMFY_PREFIX}{module}.weight"]
        alpha_value = expected_alpha(module, profile)
        if adapter.weights[2] != alpha_value:
            raise ValueError(
                f"ComfyUI parsed alpha {adapter.weights[2]} for {module}; "
                f"expected {alpha_value}"
            )
    consumed: set[str] = set()
    for adapter in patches.values():
        consumed.update(adapter.loaded_keys)
    if consumed != set(state):
        raise ValueError(
            f"ComfyUI consumed {len(consumed)}/{len(state)} source keys"
        )
    commit = subprocess.run(
        ["git", "-C", str(comfy_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return len(patches), len(consumed), ADAPTER_COUNT, commit


def run(args: argparse.Namespace) -> dict:
    for path in (args.lora, args.pruned_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = manifest_path(args.output)
    for path in (args.output, manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    hashes = {
        "lora": require_hash(
            args.lora, args.expected_lora_sha256, "source LoRA"
        ),
        "pruned_model": require_hash(
            args.pruned_model,
            args.expected_pruned_model_sha256,
            "pruned model",
        ),
    }
    source_header_length, tensors, source_metadata, profile = validate_source_lora(
        args.lora, args.inference_steps
    )
    target_metadata, table_hash, table_dtype = validate_pruned_target(
        args.pruned_model
    )
    if (
        args.expected_table_sha256 is not None
        and table_hash.lower() != args.expected_table_sha256.lower()
    ):
        raise ValueError(
            f"{CURVE_KEY} raw SHA-256 mismatch: "
            f"expected {args.expected_table_sha256}, found {table_hash}"
        )

    payload_hash, payload_bytes = data_payload_sha256(
        args.lora, source_header_length
    )
    comfy_commit = args.comfyui_commit or subprocess.run(
        ["git", "-C", str(args.comfy_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = metadata_for_binding(
        source_metadata,
        args.lora,
        args.pruned_model,
        hashes["lora"],
        hashes["pruned_model"],
        table_hash,
        payload_hash,
        comfy_commit,
        profile,
    )

    partial: Path | None = None
    try:
        partial, copied_payload_hash, copied_payload_bytes, output_hash = (
            create_repacked_partial(
                args.lora,
                args.output,
                tensors,
                metadata,
                source_header_length,
            )
        )
        if (
            copied_payload_hash != payload_hash
            or copied_payload_bytes != payload_bytes
        ):
            raise ValueError("source payload changed between validation passes")
        parsed, consumed, shapes, parsed_commit = comfyui_parser_check(
            partial, args.comfy_root, profile
        )
        if parsed_commit != comfy_commit:
            raise ValueError(
                f"ComfyUI commit changed during binding: "
                f"{comfy_commit} -> {parsed_commit}"
            )
        final_hashes = {
            "lora": require_hash(
                args.lora, hashes["lora"], "source LoRA after binding"
            ),
            "pruned_model": require_hash(
                args.pruned_model,
                hashes["pruned_model"],
                "pruned model after binding",
            ),
        }
        if final_hashes != hashes:
            raise ValueError("an input changed during metadata binding")
        publish_partial_no_overwrite(partial, args.output)
        partial = None
    finally:
        if partial is not None and partial.exists():
            partial.unlink()

    result = {
        "schema": "minimax_h3_lightx2v_pruned_binding_manifest_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "identity_tensor_payload_binding",
        "status": "static_identity_binding_validated; perceptual_render_pending",
        "inputs": {
            "lora": {
                "path": str(args.lora.resolve()),
                "sha256": hashes["lora"],
            },
            "pruned_model": {
                "path": str(args.pruned_model.resolve()),
                "sha256": hashes["pruned_model"],
            },
        },
        "target_metadata": target_metadata,
        "adaln_t_table": {
            "shape": [1025, 8],
            "dtype": table_dtype,
            "raw_sha256": table_hash,
            "used_by_lora": False,
        },
        "algorithm": {
            "name": "identity_tensor_payload_metadata_only",
            "reason": "LightX2V contains no AdaLN adapter",
            "tensor_data_payload_sha256": payload_hash,
            "tensor_data_payload_bytes": payload_bytes,
        },
        "structure": {
            "adapter_count": ADAPTER_COUNT,
            "weight_tensor_count": WEIGHT_TENSOR_COUNT,
            "alpha_tensor_count": ALPHA_TENSOR_COUNT,
            "tensor_count": TENSOR_COUNT,
            "target_shape_matches": shapes,
            "source_profile": profile.name,
            "inference_steps": profile.inference_steps,
            "regular_alpha": profile.regular_alpha,
            "qkv_alpha": profile.qkv_alpha,
            "effective_alpha_over_rank": profile.effective_alpha_over_rank,
        },
        "comfyui": {
            "commit": comfy_commit,
            "parsed_adapters": parsed,
            "consumed_keys": consumed,
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": output_hash,
        },
        "input_hashes_unchanged": True,
    }
    publish_json_no_overwrite(manifest, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="supported 208-adapter LightX2V ComfyUI LoRA",
    )
    parser.add_argument(
        "--pruned-model",
        type=Path,
        required=True,
        help="exact curve-pruned target checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new exact-checkpoint-bound LoRA; never overwritten",
    )
    parser.add_argument("--expected-lora-sha256")
    parser.add_argument("--expected-pruned-model-sha256")
    parser.add_argument("--expected-table-sha256")
    parser.add_argument(
        "--inference-steps",
        type=int,
        required=True,
        choices=(4, 8),
        help="strictly checked against the detected LightX2V source profile",
    )
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=DEFAULT_COMFY_ROOT,
    )
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
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
