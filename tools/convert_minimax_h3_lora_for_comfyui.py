#!/usr/bin/env python3
"""Convert LarryVrh's MiniMax-H3 Turbo LoRAs to native ComfyUI keys.

The conversion is intentionally limited to the two known 4-step H3 LoRAs. It
adds ``diffusion_model.`` to every adapter key and verifies every tensor name,
shape, dtype, and value. Source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


COMFY_PREFIX = "diffusion_model."
KEY_RE = re.compile(
    r"^(?:diffusion_model\.)?"
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
REFINER_SHAPES = {k: v for k, v in MAIN_SHAPES.items() if k != "adaln_proj.linear"}
FINAL_SHAPES = ((16, 2688), (10752, 16))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_prefix(key: str) -> str:
    return key[len(COMFY_PREFIX):] if key.startswith(COMFY_PREFIX) else key


def expected_modules() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    modules = {}
    for block in range(50):
        for name, shapes in MAIN_SHAPES.items():
            modules[f"blocks.{block}.{name}"] = shapes
    for block in range(2):
        for name, shapes in REFINER_SHAPES.items():
            modules[f"token_refiner.blocks.{block}.{name}"] = shapes
    modules["final_layer.adaln_proj.linear"] = FINAL_SHAPES
    return modules


EXPECTED_MODULES = expected_modules()


def validate_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    if len(state_dict) != 518:
        raise ValueError(f"expected 518 tensors, found {len(state_dict)}")

    found: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        match = KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected tensor key: {key}")
        module = match.group("module")
        side = match.group("side")
        if module not in EXPECTED_MODULES:
            raise ValueError(f"unexpected MiniMax-H3 module: {module}")
        if side in found.setdefault(module, {}):
            raise ValueError(f"duplicate LoRA side {side}: {module}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{key}: expected bfloat16, found {tensor.dtype}")
        found[module][side] = tensor

    missing_modules = sorted(set(EXPECTED_MODULES) - set(found))
    extra_modules = sorted(set(found) - set(EXPECTED_MODULES))
    if missing_modules or extra_modules:
        raise ValueError(
            f"module set mismatch; missing={missing_modules[:3]}, extra={extra_modules[:3]}"
        )

    for module, pair in found.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{module}: expected paired A/B tensors, found {sorted(pair)}")
        expected_a, expected_b = EXPECTED_MODULES[module]
        actual_a = tuple(pair["A"].shape)
        actual_b = tuple(pair["B"].shape)
        if (actual_a, actual_b) != (expected_a, expected_b):
            raise ValueError(
                f"{module}: expected A{expected_a}/B{expected_b}, "
                f"found A{actual_a}/B{actual_b}"
            )


def read_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def validate_metadata(metadata: dict[str, str]) -> None:
    required = {
        "base_model": "MiniMax-H3",
        "sampler_steps": "4",
        "dtype": "bfloat16",
        "application": "W_eff = W + lora_B @ lora_A",
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"metadata {key!r}: expected {value!r}, found {metadata.get(key)!r}"
            )


def output_path_for(source: Path, output_dir: Path | None) -> Path:
    directory = output_dir if output_dir is not None else source.parent
    return directory / f"{source.stem}_comfyui.safetensors"


def verify_value_identity(
    source_state: dict[str, torch.Tensor], output: Path
) -> None:
    converted = load_file(output, device="cpu")
    validate_state_dict(converted)
    expected_keys = {COMFY_PREFIX + strip_prefix(key) for key in source_state}
    if set(converted) != expected_keys:
        raise ValueError("converted tensor key set differs from the expected prefixed key set")
    for source_key, source_tensor in source_state.items():
        converted_key = COMFY_PREFIX + strip_prefix(source_key)
        if not torch.equal(source_tensor, converted[converted_key]):
            raise ValueError(f"tensor values changed during conversion: {source_key}")


def convert(source: Path, output: Path, force: bool) -> tuple[str, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("source and output paths must differ")
    if output.exists() and not force:
        raise FileExistsError(f"output exists (use --force to replace it): {output}")

    metadata = read_metadata(source)
    validate_metadata(metadata)
    source_state = load_file(source, device="cpu")
    validate_state_dict(source_state)
    if any(key.startswith(COMFY_PREFIX) for key in source_state):
        raise ValueError(f"input is already ComfyUI-prefixed: {source}")

    source_hash = sha256(source)
    converted = {COMFY_PREFIX + key: tensor for key, tensor in source_state.items()}
    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "comfyui_key_prefix": COMFY_PREFIX,
            "comfyui_loader": "Load LoRA (Bypass, Model Only) (for debugging)",
            "compatible_base": "MiniMax-H3 non-pruned bf16 or int8_convrot",
            "incompatible_base": "MiniMax-H3 pruned_* (AdaLN input is 8, LoRA input is 2688)",
            "conversion_source_file": source.name,
            "conversion_source_sha256": source_hash,
            "conversion_tool": Path(__file__).name,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        save_file(converted, temporary, metadata=output_metadata)
        verify_value_identity(source_state, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    final_metadata = read_metadata(output)
    for key, value in output_metadata.items():
        if final_metadata.get(key) != value:
            raise ValueError(f"output metadata verification failed for {key!r}")
    return source_hash, sha256(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="source .safetensors files")
    parser.add_argument("--output-dir", type=Path, help="defaults to each source directory")
    parser.add_argument("--force", action="store_true", help="replace existing converted files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for source in args.inputs:
            output = output_path_for(source, args.output_dir)
            source_hash, output_hash = convert(source, output, args.force)
            print(f"converted: {source} -> {output}")
            print("  259 LoRA modules / 518 BF16 tensors; values bit-identical")
            print(f"  source sha256: {source_hash}")
            print(f"  output sha256: {output_hash}")
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
