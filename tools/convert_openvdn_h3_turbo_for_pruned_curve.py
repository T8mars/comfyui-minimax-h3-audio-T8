#!/usr/bin/env python3
"""Project the published OpenVDN DMD Turbo adapter onto an H3 pruned curve.

The OpenVDN adapter is published in FastVideo/Diffusers layout.  This tool first
converts it to ComfyUI's fused H3 layout, then reuses the audited affine 1025-point
projection used by ``convert_minimax_h3_turbo_for_pruned_curve.py``.  The 208
attention/MLP updates remain tensor-identical.  Each of the 51 full-width AdaLN
updates becomes an 8-column LoRA plus an FP32 bias residual.

Inputs are read-only, outputs are never overwritten, and the output is bound to
the target ``adaln_t_table`` SHA-256 rather than a filename.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

from safetensors.torch import load_file


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_openvdn_turbo(state):
    result = {}
    for key, value in state.items():
        normalized = key.replace(".attn.orig.", ".attn.")
        normalized = normalized.replace(
            ".lora_A.turbo.weight", ".lora_A.weight"
        ).replace(".lora_B.turbo.weight", ".lora_B.weight")
        if normalized in result:
            raise ValueError(f"duplicate normalized OpenVDN key: {normalized}")
        result[normalized] = value
    return result


def comfy_source_state(state, fastvideo_module):
    normalized = normalize_openvdn_turbo(state)
    converted, report = fastvideo_module.convert_fastvideo_h3_adapter(normalized)
    prefixed = {f"diffusion_model.{key}": value for key, value in converted.items()}
    return prefixed, report


def validate_openvdn_source(source, curve_tool) -> None:
    expected_modules = set(curve_tool.EXPECTED_MODULES)
    found_modules = set()
    if len(source) != curve_tool.SOURCE_TENSOR_COUNT:
        raise ValueError(
            f"converted OpenVDN adapter: expected {curve_tool.SOURCE_TENSOR_COUNT} "
            f"tensors, found {len(source)}"
        )
    for module, (reference_a, reference_b) in curve_tool.EXPECTED_MODULES.items():
        a_key = curve_tool.module_key(module, "lora_A.weight")
        b_key = curve_tool.module_key(module, "lora_B.weight")
        if a_key not in source or b_key not in source:
            raise ValueError(f"converted OpenVDN adapter is missing {module}")
        a = source[a_key]
        b = source[b_key]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"converted OpenVDN adapter has invalid factors for {module}: "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        if a.dtype != b.dtype or str(a.dtype) != "torch.bfloat16":
            raise ValueError(f"converted OpenVDN adapter {module} must be BF16")
        if int(a.shape[1]) != reference_a[1] or int(b.shape[0]) != reference_b[0]:
            raise ValueError(
                f"converted OpenVDN adapter target dimensions differ for {module}: "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        if module in curve_tool.ADALN_MODULES and (
            tuple(a.shape) != reference_a or tuple(b.shape) != reference_b
        ):
            raise ValueError(
                f"converted OpenVDN AdaLN rank differs for {module}: "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        found_modules.add(module)
    if found_modules != expected_modules:
        raise ValueError("converted OpenVDN module set is incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pruned-model", type=Path, required=True)
    parser.add_argument("--time-embedder-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-pruned-model-sha256")
    parser.add_argument("--expected-time-reference-sha256")
    parser.add_argument("--expected-table-sha256")
    parser.add_argument("--comfyui-commit", default="unknown")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    comfy_root = args.comfy_root.resolve()
    for path in (args.adapter, args.pruned_model, args.time_embedder_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    curve_tool_path = project_root / "tools" / "convert_minimax_h3_turbo_for_pruned_curve.py"
    fastvideo_path = project_root / "h3_lora_compat_advanced.py"
    sys.path.insert(0, str(comfy_root))
    curve_tool = _load_module("openvdn_curve_projection_core", curve_tool_path)
    fastvideo = _load_module("openvdn_fastvideo_conversion", fastvideo_path)

    expected = {
        "adapter": args.expected_adapter_sha256,
        "pruned_model": args.expected_pruned_model_sha256,
        "time_reference": args.expected_time_reference_sha256,
    }
    hashes = {
        "adapter": curve_tool.require_hash(args.adapter, expected["adapter"], "OpenVDN adapter"),
        "pruned_model": curve_tool.require_hash(
            args.pruned_model, expected["pruned_model"], "pruned model"
        ),
        "time_reference": curve_tool.require_hash(
            args.time_embedder_reference, expected["time_reference"], "time reference"
        ),
    }
    raw = load_file(args.adapter, device="cpu")
    source, conversion = comfy_source_state(raw, fastvideo)
    validate_openvdn_source(source, curve_tool)
    curve, target_metadata = curve_tool.validate_pruned_model(args.pruned_model)
    table_hash = curve_tool.tensor_raw_sha256(curve)
    if args.expected_table_sha256 and table_hash.lower() != args.expected_table_sha256.lower():
        raise ValueError(
            f"adaln_t_table SHA-256 mismatch: expected {args.expected_table_sha256}, found {table_hash}"
        )
    reference = curve_tool.load_time_reference(args.time_embedder_reference)
    output_state, metrics, summary = curve_tool.project_lora(source, curve, reference)
    curve_tool.validate_curve_output(output_state, source)

    metadata_args = argparse.Namespace(
        lora=args.adapter,
        pruned_model=args.pruned_model,
        time_embedder_reference=args.time_embedder_reference,
    )
    metadata = curve_tool.metadata_for_curve(
        {},
        metadata_args,
        {
            "lora": hashes["adapter"],
            "pruned_model": hashes["pruned_model"],
            "time_reference": hashes["time_reference"],
        },
        table_hash,
        summary,
        args.comfyui_commit,
    )
    metadata.update(
        {
            "adapter_owner": "OpenVDN/vdn-minimax-h3",
            "adapter_revision": "18be6bcc4ee72585eee322ba28b5ccac2cf85ef0",
            "adapter_stage": "stage-dmd-step-250",
            "sampler_steps": "8",
            "compatibility_scope": "exact_adaln_t_table_sha256",
            "compatible_base": f"H3 pruned curve {table_hash}",
            "conversion_source_file": args.adapter.name,
            "conversion_source_sha256": hashes["adapter"],
            "validation_status": "static_projection_validated; real_openvdn_render_pending",
        }
    )
    output_hash = curve_tool.publish_safetensors_no_overwrite(
        output,
        output_state,
        metadata,
        lambda loaded: curve_tool.validate_curve_output(loaded, source),
    )
    manifest = {
        "schema": "t8.openvdn_h3.pruned_curve_adapter.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STATIC_PROJECTION_PASS_REAL_RENDER_PENDING",
        "inputs": {
            "adapter": {"file": args.adapter.name, "sha256": hashes["adapter"]},
            "pruned_model": {
                "file": args.pruned_model.name,
                "sha256": hashes["pruned_model"],
                "metadata": target_metadata,
            },
            "time_embedder_reference": {
                "file": args.time_embedder_reference.name,
                "sha256": hashes["time_reference"],
                "used_tensors": list(curve_tool.TIME_KEYS.values()),
            },
        },
        "adaln_t_table": {
            "shape": [curve_tool.GRID_SIZE, curve_tool.CURVE_WIDTH],
            "raw_sha256": table_hash,
        },
        "conversion": conversion,
        "algorithm": {
            "name": "adaln_curve_affine_lstsq_pinv1025_with_diff_b",
            "direct_adapter_count": curve_tool.DIRECT_ADAPTER_COUNT,
            "projected_adaln_count": curve_tool.PROJECTED_ADALN_COUNT,
            "bias_diff_count": curve_tool.PROJECTED_ADALN_COUNT,
        },
        "projection_summary": asdict(summary),
        "module_metrics": [asdict(metric) for metric in metrics],
        "output": {
            "file": output.name,
            "sha256": output_hash,
            "bytes": output.stat().st_size,
            "tensor_count": curve_tool.CURVE_TENSOR_COUNT,
        },
        "comfyui_commit": args.comfyui_commit,
    }
    curve_tool.publish_json_no_overwrite(manifest_path, manifest)
    final_hashes = {
        "adapter": curve_tool.require_hash(args.adapter, hashes["adapter"], "adapter after conversion"),
        "pruned_model": curve_tool.require_hash(
            args.pruned_model, hashes["pruned_model"], "pruned model after conversion"
        ),
        "time_reference": curve_tool.require_hash(
            args.time_embedder_reference, hashes["time_reference"], "time reference after conversion"
        ),
    }
    if final_hashes != hashes:
        raise ValueError("an input changed during conversion")
    return manifest


def main() -> int:
    try:
        result = run(parse_args())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
