from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as torch_functional


REPORT_SCHEMA = "h3_t8_vretoucher_source_structure_audit/v1"
UPSTREAM_REVISION = "ae25b5475680ed01958c017b32b669b4e46d7f9b"
PINNED_FILES = {
    "LICENSE": "a0350824309159a20d11eb0ada0527fbf17e54c5ec38b4e1a9584507d5036052",
    "model/VRetouchEr.py": "f67a3c93bb70b2f5e6ed627266c52d3a9034b4169fac8b881269a3781169647a",
    "model/gpen_model_video.py": "92b99d1f6b8ba47c816ce44d07c1d97800d621ed4de8b179684819d38265f27d",
    "model/network_vrt_pair_qkv_video_fuse.py": "ec69a2061028adf8a80876e5e699ab59d75aa944f1e6a4767d307afc43a7df2e",
    "model/modules/flow_comp.py": "5a07667a702885fe43ddd02d907bb53d5ea358194bbbb2a11c977fab431213b4",
    "model/modules/spectral_norm.py": "f94c800f3ecd5b54e791e251def514c8bc2b2d962cc0db709fcb8af286f46267",
    "op/__init__.py": "9ea7d54be75aa51f4761ce604c1a06b4c701998d27dfeffa8294f1dec440cf1b",
    "op/fused_act.py": "5bc2feb335fcb537ab935eba58ae13a27b0ed7cb71b3564ae4b073f84d7fb792",
    "op/upfirdn2d.py": "4ae1e299837e94e21ae894d2192343fdb6b2a565c1af8ee47c1af34be05f215a",
}


def _default_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "vretoucher_upstream"


def _sha256(path: Path) -> str:
    value = path.read_bytes()
    if b"\r" in value.replace(b"\r\n", b""):
        raise ValueError(f"pinned source contains a bare carriage return: {path}")
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


def verify_pinned_source(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    files = []
    missing = []
    mismatched = []
    for relative, expected in PINNED_FILES.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = _sha256(path)
        record = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "matches": actual == expected,
        }
        files.append(record)
        if actual != expected:
            mismatched.append(relative)
    status = "PINNED_SOURCE_PASS"
    if missing:
        status = "REJECTED_PINNED_SOURCE_MISSING"
    elif mismatched:
        status = "REJECTED_PINNED_SOURCE_HASH_MISMATCH"
    return {
        "status": status,
        "root": str(root),
        "upstream_revision": UPSTREAM_REVISION,
        "source_hash_mode": "sha256_after_crlf_to_lf_normalization",
        "files": files,
        "missing": missing,
        "mismatched": mismatched,
    }


class _MetaFusedLeakyReLU(nn.Module):
    """Construction-only stand-in with the exact persistent bias key."""

    def __init__(
        self,
        channel: int,
        negative_slope: float = 0.2,
        scale: float = 2**0.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(int(channel)))
        self.negative_slope = float(negative_slope)
        self.scale = float(scale)
        self.device_name = str(device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * torch_functional.leaky_relu(
            value + self.bias.view(1, -1, 1, 1),
            negative_slope=self.negative_slope,
        )


def _meta_fused_leaky_relu(
    value: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
    device: str = "cpu",
) -> torch.Tensor:
    del device
    return float(scale) * torch_functional.leaky_relu(
        value + bias.view(1, -1, 1, 1),
        negative_slope=float(negative_slope),
    )


def _meta_upfirdn2d(
    value: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
    device: str = "cpu",
) -> torch.Tensor:
    del device
    if value.ndim != 4 or kernel.ndim != 2:
        raise ValueError("meta upfirdn2d expects NCHW input and a 2D kernel")
    pad_left, pad_right = [int(item) for item in pad]
    output_height = (
        int(value.shape[-2]) * int(up)
        + pad_left
        + pad_right
        - int(kernel.shape[-2])
    ) // int(down) + 1
    output_width = (
        int(value.shape[-1]) * int(up)
        + pad_left
        + pad_right
        - int(kernel.shape[-1])
    ) // int(down) + 1
    if output_height < 1 or output_width < 1:
        raise ValueError("meta upfirdn2d produced invalid output geometry")
    return value.new_empty(
        (int(value.shape[0]), int(value.shape[1]), output_height, output_width)
    )


def _install_construction_stubs() -> dict[str, ModuleType | None]:
    previous = {name: sys.modules.get(name) for name in ("op", "turtle")}
    op_module = ModuleType("op")
    op_module.FusedLeakyReLU = _MetaFusedLeakyReLU
    op_module.fused_leaky_relu = _meta_fused_leaky_relu
    op_module.upfirdn2d = _meta_upfirdn2d
    turtle_module = ModuleType("turtle")
    turtle_module.up = None
    sys.modules["op"] = op_module
    sys.modules["turtle"] = turtle_module
    return previous


def _restore_modules(previous: dict[str, ModuleType | None]) -> None:
    for name, value in previous.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def state_structure_manifest(model: nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    entries = [
        {
            "key": key,
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for key, tensor in state.items()
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    parameter_numel = sum(int(item.numel()) for item in model.parameters())
    buffer_numel = sum(int(item.numel()) for item in model.buffers())
    class_counts = Counter(type(module).__name__ for module in model.modules())
    top_level_state_counts = Counter(
        entry["key"].split(".", maxsplit=1)[0] for entry in entries
    )
    return {
        "state_tensor_count": len(entries),
        "state_numel": sum(item["numel"] for item in entries),
        "parameter_numel": parameter_numel,
        "buffer_numel": buffer_numel,
        "estimated_parameter_storage_bytes": {
            "fp32": parameter_numel * 4,
            "fp16_or_bf16": parameter_numel * 2,
        },
        "state_structure_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "module_class_counts": dict(sorted(class_counts.items())),
        "top_level_state_tensor_counts": dict(sorted(top_level_state_counts.items())),
        "torchvision_deform_conv_module_count": int(
            class_counts.get("DCNv2PackFlowGuided", 0)
        ),
        "entries": entries,
        "first_entries": entries[:8],
        "last_entries": entries[-8:],
    }


def audit_meta_structure(
    root: Path,
    *,
    run_meta_forward: bool = False,
) -> dict[str, Any]:
    source = verify_pinned_source(root)
    base = {
        "schema": REPORT_SCHEMA,
        "source": source,
        "torch": torch.__version__,
        "construction_device": "meta",
        "real_parameter_storage_allocated": False,
        "checkpoint_loaded": False,
        "forward_run": False,
    }
    if source["status"] != "PINNED_SOURCE_PASS":
        return {**base, "status": source["status"]}
    root = Path(root).resolve()
    path_text = str(root)
    previous_modules = _install_construction_stubs()
    existing_path = path_text in sys.path
    if not existing_path:
        sys.path.insert(0, path_text)
    imported_names: list[str] = []
    try:
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]
        with torch.device("meta"):
            module = importlib.import_module("model.VRetouchEr")
            imported_names = [
                name for name in sys.modules if name == "model" or name.startswith("model.")
            ]
            original_spynet = module.SPyNet
            module.SPyNet = lambda: original_spynet(use_pretrain=False)
            model = module.InpaintGenerator(n_layer_t=5, frame_num=6)
        # Upstream creates SPyNet mean/std through torch.Tensor(...), which ignores
        # the default-device context and leaves those two buffers on CPU.  A normal
        # ComfyUI load moves the complete module to its execution device, so mirror
        # that contract explicitly before a meta-only shape forward.
        model = model.to(device="meta")
        manifest = state_structure_manifest(model)
        forward_report: dict[str, Any] = {
            "requested": bool(run_meta_forward),
            "executed": False,
        }
        status = "META_STRUCTURE_PASS_CHECKPOINT_NOT_VALIDATED"
        if run_meta_forward:
            inputs = [
                torch.empty((1, 3, 512, 512), device="meta", dtype=torch.float32)
                for _ in range(6)
            ]
            with torch.inference_mode():
                result, masks, flows = model(inputs)
            forward_report = {
                "requested": True,
                "executed": True,
                "input_count": len(inputs),
                "input_shape": list(inputs[0].shape),
                "result_shape": list(result.shape),
                "mask_count": len(masks),
                "mask_shapes": [list(item.shape) for item in masks],
                "flow_shape": list(flows.shape),
                "real_activation_storage_allocated": False,
                "numerics_validated": False,
            }
            status = "META_STRUCTURE_AND_FORWARD_SHAPE_PASS_CHECKPOINT_NOT_VALIDATED"
        return {
            **base,
            **manifest,
            "status": status,
            "forward_run": bool(forward_report["executed"]),
            "meta_forward": forward_report,
            "constructor_patches": {
                "unused_turtle_import_stubbed": True,
                "custom_op_construction_stubbed_without_forward": True,
                "missing_external_spynet_preload_disabled": True,
                "hardcoded_cuda_strings_do_not_allocate_under_meta": True,
            },
            "boundary": (
                "This proves only that the pinned source can describe a finite state structure "
                "under current Torch without allocating real parameter storage. It does not prove "
                "checkpoint key/shape compatibility, custom-op execution or numerical inference."
            ),
        }
    except Exception as error:
        return {
            **base,
            "status": "REJECTED_META_STRUCTURE_CONSTRUCTION_FAILED",
            "detail": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc().splitlines(),
        }
    finally:
        for name in imported_names:
            sys.modules.pop(name, None)
        if not existing_path:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass
        _restore_modules(previous_modules)
        gc.collect()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pin and meta-construct the official VRetouchEr inference structure without "
            "checkpoint load, forward execution or real parameter allocation."
        )
    )
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument(
        "--meta-forward",
        action="store_true",
        help="Also shape-propagate one six-frame 512x512 forward entirely on meta tensors.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--full-print",
        action="store_true",
        help="Print all 411 structure entries instead of the compact console summary.",
    )
    args = parser.parse_args()
    report = audit_meta_structure(args.source_root, run_meta_forward=args.meta_forward)
    if args.report is not None:
        _write_json_atomic(args.report, report)
    printed = report if args.full_print else {
        key: value for key, value in report.items() if key != "entries"
    }
    print(json.dumps(printed, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] in {
        "META_STRUCTURE_PASS_CHECKPOINT_NOT_VALIDATED",
        "META_STRUCTURE_AND_FORWARD_SHAPE_PASS_CHECKPOINT_NOT_VALIDATED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
