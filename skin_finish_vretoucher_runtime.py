from __future__ import annotations

from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from threading import RLock
from types import ModuleType
from typing import Any
import weakref

import torch
from torch import nn
from torch.nn import functional as torch_functional


# The upstream inference architecture and fallback operator formulae are from the
# MIT-licensed VRetouchEr repository by Wen Xue, pinned at the revision below.
VRETOUCHER_UPSTREAM_REVISION = "ae25b5475680ed01958c017b32b669b4e46d7f9b"
VRETOUCHER_CHECKPOINT_SIZE = 630_172_363
VRETOUCHER_STATE_TENSOR_COUNT = 411
VRETOUCHER_STATE_STRUCTURE_SHA256 = (
    "7abd9ecff0b49178fbf2cc7afecf171228ac4acddbbcfc7a5a0484020de8ceea"
)
VRETOUCHER_PINNED_FILES = {
    "LICENSE": "a0350824309159a20d11eb0ada0527fbf17e54c5ec38b4e1a9584507d5036052",
    "model/VRetouchEr.py": "f67a3c93bb70b2f5e6ed627266c52d3a9034b4169fac8b881269a3781169647a",
    "model/gpen_model_video.py": "92b99d1f6b8ba47c816ce44d07c1d97800d621ed4de8b179684819d38265f27d",
    "model/network_vrt_pair_qkv_video_fuse.py": "ec69a2061028adf8a80876e5e699ab59d75aa944f1e6a4767d307afc43a7df2e",
    "model/modules/flow_comp.py": "5a07667a702885fe43ddd02d907bb53d5ea358194bbbb2a11c977fab431213b4",
    "model/modules/spectral_norm.py": "f94c800f3ecd5b54e791e251def514c8bc2b2d962cc0db709fcb8af286f46267",
    "op/fused_act.py": "5bc2feb335fcb537ab935eba58ae13a27b0ed7cb71b3564ae4b073f84d7fb792",
    "op/upfirdn2d.py": "4ae1e299837e94e21ae894d2192343fdb6b2a565c1af8ee47c1af34be05f215a",
}
_RUNTIME_LOCK = RLock()


class VRetouchRuntimeUnavailable(RuntimeError):
    def __init__(
        self,
        status: str,
        detail: str,
        *,
        model_forward_completed: bool = False,
    ):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.model_forward_completed = bool(model_forward_completed)


def _pop_module_tree(root_name: str) -> dict[str, ModuleType]:
    prefix = f"{root_name}."
    previous = {
        name: module
        for name, module in list(sys.modules.items())
        if name == root_name or name.startswith(prefix)
    }
    for name in previous:
        sys.modules.pop(name, None)
    return previous


def _restore_module_tree(root_name: str, previous: dict[str, ModuleType]) -> None:
    prefix = f"{root_name}."
    for name in list(sys.modules):
        if name == root_name or name.startswith(prefix):
            sys.modules.pop(name, None)
    sys.modules.update(previous)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_source_sha256(path: Path) -> str:
    value = path.read_bytes()
    if b"\r" in value.replace(b"\r\n", b""):
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_PINNED_SOURCE_LINE_ENDING_INVALID",
            f"pinned source contains a bare carriage return: {path}",
        )
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


def bundled_vretoucher_source_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "vretoucher_upstream"


def verify_vretoucher_source(source_root: Path | None = None) -> dict[str, Any]:
    selected = bundled_vretoucher_source_root() if source_root is None else Path(source_root)
    root = selected.expanduser().resolve()
    files = []
    for relative, expected in VRETOUCHER_PINNED_FILES.items():
        path = root / relative
        if not path.is_file():
            raise VRetouchRuntimeUnavailable(
                "ABSTAIN_PINNED_SOURCE_MISSING",
                f"missing pinned VRetouchEr source file: {relative}",
            )
        actual = _normalized_source_sha256(path)
        if actual != expected:
            raise VRetouchRuntimeUnavailable(
                "ABSTAIN_PINNED_SOURCE_HASH_MISMATCH",
                f"pinned VRetouchEr source hash mismatch for {relative}",
            )
        files.append(
            {
                "path": relative,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "status": "PINNED_SOURCE_PASS",
        "root": str(root),
        "revision": VRETOUCHER_UPSTREAM_REVISION,
        "source_hash_mode": "sha256_after_crlf_to_lf_normalization",
        "bundled_source": root == bundled_vretoucher_source_root().resolve(),
        "files": files,
    }


class PureTorchFusedLeakyReLU(nn.Module):
    """Pure-PyTorch equivalent that retains upstream's persistent bias key."""

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
        self.requested_upstream_device = str(device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return pure_torch_fused_leaky_relu(
            value,
            self.bias,
            self.negative_slope,
            self.scale,
        )


def pure_torch_fused_leaky_relu(
    value: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
    device: str = "cpu",
) -> torch.Tensor:
    del device
    return float(scale) * torch_functional.leaky_relu(
        value + bias.view((1, -1) + (1,) * (value.ndim - 2)),
        negative_slope=float(negative_slope),
    )


def pure_torch_upfirdn2d(
    value: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: tuple[int, int] = (0, 0),
    device: str = "cpu",
) -> torch.Tensor:
    """Numerical upstream fallback without JIT compilation or platform branching."""

    del device
    if value.ndim != 4 or kernel.ndim != 2:
        raise ValueError("upfirdn2d expects NCHW input and a two-dimensional kernel")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive integers")
    pad_x0, pad_x1 = [int(item) for item in pad]
    batch, channel, input_height, input_width = [int(item) for item in value.shape]
    kernel_height, kernel_width = [int(item) for item in kernel.shape]
    output = value.permute(0, 2, 3, 1)
    output = output.reshape(batch, input_height, 1, input_width, 1, channel)
    output = torch_functional.pad(output, [0, 0, 0, up - 1, 0, 0, 0, up - 1])
    output = output.reshape(batch, input_height * up, input_width * up, channel)
    output = torch_functional.pad(
        output,
        [
            0,
            0,
            max(pad_x0, 0),
            max(pad_x1, 0),
            max(pad_x0, 0),
            max(pad_x1, 0),
        ],
    )
    height_end = int(output.shape[1]) - max(-pad_x1, 0)
    width_end = int(output.shape[2]) - max(-pad_x1, 0)
    output = output[
        :,
        max(-pad_x0, 0) : height_end,
        max(-pad_x0, 0) : width_end,
        :,
    ]
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(
        -1,
        1,
        input_height * up + pad_x0 + pad_x1,
        input_width * up + pad_x0 + pad_x1,
    )
    weight = torch.flip(kernel, [0, 1]).view(1, 1, kernel_height, kernel_width)
    output = torch_functional.conv2d(output, weight)
    output = output.reshape(
        batch,
        channel,
        input_height * up + pad_x0 + pad_x1 - kernel_height + 1,
        input_width * up + pad_x0 + pad_x1 - kernel_width + 1,
    )
    return output[:, :, ::down, ::down]


class MinimalConvModule(nn.Module):
    """The exact norm-free ConvModule subset used by VRetouchEr SPyNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_cfg: None = None,
        act_cfg: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__()
        if norm_cfg is not None or kwargs:
            raise ValueError("VRetouchEr only permits the audited norm-free ConvModule subset")
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            stride=int(stride),
            padding=int(padding),
        )
        if act_cfg is None:
            self.activate: nn.Module | None = None
        elif act_cfg == {"type": "ReLU"}:
            self.activate = nn.ReLU(inplace=False)
        else:
            raise ValueError(f"unsupported VRetouchEr act_cfg: {act_cfg!r}")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return self.activate(value) if self.activate is not None else value


@contextmanager
def _temporary_upstream_modules():
    names = ("op", "turtle", "mmcv", "mmcv.cnn")
    previous = {name: sys.modules.get(name) for name in names}
    op_module = ModuleType("op")
    op_module.FusedLeakyReLU = PureTorchFusedLeakyReLU
    op_module.fused_leaky_relu = pure_torch_fused_leaky_relu
    op_module.upfirdn2d = pure_torch_upfirdn2d
    turtle_module = ModuleType("turtle")
    turtle_module.up = None
    mmcv_module = ModuleType("mmcv")
    mmcv_cnn_module = ModuleType("mmcv.cnn")
    mmcv_cnn_module.ConvModule = MinimalConvModule
    mmcv_module.cnn = mmcv_cnn_module
    sys.modules.update(
        {
            "op": op_module,
            "turtle": turtle_module,
            "mmcv": mmcv_module,
            "mmcv.cnn": mmcv_cnn_module,
        }
    )
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _state_structure(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    entries = [
        {
            "key": key,
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for key, tensor in state.items()
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "tensor_count": len(entries),
        "numel": sum(item["numel"] for item in entries),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def construct_vretoucher_model(
    source_root: Path | None = None,
    *,
    construction_device: str = "meta",
) -> tuple[nn.Module, dict[str, Any]]:
    source = verify_vretoucher_source(source_root)
    selected_root = bundled_vretoucher_source_root() if source_root is None else Path(source_root)
    root = str(selected_root.resolve())
    added_path = root not in sys.path
    previous_model_modules: dict[str, ModuleType] = {}
    with _RUNTIME_LOCK, _temporary_upstream_modules():
        try:
            if added_path:
                sys.path.insert(0, root)
            previous_model_modules = _pop_module_tree("model")
            with torch.device(str(construction_device)):
                upstream = importlib.import_module("model.VRetouchEr")
                original_spynet = upstream.SPyNet
                upstream.SPyNet = lambda: original_spynet(use_pretrain=False)
                model = upstream.InpaintGenerator(n_layer_t=5, frame_num=6)
            model = model.to(device=str(construction_device))
            structure = _state_structure(model.state_dict())
            if (
                structure["tensor_count"] != VRETOUCHER_STATE_TENSOR_COUNT
                or structure["sha256"] != VRETOUCHER_STATE_STRUCTURE_SHA256
            ):
                raise VRetouchRuntimeUnavailable(
                    "ABSTAIN_CONSTRUCTED_MODEL_STRUCTURE_MISMATCH",
                    "constructed VRetouchEr structure differs from the pinned 411-entry contract",
                )
            return model, {
                "status": "PINNED_MODEL_STRUCTURE_PASS",
                "source": source,
                "construction_device": str(construction_device),
                "state_structure": structure,
                "external_spynet_preload_disabled": True,
                "pure_torch_custom_ops": True,
                "minimal_mmcv_dependency_shim": True,
                "preexisting_generic_model_modules_restored": True,
                "checkpoint_loaded": False,
                "forward_run": False,
            }
        finally:
            _restore_module_tree("model", previous_model_modules)
            if added_path:
                try:
                    sys.path.remove(root)
                except ValueError:
                    pass


def _precision_dtype(precision: str) -> torch.dtype:
    value = str(precision).lower()
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError("precision must be fp16, bf16 or fp32")


def load_vretoucher_model(
    source_root: Path | None,
    checkpoint: Path,
    *,
    expected_checkpoint_sha256: str,
    device: str,
    precision: str = "fp16",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    expected_hash = str(expected_checkpoint_sha256).strip().lower()
    if len(expected_hash) != 64 or any(item not in "0123456789abcdef" for item in expected_hash):
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_TRUSTED_CHECKPOINT_SHA256_REQUIRED",
            "a complete trusted checkpoint SHA-256 is required",
        )
    if not checkpoint.is_file():
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_MISSING", f"missing VRetouchEr checkpoint: {checkpoint}"
        )
    if checkpoint.stat().st_size != VRETOUCHER_CHECKPOINT_SIZE:
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_SIZE_MISMATCH",
            f"expected {VRETOUCHER_CHECKPOINT_SIZE} bytes",
        )
    actual_hash = _file_sha256(checkpoint)
    if actual_hash != expected_hash:
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_SHA256_MISMATCH", "checkpoint SHA-256 mismatch"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(key, str) for key in state)
        or any(not isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_NOT_TENSOR_STATE_DICT",
            "checkpoint must be a non-empty str-to-Tensor state dict",
        )
    structure = _state_structure(state)
    if (
        structure["tensor_count"] != VRETOUCHER_STATE_TENSOR_COUNT
        or structure["sha256"] != VRETOUCHER_STATE_STRUCTURE_SHA256
    ):
        del state
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_STRUCTURE_MISMATCH",
            "checkpoint does not match the pinned 411-entry structure",
        )
    model, construction = construct_vretoucher_model(source_root)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        del model
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CHECKPOINT_STRICT_LOAD_MISMATCH",
            "strict load returned missing or unexpected keys",
        )
    dtype = _precision_dtype(precision)
    model = model.to(device=torch.device(device), dtype=dtype).eval()
    return model, {
        **construction,
        "status": "PINNED_MODEL_AND_CHECKPOINT_LOADED_NOT_YET_INFERRED",
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": actual_hash,
            "structure": structure,
        },
        "device": str(device),
        "precision": str(precision),
        "checkpoint_loaded": True,
    }


def _forward_autocast(device_type: str, dtype: torch.dtype):
    """Match CUDA half/bfloat weights while preserving the upstream FP32 constants."""
    if str(device_type) == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return nullcontext()


def run_vretoucher_context(
    model: nn.Module,
    context: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(context, torch.Tensor) or tuple(context.shape) != (6, 3, 512, 512):
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CONTEXT_SHAPE_MISMATCH",
            "VRetouchEr context must be exactly [6,3,512,512]",
        )
    if not context.is_floating_point() or not bool(torch.isfinite(context).all()):
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_CONTEXT_VALUES_INVALID", "context must contain finite floating values"
        )
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_MODEL_HAS_NO_PARAMETERS",
            "VRetouchEr runtime model has no parameters",
        ) from error
    if parameter.device.type == "meta":
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_MODEL_NOT_CHECKPOINT_BACKED",
            "meta-constructed VRetouchEr is shape evidence only and cannot run inference",
        )
    inputs = [
        context[index : index + 1]
        .to(device=parameter.device, dtype=parameter.dtype)
        .clamp(-1.0, 1.0)
        for index in range(6)
    ]
    autocast_enabled = parameter.device.type == "cuda" and parameter.dtype in {
        torch.float16,
        torch.bfloat16,
    }
    with (
        _RUNTIME_LOCK,
        torch.inference_mode(),
        _forward_autocast(parameter.device.type, parameter.dtype),
    ):
        result, masks, flows = model(inputs)
    if tuple(result.shape) != (1, 3, 512, 512) or len(masks) != 6:
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_RUNTIME_OUTPUT_SHAPE_MISMATCH",
            "runtime output differs from the audited newest-frame contract",
            model_forward_completed=True,
        )
    output_tensors = [result, *masks, flows]
    if any(
        not isinstance(item, torch.Tensor)
        or not item.is_floating_point()
        or not bool(torch.isfinite(item).all())
        for item in output_tensors
    ):
        nonfinite_values = sum(
            int(torch.count_nonzero(~torch.isfinite(item)).item())
            for item in output_tensors
            if isinstance(item, torch.Tensor) and item.is_floating_point()
        )
        raise VRetouchRuntimeUnavailable(
            "ABSTAIN_RUNTIME_OUTPUT_NONFINITE_AFTER_FORWARD",
            f"VRetouchEr forward returned {nonfinite_values} non-finite output values",
            model_forward_completed=True,
        )
    candidate = result[0].float().clamp(-1.0, 1.0).add(1.0).mul(0.5).cpu()
    return candidate, {
        "status": "INFERENCE_CANDIDATE_REQUIRES_IDENTITY_AND_HUMAN_REVIEW",
        "input_shape": list(context.shape),
        "output_shape": list(candidate.shape),
        "mask_shapes": [list(item.shape) for item in masks],
        "flow_shape": list(flows.shape),
        "autocast_enabled": autocast_enabled,
        "compute_dtype": str(parameter.dtype),
        "automatic_accept": False,
        "audio_touched": False,
    }


class VRetoucherRuntimeSession:
    """Own one loaded model and provide an explicit, idempotent release boundary."""

    def __init__(self, model: nn.Module, load_report: dict[str, Any]):
        if not isinstance(model, nn.Module):
            raise TypeError("VRetoucherRuntimeSession requires a torch.nn.Module")
        self._model: nn.Module | None = model
        self._load_report = dict(load_report)
        self._close_report: dict[str, Any] | None = None
        self._model_weakref: weakref.ReferenceType[nn.Module] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def load_report(self) -> dict[str, Any]:
        return dict(self._load_report)

    @property
    def close_report(self) -> dict[str, Any] | None:
        return dict(self._close_report) if self._close_report is not None else None

    def run(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        with _RUNTIME_LOCK:
            if self._closed or self._model is None:
                raise VRetouchRuntimeUnavailable(
                    "ABSTAIN_RUNTIME_SESSION_CLOSED",
                    "VRetouchEr runtime session is already closed",
                )
            return run_vretoucher_context(self._model, context)

    def close(self) -> dict[str, Any]:
        with _RUNTIME_LOCK:
            if self._closed:
                gc.collect()
                object_still_referenced = bool(
                    self._model_weakref is not None and self._model_weakref() is not None
                )
                if self._close_report is not None:
                    self._close_report.update(
                        {
                            "status": (
                                "VRETOUCHER_OWNER_CLEARED_OBJECT_STILL_REFERENCED"
                                if object_still_referenced
                                else "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
                            ),
                            "object_still_referenced_elsewhere": object_still_referenced,
                        }
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {
                    **(self._close_report or {}),
                    "replay_status": "VRETOUCHER_RUNTIME_SESSION_ALREADY_CLOSED",
                    "idempotent_replay": True,
                }
            owned_model = self._model
            self._model = None
            self._closed = True
            model_reference = weakref.ref(owned_model) if owned_model is not None else None
            self._model_weakref = model_reference
            model_device = "none"
            if owned_model is not None:
                try:
                    model_device = str(next(owned_model.parameters()).device)
                except StopIteration:
                    model_device = "no_parameters"
            del owned_model
            gc.collect()
            object_still_referenced = bool(
                model_reference is not None and model_reference() is not None
            )
            cuda_cache_emptied = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_cache_emptied = True
            report = {
                "status": (
                    "VRETOUCHER_OWNER_CLEARED_OBJECT_STILL_REFERENCED"
                    if object_still_referenced
                    else "VRETOUCHER_OWNER_CLEARED_OBJECT_RELEASED"
                ),
                "owner_reference_cleared": True,
                "object_still_referenced_elsewhere": object_still_referenced,
                "model_device_before_close": model_device,
                "python_gc": True,
                "cuda_cache_emptied": cuda_cache_emptied,
                "global_comfy_models_unloaded": False,
            }
            self._close_report = report
            return dict(report)

    def __enter__(self) -> VRetoucherRuntimeSession:
        if self._closed:
            raise VRetouchRuntimeUnavailable(
                "ABSTAIN_RUNTIME_SESSION_CLOSED",
                "cannot enter a closed VRetouchEr runtime session",
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False


def load_vretoucher_session(
    source_root: Path | None,
    checkpoint: Path,
    *,
    expected_checkpoint_sha256: str,
    device: str,
    precision: str = "fp16",
) -> VRetoucherRuntimeSession:
    model, report = load_vretoucher_model(
        source_root,
        checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        device=device,
        precision=precision,
    )
    return VRetoucherRuntimeSession(model, report)


def unload_vretoucher_model(model: nn.Module | None) -> dict[str, Any]:
    had_local_reference = model is not None
    del model
    gc.collect()
    cuda_cache_emptied = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_cache_emptied = True
    return {
        "status": "LOCAL_REFERENCE_DROPPED_CALLER_MUST_RELEASE_OWN_REFERENCE",
        "local_reference_was_present": had_local_reference,
        "caller_reference_must_be_released": True,
        "python_gc": True,
        "cuda_cache_emptied": cuda_cache_emptied,
        "global_comfy_models_unloaded": False,
        "note": (
            "Python arguments are references, so this helper cannot delete a reference still held "
            "by its caller. Prefer VRetoucherRuntimeSession, whose close() clears its own strong "
            "reference and reports whether another reference remains. CUDA context may retain a "
            "small baseline; unrelated ComfyUI models are never unloaded."
        ),
    }


def runtime_capability_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pure_torch_custom_ops": True,
        "requires_mmcv": False,
        "requires_turtle_or_tkinter": False,
        "requires_external_spynet_checkpoint": False,
        "owner_scoped_runtime_session": True,
        "registered_node": False,
    }
