"""Read-only relative storage spans for H05 C01; not kernel safety approval.

The importable tensor collector reads metadata only. The bounded JSON CLI reads
declarations, never tensors, kernels, executable configuration or GPU payloads.
No contiguous(), casting, padding, attention call, fallback or model mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

MAX_INPUT_BYTES = 64 * 1024
DTYPE_BYTES = {
    "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4,
    "torch.float64": 8, "torch.int8": 1, "torch.uint8": 1,
    "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}


def _integer(value, label, maximum=2**63 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be a nonnegative bounded integer")
    return value


def describe_storage(descriptor, *, layout="unknown"):
    """Exact relative span from declared metadata, not an executable-backend gate."""
    required = {"shape", "stride", "storage_offset", "storage_bytes", "dtype", "element_size"}
    if type(descriptor) is not dict or set(descriptor) != required:
        raise ValueError("Storage descriptor must contain exactly the six metadata fields")
    shape, strides = descriptor["shape"], descriptor["stride"]
    if (type(shape) is not list or type(strides) is not list
            or not 1 <= len(shape) <= 8 or len(shape) != len(strides)):
        raise ValueError("Shape and stride must have the same bounded rank")
    shape = [_integer(value, "shape") for value in shape]
    strides = [_integer(value, "stride") for value in strides]
    offset = _integer(descriptor["storage_offset"], "storage_offset")
    storage_bytes = _integer(descriptor["storage_bytes"], "storage_bytes")
    size = _integer(descriptor["element_size"], "element_size", 8)
    dtype = descriptor["dtype"]
    if type(dtype) is not str or dtype not in DTYPE_BYTES or size != DTYPE_BYTES[dtype]:
        raise ValueError("Actual dtype and element size must agree")
    if layout not in ("NHD", "HND", "unknown"):
        raise ValueError("Layout must be explicitly NHD, HND or unknown")
    empty = any(value == 0 for value in shape)
    last = None if empty else offset + sum((length - 1) * step for length, step in zip(shape, strides, strict=True))
    # PyTorch permits a nonzero offset beyond storage on a zero-element view.
    # No element is addressed in that case; do not invent a byte access.
    end = None if empty else (last + 1) * size
    if not empty and (end > storage_bytes or last > 2**63 - 1):
        raise ValueError("Declared view reaches beyond its storage or signed-i64 element range")
    sequential = 1
    contiguous = True
    if not empty:
        for length, step in zip(reversed(shape), reversed(strides), strict=True):
            if length != 1 and step != sequential:
                contiguous = False
            sequential *= length
    rank4 = len(shape) == 4 and layout != "unknown"
    sequence_dim = (1 if layout == "NHD" else 2) if rank4 else None
    row_offset = ((shape[sequence_dim] - 1) * strides[sequence_dim]) if rank4 and not empty else None
    return {
        "schema": "t8.h05.attention-storage.v1",
        "descriptor": dict(descriptor), "layout": layout,
        "declared_view_in_storage": True, "empty": empty,
        "contiguous_from_shape_stride": contiguous,
        "first_storage_element": None if empty else offset,
        "last_storage_element": last, "end_storage_byte_exclusive": end,
        "view_span_bytes": 0 if empty else (last - offset + 1) * size,
        "rank4_sequence_descriptor_available": rank4,
        "sequence_dimension": sequence_dim,
        "last_sequence_row_offset_elements": row_offset,
        "sequence_row_offset_at_or_over_signed_i32": row_offset is not None and row_offset >= 2**31,
        "last_storage_element_at_or_over_signed_i32": last is not None and last >= 2**31,
        "end_byte_at_or_over_signed_i32": end is not None and end >= 2**31,
        "padded_or_quantized_kernel_addresses_verified": False,
        "kernel_execution_or_safety_verified": False,
        "unknown_backend_policy": "report_and_delegate_no_override_or_forced_fallback",
        "boundary": "Relative tensor-view arithmetic only. Kernel may use different layouts, padding, quantized buffers, pointer arithmetic or SM-specific code. A below-limit descriptor is not universal Sage safety.",
    }


def describe_tensor(tensor, *, layout="unknown"):
    """Read actual PyTorch metadata without reading values or relocating storage."""
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.layout is not torch.strided:
        raise ValueError("Collector requires an actual strided torch.Tensor")
    storage = tensor.untyped_storage()
    # FakeTensor advertises its simulated CPU/CUDA device, while its backing
    # storage is meta. Check the actual storage, not a subclass name or label.
    if tensor.device.type == "meta" or storage.device.type == "meta":
        raise ValueError("A meta or virtual tensor has no actual backing storage")
    descriptor = dict(shape=list(tensor.shape), stride=list(tensor.stride()),
        storage_offset=tensor.storage_offset(), storage_bytes=storage.nbytes(),
        dtype=str(tensor.dtype), element_size=tensor.element_size())
    result = describe_storage(descriptor, layout=layout)
    result.update(descriptor_origin="actual_tensor_metadata_only", actual_device=str(tensor.device),
        tensor_values_read=False, tensor_or_storage_mutated=False, tensor_allocation_performed=False,
        kernel_executed=False, model_NFE=0)
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def inspect_file(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("Descriptor JSON exceeds the 64KiB read budget")
    try:
        document = json.loads(payload, object_pairs_hook=_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (RecursionError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid descriptor JSON") from exc
    if type(document) is not dict or set(document) != {"layout", "backend", "source_sha256", "tensors"}:
        raise ValueError("Input must contain exactly layout/backend/source_sha256/tensors")
    if (type(document["backend"]) is not str or not 1 <= len(document["backend"]) <= 256
            or not re.fullmatch(r"[a-zA-Z0-9_.:-]+", document["backend"])):
        raise ValueError("Backend declaration must be a bounded identifier")
    pin = document["source_sha256"]
    if pin is not None and (type(pin) is not str or not re.fullmatch(r"[a-fA-F0-9]{64}", pin)):
        raise ValueError("Source declaration must be null or SHA256")
    tensors = document["tensors"]
    if type(tensors) is not dict or set(tensors) != {"q", "k", "v"}:
        raise ValueError("Provide exactly q/k/v descriptors")
    return dict(
        schema="t8.h05.attention-storage-set.v1", descriptor_origin="JSON_declarations_not_live_tensor_proof",
        backend_declared=document["backend"], source_sha256_declared=pin,
        source_live_code_or_kernel_pin_verified=False, descriptor_bytes_read=len(payload),
        tensors={key: describe_storage(value, layout=document["layout"]) for key, value in tensors.items()},
        tensor_payload_read=False, embedded_config_executed=False, kernel_executed=False,
        changes_backend_or_model=False, model_NFE=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path, help="Bounded metadata-only q/k/v JSON")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_file(args.descriptor), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
