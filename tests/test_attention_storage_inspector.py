"""Actual tiny CPU metadata and no-allocation boundary arithmetic; no Sage GPU."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import inspect_h3_attention_storage as audit


def fixture():
    return dict(shape=[1, 8, 2, 4], stride=[64, 8, 4, 1], storage_offset=0,
                storage_bytes=128, dtype="torch.float16", element_size=2)


def test_actual_offset_transposed_and_strided_CPU_views_preserve_storage_and_values():
    base = torch.arange(160, dtype=torch.float32)
    view = base[8:136:2].reshape(1, 8, 2, 4).transpose(1, 2)
    before = base.clone()
    result = audit.describe_tensor(view, layout="HND")
    assert result["descriptor_origin"] == "actual_tensor_metadata_only"
    assert result["descriptor"]["stride"] == [128, 8, 16, 2]
    assert result["first_storage_element"] == 8 and result["last_storage_element"] == 134
    assert result["end_storage_byte_exclusive"] == 540
    assert result["last_sequence_row_offset_elements"] == 112
    assert result["contiguous_from_shape_stride"] is False
    assert not result["kernel_executed"] and not result["tensor_values_read"]
    assert not result["tensor_allocation_performed"] and torch.equal(base, before)
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert not torch.cuda.is_initialized()


def test_actual_contiguous_NHD_and_HND_describe_distinct_row_strides():
    tensor = torch.empty(1, 8, 2, 4, dtype=torch.float16)
    result = audit.describe_tensor(tensor, layout="NHD")
    transposed = audit.describe_tensor(tensor.transpose(1, 2), layout="HND")
    assert result["last_sequence_row_offset_elements"] == transposed["last_sequence_row_offset_elements"] == 56
    assert result["contiguous_from_shape_stride"] and not transposed["contiguous_from_shape_stride"]
    assert result["end_storage_byte_exclusive"] == tensor.untyped_storage().nbytes() == 128
    assert not result["kernel_execution_or_safety_verified"]


@pytest.mark.parametrize("row_offset", [2**31 - 1, 2**31, 2**31 + 1])
def test_signed_i32_boundary_is_exact_without_allocating_large_tensor(row_offset):
    descriptor = dict(shape=[1, 2, 1, 1], stride=[row_offset * 2, row_offset, 1, 1],
                      storage_offset=17, storage_bytes=(row_offset + 18) * 2,
                      dtype="torch.float16", element_size=2)
    before = copy.deepcopy(descriptor)
    result = audit.describe_storage(descriptor, layout="NHD")
    assert result["sequence_row_offset_at_or_over_signed_i32"] == (row_offset >= 2**31)
    assert result["end_byte_at_or_over_signed_i32"] is True
    assert result["last_storage_element"] == row_offset + 17
    assert result["padded_or_quantized_kernel_addresses_verified"] is False
    assert descriptor == before


def test_unknown_layout_is_not_inferred_from_shape_or_a_safe_span():
    result = audit.describe_storage(fixture())
    assert not result["rank4_sequence_descriptor_available"]
    assert result["sequence_dimension"] is None
    assert result["unknown_backend_policy"] == "report_and_delegate_no_override_or_forced_fallback"
    assert not result["kernel_execution_or_safety_verified"]


def test_empty_and_broadcast_CPU_views_are_metadata_not_dense_allocation():
    empty = audit.describe_tensor(torch.empty(1, 0, 2, 4), layout="NHD")
    assert empty["last_storage_element"] is None and empty["view_span_bytes"] == 0
    empty_offset = torch.empty(0).as_strided((1, 0, 2, 4), (8, 8, 4, 1), 100)
    offset_result = audit.describe_tensor(empty_offset, layout="NHD")
    assert offset_result["descriptor"]["storage_offset"] == 100
    assert offset_result["descriptor"]["storage_bytes"] == 0
    assert offset_result["end_storage_byte_exclusive"] is None
    assert not offset_result["end_byte_at_or_over_signed_i32"]
    assert offset_result["view_span_bytes"] == 0
    expanded = torch.ones(1, 1, 2, 4).expand(1, 8, 2, 4)
    result = audit.describe_tensor(expanded, layout="NHD")
    assert result["last_sequence_row_offset_elements"] == 0
    assert result["descriptor"]["stride"][1] == 0
    assert not result["contiguous_from_shape_stride"]


@pytest.mark.parametrize("field,value", [
    ("storage_bytes", 127), ("storage_offset", True), ("stride", [64, -8, 4, 1]),
    ("element_size", 4), ("shape", [1, 8, 2]), ("dtype", "arbitrary_kernel_dtype"),
])
def test_invalid_descriptor_reports_a_real_data_error_not_backend_substitution(field, value):
    descriptor = fixture()
    descriptor[field] = value
    with pytest.raises(ValueError):
        audit.describe_storage(descriptor, layout="NHD")


def test_bounded_readonly_JSON_CLI_keeps_pin_and_payload_as_unverified_declarations(tmp_path):
    value = dict(layout="NHD", backend="unknown-user-kernel", source_sha256="a" * 64,
                 tensors={key: fixture() for key in ("q", "k", "v")})
    path = tmp_path / "metadata.json"
    data = json.dumps(value).encode()
    path.write_bytes(data)
    result = audit.inspect_file(path)
    assert result["descriptor_bytes_read"] == len(data)
    assert result["descriptor_origin"] == "JSON_declarations_not_live_tensor_proof"
    assert not result["source_live_code_or_kernel_pin_verified"]
    assert not result["tensor_payload_read"] and not result["embedded_config_executed"]
    assert not result["kernel_executed"] and not result["changes_backend_or_model"]
    assert path.read_bytes() == data


@pytest.mark.parametrize("payload", [b'{"layout":"NHD","layout":"HND"}', b'{"x":NaN}', b" " * (audit.MAX_INPUT_BYTES + 1)],
                         ids=["duplicate-key", "nonfinite", "over-read-budget"])
def test_duplicate_nonfinite_and_oversized_inputs_rejected(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        audit.inspect_file(path)


def test_meta_or_non_tensor_cannot_be_labelled_actual_backed_storage():
    with pytest.raises(ValueError):
        audit.describe_tensor(torch.empty(1, 8, 2, 4, device="meta"), layout="NHD")
    with pytest.raises(ValueError):
        audit.describe_tensor(fixture(), layout="NHD")
