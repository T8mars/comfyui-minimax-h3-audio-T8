"""Virtual storage must not be reported as actual backed CPU/GPU QKV."""
from pathlib import Path
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import inspect_h3_attention_storage as audit


@pytest.mark.parametrize("origin", ["factory", "from_actual"], ids=["fake-factory", "fake-from-real"])
def test_fake_CPU_view_is_not_actual_backed_storage(origin):
    actual = torch.arange(64, dtype=torch.float32).reshape(1, 8, 2, 4)
    original = actual.clone()
    storage = actual.untyped_storage()
    with FakeTensorMode() as mode:
        fake = (torch.empty(1, 8, 2, 4, dtype=torch.float32, device="cpu")
                if origin == "factory" else mode.from_tensor(actual))
    assert fake.device.type == "cpu" and fake.layout is torch.strided
    assert fake.untyped_storage().device.type == "meta"
    assert fake.untyped_storage().nbytes() == actual.untyped_storage().nbytes()
    with pytest.raises(ValueError, match="backing storage"):
        audit.describe_tensor(fake, layout="NHD")
    assert torch.equal(actual, original) and actual.untyped_storage().data_ptr() == storage.data_ptr()
    assert not torch.cuda.is_initialized()


def test_actual_Parameter_storage_is_still_valid_without_type_based_subclass_ban():
    tensor = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(1, 8, 2, 4))
    original, storage = tensor.detach().clone(), tensor.untyped_storage()
    result = audit.describe_tensor(tensor, layout="NHD")
    assert result["descriptor_origin"] == "actual_tensor_metadata_only"
    assert result["descriptor"]["storage_bytes"] == 256
    assert result["last_storage_element"] == 63 and result["actual_device"] == "cpu"
    assert not result["tensor_values_read"] and not result["kernel_execution_or_safety_verified"]
    assert torch.equal(tensor, original) and tensor.untyped_storage().data_ptr() == storage.data_ptr()
    assert not torch.cuda.is_initialized()
