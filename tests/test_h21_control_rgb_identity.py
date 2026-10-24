"""Native resized IMAGE strides remain valid control owner content."""

import hashlib

import pytest
import torch

from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.control_region_runtime import apply_h3_fun_control_region
from h3_audio_t8_pkg.modular_sampling import control_owner_identity as owners
from test_h21_control_owner_identity import real_native_encoder, real_tower
from test_h21_control_regions import source_frames
from test_modular_hyperflow import inputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_actual_native_resized_rgb_hashes_logical_bytes_without_changing_storage(dtype):
    # Core's PIL Lanczos consumes its normal FP32 IMAGE input. Cast its actual
    # noncontiguous result to test the identity adapter's storage dtypes, not
    # to claim the unmodified Core supports BF16 numpy/PIL input.
    source = torch.linspace(0, 1, 5 * 48 * 64 * 3).reshape(5, 48, 64, 3)
    prepared = fun._fit_control_video(source, width=32, height=32, length=5, fit_mode="stretch").to(dtype)
    assert prepared.shape == (5, 32, 32, 3) and not prepared.is_contiguous()
    expected = {
        "tensor_sha256": hashlib.sha256(prepared.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "dtype": str(prepared.dtype), "shape": list(prepared.shape),
    }
    pointer, strides = prepared.data_ptr(), prepared.stride()
    assert owners.prepared_image_identity(prepared) == expected
    assert (prepared.data_ptr(), prepared.stride()) == (pointer, strides)
    assert owners.prepared_image_identity(prepared.contiguous()) == expected
    prepared[4, 16, 15, 2] += 0.25
    assert owners.prepared_image_identity(prepared) != expected


def test_image_logical_copy_is_bounded_even_when_one_frame_exceeds_budget(monkeypatch):
    image = torch.arange(3 * 3 * 769 * 1025, dtype=torch.float32).reshape(3, 3, 769, 1025).movedim(1, -1)
    expected = owners.content_identity(image.contiguous())
    original = torch.Tensor.contiguous
    copied = []

    def bounded(value, *args, **kwargs):
        copied.append(value.numel() * value.element_size())
        assert copied[-1] <= 4 * 1024 * 1024
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", bounded)
    assert owners.prepared_image_identity(image) == expected
    assert copied and len(copied) > 3


def test_nonfinite_resized_image_still_rejected():
    image = torch.zeros(2, 3, 17, 19).movedim(1, -1)
    image[1, 15, 17, 2] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        owners.prepared_image_identity(image)


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_actual_native_resized_control_can_project_completed_owner(monkeypatch, backend):
    low, _, _, positive = inputs(monkeypatch)
    encoder = real_native_encoder()
    bundle, _ = real_tower(backend, 211021201)
    frames = source_frames(5, width=64, height=48)
    selected, _, _, _ = apply_h3_fun_control_region(
        low, positive, bundle, encoder, frames, 32, 32, 5, "pose", "stretch",
        0.7, 0.0, 1.0, stage="LOW",
    )
    owner = selected.get_attachment(owners.KEY)[0]
    assert owner.prepared.shape == (5, 32, 32, 3) and not owner.prepared.is_contiguous()
    _, contract = owners.project(selected)
    assert contract["controls"][0]["prepared_RGB"] == owners.content_identity(owner.prepared.contiguous())
    with torch.inference_mode():
        owner.prepared[4, 16, 15, 2] += 0.25
    with pytest.raises(ValueError, match="prepared RGB changed"):
        owners.project(selected)
