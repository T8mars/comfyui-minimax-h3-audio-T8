"""Solar file/clock contracts; no published weight or quality qualification."""
import hashlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from comfy.k_diffusion.sampling import sample_lcm

from h3_audio_t8_pkg import solar_structural as structural
from h3_audio_t8_pkg import solar_structural_sampling as sampling


def test_source_interval_and_delivered_clock_are_explicit_without_padding_tail():
    plan = structural.plan_structure(40, 36, 64, 32, source_start=3)
    assert plan["source_range"] == [3, 25]
    assert (plan["frames"], plan["video_ticks"], plan["audio_ticks"]) == (22, 7, 37)
    assert plan["qwen_source_indices"] == [3, 15]
    assert plan["warnings"] and not plan["trained_consumer_qualified"]
    plan = structural.plan_structure(400, 400, 32, 32)
    assert plan["frames"] == 396
    assert plan["qwen_source_indices"] == list(range(0, 158, 12))


@pytest.mark.parametrize("kwargs", [
    {"source_start": 40}, {"source_start": -1}, {"fps": 30},
    {"fps": float("nan")}, {"length": True}, {"width": 48},
])
def test_real_structure_clock_errors_are_explained(kwargs):
    values = dict(source_frames=40, length=22, width=32, height=32)
    values.update(kwargs)
    with pytest.raises(ValueError):
        structural.plan_structure(**values)


def test_core_lcm_and_source_grid_are_selected_without_reimplementing_sampler():
    sampler, sigmas, report = sampling.sampling_contract()
    assert sampler.sampler_function is sample_lcm
    expected = torch.tensor([1., 36/37, 12/13, 4/5, 0.], dtype=torch.float32)
    torch.testing.assert_close(sigmas, expected, rtol=0, atol=0)
    assert report["published_schedule"]
    _, other, report = sampling.sampling_contract(3)
    assert len(other) == 4 and not report["published_schedule"]


def test_silence_reader_keeps_real_encoded_values_and_checks_exact_file_receipt(tmp_path):
    original = torch.arange(2*32*37, dtype=torch.float32).reshape(2, 32, 37) / 2000
    path = tmp_path / "user-silence.safetensors"
    save_file({"silence_170f": original}, str(path), metadata={"artifact": "synthetic-test"})
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    value, identity = structural.load_silence(path, expected_sha256=sha)
    assert torch.equal(value, original.permute(1, 0, 2).unsqueeze(0))
    assert identity["sha256"] == sha and not identity["fixed_source_bytes"]
    with pytest.raises(ValueError, match="SHA256"):
        structural.load_silence(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="no tensor"):
        structural.load_silence(path, key="absent")
    save_file({"silence_170f": torch.full((2, 32, 37), float("nan"))}, str(path))
    with pytest.raises(ValueError, match="finite"):
        structural.load_silence(path)


def test_silence_owner_copies_input_and_detects_its_own_tampered_binding():
    value = torch.ones(1, 32, 2, 37)
    owner = sampling.SilenceOwner(value, 22)
    value.zero_()
    assert torch.count_nonzero(owner.silence) == 32 * 2 * 37
    owner.validate()
    owner.seed = 23
    with pytest.raises(ValueError, match="binding changed"):
        owner.validate()


@pytest.mark.parametrize("bad", [torch.zeros(1, 32, 37), torch.zeros(2, 32, 2, 37),
                                 torch.zeros(1, 31, 2, 37), torch.zeros(1, 32, 2, 0)])
def test_silence_shape_errors_are_actual_protocol_errors(bad):
    with pytest.raises(ValueError):
        sampling.SilenceOwner(bad, 0)


def test_missing_diffusion_clock_is_a_real_protocol_error_without_sampler_defaults():
    executor = SimpleNamespace(class_obj=object())
    with pytest.raises(ValueError, match="actual diffusion sigma_shift_video"):
        sampling._core_shifts(executor, {})
    options = {"minimax_h3_sigma_shift_video": 6., "minimax_h3_sigma_shift_audio": 2.}
    assert sampling._core_shifts(executor, options) == (6., 2.)
    options["minimax_h3_sigma_shift_audio"] = None
    with pytest.raises(ValueError, match="positive finite"):
        sampling._core_shifts(executor, options)


@pytest.mark.parametrize("mask", [torch.full((1, 1, 2, 37), 1.1),
                                 torch.full((1, 1, 2, 37), float("nan")),
                                 torch.ones(1, 1, 2, 36)])
def test_audio_mask_invalid_coverage_and_extent_remain_real_errors(mask):
    with pytest.raises(ValueError, match="mask"):
        sampling._replacement_rows(torch.zeros(1, 32, 2, 37), mask)
