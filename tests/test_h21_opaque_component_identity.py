"""Opaque encoders still execute; absence of patcher never certifies contents."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from helpers import FakeVideoVAE
from h3_audio_t8_pkg.patch_stack_policy import (
    model_identity_matches, nonportable_component_identity,
)
from h3_audio_t8_pkg.progressive_producers import native_producer_identity
from h3_audio_t8_pkg.reference_runtime import portable_producer
from test_progressive_producers import component


def test_actual_patcherless_encoder_retains_ordinary_execution_without_content_claim():
    encoder = FakeVideoVAE()
    pixels = torch.zeros(1, 32, 32, 3)
    before = encoder.encode(pixels)
    first = native_producer_identity(encoder, "video_vae")
    second = native_producer_identity(encoder, "video_vae")
    assert first["portable_cache_reuse"] is False
    assert first["execution_weight_sha256"] is None
    assert first["execution_weight_content_verified"] is False and first["tensor_count"] is None
    assert first["sha256"] != second["sha256"]
    assert model_identity_matches(first, second)
    torch.testing.assert_close(encoder.encode(pixels), before, rtol=0, atol=0)


def test_patcherless_encoder_cannot_be_promoted_to_portable_reference_producer():
    with pytest.raises(ValueError, match="ordinary native refs.*native CLIP/VAE"):
        portable_producer(FakeVideoVAE(), "video_vae")


def test_opaque_optional_patcher_without_weight_inspection_does_not_block_encoding():
    encoder = FakeVideoVAE()
    encoder.patcher = SimpleNamespace()
    pixels = torch.zeros(1, 32, 32, 3)
    expected = encoder.encode(pixels)
    report = native_producer_identity(encoder, "video_vae")
    assert report["portable_cache_reuse"] is False
    assert report["execution_weight_sha256"] is None
    torch.testing.assert_close(encoder.encode(pixels), expected, rtol=0, atol=0)


def test_replaced_actual_encode_method_changes_process_local_selection():
    encoder = FakeVideoVAE()
    first = native_producer_identity(encoder, "video_vae")
    original = encoder.encode

    def replacement(self, pixels):
        return original(pixels)

    encoder.encode = MethodType(replacement, encoder)
    second = native_producer_identity(encoder, "video_vae")
    assert not model_identity_matches(first, second)


def test_present_real_patcher_still_checks_actual_nan_weight_content():
    encoder = component("video_vae")
    with torch.no_grad():
        encoder.first_stage_model.probe.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite|NaN|nonfinite"):
        nonportable_component_identity(encoder, "explicit unverified state", schema="test")


def test_real_inspection_property_error_is_not_swallowed_or_retried():
    calls = []

    class Encoder:
        @property
        def patcher(self):
            calls.append(1)
            raise RuntimeError("actual accessor failure")

    with pytest.raises(RuntimeError, match="actual accessor failure"):
        nonportable_component_identity(Encoder(), "unknown", schema="test")
    assert calls == [1]
