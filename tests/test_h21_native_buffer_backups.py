"""Real native VAE / Core Dynamic registered-buffer identity, CPU only."""

from contextlib import contextmanager
from copy import deepcopy
import importlib
from types import SimpleNamespace

import comfy.model_patcher
import comfy_aimdo.control
import comfy_aimdo.host_buffer
import pytest
import torch

from h3_audio_t8_pkg import long_video_dual_identity as identity
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.progressive_producers import native_producer_identity
from h3_audio_t8_pkg.modular_sampling.control_owner_identity import integrity_snapshot
from test_h21_control_owner_identity import (
    real_native_encoder,
    build_controls,
    sample_head,
)
from test_h21_control_regions import source_frames


@pytest.fixture(autouse=True)
def actual_CPU():
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def dynamic(encoder):
    if comfy_aimdo.host_buffer.lib is None:
        # Match the real Core CPU dynamic-load fixture; no device initialization.
        assert comfy_aimdo.control.init()
        importlib.reload(comfy_aimdo.host_buffer)
    network = encoder.first_stage_model
    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcherDynamic.__init__(
        patcher, network, torch.device("cpu"), torch.device("cpu")
    )
    encoder.patcher = patcher
    network.pixel_mean_comfy_model_dtype = torch.float64
    return patcher


@contextmanager
def loaded(patcher):
    try:
        patcher.load(torch.device("cpu"))
        yield patcher
    finally:
        patcher.restore_loaded_backups()
        patcher.model.dynamic_patchers.discard(id(patcher))


def test_real_dynamic_load_nonpersistent_backups_are_not_a_malformed_state():
    encoder = real_native_encoder()
    patcher = dynamic(encoder)
    before = identity.content_identity(patcher.model_state_dict())
    source = encoder.first_stage_model.pixel_mean
    with loaded(patcher):
        assert patcher.backup_buffers["pixel_mean"] is source
        assert "pixel_mean" in patcher.model._non_persistent_buffers_set
        assert "pixel_mean" not in patcher.model_state_dict()
        live = patcher.model.pixel_mean
        assert live is not source and live.dtype is torch.float64
        actual = identity._original_state(patcher, patcher.model_state_dict())
        assert identity.content_identity(actual) == before
        assert "pixel_mean" not in actual  # Do not forge a persistent key.
        assert patcher.model.pixel_mean is live
        assert patcher.backup_buffers["pixel_mean"] is source


@pytest.mark.parametrize('corrupt_export', [False, True])
def test_registered_nonpersistent_buffer_exported_by_native_state_hook(corrupt_export):
    encoder = real_native_encoder()
    patcher = dynamic(encoder)
    def export_registered_buffer(module, state, prefix, metadata):
        state[prefix + 'pixel_mean'] = module.pixel_mean.detach()
    hook = patcher.model.register_state_dict_post_hook(export_registered_buffer)
    try:
        before = identity.content_identity(patcher.model_state_dict())
        with loaded(patcher):
            state = patcher.model_state_dict()
            live = patcher.model.pixel_mean
            assert 'pixel_mean' in patcher.model._non_persistent_buffers_set
            if corrupt_export:
                state['pixel_mean'] = state['pixel_mean'] + .125
                with pytest.raises(ValueError, match='exported buffer'):
                    identity._original_state(patcher, state)
            else:
                restored = identity._original_state(patcher, state)
                assert identity.content_identity(restored) == before
                assert restored['pixel_mean'] is patcher.backup_buffers['pixel_mean']
                assert patcher.model.pixel_mean is live
    finally:
        hook.remove()


@pytest.mark.parametrize("alias", [False, True])
def test_real_native_producer_binds_original_buffers_across_dynamic_cast_and_source_change(
    alias,
):
    encoder = real_native_encoder()
    if alias:
        encoder.first_stage_model.register_buffer(
            "pixel_alias", encoder.first_stage_model.pixel_mean, persistent=False
        )
    patcher = dynamic(encoder)
    before = native_producer_identity(encoder, "video_vae")
    assert before.get("portable_cache_reuse") is not False
    with loaded(patcher):
        assert native_producer_identity(encoder, "video_vae") == before
        live = patcher.model.pixel_mean
        original = patcher.backup_buffers["pixel_mean"]
        original.add_(0.125)
        assert patcher.model.pixel_mean is live
        assert native_producer_identity(encoder, "video_vae") != before
        original.sub_(0.125)


def test_actual_official_control_lazy_encoder_snapshot_uses_original_buffer_contents(
    monkeypatch,
):
    case = build_controls(monkeypatch, "official_model_patch")
    patcher = dynamic(case.encoder)
    owner = case.low_states[0].owner
    before = integrity_snapshot(owner)
    with loaded(patcher):
        assert integrity_snapshot(owner) == before
        original = patcher.backup_buffers["pixel_mean"]
        saved = original.clone()
        try:
            original.add_(0.125)
            assert integrity_snapshot(owner) != before
        finally:
            original.copy_(saved)


@pytest.mark.parametrize("damage", ["nan", "unregistered", "wrong_shape"])
def test_true_corrupt_backup_is_not_hidden_as_a_native_nonpersistent_buffer(damage):
    encoder = real_native_encoder()
    patcher = dynamic(encoder)
    with loaded(patcher):
        if damage == "nan":
            patcher.backup_buffers["pixel_mean"] = torch.full_like(
                patcher.backup_buffers["pixel_mean"], float("nan")
            )
        elif damage == "unregistered":
            patcher.backup_buffers["not_a_registered_buffer"] = torch.ones(1)
        else:
            patcher.backup_buffers["pixel_mean"] = torch.ones(2)
        with pytest.raises(ValueError):
            identity._original_state(patcher, patcher.model_state_dict())


def test_unknown_backup_container_is_unverified_not_a_forged_original_tensor():
    encoder = real_native_encoder()
    patcher = encoder.patcher
    state = patcher.model_state_dict()
    patcher.backup_buffers["pixel_mean"] = SimpleNamespace(
        original=encoder.first_stage_model.pixel_mean
    )
    with pytest.raises(UnverifiedModelStack):
        identity._original_state(patcher, state)
    assert set(patcher.model_state_dict()) == set(state)


def test_native_original_identity_helpers_never_unpatch_or_mutate_live_buffers():
    encoder = real_native_encoder()
    patcher = dynamic(encoder)
    with loaded(patcher):
        live = dict(patcher.model.named_buffers())
        backups = dict(patcher.backup_buffers)
        state = deepcopy(identity._original_state(patcher, patcher.model_state_dict()))
        for _ in range(2):
            assert identity.content_identity(
                identity._original_state(patcher, patcher.model_state_dict())
            ) == identity.content_identity(state)
        assert all(dict(patcher.model.named_buffers())[k] is v for k, v in live.items())
        assert all(patcher.backup_buffers[k] is v for k, v in backups.items())


def test_unknown_buffer_record_keeps_actual_native_encoder_executable_not_portable():
    encoder = real_native_encoder()
    frames = source_frames(5, width=32, height=32)
    expected = encoder.encode(frames)
    encoder.patcher.backup_buffers["pixel_mean"] = SimpleNamespace(
        original=encoder.first_stage_model.pixel_mean
    )
    report = native_producer_identity(encoder, "video_vae")
    assert report["portable_cache_reuse"] is False
    assert report["execution_weight_content_verified"] is False
    assert report["execution_weight_sha256"] is None
    assert len(report["execution_known_tensor_sha256"]) == 64
    actual = encoder.encode(frames)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unknown_lazy_vae_backup_keeps_actual_control_sampling_nonportable(monkeypatch):
    case = build_controls(monkeypatch, "official_model_patch")
    case.encoder.patcher.backup_buffers["pixel_mean"] = SimpleNamespace(
        original=case.encoder.first_stage_model.pixel_mean
    )
    snapshot = integrity_snapshot(case.low_states[0].owner)
    assert snapshot["lazy_encoder"]["unverified_buffer_backups"] is not None
    completed = sample_head(case)
    assert completed.verify()["execution"]["portable_identity"] is False
    assert case.low_states[0].gated_calls == 4


@pytest.mark.parametrize("target", ["live_weight", "adjacent_backup"])
def test_unknown_backup_record_does_not_hide_known_nonfinite_neighbor(target):
    encoder = real_native_encoder()
    patcher = encoder.patcher
    patcher.backup_buffers["pixel_mean"] = SimpleNamespace(
        original=encoder.first_stage_model.pixel_mean
    )
    if target == "live_weight":
        next(encoder.first_stage_model.parameters()).view(-1)[0] = float("nan")
    else:
        patcher.backup_buffers["latents_mean"] = torch.full_like(
            encoder.first_stage_model.latents_mean, float("nan")
        )
    with pytest.raises(ValueError) as error:
        native_producer_identity(encoder, "video_vae")
    assert not isinstance(error.value, UnverifiedModelStack)
