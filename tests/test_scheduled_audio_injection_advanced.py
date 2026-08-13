from __future__ import annotations

import copy
import json

import pytest
import torch

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.nodes_scheduled_audio_injection_advanced import (
    SCHEDULED_AUDIO_INJECTION_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.scheduled_audio_injection_advanced import (
    _progress_weight,
    sample_minimax_h3_scheduled_audio_euler,
    setup_scheduled_drive_audio_injection,
)


class FakeSampling:
    noise_scale = 1.0


class FakeBaseModel:
    def __init__(self):
        self.model_config = object()
        self.model_sampling = FakeSampling()
        self.diffusion_model = type("Diffusion", (), {})()

    def audio_scale(self):
        return 1.0

    def extra_conds(self, **_kwargs):
        return {}


class FakeModelPatcher:
    def __init__(self):
        self.model = FakeBaseModel()
        self.model_options = {}

    def clone(self):
        cloned = FakeModelPatcher()
        cloned.model = self.model
        cloned.model_options = copy.deepcopy(self.model_options)
        return cloned

    def get_model_object(self, name):
        return getattr(self.model, name)

    def add_object_patch(self, name, value):
        setattr(self.model, name, value)


class FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self, latent):
        self.latent = latent
        self.calls = 0

    def encode(self, waveform):
        assert waveform.shape[-1] in {1, 2}
        self.calls += 1
        return self.latent.clone()


def make_latent():
    return empty_av_latent(64, 64, 22)[0]


def make_audio(samples=29600):
    return {
        "waveform": torch.linspace(-0.25, 0.25, samples).reshape(1, 1, -1),
        "sample_rate": 32000,
    }


def test_progress_envelopes_and_window():
    assert _progress_weight(0.1, 0.2, 0.8, "constant") == 0.0
    assert _progress_weight(0.2, 0.2, 0.8, "constant") == 1.0
    assert _progress_weight(0.5, 0.2, 0.8, "fade_in") == pytest.approx(0.5)
    assert _progress_weight(0.5, 0.2, 0.8, "fade_out") == pytest.approx(0.5)


def test_report_only_is_latent_and_mux_passthrough(monkeypatch):
    latent = make_latent()
    audio = make_audio()
    final_audio = make_audio(100)
    sentinel_sampler = object()
    sentinel_sigmas = torch.tensor([1.0, 0.0])
    patched = object()

    monkeypatch.setattr(
        "h3_audio_t8_pkg.scheduled_audio_injection_advanced.setup_dual_clock_sampling",
        lambda *args, **kwargs: (patched, sentinel_sampler, sentinel_sigmas),
    )
    output = setup_scheduled_drive_audio_injection(
        FakeModelPatcher(),
        latent,
        audio,
        FakeAudioVAE(torch.ones_like(latent["samples"].unbind()[1])),
        4,
        12.0,
        3.0,
        mode="report_only",
        final_audio=final_audio,
    )
    assert output[0] is patched
    assert output[1] is sentinel_sampler
    assert output[2] is sentinel_sigmas
    assert output[3] is latent
    assert output[4] is final_audio
    report = json.loads(output[5])
    assert report["status"] == "bypass"
    assert report["bit_exact_bypass_claim"] is True


def test_scheduled_setup_encodes_once_and_forces_audio_mask(monkeypatch):
    latent = make_latent()
    template = latent["samples"].unbind()[1]
    encoded = torch.full_like(template, 0.125)
    vae = FakeAudioVAE(encoded)
    audio = make_audio()
    patched = FakeModelPatcher()
    sentinel_sigmas = torch.tensor([1.0, 0.5, 0.0])

    monkeypatch.setattr(
        "h3_audio_t8_pkg.scheduled_audio_injection_advanced.setup_dual_clock_sampling",
        lambda *args, **kwargs: (patched, object(), sentinel_sigmas),
    )
    output = setup_scheduled_drive_audio_injection(
        FakeModelPatcher(),
        latent,
        audio,
        vae,
        2,
        12.0,
        3.0,
        mode="scheduled_injection",
        injection_seed=42,
    )
    assert vae.calls == 1
    controlled_audio = output[3]["samples"].unbind()[1]
    audio_mask = output[3]["noise_mask"].unbind()[1]
    assert torch.equal(controlled_audio, encoded)
    assert torch.all(audio_mask == 1)
    assert output[4] is audio
    report = json.loads(output[5])
    assert report["encoded_once"] is True
    assert report["deterministic_fixed_noise"] is True
    assert report["recommended_status"].startswith("EXP")


def test_sampler_full_strength_tracks_exact_audio_path_and_locks_final():
    video_values = 2
    source = torch.tensor([[[[2.0, -1.0]]]])
    noise = torch.tensor([[[[0.5, 1.5]]]])
    initial = torch.zeros((1, 4))
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    def denoise_to_input(x, sigma, **_kwargs):
        return x

    output = sample_minimax_h3_scheduled_audio_euler(
        denoise_to_input,
        initial,
        sigmas,
        video_values=video_values,
        packed_values=4,
        shift_video=1.0,
        shift_audio=1.0,
        audio_velocity_is_raw=True,
        source_audio_x0=source,
        fixed_audio_noise=noise,
        noise_scale=1.0,
        start_percent=0.0,
        end_percent=1.0,
        strength=1.0,
        envelope="constant",
        lock_final_audio=True,
        disable=True,
    )
    assert torch.equal(output[:, :video_values], torch.zeros((1, 2)))
    assert torch.equal(output[:, video_values:], source.reshape(1, -1))


def test_same_seed_builds_same_fixed_noise(monkeypatch):
    latent = make_latent()
    template = latent["samples"].unbind()[1]
    encoded = torch.zeros_like(template)
    captured = []

    class CaptureSampler:
        def __init__(self, function):
            captured.append(function)

    monkeypatch.setattr(
        "h3_audio_t8_pkg.scheduled_audio_injection_advanced.setup_dual_clock_sampling",
        lambda *args, **kwargs: (
            FakeModelPatcher(),
            object(),
            torch.tensor([1.0, 0.0]),
        ),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.scheduled_audio_injection_advanced.comfy.samplers.KSAMPLER",
        CaptureSampler,
    )
    for _ in range(2):
        setup_scheduled_drive_audio_injection(
            FakeModelPatcher(),
            latent,
            make_audio(),
            FakeAudioVAE(encoded),
            1,
            12.0,
            3.0,
            mode="scheduled_injection",
            injection_seed=123,
        )

    x = torch.zeros((1, template.numel() + latent["samples"].unbind()[0][0].numel()))
    sigmas = torch.tensor([1.0, 0.0])

    def denoise_to_input(current, sigma, **_extra):
        return current

    assert captured[0].__name__ == "sample_minimax_h3_scheduled_audio_euler"
    assert captured[1].__name__ == "sample_minimax_h3_scheduled_audio_euler"
    first = captured[0](denoise_to_input, x.clone(), sigmas, disable=True)
    second = captured[1](denoise_to_input, x.clone(), sigmas, disable=True)
    assert torch.equal(first, second)


def test_conflicting_patch_stack_fails_closed_before_encoding(monkeypatch):
    model = FakeModelPatcher()
    model.model_options = {
        "transformer_options": {"patches_replace": {"dit": {("double_block", 0): object()}}}
    }
    latent = make_latent()
    vae = FakeAudioVAE(torch.zeros_like(latent["samples"].unbind()[1]))
    with pytest.raises(RuntimeError, match="unverified model patch stack"):
        setup_scheduled_drive_audio_injection(
            model,
            latent,
            make_audio(),
            vae,
            4,
            12.0,
            3.0,
            mode="scheduled_injection",
        )
    assert vae.calls == 0


@pytest.mark.parametrize(
    ("start", "end", "strength"),
    [(-0.1, 1.0, 1.0), (0.5, 0.4, 1.0), (0.0, 1.1, 1.0), (0.0, 1.0, 1.1)],
)
def test_invalid_schedule_rejected(start, end, strength):
    latent = make_latent()
    with pytest.raises(ValueError):
        setup_scheduled_drive_audio_injection(
            FakeModelPatcher(),
            latent,
            make_audio(),
            FakeAudioVAE(torch.zeros_like(latent["samples"].unbind()[1])),
            4,
            12.0,
            3.0,
            mode="report_only",
            start_percent=start,
            end_percent=end,
            strength=strength,
        )


def test_node_schema_is_opt_in_and_advanced():
    assert len(SCHEDULED_AUDIO_INJECTION_ADVANCED_NODE_CLASSES) == 1
    schema = SCHEDULED_AUDIO_INJECTION_ADVANCED_NODE_CLASSES[0].define_schema()
    assert schema.node_id == "MiniMaxH3ScheduledDriveAudioInjectionT8Advanced"
    assert schema.is_experimental is True
    inputs = {entry.id: entry for entry in schema.inputs}
    assert inputs["mode"].default == "report_only"
    assert inputs["lock_final_audio"].default is False
    assert inputs["allow_unverified_patch_stack"].default is False
