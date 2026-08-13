from __future__ import annotations

import pytest
import torch

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.nodes_trajectory_probe_advanced import (
    TRAJECTORY_PROBE_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.trajectory_probe_advanced import (
    build_trajectory_probe,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
)


class FakeConfig:
    sampling_settings = {}


class FakeBase:
    model_config = FakeConfig()


class FakeModel:
    def __init__(self):
        self.model = FakeBase()
        self.model_options = {}
        self.patches_uuid = "session-a"
        self._sampling = None

    def clone(self):
        clone = FakeModel()
        clone.patches_uuid = self.patches_uuid
        clone.model_options = dict(self.model_options)
        clone._sampling = self._sampling
        return clone

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        if self._sampling is None:
            import comfy.model_sampling

            sampling = type(
                "FakeSampling",
                (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST),
                {},
            )(FakeConfig())
            self._sampling = sampling
        return self._sampling

    def add_object_patch(self, name, value):
        assert name == "model_sampling"
        self._sampling = value


def _setup():
    latent, _frames = empty_av_latent(128, 128, 124)
    model = FakeModel()
    model, sampler, sigmas = setup_dual_clock_sampling(
        model,
        latent,
        4,
        12.0,
        3.0,
    )
    model.patches_uuid = "session-a"
    return model, sampler, sigmas, latent


def test_probe_splits_exactly_and_refuses_stateful_or_oversize():
    model, sampler, sigmas, latent = _setup()
    contract, high, low = build_trajectory_probe(
        model,
        sampler,
        sigmas,
        2,
        4096.0,
        latent,
    )
    assert torch.equal(high, sigmas[:3])
    assert torch.equal(low, sigmas[2:])
    assert contract["split_step"] == 2
    assert contract["resume_noise_contract"].startswith("use ComfyUI DisableNoise")
    with pytest.raises(ValueError, match="exceeds gate"):
        build_trajectory_probe(model, sampler, sigmas, 2, 0.001, latent)

    class StatefulSampler:
        sampler_function = staticmethod(lambda: None)
        extra_options = {}

    with pytest.raises(ValueError, match="only T8 stable"):
        build_trajectory_probe(model, StatefulSampler(), sigmas, 2, 4096, latent)

    model.model_options = {
        "transformer_options": {"patches_replace": {"dit": {("double_block", 0): object()}}}
    }
    with pytest.raises(ValueError, match="refuses patches_replace"):
        build_trajectory_probe(model, sampler, sigmas, 2, 4096, latent)


def test_checkpoint_requires_confirmation_and_same_session_identity(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.trajectory_probe_advanced as probe

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(probe.folder_paths, "get_output_directory", lambda: str(output))
    model, sampler, sigmas, latent = _setup()
    contract, _high, _low = build_trajectory_probe(
        model,
        sampler,
        sigmas,
        2,
        4096.0,
        latent,
    )
    path, report = save_trajectory_checkpoint(
        contract,
        latent,
        "probe",
        False,
    )
    assert path == ""
    assert report["status"] == "not_saved"

    path, report = save_trajectory_checkpoint(
        contract,
        latent,
        "probe",
        True,
    )
    assert report["status"] == "saved"
    loaded, load_report, remaining = load_trajectory_checkpoint(
        path,
        model,
        sampler,
        sigmas,
    )
    assert load_report["use_disable_noise"] is True
    assert torch.equal(remaining, sigmas[2:])
    original_video, original_audio = latent["samples"].unbind()
    loaded_video, loaded_audio = loaded["samples"].unbind()
    assert torch.equal(original_video, loaded_video)
    assert torch.equal(original_audio, loaded_audio)

    model.patches_uuid = "session-b"
    with pytest.raises(ValueError, match="MODEL identity"):
        load_trajectory_checkpoint(path, model, sampler, sigmas)


def test_checkpoint_rejects_sigma_or_metadata_tampering(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.trajectory_probe_advanced as probe

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(probe.folder_paths, "get_output_directory", lambda: str(output))
    model, sampler, sigmas, latent = _setup()
    contract, _high, _low = build_trajectory_probe(
        model,
        sampler,
        sigmas,
        1,
        4096.0,
        latent,
    )
    path, _report = save_trajectory_checkpoint(contract, latent, "tamper", True)
    changed = sigmas.clone()
    changed[1] += 0.01
    with pytest.raises(ValueError, match="sigma schedule"):
        load_trajectory_checkpoint(path, model, sampler, changed)


def test_trajectory_nodes_are_explicit_and_experimental():
    schemas = [node.define_schema() for node in TRAJECTORY_PROBE_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3TrajectoryProbeT8Advanced",
        "MiniMaxH3TrajectoryCheckpointSaveT8Advanced",
        "MiniMaxH3TrajectoryCheckpointLoadT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    save_inputs = {item.id: item for item in schemas[1].inputs}
    assert save_inputs["confirm_save"].default is False
