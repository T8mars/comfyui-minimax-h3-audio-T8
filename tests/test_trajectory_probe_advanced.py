from __future__ import annotations

import json
from pathlib import Path

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
    prepare_trajectory_model,
    MiniMaxH3TrajectorySampling,
    TrajectoryResumeNoise,
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
    return prepare_trajectory_model(model), sampler, sigmas, latent


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
    assert contract["schema"].endswith(".v2")
    assert contract["checkpoint_space"] == "internal_x_sigma_direct_transport"
    assert "without reconstruction" in contract["resume_noise_contract"]
    with pytest.raises(ValueError, match="exceeds gate"):
        build_trajectory_probe(model, sampler, sigmas, 2, 0.001, latent)

    partial = sigmas.clone()
    partial[0] = 0.99
    with pytest.raises(ValueError, match="complete sigma schedule"):
        build_trajectory_probe(model, sampler, partial, 2, 4096, latent)

    masked = dict(latent)
    masked["noise_mask"] = torch.ones(1)
    with pytest.raises(ValueError, match="refuses noise_mask"):
        build_trajectory_probe(model, sampler, sigmas, 2, 4096, masked)

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
    assert load_report["use_disable_noise"] is False
    assert load_report["resume_noise_output_required"] is True
    assert load_report["legacy_disable_noise_contract_detected"] is False
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


def test_resume_noise_preserves_internal_checkpoint_at_nonzero_sigma():
    model, _sampler, _sigmas, latent = _setup()
    sampling = model.get_model_object("model_sampling")
    assert isinstance(sampling, MiniMaxH3TrajectorySampling)
    noise = TrajectoryResumeNoise(123).generate_noise(latent)
    checkpoint = latent["samples"]
    assert noise is not checkpoint
    noise_video, noise_audio = noise.unbind()
    video, audio = checkpoint.unbind()
    sigma = torch.tensor(0.75)
    assert torch.equal(sampling.noise_scaling(sigma, noise_video, video), video)
    assert torch.equal(sampling.noise_scaling(sigma, noise_audio, audio), audio)
    assert torch.equal(sampling.inverse_noise_scaling(sigma, video), video)
    assert TrajectoryResumeNoise(123).seed == 123


def test_trajectory_nodes_are_explicit_and_experimental():
    schemas = [node.define_schema() for node in TRAJECTORY_PROBE_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3TrajectoryProbeT8Advanced",
        "MiniMaxH3TrajectoryCheckpointSaveT8Advanced",
        "MiniMaxH3TrajectoryCheckpointLoadT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    assert [item.id for item in schemas[0].outputs][-1] == "trajectory_model"
    assert [item.id for item in schemas[2].outputs] == [
        "checkpoint_latent",
        "remaining_sigmas",
        "report_json",
        "resume_noise",
    ]
    save_inputs = {item.id: item for item in schemas[1].inputs}
    assert save_inputs["confirm_save"].default is False

def test_trajectory_examples_route_the_direct_transport_contract():
    project_root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (project_root / "tests" / "fixtures" / "api" / "trajectory_probe_advanced_api.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert api["7"]["inputs"]["noise_seed"] == api["8"]["inputs"]["noise_seed"]
    assert api["9"]["inputs"]["model"] == ["7", 4]
    assert api["10"]["inputs"]["sigmas"] == ["7", 1]
    assert api["11"]["inputs"]["confirm_save"] is False

    workflow = json.loads(
        (
            project_root
            / "examples"
            / "workflows"
            / "2026-08-13_H3_Trajectory_Probe_Advanced_EXP.json"
        ).read_text(encoding="utf-8-sig")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == 18
    assert len(workflow["links"]) == 35
    assert all(
        isinstance(link, list)
        and len(link) >= 6
        and all(isinstance(link[index], int) for index in range(5))
        for link in workflow["links"]
    )

    def source(target_id, input_index):
        return next(
            (link[1], link[2])
            for link in workflow["links"]
            if link[3] == target_id and link[4] == input_index
        )

    assert source(13, 0) == (7, 0)
    assert source(13, 6) == (17, 0)
    assert source(8, 0) == (13, 4)
    assert source(16, 0) == (15, 3)
    assert source(16, 3) == (15, 1)
    assert source(16, 4) == (15, 0)
    assert nodes[14]["widgets_values"][-1] is False
    assert [nodes[node_id]["mode"] for node_id in (11, 12, 15, 16)] == [2, 2, 2, 2]

    for link_id, source_id, output_index, target_id, input_index, _kind in workflow[
        "links"
    ]:
        assert nodes[target_id]["inputs"][input_index]["link"] == link_id
        assert link_id in (nodes[source_id]["outputs"][output_index].get("links") or [])
