"""Exact source data restore, without re-SAM/encode, never weakened stage audit."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import face_source_storage as storage, face_source_nodes as nodes
from h3_audio_t8_pkg.modular_sampling.face_stage import bind_multiface_face_stage, audit_multiface_face_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_modular_face_multiface import _job
from test_face_refine_parity_advanced import _locked_av, _rehash
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning


def case(seed=319):
    parent, source, plan = _job()
    latent = _locked_av()
    prepared, _, _ = sampling.setup_dual_clock_sampling(model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    bound, selected, table, context, _ = bind_multiface_face_stage(
        plan, source, parent, prepared, sampler, sigmas, latent)
    torch.manual_seed(seed)
    result = sample_stage(RandomNoise.execute(seed).result[0],
        BasicGuider.execute(bound, conditioning()).result[0], selected, table, latent, context)[2]
    audio = {"waveform": torch.zeros((1, 2, 12000)), "sample_rate": 32000}
    values = storage.source_values(result, plan, source, parent, latent, audio)
    return values, result, parent, audio


def test_exact_data_and_independent_stage_roundtrip_without_recompute(tmp_path):
    values, result, parent, audio = case()
    before = storage.validate_source(values, result, parent, audio)
    source_path, source_sha, report = storage.save_source(values, result, parent, audio, tmp_path / "inputs")
    stage_path, stage_sha, _ = save_stage(result, tmp_path / "stages")
    restored = load_stage(tmp_path / "stages", stage_path, stage_sha, "native_high")[3]
    loaded, observed = storage.load_source(tmp_path / "inputs", source_path, source_sha,
                                           restored, parent.clone(), deepcopy(audio))
    assert storage.validate_source(loaded, restored, parent, audio) == before
    assert torch.equal(loaded["source_frames"], values["source_frames"])
    assert loaded["face_plan"] == values["face_plan"]
    assert all(torch.equal(x, y) for x, y in zip(loaded["av_latent"]["samples"].unbind(),
                                                values["av_latent"]["samples"].unbind()))
    candidate, audit = audit_multiface_face_stage(restored, loaded["face_plan"], loaded["source_frames"],
                                                  parent, loaded["av_latent"])
    assert candidate is restored.output and json.loads(audit)["automatic_accept"] is False
    assert json.loads(report)["artifact_path"] == source_path
    assert json.loads(observed)["data_only"] is True
    assert json.loads(observed)["sampling_executed"] is False
    assert json.loads(observed)["quality_accepted"] is False


@pytest.mark.parametrize("fault", ("parent", "audio", "audio_rate", "plan", "source", "latent", "stage", "extra"))
def test_stale_or_cross_character_or_changed_inputs_never_restore(tmp_path, fault):
    values, result, parent, audio = case()
    path, digest, _ = storage.save_source(values, result, parent, audio, tmp_path)
    if fault == "parent":
        parent = parent.clone()
        parent[-1, 31, 31, 0] += .001
    elif fault == "audio":
        audio["waveform"][0, 0, -1] = .001
    elif fault == "audio_rate":
        audio["sample_rate"] += 1
    elif fault == "stage":
        result = case(320)[1]
    else:
        values = deepcopy(values)
        if fault == "plan":
            values["face_plan"]["multiface"]["character_id"] = "Bob"
            _rehash(values["face_plan"])
        elif fault == "source":
            values["source_frames"][0, 31, 31, 0] += .001
        elif fault == "latent":
            values["av_latent"]["samples"].unbind()[0][0, 0, 0, 0, 0] += .001
        else:
            values["extra"] = True
        with pytest.raises(ValueError):
            storage.save_source(values, result, parent, audio, tmp_path, "bad")
        return
    with pytest.raises(ValueError):
        storage.load_source(tmp_path, path, digest, result, parent, audio)


@pytest.mark.parametrize("fault", ("manifest", "state", "digest", "traversal", "wrong_schema", "data_only", "extra"))
def test_corrupt_or_foreign_artifact_is_rejected(tmp_path, fault):
    values, result, parent, audio = case()
    path, digest, _ = storage.save_source(values, result, parent, audio, tmp_path)
    selected = tmp_path / path
    if fault == "manifest":
        selected.write_bytes(selected.read_bytes() + b" ")
    elif fault == "state":
        state = selected.parent / "state.safetensors"
        state.write_bytes(state.read_bytes()[:-1])
    elif fault == "digest":
        digest = "0" * 64
    elif fault == "traversal":
        path = "../" + path
    else:
        manifest = json.loads(selected.read_text("utf8"))
        if fault == "wrong_schema":
            manifest["schema"] = "foreign"
        elif fault == "data_only":
            manifest["data_only"] = 1
        else:
            manifest["extra"] = True
        selected.write_text(json.dumps(manifest), encoding="utf8")
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        storage.load_source(tmp_path, path, digest, result, parent, audio)


def test_cancel_never_commits_and_explicit_disabled_save_does_not_write(tmp_path, monkeypatch):
    values, result, parent, audio = case()
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        storage.save_source(values, result, parent, audio, tmp_path, interrupt=cancel)
    assert not list(tmp_path.rglob("manifest.json"))
    monkeypatch.setattr(nodes, "_root", lambda: tmp_path / "disabled")
    returned = nodes.MiniMaxH3MultiFaceSourceSaveEXPT8.execute(result, values["face_plan"],
        values["source_frames"], parent, values["av_latent"], audio).result
    assert returned[:3] == (values["face_plan"], values["source_frames"], values["av_latent"])
    assert returned[3:5] == ("", "") and not (tmp_path / "disabled").exists()
    assert json.loads(returned[5])["status"] == "save_disabled_passthrough"
    with pytest.raises(ValueError, match="boolean"):
        nodes.MiniMaxH3MultiFaceSourceSaveEXPT8.execute(result, values["face_plan"],
            values["source_frames"], parent, values["av_latent"], audio, confirm_save=1)


def test_new_nodes_only_append_distinct_explicit_contracts():
    assert [node.define_schema().node_id for node in nodes.NODES] == [
        "MiniMaxH3MultiFaceSourceSaveEXPT8", "MiniMaxH3MultiFaceSourceLoadEXPT8"]
    save, load = (node.define_schema() for node in nodes.NODES)
    assert save.is_output_node and save.inputs[-1].default is False
    assert [item.id for item in load.inputs] == ["stage_result", "parent_frames", "source_audio",
                                                "artifact_path", "artifact_sha256"]
