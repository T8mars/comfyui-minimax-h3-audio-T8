"""New producer-family contracts only; no downloads, CUDA or legacy suite rerun."""
import json

import pytest
import torch

from h3_audio_t8_pkg.freevideo_quality import profiles as p, runtime as q, assets
from h3_audio_t8_pkg.freevideo_exp import runtime as old
from h3_audio_t8_pkg.freevideo_quality.audio_nodes import NODES


def completed(runtime, role="LOW"):
    producer, selected = p.binding(runtime), p.plan("light", role)
    video, audio = torch.zeros(1, 24, 12, 16, 16), torch.ones(1, 32, 2, 65)
    roles = ["LOW", "HIGH"] if role == "HIGH" else [role]
    tables = [dict(identity=p.table_identity("3" * 64, "light", stage, "ref2va_av",
                  reference_audio_t=producer["reference_audio_t"]), index=i, bytes=100,
                  sha256="4" * 64) for stage in roles for i in range(50)]
    receipt = dict(schema=producer["stage"], plan=selected, profile="light", role=role,
        freevideo_revision=producer["revision"], vdn_revision=old.VDN_REVISION,
        model_revision=old.MODEL_REVISION, geometry=dict(width=256, height=256, frames=39),
        task="ref2va_av", clock=p.clock("light", role, "ref2va_av",
        reference_audio_t=producer["reference_audio_t"]), completed_nfe=selected["nfe"],
        sample=dict(step_seconds=[.1] * selected["nfe"]), request_sha256="0" * 64,
        output_sha256="1" * 64, source_inventory_sha256="2" * 64,
        weight_identity="3" * 64, tables=tables, tables_sha256=assets.evidence_sha(tables))
    if role == "HIGH":
        receipt.update(low_receipt_sha256="5" * 64, low_audio=old.tensor_record(audio))
    return q.make_stage(video, audio, receipt)


def test_new_audio_clock_is_explicit_and_legacy_default_unchanged():
    legacy = p.clock("light", "LOW", "ref2va_av")
    new = p.clock("light", "LOW", "ref2va_av", reference_audio_t=1.)
    assert legacy == p.clock("light", "LOW", "ref2va_av", reference_audio_t=0.)
    assert new["video_sigmas"] == legacy["video_sigmas"]
    assert new["audio_sigmas"] == legacy["audio_sigmas"]
    assert all(1. in row for row in new["modulation_timesteps"])
    assert all(0. in row for row in legacy["modulation_timesteps"])
    assert new["modulation_timesteps"][1] != legacy["modulation_timesteps"][1]
    assert p.clock("light", task="t2va") == p.clock("light", task="t2va", reference_audio_t=1.)
    with pytest.raises(ValueError):
        p.validate_clock(legacy, "light", "LOW", "ref2va_av", reference_audio_t=1.)


@pytest.mark.parametrize("runtime", [p.LEGACY_RUNTIME, p.AUDIO_RUNTIME])
def test_source_family_save_load_preserves_exact_own_receipt_and_audio(tmp_path, runtime):
    stage = completed(runtime, "HIGH")
    path, sha = q.save_stage(stage, tmp_path)
    loaded = q.load_stage(path, sha)
    assert loaded.receipt_json == stage.receipt_json
    assert torch.equal(loaded.audio, stage.audio)
    assert json.loads(open(path, encoding="utf8").read())["schema"] == p.binding(runtime)["saved"]


def test_old_receipt_cannot_be_relabelled_as_new_family_or_wrong_clock():
    stage = completed(p.LEGACY_RUNTIME)
    receipt = json.loads(stage.receipt_json)
    receipt.update(schema=p.binding(p.AUDIO_RUNTIME)["stage"], freevideo_revision=p.AUDIO_FREEVIDEO_REVISION)
    with pytest.raises(ValueError, match="clock"):
        q.make_stage(stage.video, stage.audio, receipt)
    new = completed(p.AUDIO_RUNTIME)
    changed = json.loads(new.receipt_json)
    changed["schema"] = p.binding(p.LEGACY_RUNTIME)["stage"]
    with pytest.raises(ValueError, match="producer"):
        q.make_stage(new.video, new.audio, changed)


def test_new_bank_pins_are_exact_not_a_floating_allowlist():
    bank = dict(repo="OpenVDN/vdn-minimax-h3-edge", revision="b3a8dd8d4cf215b3a90bcc20b4362318f5f4f158",
                prefix="sampling-presets-v1-audio-20261007", tables=[])
    assert assets.banks(dict(optional_adaln_sets=[bank])) == []
    with pytest.raises(ValueError, match="Unpinned"):
        assets.banks(dict(optional_adaln_sets=[dict(bank, revision="main")]))
    with pytest.raises(ValueError):
        p.clock("light", reference_audio_t=.5)


def test_new_loader_socket_reuses_existing_split_sampler_and_effect_types():
    schema = NODES[0].GET_SCHEMA().get_v1_info(NODES[0])
    assert schema.output[0] == "H3_T8_FREEVIDEO_QUALITY_MODEL"
    assert schema.input["required"]["runtime_config"][1]["default"] == ""
    assert p.binding()["revision"] == p.FREEVIDEO_REVISION
    assert p.binding(p.AUDIO_RUNTIME)["revision"] != p.FREEVIDEO_REVISION
