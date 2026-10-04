"""MID contracts and append-only schemas; numerical pinned sampler has its own CPU tool."""
import hashlib
import json

import pytest
import torch

from h3_audio_t8_pkg.freevideo_exp import runtime as old
from h3_audio_t8_pkg.freevideo_exp import split_runtime as split
from h3_audio_t8_pkg.freevideo_exp.split_nodes import NODES
from h3_audio_t8_pkg.freevideo_exp.split_sampler import data_prediction


def mid():
    base = torch.linspace(1., 0., 9)
    clock = dict(range_start=0, range_end=4, audio_policy="continue_noisy_audio_no_freeze",
        video_sigmas=(12 * base / (1 + 11 * base)).tolist(), audio_sigmas=(3 * base / (1 + 2 * base)).tolist())
    value = dict(schema="t8-freevideo-mid-v1", split_profile=split.PROFILE, role="SPLIT_LOW", complete_AV=False,
        completed_nfe=4, model_revision=old.MODEL_REVISION, freevideo_revision=old.FREEVIDEO_REVISION,
        vdn_revision=old.VDN_REVISION, geometry=dict(width=256, height=256, frames=39), split=clock,
        sample=dict(step_seconds=[.1]*4), request_sha256="0"*64, output_sha256="1"*64, source_inventory_sha256="2"*64)
    return split.make_mid(torch.zeros(1,24,12,16,16), torch.ones(1,24,12,16,16), torch.ones(1,32,2,65), value)


def test_mid_cold_roundtrip_and_separate_old_completion(tmp_path):
    state = mid()
    with pytest.raises(ValueError, match="Stage"):
        old.validate_stage(state)
    path, sha = split.save_mid(state, tmp_path)
    second, _ = split.save_mid(state, tmp_path)
    assert path != second
    loaded = split.load_mid(path, sha)
    assert loaded.receipt_sha256 == state.receipt_sha256
    assert torch.equal(loaded.audio, state.audio) and torch.equal(loaded.video_state, state.video_state)
    v,a = split.mid_av(loaded)["samples"].unbind()
    v.add_(3)
    a.add_(7)
    split.validate_mid(loaded)
    with pytest.raises(ValueError, match="SHA"):
        split.load_mid(path, "f"*64)


@pytest.mark.parametrize("key", ["video", "video_state", "audio"])
def test_mid_mutated_tensor_rejected(key):
    state = mid()
    getattr(state,key).add_(1)
    with pytest.raises(ValueError, match="content"):
        split.validate_mid(state)


@pytest.mark.parametrize("change", [{"completed_nfe":4.}, {"complete_AV":True}, {"role":"LOW"},
    {"sample":{"step_seconds":[.1]*3}}, {"vdn_revision":"foreign"}, {"request_sha256":""}])
def test_mid_forged_contract_rejected(change):
    state = mid()
    value = dict(json.loads(state.receipt_json), **change)
    text = old.canonical(value)
    state = split.FreeVideoMid(state.video,state.video_state,state.audio,text,hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError):
        split.validate_mid(state)


@pytest.mark.parametrize("key", ["video_sigmas", "audio_sigmas", "range_end", "range_start", "audio_policy"])
def test_mid_foreign_clock_rejected(key):
    value = json.loads(mid().receipt_json)["split"]
    value[key] = "foreign"
    with pytest.raises(ValueError):
        split.check_clock(value)


def test_x0_sign_and_original_timestep_not_next_grid_clock():
    sample, velocity, t = torch.tensor([1.,-1.]), torch.tensor([2.,3.]), torch.tensor(.25)
    assert torch.equal(data_prediction(sample,velocity,t),sample + .75*velocity)
    assert not torch.equal(data_prediction(sample,velocity,t),sample - .75*velocity)


def test_split_nodes_mid_has_no_original_completed_low_socket():
    high = NODES[1].define_schema().get_v1_info(NODES[1])
    assert "completed_low" not in high.input["required"]
    assert high.input["required"]["mid_state"][0] == "H3_T8_FREEVIDEO_MID"
    cold = NODES[3].define_schema().get_v1_info(NODES[3])
    assert set(cold.input["required"]) == {"manifest_path","manifest_sha256"}
    from h3_audio_t8_pkg.freevideo_exp.nodes import NODES as all_nodes
    old_high = all_nodes[3].define_schema().get_v1_info(all_nodes[3])
    assert old_high.input["required"]["tail_steps"][1]["default"] == 2
    assert old_high.input["required"]["completed_low"][0] == "H3_T8_FREEVIDEO_STAGE"
