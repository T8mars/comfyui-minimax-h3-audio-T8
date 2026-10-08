"""Actual Core position reconstruction; no model/VAE/attention forward."""
from types import SimpleNamespace

import pytest
import torch
from comfy import conds
from comfy.ldm.minimax import model as core_h3

from h3_audio_t8_pkg import sol_attn_minimax_v2 as sol
from h3_audio_t8_pkg import res_history_setup as setup
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_layout_identity import inspect_processed_layouts


def processed():
    sol._patch_packed_layout(core_h3)
    # Actual geometry/clock fields of image, audio, video-audio references and
    # nonzero image+audio guides, not an invented uniform position matrix.
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 6},
            {"kind": "audio", "ref_audio_t": 7},
            {"kind": "video_audio", "latent_t": 2, "latent_h": 6, "latent_w": 4, "ref_audio_t": 5}]
    keyframes = [{"resolved_frame_index": 72, "latent": torch.zeros(1, 24, 1, 4, 6),
                  "audio_latent": torch.zeros(1, 32, 2, 3)}]
    payload = {"refs": refs, "keyframes": keyframes, "seed": 11, "audio_scale": 1.0}
    payload["layout"] = core_h3.PackedLayout(3, 2, 4, 6, 5, refs=refs, keyframes=keyframes)
    context = torch.arange(3 * 8, dtype=torch.float32).reshape(1, 3, 8)
    selected = {"minimax_payload": conds.CONDConstant(payload),
                "latent_shapes": conds.CONDConstant([[1, 24, 2, 3, 5], [1, 32, 2, 5]]),
                "c_crossattn": conds.CONDRegular(context)}
    return SimpleNamespace(conds={"positive": [{"model_conds": selected}]},
                           original_conds={"positive": [{"cross_attn": context}]}, model_patcher=object()), payload


def test_actual_Core_all_position_fields_match_native_condition_without_Sol_cache_write(monkeypatch):
    guider, payload = processed()
    independent = processed()[0]
    layout = payload["layout"]
    before = (dict(sol._SPANS), dict(sol._PERM_CACHE), dict(sol._DEVICE_CACHE),
              {name: value.clone() for name, value in vars(layout).items() if type(value) is torch.Tensor})
    contract = inspect_processed_layouts(guider)
    assert contract == inspect_processed_layouts(guider)
    assert contract["all_position_fields_and_segments_verified"] and contract["portable_condition_layout"]
    assert inspect_processed_layouts(independent) == contract
    # Only exercise new runtime-contract wiring here. Loaded-weight and model
    # runtime qualification have their separate real-Core tests; no fake proof.
    monkeypatch.setattr(setup, "loaded_model_identity", lambda model: {"automatic_loaded_weights_verified": False})
    monkeypatch.setattr(setup, "_runtime_signature", lambda *args: {"fixture_only": True})
    actual = setup._runtime_contract(SimpleNamespace(inner_model=guider), {"seed": 11}, {}, 8 * 1024**2)
    assert actual["actual_processed_position_condition"]["sha256"] == contract["sha256"]
    assert actual["loaded_weight_fingerprint_verified"] is False
    guider.conds = {}
    unknown = setup._runtime_contract(SimpleNamespace(inner_model=guider), {"seed": 11}, {}, 8 * 1024**2)
    assert unknown["actual_processed_position_condition"]["portable_condition_layout"] is False
    assert dict(sol._SPANS) == before[0]
    assert dict(sol._PERM_CACHE) == before[1] and dict(sol._DEVICE_CACHE) == before[2]
    for name, value in before[3].items():
        assert torch.equal(getattr(layout, name), value)
    assert not torch.cuda.is_initialized()


def test_actual_position_coordinate_masks_and_source_geometry_mutations_are_rejected(monkeypatch):
    guider, payload = processed()
    layout = payload["layout"]
    assert inspect_processed_layouts(guider)["portable_condition_layout"]
    for name in ("position_ids", "img_pos", "img_update", "audio_pos", "audio_update"):
        actual = getattr(layout, name)
        changed = actual.clone()
        changed.reshape(-1)[0] = not changed.reshape(-1)[0] if changed.dtype == torch.bool else changed.reshape(-1)[0] + 1
        with monkeypatch.context() as patch:
            patch.setattr(layout, name, changed)
            with pytest.raises(UnverifiedModelStack, match="actual position content"):
                inspect_processed_layouts(guider)
            assert getattr(layout, name) is changed
    payload["keyframes"][0]["resolved_frame_index"] = 73
    with pytest.raises(UnverifiedModelStack, match="actual position content"):
        inspect_processed_layouts(guider)
    payload["keyframes"][0]["resolved_frame_index"] = 72
    payload["refs"][0]["latent_h"] = 2000000
    with pytest.raises(UnverifiedModelStack, match="bounded adapter"):
        inspect_processed_layouts(guider)
    assert not torch.cuda.is_initialized()


def test_unknown_constructor_helper_or_condition_owner_is_preserved_without_execution(monkeypatch):
    guider, payload = processed()
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unknown position execution must not be called")
    for owner, name in ((core_h3.PackedLayout, "__init__"), (core_h3, "_frame_grid")):
        with monkeypatch.context() as changed:
            changed.setattr(owner, name, foreign)
            with pytest.raises(UnverifiedModelStack):
                inspect_processed_layouts(guider)
            assert getattr(owner, name) is foreign
    selected = guider.conds["positive"][0]["model_conds"]
    selected["minimax_payload"] = SimpleNamespace(cond=payload)
    with pytest.raises(UnverifiedModelStack, match="condition owner"):
        inspect_processed_layouts(guider)
    assert not calls and not torch.cuda.is_initialized()
