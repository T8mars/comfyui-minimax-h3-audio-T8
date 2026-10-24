"""Independent actual tiny Core/physical-clock review; no trained GPU claims."""

from copy import deepcopy

import pytest
import torch
from comfy.ldm.minimax import model as native

from h3_audio_t8_pkg import reference_temporal_pack as pack
from h3_audio_t8_pkg import reference_temporal_pack_advanced as feature
from test_fast_h3_v2_core_sampler import model
import test_h21_reference_temporal_pack as existing

cpu = existing.cpu


@pytest.mark.parametrize("target", ["embedding", "audio", "keyframe"])
def test_opaque_metadata_does_not_disable_known_tensor_last_forward_seal(target):
    bare, positive = model(), existing.positive()
    opaque = object()
    positive[0][1]["user_opaque_owner"] = opaque
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, positive, "2/5")
    assert feature.audit_reference_temporal_pack(selected)["packed_condition_identity"]["opaque_paths"]
    tensor = (packed[0][0] if target == "embedding" else
              packed[0][1]["minimax_refs"][0]["audio_latent"] if target == "audio" else
              packed[0][1]["minimax_keyframes"][0]["latent"])
    before, calls = tensor.clone(), []

    def change_last(*_args):
        calls.append(1)
        if len(calls) == 2:
            tensor.add_(.1)

    handle = bare.model.diffusion_model.final_layer.register_forward_hook(change_last)
    try:
        with pytest.raises(ValueError, match="conditioning content SHA changed"):
            existing.sample(selected, packed)
    finally:
        handle.remove()
        tensor.copy_(before)
    assert calls == [1, 1]
    # Unknown internals remain unknown, but unchanged known content must run.
    result = existing.sample(selected, packed)
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    report = feature.audit_reference_temporal_pack(selected)
    assert report["completed_model_calls"] == 2 and report["cleanup_observed"]
    assert not report["portable_cache_identity"]
    assert positive[0][1]["user_opaque_owner"] is opaque


def test_physical_layout_retains_all_per_row_upstream_delta_and_source_storage():
    positive = existing.positive()
    positive[0][1]["minimax_keyframes"] = [
        {"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 6, 10)},
        {"resolved_frame_index": 17, "audio_latent": torch.ones(1, 32, 2, 3)},
    ]
    packed, _ = pack.pack_conditioning(positive, "2/5")
    refs, keyframes = packed[0][1]["minimax_refs"], packed[0][1]["minimax_keyframes"]
    signature = (3, 7, 6, 10, 13)
    full = native.PackedLayout(*signature, keyframes=keyframes, refs=positive[0][1]["minimax_refs"])
    compact = native.PackedLayout(*signature, keyframes=keyframes, refs=refs)
    delta = torch.arange(compact.seq_len * 3, dtype=torch.float64).reshape(-1, 3) / 32.
    compact.position_ids += delta
    before = compact.position_ids.clone()
    original_pointer = compact.position_ids.data_ptr()
    fixed, _ = pack.repair_payload({"layout": compact, "keyframes": keyframes, "refs": refs}, signature)
    expected, spatial_index = [], 0
    spatial_refs = [ref for ref in refs if ref["kind"] in ("image", "video", "video_audio")]
    for start, stop, kind in full.segments:
        values = full.position_ids[start:stop]
        if kind == "ref_img":
            ref = spatial_refs[spatial_index]
            spatial_index += 1
            if pack.KEY in ref:
                record = ref[pack.KEY]
                rows = ref["latent_h"] * ref["latent_w"] // 4
                values = values.reshape(record["original_t"], rows, 3)[record["indices"]].reshape(-1, 3)
        expected.append(values)
    # The input is already float64(base+delta). Only its observable stored
    # difference is available; recovering pre-addition rounding bits is not a
    # physical-layout API. The original 15-element/2.84e-14 oracle error is kept.
    baseline = native.PackedLayout(*signature, keyframes=keyframes, refs=refs)
    stored_delta = before - baseline.position_ids
    oracle = torch.cat(expected) + stored_delta
    assert torch.equal(fixed["layout"].position_ids, oracle)
    assert torch.equal(fixed["layout"].position_ids[:, 1:], before[:, 1:])
    assert torch.equal(compact.position_ids, before)
    assert compact.position_ids.data_ptr() == original_pointer
    assert fixed["layout"].position_ids.data_ptr() != original_pointer
    # Generated/audio/keyframe index maps retain the actual compact native rows.
    for field in ("img_pos", "img_update", "audio_pos", "audio_update"):
        assert torch.equal(getattr(fixed["layout"], field), getattr(compact, field))


def test_repeated_source_video_branches_have_independent_packed_tensor_storage():
    source = existing.video()
    positive = existing.conditioning()
    positive[0][1]["minimax_refs"] = [source, source]
    packed, _ = pack.pack_conditioning(positive, "2/5")
    a, b = packed[0][1]["minimax_refs"]
    source_before, b_before = source["latent"].clone(), b["latent"].clone()
    assert len({a["latent"].data_ptr(), b["latent"].data_ptr(), source["latent"].data_ptr()}) == 3
    a["latent"].add_(1)
    assert torch.equal(b["latent"], b_before) and torch.equal(source["latent"], source_before)
    with pytest.raises(ValueError, match="content SHA changed"):
        pack.verify_ref(a)
    pack.verify_ref(b)


def test_unknown_owner_internal_state_remains_executable_and_uncertified():
    class UnknownOwner:
        def __init__(self):
            self.calls = 0

    owner, bare, positive = UnknownOwner(), model(), existing.positive()
    positive[0][1]["user_opaque_owner"] = owner
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, positive, "2/5")
    handle = bare.model.diffusion_model.final_layer.register_forward_hook(
        lambda *_args: setattr(owner, "calls", owner.calls + 1))
    try:
        result = existing.sample(selected, packed)
    finally:
        handle.remove()
    assert owner.calls == 2
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    report = feature.audit_reference_temporal_pack(selected)
    assert report["packed_condition_identity"]["opaque_paths"]
    assert not report["portable_cache_identity"] and not report["compression_consumption_certified"]


@pytest.mark.parametrize("relay_enabled", [False, True])
def test_actual_multirow_positive_sampling_and_audit_report_are_independent(monkeypatch, relay_enabled):
    if relay_enabled:
        from helpers import FakeAudioVAE, FakeVideoVAE
        from h3_audio_t8_pkg import prompt_relay_advanced as relay
        from test_prompt_relay_advanced import NativeLikeFakeClip, _allow_fixture_core_contract, _plan
        _allow_fixture_core_contract(monkeypatch)
        built = relay.build_prompt_relay_conditioning(
            model=model(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
            prompt_relay_plan=_plan(length=22, local_prompts="She looks left.\nShe turns right.")[0],
            width=32, height=32, task_type="auto", audio_mode="native", audio_denoise_strength=.35,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
            ref_image_size="match", reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
            query_chunk_rows=64, ref_videos={"ref_video_0": torch.ones(56, 32, 32, 3) * .4})
        bare, positive, source = built[:3]
    else:
        bare, positive, source = model(), existing.positive(), existing.latent()
    # Synthetic text/VAEs are explicit; the sampler and DiT are actual tiny Core.
    original_count = len(positive)
    second = [[embedding.clone() + .01, deepcopy(metadata)] for embedding, metadata in positive]
    positive += second
    selected, packed, _ = feature.configure_reference_temporal_pack(bare, positive, "2/5")
    result = existing.sample(selected, packed, source)
    assert len(packed) == 2 * original_count
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    report = feature.audit_reference_temporal_pack(selected)
    assert report["completed_model_calls"] >= 2 and report["cleanup_observed"]
    report["contract"]["references"][0]["descriptor"]["indices"][0] = 999
    report["last_layout"]["signature"][0] = 999
    current = feature.audit_reference_temporal_pack(selected)
    assert current["contract"]["references"][0]["descriptor"]["indices"][0] == 0
    assert current["last_layout"]["signature"][0] != 999
    assert not current["portable_cache_identity"]
    if relay_enabled:
        contract = relay.prompt_relay_model_contract(selected)
        assert contract["execution_counts"]["completed_forwards"] >= 2
        assert contract["execution_counts"]["routed_attention_calls"] >= 2
