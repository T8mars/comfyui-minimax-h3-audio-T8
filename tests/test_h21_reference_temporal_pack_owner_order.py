"""Independent extra owner-order counterexample; no production monkeypatch."""
import torch
from comfy.patcher_extension import WrappersMP

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.reference_temporal_pack_advanced import (
    audit_reference_temporal_pack, configure_reference_temporal_pack,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_h21_reference_temporal_pack import cpu, sample  # noqa: F401
from test_fast_h3_v2_core_sampler import model
from test_prompt_relay_advanced import NativeLikeFakeClip, _allow_fixture_core_contract, _plan


def test_actual_relay_rebind_keeps_foreign_same_key_wrapper_after_original_owner(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    built = relay.build_prompt_relay_conditioning(
        model=model(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt_relay_plan=_plan(length=22, local_prompts="She looks left.\nShe turns right.")[0],
        width=32, height=32, task_type="auto", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
        query_chunk_rows=64, ref_videos={"ref_video_0": torch.ones(56, 32, 32, 3) * .4})
    original, positive, source = built[:3]
    calls = []

    def foreign(executor, x, timestep, context, transformer_options, **kwargs):
        calls.append(relay.PROMPT_RELAY_RUNTIME_KEY in transformer_options)
        return executor(x, timestep, context, transformer_options, **kwargs)

    original.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY, foreign)
    old_order = original.get_wrappers(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY)
    selected, packed, _ = configure_reference_temporal_pack(original, positive, "2/5")
    assert selected is not original
    sample(original, positive, source)
    assert calls == [True, True]
    calls.clear()
    sample(selected, packed, source)
    assert calls == [True, True]
    new_order = selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY)
    assert old_order[1] is new_order[1] is foreign
    assert original.get_wrappers(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY) == old_order


def test_opaque_previous_can_return_without_consuming_prepared_layout_not_certified():
    from test_h21_reference_temporal_pack import positive
    bare, calls, blocks = model(), [], []
    def bypass(_apply, args):
        calls.append(1)
        return args["input"]
    bare.set_model_unet_function_wrapper(bypass)
    selected, packed, _ = configure_reference_temporal_pack(bare, positive(), "2/5")
    handle = bare.model.diffusion_model.blocks[0].register_forward_pre_hook(
        lambda *_args: blocks.append(1))
    try:
        result = sample(selected, packed)
    finally:
        handle.remove()
    report = audit_reference_temporal_pack(selected)
    assert calls == [1, 1] and blocks == []
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    assert report["completed_model_calls"] == 2
    assert report["partial"] and report["status"] == "observed_partial"
    assert report["upstream_model_wrapper_present"]
    assert report["compression_consumption_certified"] is False
