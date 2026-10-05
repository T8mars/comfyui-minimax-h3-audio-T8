"""Actual shared Relay factory with tiny Core contracts; not GPU quality proof."""
from dataclasses import asdict
import json

import pytest

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
from h3_audio_t8_pkg.nodes_reference_package import MiniMaxH3ReferenceRelayConditioningEXPT8 as Node
from h3_audio_t8_pkg.reference_runtime import capture_set
from test_reference_runtime import encoders as _encoders_fixture, image
from test_prompt_relay_advanced import (
    NativeLikeFakeClip, _plan, _native_h3_model_patcher, _allow_fixture_core_contract,
)

encoders = _encoders_fixture


class CountingClip(NativeLikeFakeClip):
    def __init__(self):
        super().__init__()
        self.encode_count = 0
        self.reference_items = []

    def tokenize(self, prompt, **kwargs):
        self.reference_items = kwargs.get("minimax_ref_items", [])
        return super().tokenize(prompt, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_count += 1
        return super().encode_from_tokens_scheduled(tokens)


def inputs(encoders, *, model=None, execution_mode="report_only", width=32):
    video, audio = encoders
    a, _ = image("A", video, audio, voice=True)
    b, _ = image("B", video, audio)
    refs, _ = capture_set([a, b], json.dumps([
        {"role_id": "A", "visual": False, "voice": True},
        {"role_id": "B", "visual": True, "voice": False}]))
    plan, *_ = _plan(global_prompt="<Picture 1> listens to <Audio 1>.",
        local_prompts="She turns toward the speaker.\nShe nods once.")
    return dict(reference_set=refs, model=object() if model is None else model,
        clip=CountingClip(), video_vae=video, audio_vae=audio, prompt_relay_plan=plan,
        width=width, height=32, task_type="auto", audio_mode="native",
        audio_denoise_strength=.35, add_source_as_reference=True,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", execution_mode=execution_mode,
        query_chunk_rows=64)


def test_separate_adapter_schema_preserves_existing_relay_and_removes_ambiguous_raw_ports():
    old = MiniMaxH3PromptRelayConditioningT8Advanced
    before = asdict(old.define_schema().get_v1_info(old))
    adapted = asdict(Node.define_schema().get_v1_info(Node))
    required = adapted["input"]["required"]
    assert required["reference_set"][0] == "T8_H3_REFERENCE_SET"
    assert required["prompt_primary_audio_ordinal"][1]["default"] == 0
    assert required["execution_mode"][1]["default"] == "report_only"
    assert not {"ref_images", "ref_videos", "ref_video_audios", "ref_audios"} & set(adapted["input"].get("optional", {}))
    assert adapted["output"] == before["output"]
    assert asdict(old.define_schema().get_v1_info(old)) == before


def test_fresh_reference_relay_reuses_VAE_assets_and_keeps_voice_anchor_not_master(encoders):
    args = inputs(encoders)
    video, audio = encoders
    model, positive, latent, mux_audio, _prompt, media, report = Node.execute(**args).result
    assert model is args["model"] and args["clip"].encode_count == 1
    assert len(video.encode_calls) == 2 and len(audio.encode_calls) == 1
    assert [item["type"] for item in args["clip"].reference_items] == ["audio", "image"]
    assert [item["kind"] for item in positive[0][1]["minimax_refs"]] == ["audio", "image"]
    assert json.loads(media)["pictures"] == {"1": "role:B/visual"}
    assert mux_audio is None and latent["samples"].is_nested
    assert json.loads(report)["attention_patch_installed"] is False


def test_independent_LOW_HIGH_relay_binds_actual_layout_without_reencoding_assets(encoders, monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    args = inputs(encoders, model=_native_h3_model_patcher(), execution_mode="apply_exp")
    low = Node.execute(**args).result
    high_args = {**args, "model": _native_h3_model_patcher(), "clip": CountingClip(), "width": 64}
    high = Node.execute(**high_args).result
    assert args["clip"].encode_count == high_args["clip"].encode_count == 1
    assert low[0] is not args["model"] and high[0] is not high_args["model"]
    assert low[0] is not high[0]
    low_binding = low[1][0][1][relay.PROMPT_RELAY_BINDING_KEY]
    high_binding = high[1][0][1][relay.PROMPT_RELAY_BINDING_KEY]
    assert low_binding["binding_hash"] != high_binding["binding_hash"]
    assert json.loads(low[-1])["attention_patch_installed"] is True
    assert json.loads(high[-1])["attention_patch_installed"] is True
    assert len(encoders[0].encode_calls) == 2 and len(encoders[1].encode_calls) == 1


def test_bad_asset_or_mixed_inputs_fail_before_fresh_Qwen(encoders):
    args = inputs(encoders)
    with pytest.raises(ValueError, match="mixed raw"):
        Node.execute(**args, ref_images={"ref_image_0": object()})
    assert args["clip"].encode_count == 0
    encoders[0].revision = 1
    with pytest.raises(ValueError, match="encoder differs"):
        Node.execute(**args)
    assert args["clip"].encode_count == 0


def test_optional_native_inputs_are_forwarded_once_and_legacy_None_call_is_unchanged(encoders, monkeypatch):
    original = relay.build_conditioning
    calls = []

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **{key: value for key, value in kwargs.items() if key != "semantic_bridge"})

    monkeypatch.setattr(relay, "build_conditioning", spy)
    args = inputs(encoders)
    bridge = object()
    Node.execute(**args, semantic_bridge=bridge)
    assert len(calls) == 1 and calls[0]["semantic_bridge"] is bridge
    assert calls[0]["prepared_reference_set"] is args["reference_set"]
    # Ordinary callers do not get a new keyword or asset route silently.
    ordinary = {key: value for key, value in args.items() if key != "reference_set"}
    ordinary["prompt_relay_plan"], *_ = _plan()
    relay.build_prompt_relay_conditioning(**ordinary, prepared_reference_set=None, semantic_bridge=None)
    assert len(calls) == 2 and "prepared_reference_set" not in calls[1]
    assert calls[1]["semantic_bridge"] is None
