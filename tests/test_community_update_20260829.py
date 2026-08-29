from __future__ import annotations

from fractions import Fraction
import json

import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as chunked
from h3_audio_t8_pkg import fast_h3_advanced as fast_h3
from h3_audio_t8_pkg import h3_lora_compat_advanced as lora_compat
from h3_audio_t8_pkg import timed_references_advanced as timed


class _FakeMiniMax:
    @staticmethod
    def state_dict():
        return {
            "diffusion_model.blocks.0.attn.qkv_proj.weight": torch.zeros(1),
            "diffusion_model.blocks.0.norm.weight": torch.zeros(1),
            "other.weight": torch.zeros(1),
        }


def test_direct_h3_lora_aliases_follow_official_pr_15662(monkeypatch):
    monkeypatch.setattr(lora_compat.comfy.model_base, "MiniMaxH3", _FakeMiniMax)
    aliases = lora_compat.add_minimax_h3_direct_lora_keys(_FakeMiniMax())
    assert aliases["blocks.0.attn.qkv_proj"] == (
        "diffusion_model.blocks.0.attn.qkv_proj.weight"
    )
    assert aliases["blocks.0.norm"] == "diffusion_model.blocks.0.norm.weight"
    assert "other" not in aliases


def test_native_and_direct_h3_lora_keys_both_build_real_comfy_patches():
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    rank = 2
    down = torch.randn(rank, 4)
    up = torch.randn(6, rank)

    native_prefix = "diffusion_model.blocks.0.attn.qkv_proj"
    native = {
        f"{native_prefix}.lora_A.default.weight": down,
        f"{native_prefix}.lora_B.default.weight": up,
    }
    native_patches = lora_compat.comfy.lora.load_lora(
        native, {native_prefix: target}, log_missing=False
    )

    direct_prefix = "blocks.0.attn.qkv_proj"
    direct = {
        f"{direct_prefix}.lora_A.default.weight": down,
        f"{direct_prefix}.lora_B.default.weight": up,
    }
    direct_patches = lora_compat.comfy.lora.load_lora(
        direct, {direct_prefix: target}, log_missing=False
    )

    assert target in native_patches
    assert target in direct_patches
    assert native_patches[target].weights[0] is up
    assert native_patches[target].weights[1] is down
    assert direct_patches[target].weights[0] is up
    assert direct_patches[target].weights[1] is down


def test_fastvideo_h3_adapter_conversion_fuses_qkv_and_maps_dense_payload():
    source = {}
    for index, projection in enumerate(("q", "k", "v"), start=1):
        prefix = f"transformer_blocks.0.attn.to_{projection}"
        source[f"{prefix}.lora_A.weight"] = torch.full((2, 3), float(index))
        source[f"{prefix}.lora_B.weight"] = torch.full((4, 2), float(index + 3))
    source["transformer_blocks.0.ff.net.0.proj.lora_A.weight"] = torch.ones(2, 3)
    source["transformer_blocks.0.ff.net.0.proj.lora_B.weight"] = torch.arange(
        16, dtype=torch.float32
    ).reshape(8, 2)
    source["transformer_blocks.0.adaln_proj.linear.diff_b"] = torch.ones(6)
    source["audio_proj_in.diff"] = torch.ones(3, 2)

    converted, report = lora_compat.convert_fastvideo_h3_adapter(source)

    fused = "blocks.0.attn.qkv_proj"
    assert converted[f"{fused}.lora_A.weight"].shape == (6, 3)
    fused_b = converted[f"{fused}.lora_B.weight"]
    assert fused_b.shape == (12, 6)
    assert torch.count_nonzero(fused_b[:4, 2:]) == 0
    assert torch.count_nonzero(fused_b[4:8, :2]) == 0
    assert torch.count_nonzero(fused_b[4:8, 4:]) == 0
    assert torch.count_nonzero(fused_b[8:, :4]) == 0
    source_fc1_b = source["transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
    assert torch.equal(
        converted["blocks.0.mlp.fc1.lora_B.weight"],
        torch.cat((source_fc1_b[4:], source_fc1_b[:4])),
    )
    assert "blocks.0.adaln_proj.linear.diff_b" in converted
    assert "audio_patch_proj.diff" in converted
    assert report == {
        "conversion": "fastvideo_h3_diffusers_to_comfyui_fused",
        "fused_qkv_groups": 1,
        "direct_lora_modules": 1,
        "dense_delta_tensors": 2,
        "swiglu_half_swaps": 1,
    }


def test_lora_loader_reports_zero_hit_without_fingerprint_gate(tmp_path, monkeypatch):
    path = tmp_path / "arbitrary-user-name.safetensors"
    path.write_bytes(b"structural-test")

    class FakeInner:
        pass

    class FakePatcher:
        def __init__(self):
            self.model = FakeInner()

        def clone(self):
            return self

        @staticmethod
        def add_patches(_patches, _strength):
            return []

    monkeypatch.setattr(
        lora_compat.comfy.utils,
        "load_torch_file",
        lambda *_args, **_kwargs: ({"unknown.weight": torch.zeros(1)}, {}),
    )
    monkeypatch.setattr(
        lora_compat.comfy.lora_convert, "convert_lora", lambda state: state
    )
    monkeypatch.setattr(
        lora_compat.comfy.lora,
        "model_lora_keys_unet",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        lora_compat.comfy.lora,
        "load_lora",
        lambda *_args, **_kwargs: {},
    )
    output, report_json = lora_compat.load_minimax_h3_lora_model(
        FakePatcher(), path, 1.0
    )
    report = json.loads(report_json)
    assert output.model.__class__.__name__ == "FakeInner"
    assert report["status"] == "no_compatible_patches"
    assert report["file"]["identity_policy"] == "display_only_not_a_load_gate_no_hash_scan"
    assert "sha256" not in report["file"]


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.calls.append((text, return_word_ids, kwargs))
        return self.calls[-1]


def test_timed_reference_is_identity_outside_ref2va():
    base = _RecordingTokenizer()
    frames = torch.zeros((2, 32, 32, 3))
    wrapper = timed.MiniMaxH3TimedReferenceTokenizerT8(
        base,
        [{"prompt_tag": "face", "frames": frames, "timestamps": [2.0, 2.0]}],
    )
    wrapper.tokenize_with_weights("use #face", images=[frames[:1]])
    assert base.calls[0][0] == "use #face"
    assert "minimax_ref_items" not in base.calls[0][2]


def test_timed_reference_offsets_native_video_labels_and_keeps_tag_boundaries():
    base = _RecordingTokenizer()
    frames = torch.zeros((2, 32, 32, 3))
    wrapper = timed.MiniMaxH3TimedReferenceTokenizerT8(
        base,
        [{"prompt_tag": "face", "frames": frames, "timestamps": [1.25, 1.25]}],
    )
    native = [{"type": "image"}, {"type": "video"}]
    wrapper.tokenize_with_weights(
        "use #face but keep #face_detail", minimax_ref_items=native
    )
    text, _, kwargs = base.calls[0]
    assert text == "use <Video 2> but keep #face_detail"
    assert kwargs["minimax_ref_items"][-1]["timestamps"] == [1.25, 1.25]


def test_timed_video_sampling_uses_real_requested_rate():
    frames = torch.arange(49, dtype=torch.float32).view(49, 1, 1, 1).expand(
        -1, 32, 64, 3
    )
    sampled, timestamps = timed.prepare_timed_video_frames(
        frames, 24.0, 3.0, "source", 4.0
    )
    assert sampled[:, 0, 0, 0].tolist() == [
        0.0,
        6.0,
        12.0,
        18.0,
        24.0,
        30.0,
        36.0,
        42.0,
        48.0,
    ]
    assert timestamps[0] == Fraction(3)
    assert timestamps[-1] == Fraction(5)


def test_h3_temporal_grid_matches_17_frame_period():
    assert chunked.frames_for_tokens(5) == 17
    segments, frame_count = chunked.compute_temporal_segments(20, 34, 17)
    assert frame_count == 68
    assert segments[0][:2] == (0, 0)
    assert segments[-1][2:] == (20, 68)
    assert all(item[0] % 5 == 0 for item in segments)


def test_chunked_plan_has_no_two_megapixel_project_gate():
    plan, report = chunked.build_chunked_two_pass_plan(
        "user-selected.safetensors",
        2560,
        1440,
        136,
        17,
        0.999,
        512,
        512,
        128,
        32,
        256,
        "smoothstep",
        "fp16",
        "offload_after",
    )
    assert plan["target_width"] * plan["target_height"] > 2_000_000
    assert plan["spatial_strategy"] == "full_frame_safe"
    assert json.loads(report)["pixel_limit_policy"] == "no_project_pixel_area_limit"


def test_chunked_execute_restores_exact_input_audio(monkeypatch):
    class Nested:
        def __init__(self, tensors):
            self.tensors = tensors
            self.is_nested = True

    monkeypatch.setattr(chunked.comfy.nested_tensor, "NestedTensor", Nested)
    monkeypatch.setattr(
        chunked,
        "learned_upscale_h3_av_latent",
        lambda latent, *_args, **_kwargs: (
            latent,
            64,
            64,
            json.dumps({"status": "mock_upscale"}),
        ),
    )
    monkeypatch.setattr(
        chunked,
        "_spatial_resample",
        lambda video, _audio, *_args, **_kwargs: (video + 1, {"mock": True}),
    )
    video = torch.zeros((1, 24, 5, 4, 4))
    audio = torch.randn((1, 32, 2, 29))
    plan, _ = chunked.build_chunked_two_pass_plan(
        "mock.safetensors",
        64,
        64,
        17,
        0,
        0.999,
        64,
        64,
        0,
        0,
        32,
        "linear",
        "fp16",
        "offload_after",
    )
    output, report_json = chunked.execute_chunked_two_pass_upscale(
        object(), [], {"samples": Nested((video, audio))}, object(), object(), torch.ones(2), plan
    )
    assert output["samples"].tensors[1] is audio
    assert json.loads(report_json)["audio_preserved_by_identity"] is True


def test_fast_h3_contract_is_four_nfe_and_dense_fallback(monkeypatch):
    expected = fast_h3.native_flow_sigmas(4, 12.0)
    monkeypatch.setattr(
        fast_h3,
        "setup_dual_clock_sampling",
        lambda model, *_args, **_kwargs: (model, "sampler", expected),
    )
    monkeypatch.setattr(
        fast_h3,
        "probe_fast_h3_vsa",
        lambda: {
            "fastvideo_python_available": False,
            "triton_available": True,
            "external_vsa_executor_available": False,
            "policy": "test",
        },
    )
    model, sampler, sigmas, report_json = fast_h3.build_fast_h3_4step_setup(
        object(), object(), attention_profile="external_vsa_if_available"
    )
    report = json.loads(report_json)
    assert sampler == "sampler" and len(sigmas) == 5
    assert report["trained_contract"]["steps_nfe"] == 4
    assert report["trained_contract"]["supported_family"] == "T2VA only"
    assert report["attention_profile_effective"] == "dense_comfyui"
    assert report["model_identity_policy"].endswith("no_filename_size_or_hash_gate")


def test_fast_h3_legacy_fl2va_value_stays_load_compatible_but_warns(monkeypatch):
    expected = fast_h3.native_flow_sigmas(4, 12.0)
    monkeypatch.setattr(
        fast_h3,
        "setup_dual_clock_sampling",
        lambda model, *_args, **_kwargs: (model, "sampler", expected),
    )
    monkeypatch.setattr(
        fast_h3,
        "probe_fast_h3_vsa",
        lambda: {
            "fastvideo_python_available": False,
            "triton_available": True,
            "external_vsa_executor_available": False,
            "policy": "test",
        },
    )
    _model, _sampler, _sigmas, report_json = fast_h3.build_fast_h3_4step_setup(
        object(), object(), task_family="t2va_fl2va"
    )
    report = json.loads(report_json)
    assert report["status"] == "configured_with_warnings"
    assert "T2VA only" in report["warnings"][0]
