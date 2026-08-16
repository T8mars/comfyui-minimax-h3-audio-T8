from __future__ import annotations

import functools
import json
import types

import pytest
import torch
from comfy.ldm.minimax import model as minimax_model

from h3_audio_t8_pkg.conditioning import (
    HYBRID_KEYFRAME_SENTINEL,
    HYBRID_LAYOUT_LEGACY_SENTINEL,
    assert_hybrid_layout_contract,
    build_conditioning,
    build_packed_layout,
)
from h3_audio_t8_pkg.long_video import patch_long_video_model
from h3_audio_t8_pkg.multikeyframe_advanced import (
    ACTUAL_FRAME_INDEX,
    MULTIKEYFRAME_SCHEMA,
    MULTIKEYFRAME_SCHEMA_KEY,
    PAYLOAD_VISUAL_NOISE_AUGS_KEY,
    VISUAL_NOISE_AUGS_KEY,
    _condition_rows_with_augs,
    _segment_timestep_plan,
    append_keyframe_plan,
    build_multikeyframe_conditioning,
    native_middle_keyframe_support,
    patch_multikeyframe_model,
    resolve_keyframe_plan,
)
from h3_audio_t8_pkg.nodes_multikeyframe_advanced import (
    MiniMaxH3KeyframePlanT8Advanced,
    MiniMaxH3MultiKeyframeConditioningT8Advanced,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def make_image(value=0.5, height=128, width=128):
    return torch.full((1, height, width, 3), value)


class _CondConstant:
    def __init__(self, cond):
        self.cond = cond


class _FakeModelPatcher:
    def __init__(self, model, object_patches=None):
        self.model = model
        self.object_patches = dict(object_patches or {})

    def clone(self):
        return _FakeModelPatcher(self.model, self.object_patches)

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        current = self.model
        for part in name.split("."):
            current = getattr(current, part)
        return current

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


def make_model_patcher():
    diffusion_type = type("MiniMaxH3Model", (), {})
    diffusion = diffusion_type()
    diffusion.patch_size = (1, 2, 2)

    diffusion._forward = types.MethodType(
        minimax_model.MiniMaxH3Model._forward, diffusion
    )

    model_type = type("MiniMaxH3", (), {})
    base = model_type()
    base.diffusion_model = diffusion

    def extra_conds(_self, **kwargs):
        layout = build_packed_layout(
            7,
            37,
            8,
            8,
            207,
            keyframes=kwargs.get("minimax_keyframes"),
            refs=kwargs.get("minimax_refs"),
            frame_count=kwargs.get("minimax_frame_count"),
        )
        payload = {
            "layout": layout,
            "keyframes": kwargs.get("minimax_keyframes"),
            "refs": kwargs.get("minimax_refs"),
            "frame_count": kwargs.get("minimax_frame_count"),
        }
        return {"minimax_payload": _CondConstant(payload)}

    extra_conds.__module__ = "comfy.model_base"
    base.extra_conds = types.MethodType(extra_conds, base)
    return _FakeModelPatcher(base)


def test_advanced_node_ids_end_with_advanced_and_are_experimental():
    plan = MiniMaxH3KeyframePlanT8Advanced.define_schema()
    conditioning = MiniMaxH3MultiKeyframeConditioningT8Advanced.define_schema()
    assert plan.node_id.endswith("Advanced")
    assert conditioning.node_id.endswith("Advanced")
    assert plan.is_experimental is True
    assert conditioning.is_experimental is True
    assert plan.category == "T8/MiniMax H3/Conditioning/Experimental"
    assert conditioning.category == plan.category
    plan_inputs = {item.id: item for item in plan.inputs}
    conditioning_inputs = {item.id: item for item in conditioning.inputs}
    assert plan_inputs["visual_noise_aug"].default == 0.999
    assert "not a calibrated linear strength" in plan_inputs["visual_noise_aug"].tooltip
    for input_id in ("first_frame_noise_aug", "last_frame_noise_aug"):
        assert conditioning_inputs[input_id].default == 0.999
        assert "not a calibrated" in conditioning_inputs[input_id].tooltip
        assert "原始混噪值" in conditioning_inputs[input_id].display_name
    assert "shared" in conditioning_inputs["reference_visual_noise_aug"].tooltip


def test_plan_resolves_frame_seconds_and_percent_deterministically():
    plan = append_keyframe_plan(None, make_image(0.1), "frame", 31, 0.999, "center_crop", True)
    plan = append_keyframe_plan(plan, make_image(0.2), "seconds", 2.0, 0.995, "stretch", True)
    plan = append_keyframe_plan(plan, make_image(0.3), "percent", 75.0, 0.99, "center_crop", True)
    resolved = resolve_keyframe_plan(plan, 124)
    assert [item["frame_index"] for item in resolved] == [31, 48, 92]
    assert [item["visual_noise_aug"] for item in resolved] == [0.999, 0.995, 0.99]


@pytest.mark.parametrize(
    ("mode", "position", "match"),
    [
        ("frame", 0, "must be in 1..122"),
        ("frame", 123, "must be in 1..122"),
        ("frame", 2.5, "must be an integer"),
        ("seconds", -1, "must be non-negative"),
        ("percent", 101, "between 0 and 100"),
    ],
)
def test_plan_rejects_invalid_or_endpoint_positions(mode, position, match):
    plan = append_keyframe_plan(
        None, make_image(), mode, position, 0.999, "center_crop", True
    )
    with pytest.raises(ValueError, match=match):
        resolve_keyframe_plan(plan, 124)


def test_plan_rejects_two_entries_resolving_to_the_same_frame():
    plan = append_keyframe_plan(None, make_image(0.1), "frame", 48, 0.999, "center_crop", True)
    plan = append_keyframe_plan(plan, make_image(0.2), "seconds", 2.0, 0.995, "center_crop", True)
    with pytest.raises(ValueError, match="both resolve to frame 48"):
        resolve_keyframe_plan(plan, 124)


def _conditioning_args():
    return {
        "clip": FakeClip(),
        "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(),
        "prompt": "A dancer follows the keyframe path",
        "width": 128,
        "height": 128,
        "length": 124,
        "task_type": "FL2VA",
        "audio_mode": "native",
        "audio_denoise_strength": 0.35,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "drive_audio": None,
        "final_audio": None,
        "first_frame": make_image(0.1),
        "last_frame": make_image(0.9),
        "ref_images": None,
        "ref_videos": None,
        "ref_video_audios": None,
        "ref_audios": None,
    }


def test_empty_default_advanced_plan_is_the_exact_stable_fast_path():
    stable_args = _conditioning_args()
    stable = build_conditioning(**stable_args)

    advanced_args = _conditioning_args()
    marker_model = object()
    advanced = build_multikeyframe_conditioning(
        model=marker_model,
        keyframe_plan=None,
        first_frame_noise_aug=0.999,
        last_frame_noise_aug=0.999,
        reference_visual_noise_aug=0.999,
        **advanced_args,
    )
    assert advanced[0] is marker_model
    advanced_conditioning, advanced_latent = advanced[1], advanced[2]
    assert advanced_conditioning[0][1].keys() == stable[0][0][1].keys()
    assert MULTIKEYFRAME_SCHEMA_KEY not in advanced_conditioning[0][1]
    for actual, expected in zip(
        advanced_latent["samples"].unbind(), stable[1]["samples"].unbind()
    ):
        assert torch.equal(actual, expected)
    assert advanced[3:6] == stable[2:5]
    report = json.loads(advanced[-1])
    assert report["status"] == "stable_fast_path"
    assert report["model_cloned"] is False
    assert report["stable_conditioning_report"] == stable[5]


def test_advanced_build_adds_sorted_middle_keyframes_and_independent_augs():
    plan = append_keyframe_plan(None, make_image(0.4), "percent", 75, 0.98, "center_crop", True)
    plan = append_keyframe_plan(plan, make_image(0.3), "frame", 31, 0.995, "stretch", True)
    model = make_model_patcher()
    args = _conditioning_args()
    result = build_multikeyframe_conditioning(
        model=model,
        keyframe_plan=plan,
        first_frame_noise_aug=0.999,
        last_frame_noise_aug=0.99,
        reference_visual_noise_aug=0.999,
        **args,
    )
    patched_model, positive = result[0], result[1]
    metadata = positive[0][1]
    assert patched_model is not model
    assert "extra_conds" in patched_model.object_patches
    assert "diffusion_model._forward" in patched_model.object_patches
    assert [item[ACTUAL_FRAME_INDEX] for item in metadata["minimax_keyframes"]] == [
        0,
        31,
        92,
        123,
    ]
    assert [item["resolved_frame_index"] for item in metadata["minimax_keyframes"]] == [
        0,
        31,
        92,
        123,
    ]
    assert metadata[VISUAL_NOISE_AUGS_KEY] == [0.999, 0.995, 0.98, 0.99]
    token_kwargs = args["clip"].tokenize_calls[0][1]
    assert len(token_kwargs["images"]) == 4
    keyframe_map = json.loads(result[-2])
    assert [item["frame_index"] for item in keyframe_map] == [0, 31, 92, 123]
    report = json.loads(result[-1])
    assert report["middle_keyframe_count"] == 2
    assert report["per_condition_forward_patch"] is True
    assert report["memory_safety_tier"] == "unproven_experimental"
    assert report["recommended_default_visual_noise_aug"] == 0.999
    assert "not a VRAM percentage" in report["added_rows_estimate_scope"]
    assert any("not a VRAM percentage" in warning for warning in report["warnings"])


def test_advanced_positive_miswire_uses_native_layout_or_legacy_fails_closed():
    plan = append_keyframe_plan(
        None, make_image(0.4), "frame", 31, 0.999, "center_crop", True
    )
    model = make_model_patcher()
    result = build_multikeyframe_conditioning(
        model=model,
        keyframe_plan=plan,
        first_frame_noise_aug=0.999,
        last_frame_noise_aug=0.999,
        reference_visual_noise_aug=0.999,
        **_conditioning_args(),
    )
    metadata = result[1][0][1]
    if native_middle_keyframe_support():
        payload = model.model.extra_conds(**metadata)["minimax_payload"].cond
        cond_starts = [
            start
            for start, _stop, kind in payload["layout"].segments
            if kind == "cond"
        ]
        actual = [
            float(payload["layout"].position_ids[start, 0]) for start in cond_starts
        ]
        assert actual == pytest.approx(
            [7.0, 7.0 + 5 / 3 * 31, 7.0 + 5 / 3 * 123]
        )
    else:
        with pytest.raises(ValueError, match="only first/last keyframe anchors are supported"):
            model.model.extra_conds(**metadata)


def test_reference_only_advanced_use_is_rejected_before_model_execution():
    args = _conditioning_args()
    args.update(
        {
            "first_frame": None,
            "last_frame": None,
            "task_type": "auto",
            "ref_images": {"ref_image_1": make_image(0.5)},
        }
    )
    with pytest.raises(ValueError, match="only supports FL2VA"):
        build_multikeyframe_conditioning(
            model=make_model_patcher(),
            keyframe_plan=None,
            first_frame_noise_aug=0.999,
            last_frame_noise_aug=0.999,
            reference_visual_noise_aug=0.98,
            **args,
        )


def test_hybrid_payload_preserves_visual_and_audio_reference_order():
    plan = append_keyframe_plan(
        None, make_image(0.4), "frame", 31, 0.999, "center_crop", True
    )
    args = _conditioning_args()
    args.update(
        {
            "task_type": "Hybrid",
            "drive_audio": make_audio(5),
            "audio_mode": "reference_only",
            "add_source_as_reference": True,
            "ref_images": {"ref_image_1": make_image(0.6)},
            "ref_videos": {"ref_video_1": torch.zeros((48, 64, 64, 3))},
            "ref_video_audios": {"ref_video_audio_1": make_audio(2)},
            "ref_audios": {"ref_audio_1": make_audio(1)},
        }
    )
    model = make_model_patcher()
    result = build_multikeyframe_conditioning(
        model=model,
        keyframe_plan=plan,
        first_frame_noise_aug=0.999,
        last_frame_noise_aug=0.999,
        reference_visual_noise_aug=0.999,
        **args,
    )
    metadata = result[1][0][1]
    refs = metadata["minimax_refs"]
    if assert_hybrid_layout_contract() == HYBRID_LAYOUT_LEGACY_SENTINEL:
        assert [ref["kind"] for ref in refs] == [
            HYBRID_KEYFRAME_SENTINEL,
            HYBRID_KEYFRAME_SENTINEL,
            HYBRID_KEYFRAME_SENTINEL,
            "image",
            "video_audio",
            "audio",
            "audio",
        ]
    else:
        assert [ref["kind"] for ref in refs] == [
            "image", "video_audio", "audio", "audio"
        ]
    payload = result[0].object_patches["extra_conds"](**metadata)["minimax_payload"].cond
    assert len(payload["cond_video_latents"]) == 5
    assert len(payload["cond_audio_latents"]) == 3
    assert payload[PAYLOAD_VISUAL_NOISE_AUGS_KEY] == [0.999] * 5


def test_scoped_extra_conds_patch_repairs_positions_and_payload_order():
    keyframes = [
        {
            "resolved_frame_index": 0,
            ACTUAL_FRAME_INDEX: frame,
            "latent": torch.full((1, 24, 1, 8, 8), float(index)),
        }
        for index, frame in enumerate([0, 31, 92, 123])
    ]
    model = make_model_patcher()
    patched = patch_multikeyframe_model(model, require_per_condition_forward=True)
    out = patched.object_patches["extra_conds"](
        **{
            MULTIKEYFRAME_SCHEMA_KEY: MULTIKEYFRAME_SCHEMA,
            "minimax_keyframes": keyframes,
            "minimax_frame_count": 124,
            VISUAL_NOISE_AUGS_KEY: [0.999, 0.995, 0.98, 0.99],
        }
    )
    payload = out["minimax_payload"].cond
    layout = payload["layout"]
    cond_segments = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    actual_times = [float(layout.position_ids[start, 0]) for start, _ in cond_segments]
    assert actual_times == pytest.approx(
        [7.0, 7.0 + 5 / 3 * 31, 7.0 + 5 / 3 * 92, 7.0 + 5 / 3 * 123]
    )
    assert payload[PAYLOAD_VISUAL_NOISE_AUGS_KEY] == [0.999, 0.995, 0.98, 0.99]
    assert len(payload["cond_video_latents"]) == 4
    assert "extra_conds" not in model.object_patches


def test_per_condition_rows_and_timesteps_change_only_the_selected_condition():
    fake_model = types.SimpleNamespace(patch_size=(1, 2, 2))
    latents = [
        torch.full((1, 24, 1, 4, 4), 0.25),
        torch.full((1, 24, 1, 4, 4), 0.75),
    ]
    payload_a = {
        "cond_video_latents": latents,
        PAYLOAD_VISUAL_NOISE_AUGS_KEY: [0.999, 0.95],
        "seed": 123,
    }
    payload_b = {
        "cond_video_latents": latents,
        PAYLOAD_VISUAL_NOISE_AUGS_KEY: [0.98, 0.95],
        "seed": 123,
    }
    rows_a = _condition_rows_with_augs(fake_model, payload_a, torch.device("cpu"))
    rows_b = _condition_rows_with_augs(fake_model, payload_b, torch.device("cpu"))
    rows_per_condition = rows_a.shape[0] // 2
    assert not torch.equal(rows_a[:rows_per_condition], rows_b[:rows_per_condition])
    assert torch.equal(rows_a[rows_per_condition:], rows_b[rows_per_condition:])

    keyframes = [
        {"resolved_frame_index": 0, "latent": latents[0]},
        {"resolved_frame_index": 4, "latent": latents[1]},
    ]
    layout = build_packed_layout(
        7, 2, 4, 4, 8, keyframes=keyframes, frame_count=5
    )
    times = _segment_timestep_plan(layout, 0.1, 0.2, [0.999, 0.95], 1.0)
    cond_times = [
        times[index]
        for index, (_a, _b, kind) in enumerate(layout.segments)
        if kind == "cond"
    ]
    assert cond_times == [0.999, 0.95]


def test_uniform_default_condition_rows_match_current_comfyui_core_exactly():
    fake_model = types.SimpleNamespace(patch_size=(1, 2, 2))
    payload = {
        "cond_video_latents": [torch.rand((1, 24, 1, 4, 4)) for _ in range(2)],
        PAYLOAD_VISUAL_NOISE_AUGS_KEY: [0.999, 0.999],
        "visual_cond_noise_aug": 0.999,
        "seed": 987,
    }
    actual = _condition_rows_with_augs(fake_model, payload, torch.device("cpu"))
    expected = minimax_model.MiniMaxH3Model._cond_video_rows(
        fake_model, payload, torch.device("cpu")
    )
    assert torch.equal(actual, expected)


def test_middle_keyframes_require_both_stable_endpoints():
    plan = append_keyframe_plan(None, make_image(), "frame", 31, 0.999, "center_crop", True)
    args = _conditioning_args()
    args["last_frame"] = None
    args["task_type"] = "auto"
    with pytest.raises(ValueError, match="require both first_frame and last_frame"):
        build_multikeyframe_conditioning(
            model=make_model_patcher(), keyframe_plan=plan, **args
        )


def test_advanced_patch_rejects_long_video_patch_stacking():
    model = make_model_patcher()
    original = model.get_model_object("extra_conds")
    original.__func__._t8_long_video_patch_version = 1
    try:
        with pytest.raises(ValueError, match="cannot be stacked"):
            patch_multikeyframe_model(model, require_per_condition_forward=False)
    finally:
        del original.__func__._t8_long_video_patch_version


def test_long_video_patch_rejects_advanced_patch_stacking_in_reverse_order():
    advanced = patch_multikeyframe_model(
        make_model_patcher(), require_per_condition_forward=False
    )
    with pytest.raises(ValueError, match="cannot be stacked"):
        patch_long_video_model(advanced)


def test_different_advanced_patch_version_is_rejected():
    model = make_model_patcher()
    original = model.get_model_object("extra_conds")
    original.__func__._t8_multikeyframe_patch_version = 999
    try:
        with pytest.raises(RuntimeError, match="different MiniMax H3 Multi-Keyframe"):
            patch_multikeyframe_model(model, require_per_condition_forward=False)
    finally:
        del original.__func__._t8_multikeyframe_patch_version


def test_wrapped_process_global_packed_layout_patch_is_rejected(monkeypatch):
    original = minimax_model.PackedLayout.__init__

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    monkeypatch.setattr(minimax_model.PackedLayout, "__init__", wrapped)
    with pytest.raises(RuntimeError, match="process-global"):
        native_middle_keyframe_support()
