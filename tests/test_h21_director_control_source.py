"""Actual temporal-only RGB preparation and independent target/source geometry."""

from copy import deepcopy
from fractions import Fraction
import importlib
import json
import uuid

import av
import numpy as np
import pytest
import torch

from comfy_api.latest import InputImpl
from comfy_extras.nodes_video import GetVideoComponents

from h3_audio_t8_pkg import h3_fun_control_advanced as fun
from h3_audio_t8_pkg.control_region_runtime import apply_h3_fun_control_region
from h3_audio_t8_pkg.control_regions import prepare_control_geometry
from h3_audio_t8_pkg.core import resize_image
from h3_audio_t8_pkg.source_av import prepare_source_media_window
import test_h21_control_regions as real


def source_api():
    return importlib.import_module("h3_audio_t8_pkg.director_control_source")


def row(**values):
    return {"id": str(uuid.uuid4()), "asset_id": None, **values}


def source(count=12, width=64, height=32):
    return real.source_frames(count=count, width=width, height=height)


def prepare(frames, spec=None, **kwargs):
    return source_api().prepare_director_control_source(
        frames,
        row() if spec is None else spec,
        width=kwargs.pop("width", 64),
        height=kwargs.pop("height", 32),
        length=kwargs.pop("length", 5),
        **kwargs,
    )


def test_static_first_frame_keeps_original_spatial_size_and_lets_original_apply_repeat():
    original = source(3, width=75, height=41)
    before = original.clone()
    actual, mask, crop, report = prepare(
        original, source_kind="image", width=128, height=64
    )
    assert actual.shape == (1, 41, 75, 3) and torch.equal(actual, original[:1])
    assert mask is None and crop == ""
    assert report["source_indices"] == [0] and report["output_frame_count"] == 1
    assert (
        report["target_frame_count"] == 5
        and report["static_expansion"] == "original_fun_apply_hold_last"
    )
    assert report["ignored_image_batch_frames"] == 2
    assert torch.equal(original, before)


@pytest.mark.parametrize("fps", [15.0, 24.0, 30.0, 30000 / 1001])
@pytest.mark.parametrize("start", [0.0, 1 / 48])
def test_video_indices_match_original_source_av_ties_even_without_spatial_or_audio_work(
    fps, start
):
    original = source(30)
    spec = row(source={"start_seconds": start, "short_video_policy": "strict"})
    actual, mask, crop, report = prepare(
        original, spec, source_kind="video", source_fps=fps
    )
    expected = prepare_source_media_window(
        original, fps, 64, 32, 5, start, "strict", "pad_silence", return_frame_map=True
    )
    assert report["source_indices"] == list(expected[-1]["source_indices"])
    assert report["target_times_seconds"] == list(expected[-1]["target_times"])
    # Old SourceAV also performs Lanczos/PIL, including 8bit quantization even
    # at equal dimensions. Compare temporal selection BEFORE that spatial op.
    raw_selected = original.index_select(
        0, torch.tensor(expected[-1]["source_indices"])
    )
    assert torch.equal(actual, raw_selected) and mask is None and crop == ""
    assert torch.equal(resize_image(actual, 64, 32), expected[0])
    assert report["spatial_operation"] == "none_original_RGB_dimensions"
    assert report["source_audio_consumed"] is False
    assert report["clock_status"] == "supplied_fps_CFR_assumption_unverified"
    assert report["pts_timing_certified"] is False


def test_half_ties_use_torch_even_not_half_up():
    original = source(4)
    actual, _, _, report = prepare(original, source_kind="video", source_fps=12.0)
    assert report["source_indices"] == [0, 0, 1, 2, 2]
    assert torch.equal(actual, original[torch.tensor([0, 0, 1, 2, 2])])


def test_temporal_only_adapter_does_not_inherit_unrelated_conform_frame_or_canvas_caps():
    original = torch.zeros(1, 1, 1, 3)
    actual, mask, crop, report = prepare(
        original,
        source_kind="video",
        source_fps=24.0,
        width=65568,
        height=32,
        length=4102,
    )
    assert actual.shape == (4102, 1, 1, 3) and mask is None and crop == ""
    assert report["held_video_frames"] == 4101
    assert report["geometry"]["width"] == 65568


def test_invalid_source_kind_type_does_not_mutate_the_rgb_or_row():
    original, spec = source(), row()
    before, saved = original.clone(), deepcopy(spec)
    with pytest.raises(ValueError):
        prepare(original, spec, source_kind=[], source_fps=24.0)
    assert torch.equal(original, before) and spec == saved


def test_strict_short_source_errors_and_explicit_hold_keeps_last_without_wrapping():
    original = source(3)
    spec = row(source={"short_video_policy": "strict"})
    before, saved = original.clone(), deepcopy(spec)
    with pytest.raises(ValueError, match="too short"):
        prepare(original, spec, source_kind="video", source_fps=24.0)
    assert torch.equal(original, before) and spec == saved
    actual, _, _, report = prepare(original, source_kind="video", source_fps=24.0)
    assert report["source_indices"] == [0, 1, 2, 2, 2]
    assert report["held_video_frames"] == 2 and torch.equal(actual[-1], original[-1])


def test_large_finite_rate_has_no_invented_maximum_or_int64_index_wrap():
    original = source(3)
    actual, _, _, report = prepare(original, source_kind="video", source_fps=1e20)
    assert report["source_indices"] == [0, 2, 2, 2, 2]
    assert torch.equal(actual[1:], original[-1:].expand(4, -1, -1, -1))


@pytest.mark.parametrize(
    "fps", [None, 0.0, -1.0, float("nan"), float("inf"), True, "24"]
)
def test_real_invalid_video_rates_reject_before_any_input_mutation(fps):
    original = source()
    spec = row()
    before, saved = original.clone(), deepcopy(spec)
    with pytest.raises(ValueError):
        prepare(original, spec, source_kind="video", source_fps=fps)
    assert torch.equal(original, before) and spec == saved


@pytest.mark.parametrize(
    "fault",
    ["rank", "empty", "integer", "channels", "nan", "inf", "kind", "overflow_time"],
)
def test_invalid_actual_images_and_unrepresentable_time_fail_without_replacing_drafts(
    fault,
):
    frames, spec, kind = source(), row(), "video"
    if fault == "rank":
        frames = frames[0]
    elif fault == "empty":
        frames = frames[:0]
    elif fault == "integer":
        frames = frames.to(torch.uint8)
    elif fault == "channels":
        frames = frames[..., :2]
    elif fault in {"nan", "inf"}:
        frames[0, 0, 0, 0] = float(fault)
    elif fault == "kind":
        kind = "audio"
    else:
        spec["source"] = {"start_seconds": 1e308}
    before, saved = frames.clone(), deepcopy(spec)
    with pytest.raises(ValueError):
        prepare(frames, spec, source_kind=kind, source_fps=24.0)
    torch.testing.assert_close(frames, before, rtol=0, atol=0, equal_nan=True)
    assert spec == saved


def test_target_area_mask_and_source_crop_are_rebuilt_independently_at_each_real_stage():
    original = source(7, width=75, height=41)
    spec = row(
        region={"mode": "rect", "xyxy_normalized": [0.25, 0.25, 0.75, 0.75]},
        crop={
            "enabled": True,
            "xyxy_normalized": [0.2, 0.1, 0.8, 0.9],
            "fit": "stretch",
        },
    )
    before, saved = original.clone(), deepcopy(spec)
    low = prepare(original, spec, source_kind="video", source_fps=24.0)
    high = prepare(
        original, spec, source_kind="video", source_fps=24.0, width=128, height=64
    )
    assert low[0].shape == high[0].shape == (5, 41, 75, 3)
    assert low[1].shape == (1, 32, 64) and high[1].shape == (1, 64, 128)
    assert low[1].sum() == 512 and high[1].sum() == 2048
    assert low[2] == high[2]
    crop = json.loads(low[2])
    assert crop["source_size"] == [75, 41]
    assert crop["boxes"][0] == pytest.approx([15.0, 4.1, 60.0, 36.9])
    assert torch.equal(original, before) and spec == saved


def test_fractional_rectangle_uses_pixel_cell_area_not_center_point_rounding():
    spec = row(
        region={
            "mode": "rect",
            "xyxy_normalized": [0.5 / 64, 0.25 / 32, 1.5 / 64, 1.75 / 32],
        }
    )
    _, mask, _, _ = prepare(source(1), spec, source_kind="image")
    expected = torch.zeros(1, 32, 64)
    expected[:, :2, :2] = 0.375
    assert mask.dtype == torch.float32 and torch.equal(mask, expected)
    assert mask.sum() == 1.5


@pytest.fixture(scope="module")
def encoder():
    return real.RealTinyEncoderOwner()


@pytest.fixture
def tower_case(monkeypatch, encoder):
    return real.tower_case.__wrapped__(monkeypatch, encoder)


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
@pytest.mark.parametrize("region", ["full", "rect"])
def test_prepared_full_target_really_preserves_old_control_tower_numbers(
    tower_case, backend, region
):
    case = tower_case
    spec = row(
        region={"mode": region, "xyxy_normalized": [0, 0, 1, 1]}, fit_mode="exact"
    )
    timed, mask, crop, report = prepare(
        case.source, spec, source_kind="video", source_fps=24.0
    )
    bundle, _ = real.real_bundle(backend)
    old = real.apply_control(case, bundle, old=True)[0]
    args = (
        case.base,
        case.positive,
        bundle,
        case.vae,
        timed,
        64,
        32,
        5,
        "pose",
        "exact",
        0.7,
        0.0,
        1.0,
    )
    selected, _, runtime, _ = apply_h3_fun_control_region(
        *args, region_mask=mask, crop_plan_json=crop
    )
    baseline, actual = (
        real.actual_forward(case, old),
        real.actual_forward(case, selected),
    )
    assert all(
        torch.equal(left, right) for left, right in zip(baseline, actual, strict=True)
    )
    assert report["source_indices"] == [0, 1, 2, 3, 4]
    assert runtime.report()["residual_calls"] == 1
    assert runtime.report()["gated_calls"] == (region == "rect")
    assert not runtime.report()["portable_cache_identity"]


@pytest.mark.parametrize("backend", ["compatibility", "official_model_patch"])
def test_real_tower_and_video_encoder_consume_the_actual_timed_crop_and_target_mask(
    tower_case, backend
):
    case = tower_case
    spec = row(
        region={"mode": "rect", "xyxy_normalized": [0.0, 0.0, 0.5, 1.0]},
        crop={
            "enabled": True,
            "xyxy_normalized": [0.5, 0.0, 1.0, 1.0],
            "fit": "stretch",
            "flip_x": True,
        },
    )
    original, saved = case.source.clone(), deepcopy(spec)
    timed, mask, crop, source_report = prepare(
        case.source, spec, source_kind="video", source_fps=48.0
    )
    expected, _, geometry = prepare_control_geometry(
        timed,
        width=64,
        height=32,
        length=5,
        fit_mode="exact",
        fit_video=fun._fit_control_video,
        region_mask=mask,
        crop_plan_json=crop,
    )
    bundle, _ = real.real_bundle(backend)
    before = len(case.vae.calls)
    selected, positive, runtime, _ = apply_h3_fun_control_region(
        case.base,
        case.positive,
        bundle,
        case.vae,
        timed,
        64,
        32,
        5,
        "pose",
        "exact",
        0.7,
        0.0,
        1.0,
        region_mask=mask,
        crop_plan_json=crop,
    )
    # Compat encodes during Apply; official MODEL_PATCH defers the exact same
    # RGB dependency until its first actual diffusion-model forward.
    assert len(case.vae.calls) == before + int(backend == "compatibility")
    assert positive is case.positive and source_report["source_indices"] == [
        0,
        2,
        4,
        4,
        4,
    ]
    actual = real.actual_forward(case, selected)
    assert len(case.vae.calls) == before + 1 and torch.equal(
        case.vae.calls[-1], expected
    )
    assert all(torch.isfinite(part).all() for part in actual)
    report = runtime.report()
    assert report["crop_prepared"] and report["crop_consumed_by_observed_control"]
    assert report["residual_calls"] == report["gated_calls"] == 1
    assert report["geometry"]["geometry_sha256"] == geometry["geometry_sha256"]
    assert torch.equal(case.source, original) and spec == saved
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("pts", [[0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 8]])
def test_real_pyav_video_uses_native_get_components_fps_but_never_invents_pts_certification(
    tmp_path, pts
):
    path = tmp_path / "control.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.codec_context.thread_count = 1
        for index, point in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), index * 35, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = point, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as stream:
        actual_pts = [
            Fraction(frame.pts) * frame.time_base for frame in stream.decode(video=0)
        ]
    if pts[-1] == 8:
        assert (
            len(set(right - left for left, right in zip(actual_pts, actual_pts[1:])))
            > 1
        )
    components = GetVideoComponents.execute(InputImpl.VideoFromFile(str(path))).result
    images, audio, fps = components[:3]
    prepared, _, _, report = prepare(images, source_kind="video", source_fps=fps)
    assert audio is None and prepared.shape == (5, 32, 64, 3)
    assert report["source_fps"] == fps
    assert report["pts_timing_certified"] is False
    assert report["clock_status"] == "supplied_fps_CFR_assumption_unverified"
    assert any("VFR" in text for text in report["warnings"])
