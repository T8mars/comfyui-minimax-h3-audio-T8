import hashlib

import av
import numpy as np
import pytest

from h3_audio_t8_pkg.video_interval_screen import (
    frame_observations,
    suspected_intervals,
    screen_video,
)


def screen(images, **kwargs):
    rows = frame_observations(((i / 24, image) for i, image in enumerate(images)))
    return suspected_intervals(rows, **kwargs)


def test_sustained_cut_has_one_hint_and_explicit_editor_cut_is_not_failure():
    black = np.zeros((90, 160), np.float32)
    white = np.ones_like(black)
    images = [black] * 8 + [white] * 8
    report = screen(images)
    assert [r["kind"] for r in report["intervals"]] == ["possible_cut"]
    assert report["intervals"][0]["focus"] == 8 / 24
    assert (
        screen(images, expected_cuts=[8 / 24])["intervals"][0]["kind"] == "expected_cut"
    )
    assert report["quality_accepted"] is False and report["automatic_retry"] is False


def test_flash_is_a_transient_review_hint_not_two_hard_cuts():
    black = np.zeros((90, 160), np.float32)
    images = [black] * 8 + [np.ones_like(black)] + [black] * 8
    report = screen(images)
    assert [r["kind"] for r in report["intervals"]] == ["transient_flash_or_occlusion"]


def test_small_patch_flicker_is_localized_without_calling_it_identity_failure():
    black = np.zeros((90, 160), np.float32)
    patch = black.copy()
    patch[18:36, 40:60] = 1
    report = screen([black] * 8 + [patch] + [black] * 8)
    assert [r["kind"] for r in report["intervals"]] == ["local_transient"]
    assert report["identity_assessment"] == "manual_comparison_required"


def test_gradual_real_lighting_change_has_no_abrupt_event():
    images = [np.ones((90, 160), np.float32) * i / 40 for i in range(35)]
    assert not screen(images)["intervals"]


def test_actual_vfr_timestamps_are_kept_and_invalid_order_is_not_silently_retimed():
    image = np.zeros((90, 160), np.float32)
    rows = frame_observations([(10.0, image), (10.07, image), (10.13, image)])
    assert [r["seconds"] for r in rows] == [10.0, 10.07, 10.13]
    with pytest.raises(ValueError, match="timestamps"):
        frame_observations([(1.0, image), (0.9, image)])
    with pytest.raises(ValueError, match="finite"):
        frame_observations([(0.0, image * np.nan)])


def test_region_is_user_selected_and_only_flags_appearance_change():
    a = np.zeros((90, 160), np.float32)
    b = a.copy()
    b[20:40, 30:50] = 1
    rows = frame_observations(
        [(0.0, a), (0.05, a), (0.1, b), (0.15, b)],
        region=(30 / 160, 20 / 90, 50 / 160, 40 / 90),
    )
    report = suspected_intervals(rows, has_region=True)
    assert [r["kind"] for r in report["intervals"]] == ["region_appearance_change"]
    assert "identity_failure" not in str(report)


def test_real_encoded_video_decode_has_full_frame_count_and_leaves_source_bytes(
    tmp_path,
):
    path = tmp_path / "test.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width = 160
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for i in range(24):
            frame = av.VideoFrame.from_ndarray(
                np.full((96, 160, 3), 0 if i < 12 else 255, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report, rows = screen_video(path)
    assert len(rows) == 24 and report["frame_count"] == 24
    assert report["timestamp_source"] == "decoded_presentation_time"
    assert report["intervals"][0]["focus"] == 0.5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    def cancelled():
        raise InterruptedError("caller cancelled")

    with pytest.raises(InterruptedError, match="caller cancelled"):
        screen_video(path, cancel_check=cancelled)
