"""CPU synthetic contracts/real codecs, NOT trained pose or H3 qualification."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import av
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h21_pose_source_tested", ROOT / "tools/prepare_h21_pose_source.py")
pose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pose)


def test_192_exact_real_frame_indices_round_half_even_not_float_or_loop():
    indices = pose.source_indices(Fraction(30), Fraction(24), Fraction(8))
    assert len(indices) == len(set(indices)) == 192
    assert indices == [round(Fraction(index * 5, 4)) for index in range(192)]
    assert indices[:8] == [0, 1, 2, 4, 5, 6, 8, 9]
    assert indices[-1] == 239


@pytest.mark.parametrize("source,target,duration", [(24, 30, 8), (30, 24, Fraction(1, 5)), (0, 24, 8), (30, -24, 8), (30, 24, 0)])
def test_invalid_real_unique_frame_mapping_is_not_silently_repaired(source, target, duration):
    with pytest.raises(ValueError):
        pose.source_indices(source, target, duration)


def test_portrait_exact_aspect_is_preserved_and_landscape_squeeze_rejected():
    rgb = np.zeros((1024, 576, 3), dtype=np.uint8)
    rgb[:, :, :] = [11, 71, 199]
    actual = pose.resize_source(rgb, 288, 512)
    assert actual.shape == (512, 288, 3)
    assert np.array_equal(actual, np.broadcast_to([11, 71, 199], actual.shape))
    with pytest.raises(ValueError, match="aspect"):
        pose.resize_source(rgb, 256, 144)
    with pytest.raises(ValueError, match="uint8"):
        pose.resize_source(rgb.astype(np.float32), 288, 512)


def prediction(people=1):
    result = np.zeros((1, 8400, 85), np.float32)
    for index in range(people):
        center = 70 + 80 * index
        result[0, index, :6] = [center / 8 - index, 100 / 8, np.log(20 / 8), np.log(50 / 8), .9, .9]
    return result


def test_real_raw_head_contract_no_detection_or_more_than_fixed5_fails_without_truncation():
    boxes, scores = pose.person_boxes(prediction(1), 1.)
    assert boxes.shape == (1, 4) and scores.shape == (1,)
    np.testing.assert_allclose(boxes[0], [60, 75, 80, 125], atol=1e-5)
    with pytest.raises(ValueError, match="found0"):
        pose.person_boxes(prediction(0), 1.)
    with pytest.raises(ValueError, match="found6"):
        pose.person_boxes(prediction(6), 1.)


@pytest.mark.parametrize("which", ["shape", "nan", "ratio", "overflow"])
def test_bad_raw_detector_protocol_or_nonfinite_math_remains_explicit(which):
    raw, ratio = prediction(), 1.
    if which == "shape":
        raw = raw[:, :-1]
    elif which == "nan":
        raw[0, 0, 0] = np.nan
    elif which == "ratio":
        ratio = 0.
    else:
        raw[0, 0, 2] = 10000
    with pytest.raises(ValueError):
        pose.person_boxes(raw, ratio)


def test_actual_crop_affine_maps_real_box_center_to_pinned_network_center():
    matrix, center, scale = pose.crop_transform([10, 20, 90, 220])
    np.testing.assert_allclose(matrix @ np.array([*center, 1]), [144, 192], atol=2e-5)
    assert scale[0] / scale[1] == pytest.approx(.75)
    with pytest.raises(ValueError, match="box"):
        pose.crop_transform([10, 20, 10, 21])


class SyntheticDetector:
    """Only a protocol fixture. This is never a trained detector result."""
    def __call__(self, tensor):
        assert tensor.device.type == "cpu" and tensor.shape == (1, 3, 640, 640)
        assert bool((tensor[:, :, :, 360:] == 114).all())
        return torch.from_numpy(prediction())


class SyntheticSimCC:
    """Only actual tensor-shape/affine math; no pose accuracy claim."""
    def __init__(self, broken=None):
        self.broken = broken

    def __call__(self, tensor):
        assert tensor.device.type == "cpu" and tensor.shape == (5, 3, 384, 288)
        assert torch.count_nonzero(tensor[1:]) == 0  # Padding is not extra people.
        sx, sy = torch.zeros(5, 133, 576), torch.zeros(5, 133, 768)
        if self.broken != "confidence":
            sx[:, :, 200], sy[:, :, 300] = .8, .7
        if self.broken == "shape":
            sx = sx[:, :, :-1]
        if self.broken == "nan":
            sy[0, 0, 0] = float("nan")
        return sx, sy


def test_infer_frame_actual_CPU_tensors_portrait_letterbox_fixed_batch_and_133_mapping():
    rgb = np.full((1024, 576, 3), 100, np.uint8)
    small, boxes, scores, points, conf = pose.infer_frame(rgb, SyntheticDetector(), SyntheticSimCC(), width=288, height=512)
    assert small.shape == (512, 288, 3) and len(boxes) == 1
    assert points.shape == (1, 133, 2) and conf.shape == (1, 133)
    np.testing.assert_allclose(conf, .7)
    assert scores[0] == pytest.approx(.81)
    assert np.isfinite(points).all() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("broken", ["confidence", "shape", "nan"])
def test_bad_actual_SimCC_or_no_confident_body_is_not_filled_or_skipped(broken):
    with pytest.raises(ValueError):
        pose.infer_frame(np.full((1024, 576, 3), 100, np.uint8), SyntheticDetector(), SyntheticSimCC(broken), width=288, height=512)


def test_own_primitive_raster_portrait_and_zero_confidence_never_invents_body():
    points = np.full((1, 133, 2), [70., 200.])
    scores = np.ones((1, 133))
    drawn = pose.raster(points, scores, width=288, height=512)
    assert drawn.mode == "RGB" and drawn.size == (288, 512)
    assert np.asarray(drawn).any()
    empty = pose.raster(points, scores * 0, width=288, height=512)
    assert not np.asarray(empty).any()
    with pytest.raises(ValueError, match="133"):
        pose.raster(points[:, :-1], scores[:, :-1], width=288, height=512)


@pytest.fixture
def source(tmp_path):
    target = tmp_path / "source.mp4"
    arrays = [np.full((32, 18, 3), index * 23, np.uint8) for index in range(6)]
    with av.open(str(target), mode="w", format="mp4") as movie:
        stream = movie.add_stream("libx264rgb", rate=30)
        stream.width, stream.height, stream.pix_fmt = 18, 32, "rgb24"
        stream.options = {"crf": "0", "preset": "medium", "threads": "1"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 30)
        for index, array in enumerate(arrays):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode(None):
            movie.mux(packet)
    return target, arrays, target.read_bytes()


def test_real_lossless_source_complete_decode_PTS_and_exact_selected_RGB_source_provenance(source):
    path, arrays, before = source
    plan = pose.inspect_source(path, pose.sha(path), duration=Fraction(1, 6), target_rate=Fraction(24))
    assert plan["frames"] == 6 and plan["target_frames"] == 4
    assert plan["source_indices"] == [0, 1, 2, 4]
    assert plan["all_CFR_PTS_verified"]
    assert plan["all_frame_RGB_sha256"] == [pose.rgb_sha(array) for array in arrays]
    assert path.read_bytes() == before


def test_bad_source_SHA_short_source_and_missing_optin_never_load_models_or_invent_tail(source, tmp_path):
    path, _, before = source
    with pytest.raises(ValueError, match="SHA"):
        pose.inspect_source(path, "0" * 64, duration=Fraction(1, 6))
    with pytest.raises(ValueError, match="too short"):
        pose.inspect_source(path, pose.sha(path), duration=Fraction(8))
    with pytest.raises(ValueError, match="opt-in"):
        pose.prepare(path, tmp_path / "absent", expected_source_sha=pose.sha(path))
    assert not (tmp_path / "absent").exists()
    assert path.read_bytes() == before and not torch.cuda.is_initialized()


def test_non_CFR_real_encoded_timestamps_are_rejected_without_media_normalization(tmp_path):
    target = tmp_path / "non-cfr.mp4"
    with av.open(str(target), mode="w", format="mp4") as movie:
        stream = movie.add_stream("libx264rgb", rate=30)
        stream.width, stream.height, stream.pix_fmt = 18, 32, "rgb24"
        stream.options = {"crf": "0", "threads": "1"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 30)
        for pts in (0, 1, 3, 4, 5, 6):
            frame = av.VideoFrame.from_ndarray(np.full((32, 18, 3), pts * 20, np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode(None):
            movie.mux(packet)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="CFR"):
        pose.inspect_source(target, pose.sha(target), duration=Fraction(1, 6))
    assert target.read_bytes() == before


def test_existing_output_never_overwritten_and_real_resource_not_loaded(source, tmp_path, monkeypatch):
    monkeypatch.setattr(pose, "EVIDENCE", tmp_path)
    existing = tmp_path / "previous-failure"
    existing.mkdir()
    original = existing / "failure.json"
    original.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="prior evidence"):
        pose.prepare(source[0], existing, expected_source_sha=pose.sha(source[0]), allow_pinned_cpu_models=True)
    assert original.read_bytes() == b"original evidence"
    assert not torch.cuda.is_initialized()


def test_one_good_person_does_not_hide_another_with_no_confident_joints():
    def detector(tensor):
        return torch.from_numpy(prediction(2))

    def estimator(tensor):
        sx, sy = torch.zeros(5, 133, 576), torch.zeros(5, 133, 768)
        sx[0, :, 200], sy[0, :, 300] = .8, .7
        return sx, sy

    with pytest.raises(ValueError, match="person has no confident"):
        pose.infer_frame(np.full((1024, 576, 3), 100, np.uint8), detector, estimator, width=288, height=512)


def test_native_RGB_PNG_MOV_actual_codec_avoids_hidden_pixel_format_conversion(tmp_path):
    from io import BytesIO
    from PIL import Image

    rng = np.random.default_rng(81224)
    arrays = [rng.integers(0, 256, (32, 18, 3), dtype=np.uint8) for _ in range(17)]
    movie = tmp_path / "nativeRGB.mov"
    pose.encode_rgb_movie(movie, arrays, Fraction(24))
    with av.open(str(movie)) as container:
        assert container.streams.video[0].codec_context.name == "png"
        assert not container.streams.audio
        frames = list(container.decode(video=0))
    assert len(frames) == len(arrays)
    for index, (frame, rgb) in enumerate(zip(frames, arrays)):
        assert frame.format.name == "rgb24"
        assert frame.pts * frame.time_base == Fraction(index, 24)
        assert np.array_equal(frame.to_ndarray(format="rgb24"), rgb)
    with av.open(str(movie)) as incoming:
        packets = [bytes(packet) for packet in incoming.demux(video=0) if packet.dts is not None]
    assert len(packets) == len(arrays)
    for packet, rgb in zip(packets, arrays):
        png = BytesIO()
        Image.fromarray(rgb).save(png, format="PNG")
        assert packet == png.getvalue()


def test_actual_pose_manifest_reader_and_selected_RGB_companions_remain_compatible(tmp_path):
    import json
    from PIL import Image

    media_spec = importlib.util.spec_from_file_location("h21_pose_media_contract", ROOT / "tools/prepare_h21_control_media.py")
    media = importlib.util.module_from_spec(media_spec)
    media_spec.loader.exec_module(media)
    arrays = [np.full((32, 18, 3), (index + 1) * 41, np.uint8) for index in range(4)]
    source = tmp_path / "selected_source.mov"
    pose.encode_rgb_movie(source, arrays, Fraction(24))
    images = []
    for index, rgb in enumerate(arrays):
        path = tmp_path / f"pose{index}.png"
        Image.fromarray(pose.resize_source(rgb, 9, 16)).save(path)
        images.append({"frame": index, "path": str(path), "sha256": pose.sha(path)})
    manifest = tmp_path / "report.json"
    manifest.write_text(json.dumps({"source": str(source), "source_sha256": pose.sha(source), "frames": 4,
                                   "fps": 24, "width": 9, "height": 16, "images": images}), encoding="utf-8")
    _, _, _, _, rate, pose_arrays, _ = media._manifest(manifest, pose.sha(manifest))
    control = tmp_path / "pose.mov"
    pose.encode_rgb_movie(control, pose_arrays, rate)
    assert len(media.verify_video(control, pose_arrays, rate)) == 4
    companions = tmp_path / "companions"
    companions.mkdir()
    saved, audio = media._source_frames(source, companions, 4, rate)
    assert audio == []
    assert [saved[role]["rgb_sha256"] for role in ("first", "middle", "last")] == [pose.rgb_sha(arrays[index]) for index in (0, 1, 3)]


def test_pure_source_decode_materializes_independent_PNG_pixels_before_model_or_encoder(source, tmp_path):
    from PIL import Image

    original, arrays, before = source
    plan = pose.inspect_source(original, pose.sha(original), duration=Fraction(1, 6))
    saved = pose.materialize_source_pngs(original, plan, tmp_path / "selected")
    assert [row["source_frame"] for row in saved] == [0, 1, 2, 4]
    for row in saved:
        with Image.open(row["path"]) as image:
            rgb = np.array(image)
        assert np.array_equal(rgb, arrays[row["source_frame"]])
        assert pose.sha(row["path"]) == row["sha256"]
    assert original.read_bytes() == before
    with pytest.raises(FileExistsError):
        pose.materialize_source_pngs(original, plan, tmp_path / "selected")


def test_materialization_does_not_accept_changed_frame_RGB_or_skip_it(source, tmp_path):
    original, _, before = source
    plan = pose.inspect_source(original, pose.sha(original), duration=Fraction(1, 6))
    plan["selected_frame_RGB_sha256"][1] = "0" * 64
    with pytest.raises(ValueError, match="RGB/PTS differs"):
        pose.materialize_source_pngs(original, plan, tmp_path / "failed")
    assert len(list((tmp_path / "failed").glob("*.png"))) == 1
    assert original.read_bytes() == before


def test_materialization_preserves_full_decode_RGB_conversion_order_before_selection(source, tmp_path, monkeypatch):
    """Trace actual tiny codec frames; not a trained pose/model fixture."""
    original, _, _ = source
    plan = pose.inspect_source(original, pose.sha(original), duration=Fraction(1, 6))
    real_open, conversions = pose.av.open, []

    class ObservedFrame:
        def __init__(self, index, frame):
            self.index, self.frame = index, frame
            self.pts, self.time_base = frame.pts, frame.time_base

        def to_ndarray(self, *, format):
            conversions.append(self.index)
            return self.frame.to_ndarray(format=format)

    class ObservedContainer:
        def __init__(self, *args, **kwargs):
            self.container = real_open(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def decode(self, **kwargs):
            for index, frame in enumerate(self.container.decode(**kwargs)):
                yield ObservedFrame(index, frame)

    monkeypatch.setattr(pose.av, "open", ObservedContainer)
    rows = pose.materialize_source_pngs(original, plan, tmp_path / "observed")
    assert len(rows) == 4
    assert conversions == list(range(plan["source_indices"][-1] + 1))
