"""Real CPU VAE/codec consumers; no trained GPU or perceptual qualification."""
import json
from fractions import Fraction
import shutil

import av
import pytest
import torch

from h3_audio_t8_pkg import audio_ops, h3_av_delivery
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3OutputTrimT8
from test_h21_core_vae_contracts import real_tiny_vae_owners  # noqa: F401


@pytest.mark.parametrize("broken", ["video", "audio"])
def test_basic_decode_actual_core_nonfinite_then_clean_next_request(real_tiny_vae_owners, broken):  # noqa: F811
    video, audio = real_tiny_vae_owners
    value = empty_av_latent(64, 32, 5)[0]
    for tensor in value["samples"].unbind():
        tensor.normal_(generator=torch.Generator().manual_seed(211))
    with torch.no_grad():
        baseline = MiniMaxH3AVDecodeT8.execute(value, video, audio).result
        parameter = (video.first_stage_model.post_quant_conv.weight if broken == "video"
                     else audio.first_stage_model.dec_in_proj.weight)
        original = parameter.clone()
        parameter.fill_(float("nan"))
        try:
            with pytest.raises(ValueError, match="nonfinite|invalid"):
                MiniMaxH3AVDecodeT8.execute(value, video, audio)
        finally:
            parameter.copy_(original)
        clean = MiniMaxH3AVDecodeT8.execute(value, video, audio).result
    torch.testing.assert_close(clean[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(clean[1]["waveform"], baseline[1]["waveform"], rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_absolute_pcm_endpoints_reassemble_without_legacy_one_sample_overlap():
    frames = torch.arange(24).float().reshape(-1, 1, 1, 1)
    sound = {"waveform": torch.arange(44100).float().reshape(1, 1, -1), "sample_rate": 44100}
    legacy = [audio_ops.trim_av_output(frames, start, 1/24, sound)[1]["waveform"]
              for start in (0., 1/24)]
    assert sum(x.shape[-1] for x in legacy) == 3676  # Retained historical behavior.
    legacy_mapping = json.loads(audio_ops.trim_av_output(frames, 1/24, 1/24, sound)[2])["audio_mapping"]
    assert legacy_mapping["absolute_end_sample"] == 3676
    assert legacy_mapping["requested_absolute_end_sample"] == 3675
    exact = [audio_ops.trim_av_output(frames, start, 1/24, sound,
              audio_boundary_policy="absolute_endpoints")[1]["waveform"] for start in (0., 1/24)]
    assert torch.equal(torch.cat(exact, -1), sound["waveform"][..., :3675])


def media(count=24, rate=44100):
    frames = torch.linspace(0, 1, count * 32 * 64 * 3).reshape(count, 32, 64, 3)
    n = round(count * rate / 24)
    sound = {"sample_rate": rate, "waveform": (.1 * torch.sin(torch.arange(n) * .1)).reshape(1, 1, -1)}
    return frames, sound


def test_valid_but_wrong_completed_file_is_not_published(tmp_path, monkeypatch):
    # A valid real codec result for a different take is not this request's output.
    wrong = tmp_path / "earlier-take.mp4"
    h3_av_delivery.save_h3_av_safe(*media(12), wrong)
    original = wrong.read_bytes()
    monkeypatch.setattr(h3_av_delivery, "_mux_h3_world_audio",
                        lambda video, raw, output, **kw: shutil.copyfile(wrong, output))
    result = tmp_path / "new-take.mp4"
    with pytest.raises(ValueError, match="delivery|Delivery"):
        h3_av_delivery.save_h3_av_safe(*media(), result)
    assert not result.exists() and wrong.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["earlier-take.mp4"]


@pytest.mark.parametrize("origin", [0., 10., -2.])
@pytest.mark.parametrize("rate", [32000, 44100, 48000])
def test_many_adjacent_global_windows_cover_exact_original_pcm(origin, rate):
    frames, sound = media(96, rate)
    # Include negative source origin without requesting negative global time.
    first = 48 if origin < 0 else 0
    pieces, video_pieces = [], []
    for i in range(first, first + 48):
        clip, audio, raw = MiniMaxH3OutputTrimT8.execute(frames, origin+i/24, 1/24, 24., sound,
            source_time_origin_seconds=origin, audio_boundary_policy="absolute_endpoints").result
        report = json.loads(raw)
        assert report["start_frame"] == i
        assert report["audio_mapping"]["right_padding_samples"] == 0
        pieces.append(audio["waveform"])
        video_pieces.append(clip)
    assert torch.equal(torch.cat(video_pieces), frames[first:first+48])
    assert torch.equal(torch.cat(pieces, -1), sound["waveform"][..., round(first*rate/24):round((first+48)*rate/24)])


def test_origin_underflow_and_video_overflow_are_real_errors_and_audio_padding_is_reported():
    frames, sound = media(24, 32000)
    with pytest.raises(ValueError, match="before the source"):
        audio_ops.trim_av_output(frames, 9., 1., sound, source_time_origin_seconds=10.)
    with pytest.raises(ValueError, match="exceed decoded"):
        audio_ops.trim_av_output(frames, 10.5, 1., sound, source_time_origin_seconds=10.)
    short = {**sound, "waveform": sound["waveform"][..., :20000]}
    _, audio, raw = audio_ops.trim_av_output(frames, 10., 1., short,
        source_time_origin_seconds=10., audio_boundary_policy="absolute_endpoints")
    row = json.loads(raw)["audio_mapping"]
    assert row["right_padding_samples"] == 12000 and row["copied_samples"] == 20000
    assert row["absolute_start_sample"] == 320000 and row["local_start_sample"] == 0
    assert torch.equal(audio["waveform"][..., :20000], short["waveform"])
    assert torch.count_nonzero(audio["waveform"][..., 20000:]) == 0


@pytest.mark.parametrize("field", ["start_seconds", "duration_seconds", "fps", "source_time_origin_seconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_clock_has_concrete_error_before_slicing(field, value):
    kwargs = dict(start_seconds=0., duration_seconds=1., fps=24., source_time_origin_seconds=0.)
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite"):
        audio_ops.trim_av_output(torch.zeros(24, 2, 2, 3), **kwargs)


@pytest.mark.parametrize("rate", [32000, 44100, 48000])
@pytest.mark.parametrize("preserve", [False, True])
def test_actual_codec_pcm_endpoint_and_padding_receipt(tmp_path, rate, preserve):
    frames, sound = media(25, rate)
    if preserve:
        sound["waveform"] = sound["waveform"][..., :10001]
    path = tmp_path / "clock.mp4"
    report = h3_av_delivery.save_h3_av_safe(frames, sound, path, fps=24., preserve_audio_length=preserve)
    check = report["delivery_verification"]
    assert check["full_decode_verified"] and not check["quality_reviewed"]
    assert 0 <= check["aac_decode_padding_samples"] < 1024
    assert check["logical_audio_samples"] == (10001 if preserve else (25*rate+23)//24)
    with av.open(str(path)) as container:
        decoded = list(container.decode(video=0))
        assert len(decoded) == 25
        assert [round(float(f.pts*f.time_base)*24) for f in decoded] == list(range(25))


def test_interrupted_completed_private_encode_cleans_then_retry_succeeds(tmp_path, monkeypatch):
    from comfy.model_management import InterruptProcessingException
    original = h3_av_delivery.delivery._encode_rgb_frames_isolated
    def interrupt_after_encoding(*args, **kwargs):
        original(*args, **kwargs)
        raise InterruptProcessingException()
    monkeypatch.setattr(h3_av_delivery.delivery, "_encode_rgb_frames_isolated", interrupt_after_encoding)
    result = tmp_path / "retry.mp4"
    with pytest.raises(InterruptProcessingException):
        h3_av_delivery.save_h3_av_safe(*media(), result)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(h3_av_delivery.delivery, "_encode_rgb_frames_isolated", original)
    assert h3_av_delivery.save_h3_av_safe(*media(), result)["status"] == "pass"


def test_unknown_vae_class_keeps_actual_core_decode_delegated(real_tiny_vae_owners):  # noqa: F811
    video, audio = real_tiny_vae_owners
    video.__class__ = type("UnlistedVideoOwner", (type(video),), {})
    value = empty_av_latent(64, 32, 5)[0]
    with torch.no_grad():
        frames, sound, *_ = MiniMaxH3AVDecodeT8.execute(value, video, audio).result
    assert frames.shape == (5, 32, 64, 3) and torch.isfinite(sound["waveform"]).all()
    assert video.calls == [("regular", None)]


def test_real_nonzero_pts_source_decode_trim_and_zero_origin_delivery(tmp_path):
    from h3_audio_t8_pkg.director_media import run_media
    frames, sound = media(24, 48000)
    source = tmp_path / "source-offset.mkv"
    with av.open(str(source), "w") as container:
        video = container.add_stream("ffv1", rate=24)
        video.width, video.height, video.pix_fmt = 64, 32, "bgr0"
        video.codec_context.thread_count = 1
        audio = container.add_stream("pcm_f32le", rate=48000)
        audio.layout = "mono"
        audio.codec_context.thread_count = 1
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray((pixels*255).round().byte().numpy(), format="rgb24")
            frame.pts, frame.time_base = 120+index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(sound["waveform"][0].numpy(), format="fltp", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 48000, 240000, Fraction(1, 48000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    evidence = run_media("external", source)
    assert not evidence["decoded_clock"]["zero_origin_film_compatible"]
    assert evidence["decoded_clock"]["video"]["first_time"] == {"num": 5, "den": 1}
    with av.open(str(source)) as container:
        rgb = torch.stack([torch.from_numpy(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]).float()/255
    with av.open(str(source)) as container:
        pcm = torch.cat([torch.from_numpy(f.to_ndarray()) for f in container.decode(audio=0)], -1).unsqueeze(0)
    cut, selected, report = MiniMaxH3OutputTrimT8.execute(rgb, 5.25, .5, 24.,
        {"waveform": pcm, "sample_rate": 48000}, source_time_origin_seconds=5.,
        audio_boundary_policy="absolute_endpoints").result
    assert torch.equal(cut, rgb[6:18])
    assert torch.equal(selected["waveform"], pcm[..., 12000:36000])
    assert json.loads(report)["audio_mapping"]["absolute_start_sample"] == 252000
    delivered = h3_av_delivery.save_h3_av_safe(cut, selected, tmp_path / "rebased.mp4")
    assert delivered["delivery_verification"]["decoded"]["frames"] == 12
    assert all(row["start_pts"] == 0 for row in delivered["delivery_verification"]["source_timing"]["streams"])
