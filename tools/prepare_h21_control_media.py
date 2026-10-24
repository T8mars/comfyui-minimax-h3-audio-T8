"""CPU-only, lossless repack of an existing ordered pose PNG manifest.

No pose model is run. The original images/video are read-only. The stored
control movie contains only the original frames; Director Source, not this
utility, owns any explicit hold-last padding required by the generated canvas.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _manifest(path, expected_sha):
    path = Path(path).resolve(strict=True)
    identity = file_sha(path)
    if expected_sha is not None and identity != expected_sha:
        raise ValueError("Pose manifest SHA changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pose manifest must be an object")
    frames = value.get("frames")
    if type(frames) is not int or frames <= 0:
        raise ValueError("Pose frame count must be a positive integer")
    rate = Fraction(str(value.get("fps")))
    if rate <= 0:
        raise ValueError("Pose fps must be positive")
    rows = value.get("images")
    if not isinstance(rows, list) or len(rows) != frames:
        raise ValueError("Pose manifest count differs from its ordered images")
    width, height = value.get("width"), value.get("height")
    if any(type(size) is not int or size <= 0 for size in (width, height)):
        raise ValueError("Pose dimensions must be positive integers")
    source = Path(value["source"]).resolve(strict=True)
    if file_sha(source) != value.get("source_sha256"):
        raise ValueError("Original RGB source SHA changed")
    arrays, identities, seen = [], [], set()
    for number, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("frame")) is not int or row["frame"] != number:
            raise ValueError("Pose image order must be contiguous original frame indices")
        image_path = Path(row["path"]).resolve(strict=True)
        if image_path in seen:
            raise ValueError("Pose image paths must not repeat")
        seen.add(image_path)
        actual = file_sha(image_path)
        if actual != row.get("sha256"):
            raise ValueError(f"Pose image SHA changed at frame {number}")
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"Pose image is not the declared original RGB geometry at frame {number}")
            array = np.array(image)
        arrays.append(array)
        identities.append({"frame": number, "path": str(image_path), "png_sha256": actual, "rgb_sha256": rgb_sha(array)})
    return path, identity, value, source, rate, arrays, identities


def _encode(path, arrays, rate):
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264rgb", rate=rate)
        stream.width = arrays[0].shape[1]
        stream.height = arrays[0].shape[0]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "medium", "threads": "1"}
        stream.time_base = Fraction(1, 1) / rate
        stream.codec_context.time_base = stream.time_base
        for number, array in enumerate(arrays):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = number
            frame.time_base = Fraction(1, 1) / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def verify_video(path, arrays, rate):
    checked = []
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        if container.streams.audio or video.average_rate != rate:
            raise ValueError("Control movie must be CFR at original fps and contain no audio")
        for number, frame in enumerate(container.decode(video)):
            if number >= len(arrays):
                raise ValueError("Control movie contains invented extra frames")
            pixels = frame.to_ndarray(format="rgb24")
            if not np.array_equal(pixels, arrays[number]):
                raise ValueError(f"Lossless RGB check failed at control frame {number}")
            if frame.pts is None or frame.pts * frame.time_base != Fraction(number, 1) / rate:
                raise ValueError(f"Control frame {number} has a non-CFR timestamp")
            checked.append({"frame": number, "pts": frame.pts, "time_base": str(frame.time_base), "rgb_sha256": rgb_sha(pixels)})
    if len(checked) != len(arrays):
        raise ValueError("Control movie is incomplete")
    return checked


def _source_frames(source, output, count, rate):
    wanted = {"first": 0, "middle": (count - 1) // 2, "last": count - 1}
    result, total = {}, 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != rate:
            raise ValueError("Original RGB video fps does not match pose provenance")
        for number, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.pts * frame.time_base != Fraction(number, 1) / rate:
                raise ValueError("Original RGB video is not the declared complete CFR timeline")
            for role, index in wanted.items():
                if number == index:
                    pixels = frame.to_ndarray(format="rgb24")
                    target = output / f"rgb_{role}.png"
                    Image.fromarray(pixels).save(target)
                    result[role] = {"path": str(target), "frame": number, "fps": float(rate), "width": frame.width, "height": frame.height, "sha256": file_sha(target), "rgb_sha256": rgb_sha(pixels)}
            total += 1
        audio = [{"sample_rate": item.codec_context.sample_rate, "channels": item.codec_context.channels} for item in container.streams.audio]
    if total != count:
        raise ValueError("Original RGB video count differs from pose provenance")
    return result, audio


def prepare(manifest, output, *, expected_manifest_sha256=None):
    path, identity, value, source, rate, arrays, inputs = _manifest(manifest, expected_manifest_sha256)
    output = Path(output).resolve()
    # Never reuse/overwrite a prior success or failure directory.
    output.mkdir(parents=True, exist_ok=False)
    movie = output / "control_pose_cfr_rgb_lossless.mp4"
    _encode(movie, arrays, rate)
    verified = verify_video(movie, arrays, rate)
    source_images, audio = _source_frames(source, output, len(arrays), rate)
    for row in inputs:
        if file_sha(row["path"]) != row["png_sha256"]:
            raise ValueError("Pose PNG changed during preparation")
    if file_sha(source) != value["source_sha256"] or file_sha(path) != identity:
        raise ValueError("Provenance source changed during preparation")
    report = {
        "schema": "t8.local.h21.control-media.v1",
        "status": "CPU_lossless_RGB_and_CFR_verified_not_model_or_GPU_qualified",
        "manifest": {"path": str(path), "sha256": identity},
        "raw_rgb": {"path": str(source), "sha256": value["source_sha256"], "frames": len(arrays), "fps": str(rate), "audio_streams": audio},
        "control_video": {"path": str(movie), "sha256": file_sha(movie), "bytes": movie.stat().st_size, "frames": len(arrays), "fps": str(rate), "width": value["width"], "height": value["height"], "codec": "libx264rgb_crf0", "audio": False},
        "source_images": source_images,
        "input_pngs": inputs,
        "decoded_control_frames": verified,
        "roles": {"raw_rgb": "read-only provenance; not automatically a reference, first/end or audio source", "control_video": "pose IMAGE control only, no embedded audio or hidden frames", "first_middle_last": "exact decoded original RGB, optional explicit conditioning; not pose frames", "original_audio": "retained in raw RGB source; not automatically driven or muxed"},
        "actions": {"pose_model_rerun": False, "Torch_or_Core_import": False, "GPU": False, "service": False, "download": False, "environment_change": False, "source_modification": False},
    }
    _write_json(output / "report.json", report)
    return report


def extract_float_audio(source, output, *, expected_source_sha256):
    """Keep the decoded original float PCM in a separate explicit audio asset.

    Decoder padding is retained, not silently trimmed. This is not a speech
    alignment pass or source-audio adoption by any generated-video recipe.
    """
    source = Path(source).resolve(strict=True)
    if file_sha(source) != expected_source_sha256:
        raise ValueError("Original audio-bearing RGB source SHA changed")
    blocks = []
    with av.open(str(source)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("Explicit audio extraction requires one original audio stream")
        stream = container.streams.audio[0]
        rate, layout = stream.codec_context.sample_rate, stream.codec_context.layout.name
        channels = len(stream.codec_context.layout.channels)
        for frame in container.decode(stream):
            if frame.format.name not in {"flt", "fltp"} or frame.sample_rate != rate or len(frame.layout.channels) != channels:
                raise ValueError("Original decoder must expose stable float32 PCM; no implicit resample or downmix")
            pixels = frame.to_ndarray()
            canonical = pixels if frame.format.is_planar else pixels.reshape(-1, channels).T
            blocks.append(np.ascontiguousarray(canonical))
    if not blocks:
        raise ValueError("Original audio contains no decoded samples")
    expected = np.concatenate(blocks, axis=1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    wav = output / "original_decoded_float32.wav"
    with av.open(str(wav), mode="w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=rate)
        stream.layout = layout
        position = 0
        for block in blocks:
            packed = np.ascontiguousarray(block.T).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(packed, format="flt", layout=layout)
            frame.sample_rate, frame.pts, frame.time_base = rate, position, Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
            position += block.shape[1]
        for packet in stream.encode(None):
            container.mux(packet)
    decoded = []
    with av.open(str(wav)) as container:
        for frame in container.decode(audio=0):
            array = frame.to_ndarray()
            decoded.append(np.ascontiguousarray(array if frame.format.is_planar else array.reshape(-1, channels).T))
    actual = np.concatenate(decoded, axis=1)
    if expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
        raise ValueError("Original float32 PCM did not survive explicit audio asset bit-exactly")
    if file_sha(source) != expected_source_sha256:
        raise ValueError("Original audio source changed during extraction")
    report = {"schema": "t8.local.h21.original-audio.v1", "status": "CPU_decoded_PCM_bit_exact_not_generated_audio_or_lipsync_qualified", "source": {"path": str(source), "sha256": expected_source_sha256}, "audio": {"path": str(wav), "bytes": wav.stat().st_size, "sha256": file_sha(wav), "sample_rate": rate, "channels": channels, "samples": expected.shape[1], "duration": expected.shape[1] / rate, "source_decoded_pcm_sha256": rgb_sha(expected), "wav_decoded_pcm_sha256": rgb_sha(actual)}, "policy": "No resample, downmix, ASR or implicit trim; codec padding retained. Explicit record window must select original timeline. Not a control signal or automatic @ref."}
    _write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    report = prepare(args.manifest, args.output, expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps({"report": str(Path(args.output).resolve() / "report.json"), "control_video": report["control_video"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
