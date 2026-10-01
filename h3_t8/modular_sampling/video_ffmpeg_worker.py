"""Trusted, data-only raw-YUV encoder worker. No ComfyUI, PyAV or torch imports."""
from fractions import Fraction
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local(root, name):
    if name not in {"frames.yuv", "audio.f32le", "metadata.txt", "video.partial.mp4", "worker-report.json"}:
        raise ValueError("Unknown owned worker file")
    path = root / name
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
        raise ValueError("Worker file is a link or junction")
    return path


def run(command, stage="codec"):
    # This process and all children belong to the controller's kill-on-close Job.
    tails = [deque(maxlen=512), deque(maxlen=16)]  # 4 MiB output, 128 KiB diagnostics.
    counts, errors = [0, 0], []
    def drain(pipe, index):
        try:
            while chunk := pipe.read(8192):
                counts[index] += len(chunk)
                tails[index].append(chunk)
        except (OSError, ValueError) as error:
            errors.append(error)
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) as process:
        threads = [threading.Thread(target=drain, args=(pipe, index), daemon=True)
                   for index, pipe in enumerate((process.stdout, process.stderr))]
        for thread in threads:
            thread.start()
        code = process.wait()
        for thread in threads:
            thread.join()  # Whole worker/tree still bounded by the controller's Job deadline.
    if errors or code:
        error = b"".join(tails[1])[-4000:].decode("utf8", "replace")
        raise RuntimeError(f"{stage} exited {code} (0x{code & 0xffffffff:08x}): {error}; drain_errors={len(errors)}")
    if counts[0] > 4 * 1024**2:
        raise ValueError(stage + " output exceeded the bounded response limit")
    return b"".join(tails[0]).decode("utf8")


def escape_metadata(value):
    return str(value).replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", "\\\n")


def execute(request, expected_sha):
    request = Path(request).absolute()
    if request.is_symlink() or request.stat().st_size > 4 * 1024**2 or sha(request) != expected_sha:
        raise ValueError("Worker request identity differs")
    root = request.parent
    for directory in (root, *root.parents):
        if directory.is_symlink() or getattr(directory.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Worker directory cannot traverse a link or junction")
    data = json.loads(request.read_text(encoding="utf8"))
    if data["schema"] != "t8.raw-yuv-isolated-export.v1":
        raise ValueError("Unknown raw video contract")
    raw = local(root, "frames.yuv")
    audio = local(root, "audio.f32le")
    partial = local(root, "video.partial.mp4")
    if partial.exists() or data["bit_depth"] not in (8, 10) or data["color_space"] not in ("sRGB", "HDR", "HDR PQ"):
        raise ValueError("Existing output or unsupported depth/color")
    def verify_inputs():
        if sha(request) != expected_sha or raw.stat().st_size != data["video_bytes"] or sha(raw) != data["video_sha"]:
            raise ValueError("Raw YUV changed")
        if data["audio"] and (audio.stat().st_size != data["audio"]["bytes"] or sha(audio) != data["audio"]["sha"]):
            raise ValueError("Raw audio changed")
    verify_inputs()
    pix_fmt = "yuv420p10le" if data["bit_depth"] == 10 else "yuv420p"
    fps = str(Fraction(data["fps"]))
    metadata_path = local(root, "metadata.txt")
    with metadata_path.open("x", encoding="utf8", newline="\n") as stream:
        stream.write(";FFMETADATA1\n")
        for key, value in data["metadata"].items():
            stream.write(escape_metadata(key) + "=" + escape_metadata(json.dumps(value, ensure_ascii=False, allow_nan=False)) + "\n")
    command = [data["ffmpeg"], "-hide_banner", "-v", "error", "-n", "-nostdin", "-threads", "1",
               "-f", "rawvideo", "-pixel_format", pix_fmt, "-video_size", f"{data['width']}x{data['height']}",
               "-framerate", fps, "-i", str(raw)]
    if data["audio"]:
        a = data["audio"]
        command += ["-f", "f32le", "-ar", str(a["sample_rate"]), "-ac", str(a["channels"]), "-i", str(audio)]
    command += ["-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:v:0"]
    if data["audio"]:
        command += ["-map", "1:a:0", "-c:a", "aac", "-threads:a", "1"]
    color, transfer = ("bt709", "iec61966-2-1") if data["color_space"] == "sRGB" else (
        "bt2020", "arib-std-b67" if data["color_space"] == "HDR" else "smpte2084")
    matrix = "bt709" if color == "bt709" else "bt2020nc"
    command += ["-map_metadata", "2" if data["audio"] else "1", "-c:v", "libx264", "-threads:v", "1",
                "-filter_threads", "1", "-vf", f"setparams=range=limited:color_primaries={color}:color_trc={transfer}:colorspace={matrix}",
                "-pix_fmt", pix_fmt, "-crf", "23", "-preset", "medium", "-color_range", "tv",
                "-color_primaries", color, "-color_trc", transfer,
                "-colorspace", matrix,
                "-movflags", "use_metadata_tags+faststart", str(partial)]
    run(command, "encode_h264_aac")
    details = json.loads(run([data["ffprobe"], "-v", "error", "-show_streams", "-of", "json", str(partial)], "probe_streams"))
    video = [s for s in details["streams"] if s["codec_type"] == "video"]
    sound = [s for s in details["streams"] if s["codec_type"] == "audio"]
    if (len(video) != 1 or (video[0]["width"], video[0]["height"], int(video[0]["nb_frames"])) !=
            (data["width"], data["height"], data["frames"]) or Fraction(video[0]["avg_frame_rate"]) != Fraction(fps)
            or video[0]["pix_fmt"] != pix_fmt or video[0]["color_transfer"] != transfer
            or len(sound) != int(bool(data["audio"]))):
        raise ValueError("Encoded media geometry, depth/color, fps or audio presence changed")
    if sound and (int(sound[0]["sample_rate"]), sound[0]["channels"]) != (data["audio"]["sample_rate"], data["audio"]["channels"]):
        raise ValueError("Encoded audio rate or channels changed")
    decoded = {}
    decoded_format = "rgb48le" if data["bit_depth"] == 10 else "rgb24"
    for kind, options in (("video", ["-map", "0:v:0", "-pix_fmt", decoded_format]),
                          ("audio", ["-map", "0:a:0", "-c:a", "pcm_f32le"])):
        if kind == "audio" and not data["audio"]:
            continue
        text = run([data["ffmpeg"], "-v", "error", "-xerror", "-i", str(partial), *options,
                    "-f", "hash", "-hash", "sha256", "-"], "strict_decode_" + kind).strip()
        if not text.startswith("SHA256=") or len(text) != 71:
            raise ValueError("Strict full decode did not return a SHA256")
        decoded[kind] = text[7:]
    verify_inputs()
    result = {"schema": data["schema"], "status": "validated_not_published", "request_sha": expected_sha,
              "file_sha256": sha(partial), "streams": details["streams"], "decoded_sha256": decoded,
              "decoded_video_format": decoded_format, "strict_decode_verified": True, "quality_accepted": False}
    with local(root, "worker-report.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    execute(sys.argv[1], sys.argv[2])
