from __future__ import annotations

import argparse
from fractions import Fraction
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import av
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "h3_audio_t8_pkg"


def _load_runtime():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.flashvsr_advanced")


def _read_frames(path: Path, frame_count: int, width: int, height: int):
    decoded = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24.0)
        for frame in container.decode(stream):
            decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0)
            if len(decoded) == frame_count:
                break
    if not decoded:
        raise RuntimeError(f"no video frames decoded from {path}")
    frames = torch.stack(decoded).permute(0, 3, 1, 2)
    frames = F.interpolate(frames, size=(height, width), mode="bicubic", align_corners=False)
    return frames.permute(0, 2, 3, 1).clamp(0.0, 1.0), fps


def _write_video(path: Path, frames: torch.Tensor, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for tensor in frames:
            array = tensor.mul(255).round().byte().cpu().numpy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _mux_source_audio(video: Path, source: Path, output: Path, duration: float):
    ffmpeg = Path(r"F:\AI-T8-video-onekey\ffmpeg\bin\ffmpeg.exe")
    command = [
        str(ffmpeg if ffmpeg.is_file() else "ffmpeg"),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-t",
        f"{duration:.9f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]
    subprocess.run(command, check=True)


def _strict_decode(path: Path):
    ffmpeg = Path(r"F:\AI-T8-video-onekey\ffmpeg\bin\ffmpeg.exe")
    command = [
        str(ffmpeg if ffmpeg.is_file() else "ffmpeg"),
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="One serial low-resolution FlashVSR probe")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["quality_locked", "balanced_dynamic_exp", "memory_safe"], default="quality_locked")
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--seed", type=int, default=26083001)
    args = parser.parse_args()

    runtime = _load_runtime()
    frames, fps = _read_frames(args.input, args.frames, args.width, args.height)
    audio_sentinel = {"source_file": str(args.input), "passthrough_probe": True}
    handle, load_report = runtime.load_flashvsr_model(
        model_dir=args.model_dir,
        model_name=args.model_dir.name,
        mode="tiny",
        precision="bf16",
    )
    plan, plan_report = runtime.build_flashvsr_plan(
        frames,
        quality_profile=args.profile,
        tile_size=128,
        tile_overlap=16,
    )
    restored, source, returned_audio, run_report = runtime.restore_flashvsr(
        handle,
        plan,
        frames,
        audio_sentinel,
        scale=args.scale,
        seed=args.seed,
        color_fix=True,
        release_policy="clear_after",
    )
    if returned_audio is not audio_sentinel:
        raise RuntimeError("FlashVSR did not return the exact input audio object")
    if restored.shape[0] != source.shape[0] or not bool(torch.isfinite(restored).all()):
        raise RuntimeError("FlashVSR returned invalid frames")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    silent = args.output_dir / f"flashvsr_{args.profile}_silent.mp4"
    final = args.output_dir / f"flashvsr_{args.profile}.mp4"
    _write_video(silent, restored, fps)
    _mux_source_audio(silent, args.input, final, restored.shape[0] / fps)
    _strict_decode(final)
    report = {
        "input": str(args.input),
        "output": str(final),
        "profile": args.profile,
        "source_shape": list(source.shape),
        "output_shape": list(restored.shape),
        "fps": fps,
        "load": json.loads(load_report),
        "plan": json.loads(plan_report),
        "run": json.loads(run_report),
        "audio_object_identity": True,
        "strict_decode": True,
    }
    (args.output_dir / f"flashvsr_{args.profile}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    silent.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
