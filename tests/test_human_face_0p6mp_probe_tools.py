from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    tools_root = str(TOOLS)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _media_report(*, frames: int, width: int, height: int):
    return {
        "strict_decode_passed": True,
        "probe": {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": width,
                    "height": height,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "32000",
                    "channels": 2,
                },
            ]
        },
        "decoded_video": {"bytes": frames * width * height * 3},
        "decoded_audio": {"bytes": 1},
    }


def test_clipproj_0p6mp_contract_changes_only_canvas():
    tool = _load("run_human_face_0p6mp_clipproj_probe")
    contract = tool._contract()
    assert (contract["width"], contract["height"]) == (1088, 544)
    assert contract["pixels"] == 591_872
    assert contract["multiple_of_32"] is True
    assert contract["relative_aspect_error"] < 0.02
    assert contract["frame_count"] == 124
    assert contract["steps"] == 8
    assert contract["seed"] == 2608245001
    assert contract["task_type"] == "I2VA"
    for arm in tool.ARMS:
        graph = tool.build_prompt(arm=arm, run_id="unit")
        assert graph["8"]["inputs"]["width"] == 1088
        assert graph["8"]["inputs"]["height"] == 544
        assert graph["8"]["inputs"]["length"] == 124
        assert graph["9"]["inputs"]["steps"] == 8
    native = tool.build_prompt(arm="native_32b", run_id="unit")
    assert native["3"]["inputs"]["clip_name"] == (
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    assert native["8"]["inputs"]["first_frame"] == ["15", 0]
    assert "5" not in native


def test_clipproj_0p6mp_media_and_resource_contracts():
    tool = _load("run_human_face_0p6mp_clipproj_probe")
    assert all(tool._media_checks(_media_report(frames=124, width=1088, height=544)).values())
    assert tool.parse_args(["--arm", "clipproj_4b"]).min_free_vram_mib == 14_500
    assert tool.parse_args(["--arm", "native_32b"]).min_free_vram_mib == 14_500
    with pytest.raises(SystemExit):
        tool.parse_args(["--arm", "clipproj_4b", "--min-free-vram-mib", "14499"])


def test_creator_0p6mp_reuses_one_sampler_and_preserves_av_counts():
    tool = _load("run_human_face_0p6mp_creator_av_probe")
    graph = tool.build_prompt(run_id="unit")
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) == 1
    assert graph["8"]["inputs"]["width"] == 1088
    assert graph["8"]["inputs"]["height"] == 544
    assert graph["8"]["inputs"]["length"] == 124
    assert graph["23"]["inputs"]["first_segment"] == ["12", 0]
    assert graph["23"]["inputs"]["second_segment"] == ["12", 0]
    assert tool.legacy.OUTPUT_FRAMES == 243
    assert tool.legacy.OUTPUT_AUDIO_SAMPLES == 324_000
    with tool._configured_legacy():
        checks = tool.legacy._review_media_checks(
            _media_report(frames=243, width=1088, height=544)
        )
    assert all(checks.values())
    assert tool.parse_args([]).min_free_vram_mib == 14_500


def test_creator_file_backed_encoder_avoids_large_stdin_pipe(monkeypatch, tmp_path):
    tool = _load("run_human_face_0p6mp_creator_av_probe")
    frames = [type("Frame", (), {"tobytes": lambda self: b"rgb"})()]
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    output = tmp_path / "out.mp4"
    observed = {}

    class Completed:
        returncode = 0
        stderr = b""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        assert Path(command[command.index("-i") + 1]).read_bytes() == b"rgb"
        return Completed()

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    tool._encode_review_arm_from_file(
        frames=frames, audio_path=audio, output_path=output, ffmpeg="ffmpeg"
    )
    assert "input" not in observed["kwargs"]
    assert observed["kwargs"]["timeout"] == 600
    assert not output.with_suffix(".rgb24.tmp").exists()


def test_creator_png_sequence_route_uses_x264_without_rawvideo_pipe():
    tool = _load("run_human_face_0p6mp_creator_av_probe")
    args = tool._video_encode_args()
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-threads") + 1] == "1"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert "rawvideo" not in args
    with tool._configured_legacy():
        assert tool.legacy._prepare_review_media is tool._prepare_review_media_from_pngs
