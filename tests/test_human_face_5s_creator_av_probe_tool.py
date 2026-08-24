from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_human_face_5s_creator_av_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "run_human_face_5s_creator_av_probe", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_reuses_one_124_frame_human_face_latent_twice():
    tool = _load_tool()
    prompt = tool.build_prompt(run_id="unit")

    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in prompt.values()) == 1
    assert prompt["8"]["inputs"]["task_type"] == "I2VA"
    assert prompt["8"]["inputs"]["length"] == 124
    assert prompt["8"]["inputs"]["first_frame"] == ["15", 0]
    assert prompt["15"]["inputs"]["image"] == "10A.jpg"
    assert prompt["23"]["inputs"]["first_segment"] == ["12", 0]
    assert prompt["23"]["inputs"]["second_segment"] == ["12", 0]
    assert prompt["23"]["inputs"]["require_identical_metadata"] is True
    assert prompt["9"]["inputs"]["steps"] == 8


def test_exact_long_dual_clock_composition_counts():
    tool = _load_tool()

    assert tool.SOURCE_FRAMES == 124
    assert tool.DROP_VIDEO_FRAMES == 5
    assert tool.OUTPUT_FRAMES == 243
    assert tool.SOURCE_AUDIO_LATENT_STEPS == 207
    assert tool.DROP_AUDIO_LATENT_STEPS == 9
    assert tool.OUTPUT_AUDIO_LATENT_STEPS == 405
    assert tool.SOURCE_AUDIO_SAMPLES == 165_600
    assert tool.DROP_AUDIO_SAMPLES == 7_200
    assert tool.OUTPUT_AUDIO_SAMPLES == 324_000
    assert tool.OUTPUT_FRAMES / tool.face.FPS >= 5.0


def test_review_media_contract_uses_243_frames_at_512x256():
    tool = _load_tool()
    report = {
        "strict_decode_passed": True,
        "probe": {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 512,
                    "height": 256,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "32000",
                    "channels": 2,
                },
            ]
        },
        "decoded_video": {"bytes": 243 * 512 * 256 * 3},
        "decoded_audio": {"bytes": 1},
    }

    assert all(tool._review_media_checks(report).values())
    report["decoded_video"]["bytes"] = 124 * 512 * 256 * 3
    assert tool._review_media_checks(report)["decoded_video_exact_frames"] is False


def test_cli_cannot_lower_creator_reviewed_floor():
    tool = _load_tool()
    args = tool.parse_args([])
    assert args.arm == "clipproj_4b"
    assert args.min_free_vram_mib == 13_500
    with pytest.raises(SystemExit):
        tool.parse_args(["--min-free-vram-mib", "13499"])
