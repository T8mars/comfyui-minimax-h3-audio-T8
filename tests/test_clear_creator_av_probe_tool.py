from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_clear_creator_av_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_clear_creator_av_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_samples_once_reuses_exact_latent_and_saves_lossless_sources():
    tool = _load_tool()
    prompt = tool.build_prompt(run_id="unit")

    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in prompt.values()) == 1
    assert prompt["23"]["inputs"]["first_segment"] == ["12", 0]
    assert prompt["23"]["inputs"]["second_segment"] == ["12", 0]
    assert prompt["23"]["inputs"]["require_identical_metadata"] is True
    assert prompt["24"]["inputs"]["av_latent"] == ["23", 0]
    assert prompt["26"]["inputs"]["av_latent"] == ["12", 0]
    assert prompt["25"]["class_type"] == "SaveImage"
    assert prompt["27"]["class_type"] == "SaveImage"
    assert prompt["30"]["class_type"] == "SaveAudio"
    assert prompt["31"]["class_type"] == "SaveAudio"
    assert prompt["9"]["inputs"]["steps"] == 8
    assert prompt["8"]["inputs"]["prompt"] == tool.triplet.CLEAR_PROMPT


def test_exact_dual_clock_composition_counts():
    tool = _load_tool()
    assert tool.SOURCE_FRAMES == 22
    assert tool.DROP_VIDEO_FRAMES == 5
    assert tool.OUTPUT_FRAMES == 39
    assert tool.SOURCE_AUDIO_SAMPLES == 29_600
    assert tool.DROP_AUDIO_SAMPLES == 7_200
    assert tool.OUTPUT_AUDIO_SAMPLES == 52_000


def test_review_media_contract_uses_39_frames_not_22():
    tool = _load_tool()
    report = {
        "strict_decode_passed": True,
        "probe": {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 256,
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
        "decoded_video": {"bytes": 39 * 256 * 256 * 3},
        "decoded_audio": {"bytes": 1},
    }

    assert all(tool._review_media_checks(report).values())
    report["decoded_video"]["bytes"] = 22 * 256 * 256 * 3
    assert tool._review_media_checks(report)["decoded_video_exact_frames"] is False


def test_cli_is_fixed_to_4b_low_load_arm():
    tool = _load_tool()
    args = tool.parse_args([])
    assert args.arm == "clipproj_4b"
    assert args.min_free_vram_mib == 13_000
