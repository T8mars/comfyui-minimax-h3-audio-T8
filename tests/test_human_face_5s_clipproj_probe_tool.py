from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_human_face_5s_clipproj_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "run_human_face_5s_clipproj_probe", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_arms_share_the_fixed_long_face_i2va_contract():
    tool = _load_tool()
    graphs = {arm: tool.build_prompt(arm=arm, run_id="unit") for arm in tool.ARMS}
    contracts = {
        arm: (
            graph["8"]["inputs"]["prompt"],
            graph["8"]["inputs"]["width"],
            graph["8"]["inputs"]["height"],
            graph["8"]["inputs"]["length"],
            graph["8"]["inputs"]["task_type"],
            graph["9"]["inputs"]["steps"],
            graph["10"]["inputs"]["noise_seed"],
        )
        for arm, graph in graphs.items()
    }
    assert len(set(contracts.values())) == 1
    assert next(iter(contracts.values())) == (
        tool.PROMPT,
        512,
        256,
        124,
        "I2VA",
        8,
        2608245001,
    )
    assert graphs["clipproj_4b"]["5"]["inputs"]["has_reference_images"] is True
    assert graphs["clipproj_8b"]["5"]["inputs"]["has_reference_images"] is True
    assert graphs["native_32b"]["8"]["inputs"]["first_frame"] == ["15", 0]
    assert graphs["native_32b"]["15"]["inputs"]["image"] == "10A.jpg"


def test_contract_is_at_least_five_seconds_and_has_spoken_dialogue():
    tool = _load_tool()
    contract = tool._contract()
    assert contract["frame_count"] == 124
    assert contract["duration_seconds"] >= 5.0
    assert "<d>" in contract["prompt"] and "</d>" in contract["prompt"]
    assert "no subtitles" in contract["prompt"]


def test_media_contract_checks_124_frames_at_512x256():
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
        "decoded_video": {"bytes": 124 * 512 * 256 * 3},
        "decoded_audio": {"bytes": 1},
    }
    assert all(tool._media_checks(report).values())


def test_cli_cannot_lower_native_32b_reviewed_floor():
    tool = _load_tool()
    assert tool.parse_args(["--arm", "native_32b"]).min_free_vram_mib == 14_500
    with pytest.raises(SystemExit):
        tool.parse_args(
            ["--arm", "native_32b", "--min-free-vram-mib", "13000"]
        )
