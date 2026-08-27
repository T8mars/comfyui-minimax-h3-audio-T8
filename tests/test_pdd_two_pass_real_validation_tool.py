from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_pdd_two_pass_real_validation.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "run_pdd_two_pass_real_validation", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", ["FL2VA", "Ref2VA"])
def test_prompt_splits_one_pdd_trajectory_into_four_plus_four(variant):
    tool = _load_tool()
    args = tool._parser().parse_args(["--variant", variant])
    prompt = tool._prompt(args, "unit-test")

    assert prompt["8"]["class_type"] == "MiniMaxH3PDD8StepSetupT8Advanced"
    assert prompt["9"]["inputs"] == {"sigmas": ["8", 2], "step": 4}
    assert prompt["12"]["inputs"]["sigmas"] == ["9", 0]
    assert prompt["13"]["inputs"]["av_latent"] == ["12", 1]
    assert prompt["14"]["inputs"]["width"] == ["13", 1]
    assert prompt["14"]["inputs"]["height"] == ["13", 2]
    assert prompt["16"]["inputs"]["model"] == ["8", 0]
    assert prompt["16"]["inputs"]["av_latent"] == ["15", 0]
    assert prompt["19"]["inputs"]["sigmas"] == ["9", 1]
    assert prompt["19"]["inputs"]["latent_image"] == ["15", 0]


def test_prompt_keeps_native_joint_audio_continuation():
    tool = _load_tool()
    args = tool._parser().parse_args(["--variant", "Ref2VA"])
    prompt = tool._prompt(args, "unit-test")

    assert prompt["15"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    assert prompt["15"]["inputs"]["second_pass_audio_strength"] == 0.0
    assert prompt["20"]["inputs"]["av_latent"] == ["19", 0]


def test_isolated_server_command_honors_lowvram_and_reserve(tmp_path):
    tool = _load_tool()
    args = tool._parser().parse_args([])
    args.python = args.python.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.lowvram = True
    command = tool.shared._server_command(args, tmp_path)

    assert "--lowvram" in command
    reserve_index = command.index("--reserve-vram")
    assert command[reserve_index + 1] == "2.0"


def test_guarded_contract_accepts_1p5_and_rejects_out_of_range_scale():
    tool = _load_tool()
    args = tool._parser().parse_args(["--scale-by", "1.5"])
    tool._validate_contract(args)
    assert tool._aligned_output_geometry(864, 480, 1.5) == (1312, 736)

    args = tool._parser().parse_args(["--scale-by", "0.5"])

    with pytest.raises(ValueError, match=r"within \[1.0, 4.0\]"):
        tool._validate_contract(args)
