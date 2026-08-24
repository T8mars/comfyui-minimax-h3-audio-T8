from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "analyze_clipproj_same_input_three_way.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "analyze_clipproj_same_input_three_way", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _graph(seed: int = 7):
    return {
        "1": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "prompt": "rain",
                "width": 256,
                "height": 256,
                "length": 22,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
            },
        },
        "2": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {"steps": 4, "shift_video": 12.0, "shift_audio": 3.0},
        },
        "3": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }


def test_generation_contract_extracts_only_common_sampling_inputs():
    tool = _load_tool()

    contract = tool.extract_generation_contract(_graph())

    assert contract == {
        "prompt": "rain",
        "seed": 7,
        "width": 256,
        "height": 256,
        "frame_count": 22,
        "task_type": "T2VA",
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "steps": 4,
        "shift_video": 12.0,
        "shift_audio": 3.0,
    }


def test_contract_comparison_is_exact_and_reports_a_stable_hash():
    tool = _load_tool()
    contract = tool.extract_generation_contract(_graph())

    result = tool.require_equal_contracts(
        {arm: dict(contract) for arm in tool.ARM_ORDER}
    )

    assert len(result["contract_sha256"]) == 64
    changed = {arm: dict(contract) for arm in tool.ARM_ORDER}
    changed["clipproj_8b"]["seed"] = 8
    with pytest.raises(ValueError, match="not identical"):
        tool.require_equal_contracts(changed)


def test_runtime_observations_preserve_scope_and_do_not_infer_superiority():
    tool = _load_tool()
    guarded = {
        "status": "PASS",
        "passed": True,
        "phase": {"elapsed_seconds": 36.0},
        "gpu": {
            "baseline": {"used_mib": 2200},
            "monitor": {"peak_used_mib": 15000, "minimum_free_mib": 1100},
        },
    }
    trace = {
        "runtime": {
            "status": "success",
            "duration_seconds": 38.0,
            "server_snapshot": {"devices": [{"vram_total": 16 * 1024**3}]},
            "summary": {
                "peak_vram_used_bytes": 14 * 1024**3,
                "baseline_vram_used_bytes": 2 * 1024**3,
            },
        }
    }

    first = tool.runtime_observation(guarded, source_kind="guarded_probe")
    second = tool.runtime_observation(trace, source_kind="runtime_trace")

    assert first["prompt_to_terminal_seconds"] == 36.0
    assert first["terminal_success"] is True
    assert second["whole_device_peak_mib"] == 14 * 1024
    assert second["minimum_free_mib"] == 2 * 1024
    assert second["terminal_success"] is True
