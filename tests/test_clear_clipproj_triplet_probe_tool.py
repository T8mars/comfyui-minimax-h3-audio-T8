from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_clear_clipproj_triplet_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "run_clear_clipproj_triplet_probe", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_tree(tool, tmp_path: Path, arm: str):
    comfy = tmp_path / "ComfyUI"
    python = tmp_path / "python.exe"
    for path in (comfy / "main.py", python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    for name in (
        "minimax-h3-audio-T8",
        "ComfyUI-ClipProj",
        "ComfyUI-VideoHelperSuite",
    ):
        (comfy / "custom_nodes" / name).mkdir(parents=True, exist_ok=True)
    spec = tool.ARM_ASSETS[arm]
    for role, path in tool._model_paths(comfy, arm).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "clip":
            size = int(spec["clip_bytes"])
        elif role == "projection":
            size = int(spec["projection_bytes"])
        else:
            size = 1
        path.write_bytes(b"x" * size)
    return comfy, python


def test_all_arms_share_the_exact_clear_eight_nfe_contract():
    tool = _load_tool()
    graphs = {arm: tool.build_prompt(arm=arm, run_id="unit") for arm in tool.ARMS}

    contracts = {
        arm: (
            graph["8"]["inputs"]["prompt"],
            graph["8"]["inputs"]["width"],
            graph["8"]["inputs"]["height"],
            graph["8"]["inputs"]["length"],
            graph["9"]["inputs"]["steps"],
            graph["9"]["inputs"]["shift_video"],
            graph["9"]["inputs"]["shift_audio"],
            graph["10"]["inputs"]["noise_seed"],
        )
        for arm, graph in graphs.items()
    }
    assert len(set(contracts.values())) == 1
    assert next(iter(contracts.values())) == (
        tool.CLEAR_PROMPT,
        256,
        256,
        22,
        8,
        12.0,
        3.0,
        2608241001,
    )
    assert graphs["clipproj_4b"]["5"]["inputs"]["encoder_family"] == "4B"
    assert graphs["clipproj_8b"]["5"]["inputs"]["encoder_family"] == "8B"
    assert "4" not in graphs["native_32b"] and "5" not in graphs["native_32b"]
    assert graphs["native_32b"]["8"]["inputs"]["clip"] == ["3", 0]


def test_default_resource_gates_are_arm_specific_and_conservative():
    tool = _load_tool()
    assert tool.parse_args(["--arm", "clipproj_4b"]).min_free_vram_mib == 12_500
    assert tool.parse_args(["--arm", "clipproj_8b"]).min_free_vram_mib == 13_000
    assert tool.parse_args(["--arm", "native_32b"]).min_free_vram_mib == 14_500
    basis = tool.ARM_ASSETS["native_32b"]["free_vram_gate_basis"]
    assert basis["observed_incremental_used_mib"] == pytest.approx(
        basis["observed_peak_used_mib"] - basis["observed_baseline_used_mib"]
    )
    assert basis["unrounded_required_free_mib"] == pytest.approx(
        basis["observed_incremental_used_mib"]
        + basis["required_remaining_headroom_mib"]
    )
    assert basis["enforced_rounded_floor_mib"] == 14_500


def test_resource_floor_cannot_be_lowered_from_the_cli():
    tool = _load_tool()
    with pytest.raises(SystemExit):
        tool.parse_args(
            ["--arm", "native_32b", "--min-free-vram-mib", "13200"]
        )


def test_preflight_refuses_active_user_service_without_starting(monkeypatch, tmp_path):
    tool = _load_tool()
    arm = "clipproj_4b"
    monkeypatch.setitem(tool.ARM_ASSETS[arm], "clip_bytes", 1)
    monkeypatch.setitem(tool.ARM_ASSETS[arm], "projection_bytes", 1)
    comfy, python = _fake_tree(tool, tmp_path, arm)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "used_mib": 1, "free_mib": 16_000},
    )
    monkeypatch.setattr(
        tool.shared,
        "port_is_listening",
        lambda _host, port, **_kwargs: port == 8188,
    )
    args = tool.parse_args(
        [
            "--arm",
            arm,
            "--comfy-root",
            str(comfy),
            "--python",
            str(python),
            "--ffmpeg",
            sys.executable,
            "--ffprobe",
            sys.executable,
        ]
    )

    report = tool.preflight(args)

    assert report["status"] == "ABSTAIN_USER_SERVICE_8188_ACTIVE"
    assert report["ready_for_real_run"] is False
    assert report["checks"]["user_service_8188_inactive"] is False


def test_preflight_enforces_free_vram_and_asset_size(monkeypatch, tmp_path):
    tool = _load_tool()
    arm = "clipproj_8b"
    monkeypatch.setitem(tool.ARM_ASSETS[arm], "clip_bytes", 1)
    monkeypatch.setitem(tool.ARM_ASSETS[arm], "projection_bytes", 1)
    comfy, python = _fake_tree(tool, tmp_path, arm)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "used_mib": 4_000, "free_mib": 12_380},
    )
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    args = tool.parse_args(
        [
            "--arm",
            arm,
            "--comfy-root",
            str(comfy),
            "--python",
            str(python),
            "--ffmpeg",
            sys.executable,
            "--ffprobe",
            sys.executable,
        ]
    )
    report = tool.preflight(args)
    assert report["status"] == "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    assert report["checks"]["reviewed_asset_sizes_match"] is True

    tool._model_paths(comfy, arm)["projection"].write_bytes(b"wrong")
    report = tool.preflight(args)
    assert report["status"] == "ABSTAIN_ASSET_SIZE_MISMATCH"
