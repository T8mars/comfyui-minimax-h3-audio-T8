from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_audio_refine_smoke.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_audio_refine_smoke", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_required_tree(tool, root: Path) -> tuple[Path, Path, Path]:
    comfy_root = root / "ComfyUI"
    python = root / "python.exe"
    plugin_root = root / "audio-refine"
    for path in (
        comfy_root / "main.py",
        python,
        plugin_root / "__init__.py",
        plugin_root / "nodes_audio_refine_advanced.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (comfy_root / "custom_nodes" / "ComfyUI-ClipProj").mkdir(
        parents=True, exist_ok=True
    )
    for path in tool._model_paths(comfy_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weight")
    return comfy_root, python, plugin_root


def test_prompt_is_one_fixed_turbo4_refine4_dual_clock_job():
    tool = _load_tool()
    prompt = tool.build_prompt(run_id="unit", seed=2608260404)

    samplers = [
        node for node in prompt.values() if node["class_type"] == "SamplerCustomAdvanced"
    ]
    assert tool.MAX_PROMPTS_PER_INVOCATION == 1
    assert len(samplers) == 2
    assert prompt["8"]["inputs"]["width"] == 256
    assert prompt["8"]["inputs"]["height"] == 256
    assert prompt["8"]["inputs"]["length"] == 22
    assert prompt["9"]["inputs"]["steps"] == 4
    assert prompt["9"]["inputs"]["shift_video"] == 12.0
    assert prompt["9"]["inputs"]["shift_audio"] == 3.0
    assert prompt["15"]["class_type"] == "MiniMaxH3AudioRefineAuditT8Advanced"
    assert prompt["15"]["inputs"]["minimum_free_vram_mib"] == 512
    assert prompt["15"]["inputs"]["minimum_commit_headroom_gib"] == 16.0
    assert prompt["16"]["inputs"] == {
        "audit": ["15", 0],
        "refine_steps": 4,
        "audio_denoise": 0.5,
        "refine_seed": 2608260404,
        "model_strategy": "connected_model_explicit",
    }
    assert prompt["17"]["class_type"] == "MiniMaxH3AudioRefineDualClockSetupT8Advanced"
    assert prompt["18"]["inputs"] == {
        "noise": ["17", 1],
        "guider": ["17", 2],
        "sampler": ["17", 3],
        "sigmas": ["17", 4],
        "latent_image": ["17", 5],
    }
    assert prompt["19"]["class_type"] == "MiniMaxH3TwoPassAudioAuditT8Advanced"
    assert prompt["19"]["inputs"]["expected_audio_strength"] == 1.0
    assert prompt["26"]["class_type"] == "MiniMaxH3AudioRefineQualityGateT8Advanced"
    assert prompt["26"]["inputs"]["accept_candidate"] is False
    assert sum(node["class_type"] == "SaveVideo" for node in prompt.values()) == 3


def test_quality_pair_profile_is_fixed_to_one_0p64mp_five_second_prompt():
    tool = _load_tool()
    args = tool.parse_args(["--quality-pair"])
    prompt = tool.build_prompt(
        run_id="quality",
        seed=args.seed,
        width=args.width,
        height=args.height,
        frames=args.frames,
    )

    assert (args.width, args.height, args.frames) == (1056, 608, 124)
    assert prompt["8"]["inputs"]["width"] == 1056
    assert prompt["8"]["inputs"]["height"] == 608
    assert prompt["8"]["inputs"]["length"] == 124


def test_preflight_rejects_active_user_service_and_low_commit(monkeypatch, tmp_path):
    tool = _load_tool()
    comfy_root, python, plugin_root = _write_required_tree(tool, tmp_path)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16380, "used_mib": 1000, "free_mib": 15380},
    )
    monkeypatch.setattr(
        tool.shared,
        "port_is_listening",
        lambda _host, port, **_kwargs: port == 8188,
    )
    monkeypatch.setattr(
        tool,
        "host_memory_snapshot",
        lambda: {"commit_headroom_gib": 64.0, "ram_available_gib": 64.0},
    )
    args = tool.parse_args(
        [
            "--comfy-root",
            str(comfy_root),
            "--python",
            str(python),
            "--plugin-root",
            str(plugin_root),
            "--ffmpeg",
            sys.executable,
            "--ffprobe",
            sys.executable,
        ]
    )

    report = tool.preflight(args)

    assert report["status"] == "ABSTAIN_USER_SERVICE_ACTIVE"
    assert report["ready_for_real_run"] is False
    assert report["checks"]["user_service_8188_inactive"] is False
    assert report["checks"]["target_port_free"] is True

    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool,
        "host_memory_snapshot",
        lambda: {"commit_headroom_gib": 15.9, "ram_available_gib": 64.0},
    )
    report = tool.preflight(args)
    assert report["status"] == "ABSTAIN_INSUFFICIENT_HOST_COMMIT"
    assert report["checks"]["commit_headroom_at_least_16_gib"] is False


def test_default_invocation_is_dry_run_and_never_calls_real_runner(
    monkeypatch, tmp_path
):
    tool = _load_tool()
    called = []
    monkeypatch.setattr(
        tool,
        "preflight",
        lambda _args: {"status": "READY", "ready_for_real_run": True},
    )
    monkeypatch.setattr(tool, "run_real_probe", lambda *_args: called.append(True))

    result = tool.main(["--artifact-root", str(tmp_path / "artifact")])

    assert result == 0
    assert called == []


def test_isolated_command_uses_private_paths_and_only_one_feature_worktree(tmp_path):
    tool = _load_tool()
    comfy_root, python, plugin_root = _write_required_tree(tool, tmp_path)
    args = tool.parse_args(
        [
            "--comfy-root",
            str(comfy_root),
            "--python",
            str(python),
            "--plugin-root",
            str(plugin_root),
        ]
    )
    run_root = tmp_path / "run"
    command = tool._server_command(args, run_root)

    assert "--disable-all-custom-nodes" in command
    whitelist = command.index("--whitelist-custom-nodes")
    assert command[whitelist + 1 : whitelist + 3] == [
        plugin_root.name,
        "ComfyUI-ClipProj",
    ]
    assert "--cache-none" in command
    assert str(run_root / "output") in command
    assert str(run_root / "user") in command


def test_mechanical_checks_use_completed_execution_events_when_history_is_disabled():
    tool = _load_tool()
    phase = {
        "prompt_id": "prompt",
        "terminal": {"type": "execution_success"},
        "history": None,
        "events": [
            {"type": "executing", "node": "19"},
            {"type": "executing", "node": "24"},
            {"type": "executing", "node": "25"},
            {"type": "executing", "node": "26"},
            {"type": "execution_success", "node": None},
        ],
    }
    media = {
        "strict_decode": True,
        "video_256x256": True,
        "video_22_frames": True,
        "audio_32khz_stereo": True,
        "audio_nonempty": True,
    }

    checks = tool._mechanical_checks(phase, media, media, media)

    assert all(checks.values())
