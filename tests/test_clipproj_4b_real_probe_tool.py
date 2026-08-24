from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_clipproj_4b_real_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_clipproj_4b_real_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_required_tree(tool, root: Path) -> tuple[Path, Path]:
    comfy_root = root / "ComfyUI"
    python = root / "python.exe"
    file_paths = [comfy_root / "main.py", python]
    directory_paths = [
        comfy_root / "custom_nodes" / "minimax-h3-audio-T8",
        comfy_root / "custom_nodes" / "ComfyUI-ClipProj",
        comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    ]
    for path in file_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for path in directory_paths:
        path.mkdir(parents=True, exist_ok=True)
    for role, path in tool._model_paths(comfy_root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        size = int(tool.EXPECTED_ASSETS[role]["bytes"]) if role in tool.EXPECTED_ASSETS else 5
        path.write_bytes(b"x" * size)
    return comfy_root, python


def test_prompt_uses_reviewed_4b_assets_and_fixed_low_load_contract():
    tool = _load_tool()
    prompt = tool.build_prompt(run_id="unit-test")

    assert prompt["3"]["inputs"] == {
        "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
        "type": "krea2",
        "device": "default",
    }
    assert prompt["4"]["inputs"]["projection"] == "mmh3-4b-ClipProj-v3.1.safetensors"
    assert prompt["5"]["inputs"]["encoder_family"] == "4B"
    assert prompt["5"]["inputs"]["projection_path"] == tool.PROJECTION_NAME
    assert prompt["8"]["inputs"]["width"] == 256
    assert prompt["8"]["inputs"]["height"] == 256
    assert prompt["8"]["inputs"]["length"] == 22
    assert prompt["8"]["inputs"]["task_type"] == "T2VA"
    assert prompt["9"]["class_type"] == "MiniMaxH3DualClockSamplerT8"
    assert prompt["9"]["inputs"]["steps"] == 4
    assert prompt["9"]["inputs"]["shift_video"] == 12.0
    assert prompt["9"]["inputs"]["shift_audio"] == 3.0
    assert prompt["10"]["inputs"]["noise_seed"] == 123456789
    assert prompt["14"]["inputs"]["save_metadata"] is False


def test_prompt_and_cli_accept_an_explicit_comparison_seed():
    tool = _load_tool()

    prompt = tool.build_prompt(run_id="comparison", seed=2608228001)
    args = tool.parse_args(["--seed", "2608228001"])

    assert prompt["10"]["inputs"]["noise_seed"] == 2608228001
    assert args.seed == 2608228001


def test_preflight_abstains_without_starting_when_user_gpu_is_busy(monkeypatch, tmp_path):
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "EXPECTED_ASSETS",
        {
            "clip": {"bytes": 4, "sha256": "A" * 64, "revision": "clip"},
            "projection": {"bytes": 4, "sha256": "B" * 64, "revision": "projection"},
        },
    )
    comfy_root, python = _write_required_tree(tool, tmp_path)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {
            "available": True,
            "total_mib": 16_380,
            "used_mib": 13_500,
            "free_mib": 2_880,
        },
    )
    monkeypatch.setattr(
        tool.shared,
        "port_is_listening",
        lambda _host, port, **_kwargs: port == 8188,
    )
    args = tool.parse_args(
        [
            "--comfy-root",
            str(comfy_root),
            "--python",
            str(python),
            "--artifact-root",
            str(tmp_path / "artifact"),
            "--ffmpeg",
            sys.executable,
            "--ffprobe",
            sys.executable,
        ]
    )

    report = tool.preflight(args)

    assert report["status"] == "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    assert report["ready_for_real_run"] is False
    assert report["checks"]["reviewed_asset_sizes_match"] is True
    assert report["checks"]["free_vram_gate"] is False
    assert report["target"]["already_listening"] is False
    assert report["user_service_8188_observed_only"] is True


def test_preflight_rejects_wrong_reviewed_asset_size_before_gpu_use(monkeypatch, tmp_path):
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "EXPECTED_ASSETS",
        {
            "clip": {"bytes": 4, "sha256": "A" * 64, "revision": "clip"},
            "projection": {"bytes": 4, "sha256": "B" * 64, "revision": "projection"},
        },
    )
    comfy_root, python = _write_required_tree(tool, tmp_path)
    tool._model_paths(comfy_root)["projection"].write_bytes(b"wrong")
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_380, "used_mib": 1, "free_mib": 16_379},
    )
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    args = tool.parse_args(
        [
            "--comfy-root",
            str(comfy_root),
            "--python",
            str(python),
            "--ffmpeg",
            sys.executable,
            "--ffprobe",
            sys.executable,
        ]
    )

    report = tool.preflight(args)

    assert report["status"] == "ABSTAIN_ASSET_SIZE_MISMATCH"
    assert report["ready_for_real_run"] is False
    assert report["reviewed_asset_size_checks"]["projection"] is False


def test_media_contract_requires_exact_frame_geometry_and_audio_stream():
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
        "decoded_video": {"bytes": 22 * 256 * 256 * 3, "sha256": "A"},
        "decoded_audio": {"bytes": 1024, "sha256": "B"},
    }

    checks = tool.media_contract_checks(report)

    assert all(checks.values())
    report["decoded_video"]["bytes"] -= 3
    assert tool.media_contract_checks(report)["decoded_video_exactly_22_frames"] is False


def test_hash_mismatch_never_starts_isolated_server(monkeypatch, tmp_path):
    tool = _load_tool()
    clip_bytes = b"clip"
    projection_bytes = b"projection"
    monkeypatch.setattr(
        tool,
        "EXPECTED_ASSETS",
        {
            "clip": {
                "bytes": len(clip_bytes),
                "sha256": hashlib.sha256(b"different").hexdigest().upper(),
                "revision": "clip",
            },
            "projection": {
                "bytes": len(projection_bytes),
                "sha256": hashlib.sha256(projection_bytes).hexdigest().upper(),
                "revision": "projection",
            },
        },
    )
    comfy_root, python = _write_required_tree(tool, tmp_path)
    paths = tool._model_paths(comfy_root)
    paths["clip"].write_bytes(clip_bytes)
    paths["projection"].write_bytes(projection_bytes)
    args = tool.parse_args(
        [
            "--comfy-root",
            str(comfy_root),
            "--python",
            str(python),
            "--artifact-root",
            str(tmp_path / "artifact"),
        ]
    )

    class RefuseServer:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("server must not be constructed after an asset hash mismatch")

    monkeypatch.setattr(tool.shared, "IsolatedServer", RefuseServer)
    result = tool.run_real_probe(args, {"status": "READY"})

    assert result["status"] == "ABSTAIN_ASSET_HASH_MISMATCH"
    assert result["process_ids"] == []
    assert result["checks"]["no_isolated_server_started"] is True
    assert json.loads((Path(result["run_root"]) / "validation_report.json").read_text())["status"] \
        == "ABSTAIN_ASSET_HASH_MISMATCH"


def test_confirm_run_still_refuses_when_preflight_is_not_ready(monkeypatch, tmp_path):
    tool = _load_tool()
    args = [
        "--artifact-root",
        str(tmp_path / "artifact"),
        "--confirm-run",
    ]
    monkeypatch.setattr(
        tool,
        "preflight",
        lambda _args: {
            "status": "ABSTAIN_INSUFFICIENT_FREE_VRAM",
            "ready_for_real_run": False,
        },
    )
    monkeypatch.setattr(
        tool,
        "run_real_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("real run must not start after a failed preflight")
        ),
    )

    assert tool.main(args) == 3
