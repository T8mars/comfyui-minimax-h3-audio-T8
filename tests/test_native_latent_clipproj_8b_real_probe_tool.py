from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_native_latent_clipproj_8b_real_probe.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "run_native_latent_clipproj_8b_real_probe", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_required_tree(tool, root: Path) -> tuple[Path, Path]:
    comfy_root = root / "ComfyUI"
    python = root / "python.exe"
    for path in (comfy_root / "main.py", python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for path in (
        comfy_root / "custom_nodes" / "minimax-h3-audio-T8",
        comfy_root / "custom_nodes" / "ComfyUI-ClipProj",
        comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for role, path in tool._model_paths(comfy_root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        size = int(tool.EXPECTED_ASSETS[role]["bytes"]) if role in tool.EXPECTED_ASSETS else 5
        path.write_bytes(b"x" * size)
    return comfy_root, python


def test_prompt_preserves_two_segment_eight_nfe_contract_and_decodes_once():
    tool = _load_tool()
    prompt = tool.build_prompt(run_id="unit-test")

    assert prompt["3"]["inputs"] == {
        "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
        "type": "boogu",
        "device": "default",
    }
    assert prompt["5"]["inputs"]["encoder_family"] == "8B"
    assert prompt["8"]["inputs"]["clip"] == ["5", 0]
    assert prompt["18"]["inputs"]["clip"] == ["5", 0]
    assert prompt["9"]["inputs"]["steps"] == 8
    assert prompt["19"]["inputs"]["steps"] == 8
    assert prompt["9"]["inputs"]["shift_video"] == 12.0
    assert prompt["19"]["inputs"]["shift_audio"] == 3.0
    assert prompt["10"]["inputs"]["noise_seed"] == 2608229101
    assert prompt["20"]["inputs"]["noise_seed"] == 2608229102
    assert prompt["23"]["class_type"] == "MiniMaxH3NativeLatentTimelineConcatT8Advanced"
    assert prompt["23"]["inputs"]["output_device"] == "cpu"
    assert sum(node["class_type"] == "MiniMaxH3AVDecodeT8" for node in prompt.values()) == 1
    assert sum(node["class_type"] == "VHS_VideoCombine" for node in prompt.values()) == 1


def test_preflight_refuses_while_user_comfyui_8188_is_active(monkeypatch, tmp_path):
    tool = _load_tool()
    monkeypatch.setattr(
        tool,
        "EXPECTED_ASSETS",
        {
            "clip": {"bytes": 4, "sha256": "A" * 64, "identity": "clip"},
            "projection": {"bytes": 4, "sha256": "B" * 64, "identity": "projection"},
        },
    )
    comfy_root, python = _write_required_tree(tool, tmp_path)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_380, "used_mib": 1, "free_mib": 16_379},
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
    assert report["checks"]["target_port_free"] is True


def test_media_contract_requires_exact_combined_video_and_audio_lengths():
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
        "decoded_audio": {"bytes": 51_200 * 2 * 4},
    }

    assert all(tool.media_contract_checks(report).values())
    report["decoded_audio"]["bytes"] -= 8
    assert (
        tool.media_contract_checks(report)[
            "decoded_aac_audio_matches_32b_reference_51200_samples"
        ]
        is False
    )


def test_hash_mismatch_never_constructs_private_server(monkeypatch, tmp_path):
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
                "identity": "clip",
            },
            "projection": {
                "bytes": len(projection_bytes),
                "sha256": hashlib.sha256(projection_bytes).hexdigest().upper(),
                "identity": "projection",
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
            raise AssertionError("private server must not start after hash mismatch")

    monkeypatch.setattr(tool.shared, "IsolatedServer", RefuseServer)
    result = tool.run_real_probe(args, {"status": "READY"})

    assert result["status"] == "ABSTAIN_ASSET_HASH_MISMATCH"
    assert result["process_ids"] == []
    assert result["checks"]["no_isolated_server_started"] is True


def test_confirm_run_never_bypasses_failed_preflight(monkeypatch, tmp_path):
    tool = _load_tool()
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

    assert tool.main(["--artifact-root", str(tmp_path / "artifact"), "--confirm-run"]) == 3
