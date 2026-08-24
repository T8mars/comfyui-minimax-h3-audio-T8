from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_nfe_resume_real_probe.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_nfe_resume_real_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_contract() -> str:
    return json.dumps(
        {
            "base": "unit-test-base",
            "lora": "unit-test-lora",
            "sampler": "dual_clock_euler/native_flow",
        },
        separators=(",", ":"),
    )


def test_build_control_prompt_is_same_math_without_checkpoint_io():
    tool = _load_tool()
    prompt = tool.build_prompt(
        mode="disabled",
        run_id="unit-test",
        checkpoint_path="unit-test/state.h3nfe.safetensors",
        model_contract_id=_fake_contract(),
    )

    sampler_setup = prompt["9"]["inputs"]
    assert prompt["9"]["class_type"] == "MiniMaxH3NFEResumeSamplerT8Advanced"
    assert sampler_setup["mode"] == "disabled"
    assert sampler_setup["steps"] == 4
    assert sampler_setup["shift_video"] == 12.0
    assert sampler_setup["shift_audio"] == 3.0
    assert sampler_setup["confirm_checkpoint_write"] is False
    assert sampler_setup["model_contract_id"] == ""
    assert sampler_setup["run_contract_json"] == "{}"
    assert prompt["15"]["inputs"]["confirm_save"] is True


def test_build_checkpoint_and_resume_prompts_lock_same_contract():
    tool = _load_tool()
    contract = _fake_contract()
    checkpoint_path = "unit-test/state.h3nfe.safetensors"
    checkpoint = tool.build_prompt(
        mode="checkpoint_each_step",
        run_id="unit-test",
        checkpoint_path=checkpoint_path,
        model_contract_id=contract,
    )
    resume = tool.build_prompt(
        mode="resume",
        run_id="unit-test",
        checkpoint_path=checkpoint_path,
        model_contract_id=contract,
    )

    for prompt in (checkpoint, resume):
        node = prompt["9"]["inputs"]
        assert node["checkpoint_path"] == checkpoint_path
        assert node["model_contract_id"] == contract
        assert node["run_contract_json"] == ["16", 0]
        assert prompt["16"] == {
            "inputs": {
                "positive": ["8", 0],
                "conditioned_prompt": ["8", 3],
                "media_map_json": ["8", 4],
                "conditioning_report": ["8", 5],
                "hash_chunk_megabytes": 4,
            },
            "class_type": "MiniMaxH3NFERunContractT8Advanced",
        }
        assert prompt["10"]["inputs"]["noise_seed"] == 2608228001
        assert prompt["12"]["inputs"]["sampler"] == ["9", 1]
        assert prompt["12"]["inputs"]["sigmas"] == ["9", 2]

    assert checkpoint["9"]["inputs"]["confirm_checkpoint_write"] is True
    assert checkpoint["15"]["inputs"]["confirm_save"] is False
    assert resume["9"]["inputs"]["confirm_checkpoint_write"] is False
    assert resume["15"]["inputs"]["confirm_save"] is True


def test_model_contract_uses_names_sizes_and_optional_base_sidecar(tmp_path):
    tool = _load_tool()
    paths = {}
    for key in ("base", "lora", "clip", "projection", "video_vae", "audio_vae"):
        path = tmp_path / f"{key}.safetensors"
        path.write_bytes(key.encode("utf-8"))
        paths[key] = path
    paths["base"].with_suffix(".sha256").write_text("A" * 64, encoding="utf-8")

    payload = json.loads(tool.model_contract_json(paths))

    assert payload["base"]["sha256"] == "A" * 64
    assert payload["lora"]["strength"] == 1.0
    assert payload["weight_hash_scope"] == "sidecar_or_identity_only"
    assert payload["steps"] == 4
    assert payload["attention"] == "stock"
    assert payload["sampler"] == "dual_clock_euler/native_flow"
    assert len(tool.model_contract_json(paths).encode("utf-8")) < 4096


def test_real_probe_full_hash_contract_covers_every_declared_weight(tmp_path):
    tool = _load_tool()
    paths = {}
    for key in ("base", "lora", "clip", "projection", "video_vae", "audio_vae"):
        path = tmp_path / f"{key}.safetensors"
        path.write_bytes((key + "-weight").encode("utf-8"))
        paths[key] = path

    hashes = tool.hash_model_files(paths)
    payload = json.loads(tool.model_contract_json(paths, verified_hashes=hashes))

    assert payload["weight_hash_scope"] == "full_sha256_all_declared_files"
    assert len(tool.model_contract_json(paths, verified_hashes=hashes).encode("utf-8")) < 4096
    assert set(hashes) == set(paths)
    assert all(payload[role]["sha256"] == hashes[role] for role in paths)
    paths["lora"].write_bytes(b"changed-weight")
    changed = tool.hash_model_files(paths)
    assert changed["lora"] != hashes["lora"]
    with pytest.raises(ValueError, match="cover every declared model role"):
        tool.model_contract_json(paths, verified_hashes={"base": hashes["base"]})


def test_isolated_start_gate_rechecks_port_and_vram(monkeypatch, tmp_path):
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--comfy-root",
            str(tmp_path / "ComfyUI"),
            "--python",
            str(tmp_path / "python.exe"),
            "--min-free-vram-mib",
            "12000",
        ]
    )
    monkeypatch.setattr(
        tool,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16384, "used_mib": 5000, "free_mib": 11000},
    )
    monkeypatch.setattr(tool, "port_is_listening", lambda *_args, **_kwargs: False)

    gate = tool.isolated_start_gate(args)

    assert gate["ready"] is False
    assert gate["checks"]["target_port_free"] is True
    assert gate["checks"]["free_vram_gate"] is False


def test_latent_and_media_comparators_require_exact_streams():
    tool = _load_tool()
    latent_a = {
        "tensors": {
            "samples_video": {"shape": [1, 2], "dtype": "torch.float16", "sha256": "A"},
            "samples_audio": {"shape": [1, 3], "dtype": "torch.float32", "sha256": "B"},
        }
    }
    latent_b = json.loads(json.dumps(latent_a))
    assert tool.compare_latent_reports(latent_a, latent_b)["all_tensors_exact"] is True
    latent_b["tensors"]["samples_audio"]["sha256"] = "C"
    assert tool.compare_latent_reports(latent_a, latent_b)["all_tensors_exact"] is False

    media_a = {
        "strict_decode_passed": True,
        "probe": {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
        "decoded_video": {"bytes": 99, "sha256": "D"},
        "decoded_audio": {"bytes": 55, "sha256": "E"},
    }
    media_b = json.loads(json.dumps(media_a))
    assert all(tool.compare_media_reports(media_a, media_b).values())
    media_b["decoded_audio"]["sha256"] = "F"
    assert tool.compare_media_reports(media_a, media_b)["decoded_audio_exact"] is False


def test_preflight_abstains_before_start_when_vram_is_busy(monkeypatch, tmp_path):
    tool = _load_tool()
    comfy_root = tmp_path / "ComfyUI"
    python = tmp_path / "python.exe"
    required = [
        comfy_root / "main.py",
        python,
        comfy_root / "custom_nodes" / "minimax-h3-audio-T8",
        comfy_root / "custom_nodes" / "ComfyUI-ClipProj",
        comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    ]
    for path in required:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
    for path in tool._model_paths(comfy_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

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
    monkeypatch.setattr(
        tool,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16384, "used_mib": 15000, "free_mib": 1000},
    )
    monkeypatch.setattr(tool, "port_is_listening", lambda _host, port, **_kwargs: port == 8188)

    report = tool.preflight(args)

    assert report["ready_for_real_run"] is False
    assert report["checks"]["free_vram_gate"] is False
    assert report["user_service_8188_observed_only"] is True
    assert report["target"]["already_listening"] is False
    assert report["status"] == "ABSTAIN_RESOURCE_BUSY"


def test_isolated_server_command_uses_private_in_memory_database(tmp_path):
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--comfy-root",
            str(tmp_path / "ComfyUI"),
            "--python",
            str(tmp_path / "python.exe"),
        ]
    )

    command = tool._server_command(args, tmp_path / "run")

    database_index = command.index("--database-url")
    assert command[database_index + 1] == "sqlite:///:memory:"
    assert "--disable-all-custom-nodes" in command
    assert "--reserve-vram" in command
