from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import wave

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_speech_multilingual_formal_batch.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("run_speech_multilingual_formal_batch", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _prompt(case_id: str, output_prefix: str) -> dict:
    return {
        "1": {
            "inputs": {"unet_name": "base.safetensors", "weight_dtype": "default"},
            "class_type": "UNETLoader",
        },
        "2": {
            "inputs": {"clip_name": "clip.safetensors", "type": "minimax"},
            "class_type": "CLIPLoader",
        },
        "3": {
            "inputs": {"vae_name": "video.safetensors"},
            "class_type": "VAELoader",
        },
        "4": {
            "inputs": {"vae_name": "audio.safetensors"},
            "class_type": "VAELoader",
        },
        "5": {
            "inputs": {
                "seed": 1,
                "release_policy": "unload_all_models",
            },
            "class_type": "MiniMaxH3SpeechStudioT8",
        },
        "6": {
            "inputs": {
                "audio": ["5", 0],
                "filename_prefix": output_prefix,
            },
            "class_type": "SaveAudio",
        },
        "_case": case_id,
    }


def _write_plan(tool, root: Path, case_count: int = 2) -> tuple[Path, list[dict]]:
    plan_root = root / "plan"
    prompt_root = plan_root / "api_prompts"
    prompt_root.mkdir(parents=True)
    source = root / "reviewed-source.json"
    source.write_text('{"reviewed":true}\n', encoding="utf-8")
    cases = []
    for index in range(case_count):
        case_id = f"en-u{index + 1:02d}-described-v1-s{index + 1}"
        output_prefix = f"MiniMaxH3_T8_Speech/formal-test/{case_id}"
        prompt = _prompt(case_id, output_prefix)
        relative = f"api_prompts/{case_id}.json"
        prompt_path = plan_root / PurePosixPath(relative)
        prompt_path.write_bytes(tool.matrix._json_bytes(prompt))
        cases.append(
            {
                "case_id": case_id,
                "language_code": "en",
                "generation_mode": "described",
                "utterance_id": f"u{index + 1:02d}",
                "seed": index + 1,
                "speaker_id": "",
                "voice_profile_id": "v1",
                "expected_text": f"Sentence {index + 1}.",
                "prompt_path": relative,
                "prompt_sha256": hashlib.sha256(tool.matrix._json_bytes(prompt)).hexdigest().upper(),
                "output_prefix": output_prefix,
                "reference_audio": None,
                "status": "PENDING_NOT_RUN",
            }
        )
    plan = {
        "schema": tool.matrix.SCHEMA,
        "plan_id": "unit",
        "case_count": len(cases),
        "source_files": {
            "reviewed": {"path": str(source.resolve()), "sha256": _sha(source)}
        },
        "cases": cases,
    }
    (plan_root / "plan.json").write_bytes(tool.matrix._json_bytes(plan))
    return plan_root, cases


def _write_comfy_tree(root: Path) -> tuple[Path, Path]:
    comfy = root / "ComfyUI"
    python = root / "python.exe"
    (comfy / "custom_nodes" / "minimax-h3-audio-T8").mkdir(parents=True)
    (comfy / "input").mkdir(parents=True)
    (comfy / "main.py").write_text("", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    for folder, name in (
        ("diffusion_models", "base.safetensors"),
        ("text_encoders", "clip.safetensors"),
        ("vae", "video.safetensors"),
        ("vae", "audio.safetensors"),
    ):
        path = comfy / "models" / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
    return comfy, python


def _args(tool, plan_root: Path, comfy: Path, python: Path, *extra: str):
    return tool.parse_args(
        [
            "--plan-root",
            str(plan_root),
            "--comfy-root",
            str(comfy),
            "--python",
            str(python),
            *extra,
        ]
    )


def _write_audio(path: Path, identity: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8_000
    sample_count = int(sample_rate * 2.1)
    first = int(identity).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(first + b"\x00\x00" * (sample_count - 1))


def test_defaults_are_preflight_only_private_and_bounded():
    tool = _load_tool()
    args = tool.parse_args([])

    assert args.confirm_run is False
    assert args.host == "127.0.0.1"
    assert args.port == 8197
    assert args.port != 8188
    assert args.max_cases == 1
    assert tool.MAX_CASES_PER_INVOCATION == 6


def test_plan_loader_rejects_prompt_drift_and_missing_unload(tmp_path):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 1)
    prompt_path = plan_root / cases[0]["prompt_path"]
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompt["5"]["inputs"]["release_policy"] = "keep_models"
    prompt_path.write_bytes(tool.matrix._json_bytes(prompt))

    with pytest.raises(ValueError, match="prompt SHA-256 drift"):
        tool.load_plan(plan_root)

    cases[0]["prompt_sha256"] = hashlib.sha256(tool.matrix._json_bytes(prompt)).hexdigest().upper()
    plan = json.loads((plan_root / "plan.json").read_text(encoding="utf-8"))
    plan["cases"] = cases
    (plan_root / "plan.json").write_bytes(tool.matrix._json_bytes(plan))
    with pytest.raises(ValueError, match="unload_all_models"):
        tool.load_plan(plan_root)


def test_preflight_rejects_8188_before_gpu_or_model_work(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, _ = _write_plan(tool, tmp_path, 1)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: (_ for _ in ()).throw(AssertionError("GPU query must not run")),
    )
    args = _args(tool, plan_root, comfy, python, "--port", "8188")

    report = tool.preflight(args)

    assert report["status"] == "ABSTAIN_INVALID_CONFIGURATION"
    assert report["real_run_started"] is False


def test_preflight_is_ready_only_with_hashes_files_port_and_gpu(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 2)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )
    args = _args(tool, plan_root, comfy, python)

    report = tool.preflight(args)

    assert report["status"] == "READY"
    assert report["ready_for_real_run"] is True
    assert report["selected_case_ids"] == [cases[0]["case_id"]]
    assert all(report["checks"].values())


def test_main_dry_run_never_executes_and_confirm_refuses_failed_gate(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root = tmp_path / "plan"
    monkeypatch.setattr(
        tool,
        "preflight",
        lambda _args: {
            "status": "ABSTAIN_INSUFFICIENT_FREE_VRAM",
            "ready_for_real_run": False,
            "selected_case_ids": [],
        },
    )
    monkeypatch.setattr(
        tool,
        "run_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("run must not start")),
    )

    assert tool.main(["--plan-root", str(plan_root)]) == 0
    assert tool.main(["--plan-root", str(plan_root), "--confirm-run"]) == 3


def test_serial_execution_collects_once_and_resumes_without_duplicates(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 2)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )

    class FakeServer:
        def __init__(self, _args, _session_root, _output_root):
            self.stopped = False

        def start(self):
            return 321

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(tool, "IsolatedSpeechServer", FakeServer)
    call_count = 0

    async def fake_submit(*, server, prompt, timeout_seconds):
        nonlocal call_count
        assert server.endswith(":8197")
        assert timeout_seconds > 0
        call_count += 1
        save = tool._find_one(prompt, "SaveAudio")
        prefix = PurePosixPath(save["inputs"]["filename_prefix"])
        output = plan_root / "comfy_output"
        path = output.joinpath(*prefix.parent.parts) / f"{prefix.name}_00001_.wav"
        _write_audio(path, call_count)
        return {
            "prompt_id": f"prompt-{call_count}",
            "terminal": {"type": "execution_success"},
            "elapsed_seconds": 1.0,
        }

    monkeypatch.setattr(tool.shared, "submit_prompt", fake_submit)
    args = _args(tool, plan_root, comfy, python, "--confirm-run")

    first_preflight = tool.preflight(args)
    first = tool.run_batch(args, first_preflight)
    second_preflight = tool.preflight(args)
    second = tool.run_batch(args, second_preflight)

    assert first["status"] == "PASS_PARTIAL_COLLECTION_PENDING_MORE_CASES"
    assert first["collection"]["collected_unique_case_count"] == 1
    assert second["status"] == "PASS_COMPLETE_COLLECTION_PENDING_EVALUATION"
    assert second["collection"]["collected_unique_case_count"] == 2
    assert call_count == 2
    assert (plan_root / "multilingual_manifest.json").is_file()
    assert not (plan_root / "execution.lock").exists()
    state = json.loads((plan_root / "execution_state.json").read_text(encoding="utf-8"))
    assert len(state["attempts"][cases[0]["case_id"]]) == 1
    assert len(state["attempts"][cases[1]["case_id"]]) == 1


def test_execution_failure_stops_before_next_case_and_keeps_recoverable_state(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 2)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )

    class FakeServer:
        def __init__(self, _args, _session_root, _output_root):
            pass

        def start(self):
            return 654

        def stop(self):
            pass

    monkeypatch.setattr(tool, "IsolatedSpeechServer", FakeServer)
    calls = 0

    async def fake_submit(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "prompt_id": "failed-prompt",
            "terminal": {"type": "execution_error"},
            "elapsed_seconds": 0.5,
        }

    monkeypatch.setattr(tool.shared, "submit_prompt", fake_submit)
    args = _args(tool, plan_root, comfy, python, "--confirm-run", "--max-cases", "2")
    report = tool.run_batch(args, tool.preflight(args))

    assert report["status"] == "FAIL_EXECUTION_OR_OUTPUT_CONTRACT"
    assert report["passed"] is False
    assert calls == 1
    assert len(report["results"]) == 1
    assert not (plan_root / "execution.lock").exists()
    state = json.loads((plan_root / "execution_state.json").read_text(encoding="utf-8"))
    assert state["attempts"][cases[0]["case_id"]][0]["status"] \
        == "FAILED_TERMINAL_OR_OUTPUT_CONTRACT"
    assert cases[1]["case_id"] not in state["attempts"]


def test_submission_exception_stops_owned_server_releases_lock_and_records_state(
    monkeypatch, tmp_path
):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 1)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )
    server_instances = []

    class FakeServer:
        def __init__(self, _args, _session_root, _output_root):
            self.stopped = False
            server_instances.append(self)

        def start(self):
            return 777

        def stop(self):
            self.stopped = True

    async def fake_submit(**_kwargs):
        raise TimeoutError("synthetic bounded timeout")

    monkeypatch.setattr(tool, "IsolatedSpeechServer", FakeServer)
    monkeypatch.setattr(tool.shared, "submit_prompt", fake_submit)
    args = _args(tool, plan_root, comfy, python, "--confirm-run")
    report = tool.run_batch(args, tool.preflight(args))

    assert report["status"] == "FAIL_EXECUTION_OR_OUTPUT_CONTRACT"
    assert report["runtime_error"] == {
        "type": "TimeoutError",
        "message": "synthetic bounded timeout",
    }
    assert len(server_instances) == 1 and server_instances[0].stopped is True
    assert not (plan_root / "execution.lock").exists()
    state = json.loads((plan_root / "execution_state.json").read_text(encoding="utf-8"))
    attempt = state["attempts"][cases[0]["case_id"]][0]
    assert attempt["status"] == "EXECUTION_EXCEPTION"
    assert attempt["error"]["type"] == "TimeoutError"


def test_server_start_failure_is_reported_without_submitting_or_leaking_lock(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, _ = _write_plan(tool, tmp_path, 1)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )
    servers = []

    class FailingServer:
        def __init__(self, _args, _session_root, _output_root):
            self.stopped = False
            servers.append(self)

        def start(self):
            raise RuntimeError("synthetic startup failure")

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(tool, "IsolatedSpeechServer", FailingServer)
    monkeypatch.setattr(
        tool.shared,
        "submit_prompt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("prompt must not submit")),
    )
    args = _args(tool, plan_root, comfy, python, "--confirm-run")
    report = tool.run_batch(args, tool.preflight(args))

    assert report["status"] == "FAIL_EXECUTION_OR_OUTPUT_CONTRACT"
    assert report["real_run_started"] is False
    assert report["runtime_error"] == {
        "type": "RuntimeError",
        "message": "synthetic startup failure",
    }
    assert len(servers) == 1 and servers[0].stopped is True
    assert not (plan_root / "execution.lock").exists()


def test_existing_lock_and_output_conflict_fail_closed(monkeypatch, tmp_path):
    tool = _load_tool()
    plan_root, cases = _write_plan(tool, tmp_path, 1)
    comfy, python = _write_comfy_tree(tmp_path)
    monkeypatch.setattr(tool.shared, "port_is_listening", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        tool.shared,
        "gpu_memory_mib",
        lambda: {"available": True, "total_mib": 16_000, "used_mib": 1_000, "free_mib": 15_000},
    )
    args = _args(tool, plan_root, comfy, python)
    (plan_root / "execution.lock").write_text("locked", encoding="utf-8")
    assert tool.preflight(args)["status"] == "ABSTAIN_EXECUTION_LOCK_PRESENT"
    (plan_root / "execution.lock").unlink()

    prefix = PurePosixPath(cases[0]["output_prefix"])
    parent = plan_root / "comfy_output"
    first = parent.joinpath(*prefix.parent.parts) / f"{prefix.name}_a.wav"
    second = parent.joinpath(*prefix.parent.parts) / f"{prefix.name}_b.wav"
    _write_audio(first, 1)
    _write_audio(second, 2)
    assert tool.preflight(args)["status"] == "ABSTAIN_OUTPUT_CONFLICT"
