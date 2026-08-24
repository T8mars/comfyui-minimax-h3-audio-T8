from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "materialize_voice_clone_abx_standardized.py"
ABX_TOOL_PATH = ROOT / "tools" / "build_voice_clone_abx_review.py"


def _load_path(name: str, path: Path):
    tools_root = str(path.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_tool():
    return _load_path("h3_voice_clone_abx_standardization_test_tool", TOOL_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_wav(
    path: Path,
    *,
    identity: int,
    sample_rate: int = 16_000,
    channels: int = 1,
    seconds: float = 0.22,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(sample_rate * seconds)
    mono = [((index * (identity + 11)) % 5000) - 2500 for index in range(frame_count)]
    if channels == 1:
        samples = array("h", mono)
    else:
        samples = array(
            "h",
            (
                sample
                for value in mono
                for sample in (value, max(-32768, min(32767, value // 2 + identity)))
            ),
        )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _jobs_payload(tmp_path: Path) -> tuple[dict, Path]:
    references = []
    for index in range(2):
        path = tmp_path / "inputs" / f"reference_{index}.wav"
        _write_wav(path, identity=index + 1, sample_rate=16_000, channels=1)
        references.append(path)
    candidates = []
    for index in range(6):
        path = tmp_path / "inputs" / f"candidate_{index}.wav"
        _write_wav(path, identity=100 + index, sample_rate=22_050, channels=2)
        candidates.append(path)

    jobs = []
    for index, candidate in enumerate(candidates):
        target = index // 3
        impostor = 1 - target
        case_id = f"abx-speaker_{target}-vs-speaker_{impostor}-u{index + 1:02d}-s{201 + index % 3}"
        jobs.append(
            {
                "case_id": case_id,
                "target_speaker_id": f"speaker_{target}",
                "impostor_speaker_id": f"speaker_{impostor}",
                "condition_id": f"speaker_{target}",
                "utterance_id": f"u{index + 1:02d}",
                "language_code": "en",
                "seed": 201 + index % 3,
                "seed_known": True,
                "inputs": {
                    "target_reference": {
                        "path": str(references[target]),
                        "sha256": _sha(references[target]),
                    },
                    "impostor_reference": {
                        "path": str(references[impostor]),
                        "sha256": _sha(references[impostor]),
                    },
                    "candidate": {"path": str(candidate), "sha256": _sha(candidate)},
                },
                "outputs": {
                    "target_reference": f"standardized/references/speaker_{target}.flac",
                    "impostor_reference": f"standardized/references/speaker_{impostor}.flac",
                    "candidate": f"standardized/candidates/{case_id}.flac",
                },
            }
        )
    payload = {
        "schema": "minimax_h3_t8_voice_clone_abx_standardization_jobs_v1",
        "plan_id": "identity_unit_v1",
        "generation_plan_sha256": "A" * 64,
        "identity_design_sha256": "B" * 64,
        "review_id": "identity-unit-abx-v1",
        "contract": {
            "sample_rate": 32000,
            "channels": 1,
            "codec": "flac",
            "container": "flac",
            "loudness_normalization": False,
            "duration_policy": "preserve_each_source",
        },
        "job_count": len(jobs),
        "jobs": jobs,
        "execution_started": False,
        "abx_manifest_written": False,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
    }
    jobs_path = tmp_path / "abx_standardization_jobs.json"
    jobs_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, jobs_path


def _require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media-contract tests")


def test_default_invocation_is_preflight_only(tmp_path):
    _require_ffmpeg()
    tool = _load_tool()
    _, jobs_path = _jobs_payload(tmp_path)
    output = tmp_path / "materialized"

    result = tool.main([str(jobs_path), "--output-root", str(output)])

    assert result == 0
    report = json.loads(
        (output / "latest_standardization_preflight.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "READY"
    assert report["ready_for_materialization"] is True
    assert report["registered_abx_cases"] == 6
    assert report["registered_unique_files"] == 8
    assert report["materialization_started"] is False
    assert not (output / "standardized").exists()
    assert not (output / "materialization_state.json").exists()


def test_bounded_resume_finishes_fair_manifest_and_existing_blind_builder_accepts_it(tmp_path):
    _require_ffmpeg()
    tool = _load_tool()
    _, jobs_path = _jobs_payload(tmp_path)
    output = tmp_path / "materialized"

    first = tool.main(
        [
            str(jobs_path),
            "--output-root",
            str(output),
            "--max-files",
            "2",
            "--confirm-run",
        ]
    )
    assert first == 0
    first_result = json.loads(
        (output / "latest_standardization_result.json").read_text(encoding="utf-8")
    )
    assert first_result["status"] == "PARTIAL_PROGRESS"
    assert first_result["processed_file_count"] == 2
    assert first_result["remaining_file_count"] == 6
    assert not (output / "abx_manifest.json").exists()

    second = tool.main(
        [
            str(jobs_path),
            "--output-root",
            str(output),
            "--max-files",
            "25",
            "--confirm-run",
        ]
    )
    assert second == 0
    second_result = json.loads(
        (output / "latest_standardization_result.json").read_text(encoding="utf-8")
    )
    assert second_result["status"] == "COMPLETE"
    assert second_result["processed_file_count"] == 6
    assert second_result["remaining_file_count"] == 0
    assert second_result["abx_case_count"] == 6
    assert len(list((output / "standardized").rglob("*.flac"))) == 8

    manifest_path = output / "abx_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "minimax_h3_t8_voice_clone_abx_manifest_v1"
    assert manifest["target_position_policy"] == "balanced_by_target_and_global"
    assert len(manifest["cases"]) == 6
    assert manifest["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    for path in (output / "standardized").rglob("*.flac"):
        contract = tool._probe_audio(path, Path(shutil.which("ffprobe")))
        tool._assert_output_contract(contract, path.name)

    abx_tool = _load_path("h3_standardized_abx_builder_test_tool", ABX_TOOL_PATH)
    key = abx_tool.build_package(manifest, output, tmp_path / "blind", random_seed=260823)
    assert len(key["cases"]) == 6
    assert key["target_position_policy"] == "balanced_by_target_and_global"
    assert Counter(row["target_code"] for row in key["cases"]) == {"A": 3, "B": 3}
    assert (tmp_path / "blind" / "blind_review.html").is_file()

    final_report = tool.preflight(
        tool.parse_args([str(jobs_path), "--output-root", str(output)])
    )
    assert final_report["status"] == "COMPLETE_ALREADY"
    assert final_report["ready_for_materialization"] is False
    assert not (output / "standardization.lock").exists()


def test_preflight_rejects_input_identity_drift(tmp_path):
    _require_ffmpeg()
    tool = _load_tool()
    payload, jobs_path = _jobs_payload(tmp_path)
    candidate = Path(payload["jobs"][0]["inputs"]["candidate"]["path"])
    _write_wav(candidate, identity=999, sample_rate=22_050, channels=2)

    report = tool.preflight(
        tool.parse_args([str(jobs_path), "--output-root", str(tmp_path / "output")])
    )
    assert report["status"] == "ABSTAIN_INPUT_IDENTITY_DRIFT"
    assert report["materialization_started"] is False


def test_preflight_rejects_untracked_output_and_path_escape(tmp_path):
    _require_ffmpeg()
    tool = _load_tool()
    payload, jobs_path = _jobs_payload(tmp_path)
    output = tmp_path / "materialized"
    untracked = output / "standardized" / "references" / "speaker_0.flac"
    untracked.parent.mkdir(parents=True)
    untracked.write_bytes(b"untracked")

    report = tool.preflight(
        tool.parse_args([str(jobs_path), "--output-root", str(output)])
    )
    assert report["status"] == "ABSTAIN_OUTPUT_CONFLICT"
    assert report["output_conflicts"]

    untracked.unlink()
    payload["jobs"][0]["outputs"]["candidate"] = "../escape.flac"
    jobs_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report = tool.preflight(
        tool.parse_args([str(jobs_path), "--output-root", str(output)])
    )
    assert report["status"] == "ABSTAIN_JOBS_INVALID"


def test_runtime_failure_keeps_atomic_state_and_never_writes_manifest(tmp_path):
    _require_ffmpeg()
    tool = _load_tool()
    payload, jobs_path = _jobs_payload(tmp_path)
    bad = Path(payload["jobs"][0]["inputs"]["candidate"]["path"])
    bad.write_bytes(b"not-decodable-audio")
    payload["jobs"][0]["inputs"]["candidate"]["sha256"] = _sha(bad)
    jobs_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "materialized"

    result = tool.main(
        [
            str(jobs_path),
            "--output-root",
            str(output),
            "--max-files",
            "25",
            "--confirm-run",
        ]
    )
    assert result == 2
    failure = json.loads(
        (output / "latest_standardization_result.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "ABSTAIN_RUNTIME_FAILURE"
    assert failure["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    assert not (output / "abx_manifest.json").exists()
    assert not (output / "standardization.lock").exists()
    assert not list(output.rglob("*.tmp-*.flac"))
