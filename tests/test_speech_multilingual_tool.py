from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from h3_audio_t8_pkg.speech_verification import transcript_metrics


ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    name = "h3_audio_t8_multilingual_validation_test_tool"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "validate_speech_multilingual.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, *, utterances=2, seeds=(11, 12)) -> dict:
    cases = []
    index = 0
    for utterance_index in range(utterances):
        for mode in ("described", "clone"):
            for seed in seeds:
                audio_path = tmp_path / f"case_{index}.wav"
                audio_path.write_bytes(f"unique-audio-{index}".encode())
                cases.append(
                    {
                        "case_id": f"zh_{utterance_index}_{mode}_{seed}",
                        "language_code": "zh",
                        "generation_mode": mode,
                        "utterance_id": f"utterance_{utterance_index}",
                        "seed": seed,
                        "speaker_id": "speaker_1" if mode == "clone" else "",
                        "voice_profile_id": "calm_voice" if mode == "described" else "",
                        "audio_path": audio_path.name,
                        "expected_text": f"第{utterance_index + 1}条测试文本",
                    }
                )
                index += 1
    return {"schema": "minimax_h3_t8_multilingual_manifest_v1", "cases": cases}


def _validate(tool, payload, manifest_path):
    return tool.validate_manifest_design(
        payload,
        manifest_path,
        transcript_metrics,
        strict_design=True,
        minimum_samples_per_language=8,
        minimum_utterances_per_language=2,
        minimum_seeds_per_utterance_mode=2,
        required_generation_modes=("described", "clone"),
    )


def test_multilingual_manifest_design_requires_real_mode_seed_and_audio_coverage(tmp_path):
    tool = _load_tool()
    payload = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    cases, report = _validate(tool, payload, manifest_path)
    assert len(cases) == 8
    assert report["design_gate_pass"] is True
    assert report["unique_audio_count"] == 8
    language = report["language_design"]["zh"]
    assert language["generation_modes"] == ["clone", "described"]
    assert language["utterance_count"] == 2
    assert language["insufficient_replicates"] == []


def test_multilingual_manifest_denies_reused_audio_and_missing_replicates(tmp_path):
    tool = _load_tool()
    payload = _manifest(tmp_path)
    first = tmp_path / payload["cases"][0]["audio_path"]
    second = tmp_path / payload["cases"][1]["audio_path"]
    second.write_bytes(first.read_bytes())
    payload["cases"].pop()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    _, report = _validate(tool, payload, manifest_path)
    assert report["design_gate_pass"] is False
    assert any("audio content reused" in item for item in report["global_findings"])
    assert report["language_design"]["zh"]["insufficient_replicates"]


def test_multilingual_manifest_fails_closed_on_ambiguous_fields(tmp_path):
    tool = _load_tool()
    payload = _manifest(tmp_path)
    payload["cases"][0].pop("seed")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="integer seed"):
        _validate(tool, payload, manifest_path)

    payload = _manifest(tmp_path)
    payload["cases"][0]["language_code"] = "auto"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit language_code"):
        _validate(tool, payload, manifest_path)

    payload = _manifest(tmp_path)
    payload["cases"][1]["expected_text"] = "同一utterance不应更换文本"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent expected_text"):
        _validate(tool, payload, manifest_path)

    payload = _manifest(tmp_path)
    payload["cases"][1]["seed"] = payload["cases"][0]["seed"]
    payload["cases"][1]["voice_profile_id"] = payload["cases"][0]["voice_profile_id"]
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate language/utterance/mode/condition/seed"):
        _validate(tool, payload, manifest_path)


def test_validate_only_audits_design_without_asr_import(monkeypatch, tmp_path):
    tool = _load_tool()
    payload = _manifest(tmp_path, utterances=1, seeds=(7,))
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "audit.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_speech_multilingual.py",
            str(manifest_path),
            "--output",
            str(output_path),
            "--validate-only",
            "--minimum-samples-per-language",
            "2",
            "--minimum-utterances-per-language",
            "1",
            "--minimum-seeds-per-utterance-mode",
            "1",
        ],
    )
    assert tool.main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evaluation_executed"] is False
    assert report["experimental_design"]["design_gate_pass"] is True
    assert report["stable_multilingual_gate_pass"] is False


def test_result_breakdowns_expose_mode_condition_and_outliers():
    tool = _load_tool()
    results = [
        {
            "case_id": "described_ok",
            "language_code": "en",
            "generation_mode": "described",
            "condition_id": "warm_voice",
            "utterance_id": "line_1",
            "metrics": {"primary_metric": "WER", "primary_error_rate": 0.0},
        },
        {
            "case_id": "clone_ok",
            "language_code": "en",
            "generation_mode": "clone",
            "condition_id": "speaker_a",
            "utterance_id": "line_1",
            "metrics": {"primary_metric": "WER", "primary_error_rate": 0.1},
        },
        {
            "case_id": "clone_hallucination",
            "language_code": "en",
            "generation_mode": "clone",
            "condition_id": "speaker_b",
            "utterance_id": "line_1",
            "metrics": {"primary_metric": "WER", "primary_error_rate": 2.0},
        },
    ]

    modes, conditions, outliers = tool.summarize_result_breakdowns(results, 0.15)
    clone = modes["en"]["clone"]
    assert clone["sample_count"] == 2
    assert clone["mean_primary_error_rate"] == pytest.approx(1.05)
    assert clone["case_pass_rate_at_threshold"] == 0.5
    assert clone["cases_over_threshold"] == ["clone_hallucination"]
    assert [item["condition_id"] for item in conditions] == [
        "speaker_a",
        "speaker_b",
        "warm_voice",
    ]
    assert outliers == [
        {
            "case_id": "clone_hallucination",
            "language_code": "en",
            "generation_mode": "clone",
            "condition_id": "speaker_b",
            "utterance_id": "line_1",
            "primary_metric": "WER",
            "primary_error_rate": 2.0,
        }
    ]
