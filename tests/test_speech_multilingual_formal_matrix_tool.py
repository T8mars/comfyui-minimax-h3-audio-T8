from __future__ import annotations

from array import array
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "build_speech_multilingual_formal_matrix.py"
DESCRIBED_TEMPLATE = ROOT / "tests" / "fixtures" / "api" / "speech_described_api.json"
CLONE_TEMPLATE = ROOT / "tests" / "fixtures" / "api" / "speech_reference_clone_api.json"


def _load_tool():
    name = "h3_audio_t8_speech_multilingual_formal_matrix_test_tool"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, *, identity: int, seconds: float = 2.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(16_000 * seconds)
    samples = array(
        "h",
        (((index * (identity + 3)) % 2000) - 1000 for index in range(frame_count)),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _fixture_payloads(
    tmp_path: Path,
    *,
    utterance_count: int = 10,
    speaker_count: int = 10,
) -> tuple[dict, dict, Path, Path]:
    spec = {
        "schema": "minimax_h3_t8_multilingual_speech_formal_spec_v1",
        "plan_id": "formal_unit_v1",
        "seeds": [101, 102, 103],
        "render_seconds": 8.0,
        "steps": 20,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "text_set_scope": "reviewed_lexically_varied_not_phonetically_balanced",
        "languages": [
            {
                "language_code": "en",
                "studio_language": "English",
                "described_voice_profile_id": "neutral_en_v1",
                "described_voice_description": "a neutral adult English speaker",
                "utterances": [
                    {"utterance_id": f"u{index:02d}", "text": f"Reviewed line number {index}."}
                    for index in range(1, utterance_count + 1)
                ],
            }
        ],
    }
    references = tmp_path / "references"
    speakers = []
    for index in range(speaker_count):
        reference = references / f"speaker_{index}.wav"
        _write_wav(reference, identity=index + 1)
        speakers.append(
            {
                "speaker_id": f"speaker_{index}",
                "source_id": f"source_{index}",
                "input_name": f"formal_refs/speaker_{index}.wav",
                "reference_audio": str(reference),
                "source_language_code": "en",
            }
        )
    sources = {
        "schema": "minimax_h3_t8_licensed_voice_sources_v1",
        "dataset": "unit-test-licensed-corpus",
        "dataset_revision": "fixed-revision",
        "license": "CC-BY-4.0",
        "license_source": "https://example.invalid/license",
        "speakers": speakers,
    }
    spec_path = tmp_path / "spec.json"
    sources_path = tmp_path / "sources.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    sources_path.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    return spec, sources, spec_path, sources_path


def _build(
    tool,
    tmp_path: Path,
    *,
    utterance_count: int = 10,
    speaker_count: int = 10,
):
    spec, sources, spec_path, sources_path = _fixture_payloads(
        tmp_path,
        utterance_count=utterance_count,
        speaker_count=speaker_count,
    )
    described = json.loads(DESCRIBED_TEMPLATE.read_text(encoding="utf-8"))
    clone = json.loads(CLONE_TEMPLATE.read_text(encoding="utf-8"))
    plan, prompts = tool.build_plan(
        spec_payload=spec,
        source_payload=sources,
        described_template=described,
        clone_template=clone,
        spec_path=spec_path,
        sources_path=sources_path,
        described_template_path=DESCRIBED_TEMPLATE,
        clone_template_path=CLONE_TEMPLATE,
        minimum_utterances=utterance_count,
        minimum_clone_speakers=speaker_count,
    )
    return plan, prompts, spec_path, sources_path


def _node(prompt: dict, class_type: str) -> dict:
    matches = [node for node in prompt.values() if node.get("class_type") == class_type]
    assert len(matches) == 1
    return matches[0]


def _write_case_outputs(tool, plan: dict, output_root: Path) -> list[Path]:
    paths = []
    for index, case in enumerate(plan["cases"]):
        prefix = Path(*tool.PurePosixPath(case["output_prefix"]).parts)
        path = output_root / prefix.parent / f"{prefix.name}_00001_.wav"
        _write_wav(path, identity=100 + index)
        paths.append(path)
    return paths


def test_plan_is_balanced_deterministic_and_patches_both_templates(tmp_path):
    tool = _load_tool()
    plan, prompts, _, _ = _build(tool, tmp_path)

    assert plan["case_count"] == 60
    assert plan["execution_started"] is False
    assert plan["evaluation_executed"] is False
    assert plan["stable_multilingual_gate_pass"] is False
    assert "created_at" not in plan
    assert len(prompts) == 60
    assert len({case["case_id"].casefold() for case in plan["cases"]}) == 60
    assert len({case["output_prefix"].casefold() for case in plan["cases"]}) == 60

    described_cases = [case for case in plan["cases"] if case["generation_mode"] == "described"]
    clone_cases = [case for case in plan["cases"] if case["generation_mode"] == "clone"]
    assert len(described_cases) == len(clone_cases) == 30
    assert len({case["speaker_id"] for case in clone_cases}) == 10
    for utterance_id in {case["utterance_id"] for case in plan["cases"]}:
        for mode in ("described", "clone"):
            subset = [
                case
                for case in plan["cases"]
                if case["utterance_id"] == utterance_id and case["generation_mode"] == mode
            ]
            assert {case["seed"] for case in subset} == {101, 102, 103}

    described_case = described_cases[0]
    described_prompt = prompts[described_case["prompt_path"]]
    described_profile = _node(described_prompt, "MiniMaxH3VoiceProfileT8")["inputs"]
    described_studio = _node(described_prompt, "MiniMaxH3SpeechStudioT8")["inputs"]
    assert described_profile["voice_mode"] == "described_voice"
    assert described_profile["rights_confirmed"] is False
    assert described_studio["seed"] == described_case["seed"]
    assert described_studio["verify_mode"] == "off"
    assert described_studio["speaker_check_mode"] == "off"
    assert described_studio["release_policy"] == "unload_all_models"

    clone_case = clone_cases[0]
    clone_prompt = prompts[clone_case["prompt_path"]]
    clone_profile = _node(clone_prompt, "MiniMaxH3VoiceProfileT8")["inputs"]
    clone_load = _node(clone_prompt, "LoadAudio")["inputs"]
    clone_save = _node(clone_prompt, "SaveAudio")["inputs"]
    assert clone_profile["voice_mode"] == "reference_voice"
    assert clone_profile["rights_confirmed"] is True
    assert clone_profile["reference_duration_seconds"] == pytest.approx(2.05, abs=0.01)
    assert clone_load["audio"].startswith("formal_refs/")
    assert clone_save["filename_prefix"] == clone_case["output_prefix"]
    assert clone_case["reference_audio"]["duration_seconds"] >= 2.0

    for case in plan["cases"]:
        prompt_bytes = tool._json_bytes(prompts[case["prompt_path"]])
        assert hashlib.sha256(prompt_bytes).hexdigest().upper() == case["prompt_sha256"]


def test_plan_write_is_byte_idempotent_and_rejects_prompt_drift(tmp_path):
    tool = _load_tool()
    plan, prompts, _, _ = _build(tool, tmp_path / "source")
    output = tmp_path / "plan"

    plan_path = tool.write_plan(output, plan, prompts)
    original_plan = plan_path.read_bytes()
    original_prompt = (output / plan["cases"][0]["prompt_path"]).read_bytes()
    assert tool.write_plan(output, plan, prompts) == plan_path
    assert plan_path.read_bytes() == original_plan
    assert (output / plan["cases"][0]["prompt_path"]).read_bytes() == original_prompt

    drifted = output / plan["cases"][0]["prompt_path"]
    drifted.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing prompt differs"):
        tool.write_plan(output, plan, prompts)


def test_collector_requires_decodable_unique_complete_audio(tmp_path):
    tool = _load_tool()
    plan, _, _, _ = _build(
        tool,
        tmp_path / "source",
        utterance_count=1,
        speaker_count=1,
    )
    output_root = tmp_path / "comfy_output"

    report, manifest = tool.collect_outputs(plan, output_root)
    assert manifest is None
    assert report["status_counts"] == {"PENDING_MISSING_OUTPUT": 6}

    paths = _write_case_outputs(tool, plan, output_root)
    report, manifest = tool.collect_outputs(plan, output_root)
    assert report["all_outputs_collected"] is True
    assert report["status_counts"] == {"COLLECTED_UNEVALUATED": 6}
    assert manifest is not None and len(manifest["cases"]) == 6
    assert all(len(row["audio_sha256"]) == 64 for row in manifest["cases"])
    assert all(row["audio_bytes"] > 0 for row in manifest["cases"])
    assert all(row["audio_contract"]["duration_seconds"] >= 2.0 for row in manifest["cases"])
    assert all(row["audio_contract"]["duration_seconds"] >= 2.0 for row in report["rows"])

    shutil.copyfile(paths[0], paths[1])
    report, manifest = tool.collect_outputs(plan, output_root)
    assert manifest is None
    assert report["status_counts"]["ABSTAIN_DUPLICATE_AUDIO_CONTENT"] == 1

    _write_wav(paths[1], identity=999)
    paths[2].write_bytes(b"not audio")
    report, manifest = tool.collect_outputs(plan, output_root)
    assert manifest is None
    assert report["status_counts"]["ABSTAIN_INVALID_AUDIO"] == 1


def test_collect_cli_removes_stale_manifest_when_outputs_are_incomplete(tmp_path):
    tool = _load_tool()
    plan, prompts, _, sources_path = _build(
        tool,
        tmp_path / "source",
        utterance_count=1,
        speaker_count=1,
    )
    output = tmp_path / "plan"
    tool.write_plan(output, plan, prompts)
    stale = output / "multilingual_manifest.json"
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    result = tool.main(
        [
            "--clone-sources",
            str(sources_path),
            "--output",
            str(output),
            "--collect-from",
            str(tmp_path / "empty_comfy_output"),
        ]
    )

    assert result == 3
    assert not stale.exists()
    report = json.loads((output / "collection_report.json").read_text(encoding="utf-8"))
    assert report["all_outputs_collected"] is False
    assert report["stable_multilingual_gate_pass"] is False


def test_spec_and_source_contracts_fail_closed(tmp_path):
    tool = _load_tool()
    spec, sources, spec_path, sources_path = _fixture_payloads(tmp_path)
    described = json.loads(DESCRIBED_TEMPLATE.read_text(encoding="utf-8"))
    clone = json.loads(CLONE_TEMPLATE.read_text(encoding="utf-8"))

    def build(candidate_spec=spec, candidate_sources=sources):
        return tool.build_plan(
            spec_payload=candidate_spec,
            source_payload=candidate_sources,
            described_template=described,
            clone_template=clone,
            spec_path=spec_path,
            sources_path=sources_path,
            described_template_path=DESCRIBED_TEMPLATE,
            clone_template_path=CLONE_TEMPLATE,
        )

    invalid_spec = json.loads(json.dumps(spec))
    invalid_spec["seeds"] = [1, 1, 2]
    with pytest.raises(ValueError, match="distinct integer seeds"):
        build(candidate_spec=invalid_spec)

    invalid_spec = json.loads(json.dumps(spec))
    invalid_spec["steps"] = 20.5
    with pytest.raises(ValueError, match="steps must be an integer"):
        build(candidate_spec=invalid_spec)

    invalid_spec = json.loads(json.dumps(spec))
    invalid_spec["render_seconds"] = 16.0
    with pytest.raises(ValueError, match="render_seconds"):
        build(candidate_spec=invalid_spec)

    invalid_sources = json.loads(json.dumps(sources))
    invalid_sources["speakers"][1]["reference_audio"] = invalid_sources["speakers"][0][
        "reference_audio"
    ]
    with pytest.raises(ValueError, match="unique audio content"):
        build(candidate_sources=invalid_sources)

    invalid_sources = json.loads(json.dumps(sources))
    invalid_sources["speakers"][0]["input_name"] = "../outside.wav"
    with pytest.raises(ValueError, match="relative ComfyUI input audio path"):
        build(candidate_sources=invalid_sources)

    short_reference = tmp_path / "short.wav"
    _write_wav(short_reference, identity=1000, seconds=0.5)
    invalid_sources = json.loads(json.dumps(sources))
    invalid_sources["speakers"][0]["reference_audio"] = str(short_reference)
    with pytest.raises(ValueError, match="shorter than 2 seconds"):
        build(candidate_sources=invalid_sources)


def test_collector_rejects_tampered_plan_identity_and_path(tmp_path):
    tool = _load_tool()
    plan, _, _, _ = _build(
        tool,
        tmp_path / "source",
        utterance_count=1,
        speaker_count=1,
    )
    plan["case_count"] += 1
    with pytest.raises(ValueError, match="case_count"):
        tool.collect_outputs(plan, tmp_path / "output")

    plan["case_count"] -= 1
    plan["cases"][0]["output_prefix"] = "../outside"
    with pytest.raises(ValueError, match="unsafe output_prefix"):
        tool.collect_outputs(plan, tmp_path / "output")
