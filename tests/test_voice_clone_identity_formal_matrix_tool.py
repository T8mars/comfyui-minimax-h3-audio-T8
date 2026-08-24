from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "build_voice_clone_identity_formal_matrix.py"
BATCH_TOOL_PATH = ROOT / "tools" / "run_speech_multilingual_formal_batch.py"
CLONE_TEMPLATE = ROOT / "tests" / "fixtures" / "api" / "speech_reference_clone_api.json"


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
    return _load_path("h3_voice_clone_identity_formal_matrix_test_tool", TOOL_PATH)


def _write_wav(path: Path, *, identity: int, seconds: float = 2.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(16_000 * seconds)
    samples = array(
        "h",
        (((index * (identity + 7)) % 3000) - 1500 for index in range(frame_count)),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _payloads(
    tmp_path: Path,
    *,
    target_count: int = 10,
    utterance_count: int = 10,
) -> tuple[dict, dict, Path, Path]:
    if target_count == 10:
        labels = ["M", "M", "F", "F", "M", "M", "F", "M", "M", "F"]
    else:
        labels = ["M"] * target_count
    targets = [
        {
            "speaker_id": f"speaker_{index:02d}",
            "gender_label": labels[index],
            "split": "test-clean",
        }
        for index in range(target_count)
    ]
    spec = {
        "schema": "minimax_h3_t8_voice_clone_identity_formal_spec_v1",
        "plan_id": "identity_unit_v1",
        "seeds": [201, 202, 203],
        "render_seconds": 8.0,
        "steps": 20,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "language_code": "en",
        "studio_language": "English",
        "text_set_scope": "reviewed_unit_test",
        "minimum_independent_reviewers": 3,
        "speaker_metadata": {
            "dataset": "LibriSpeech SPEAKERS.TXT",
            "revision": "fixed-revision",
            "relative_path": "SPEAKERS.TXT",
            "source_url": "https://example.invalid/SPEAKERS.TXT",
            "sha256": "A" * 64,
            "label_semantics": "Unit-test blocking labels only.",
        },
        "target_speakers": targets,
        "utterances": [
            {"utterance_id": f"u{index:02d}", "text": f"Reviewed sentence number {index}."}
            for index in range(1, utterance_count + 1)
        ],
        "abx_standardization": {
            "sample_rate": 32000,
            "channels": 1,
            "codec": "flac",
            "container": "flac",
            "loudness_normalization": False,
            "duration_policy": "preserve_each_source",
        },
    }
    references = tmp_path / "references"
    speakers = []
    for index, target in enumerate(targets):
        reference = references / f"speaker_{index:02d}.wav"
        _write_wav(reference, identity=index + 1)
        speakers.append(
            {
                "speaker_id": target["speaker_id"],
                "source_id": f"source_{index:02d}",
                "input_name": f"identity_refs/speaker_{index:02d}.wav",
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
    target_count: int = 10,
    utterance_count: int = 10,
    impostors_per_target: int = 3,
):
    spec, sources, spec_path, sources_path = _payloads(
        tmp_path, target_count=target_count, utterance_count=utterance_count
    )
    clone_template = json.loads(CLONE_TEMPLATE.read_text(encoding="utf-8"))
    plan, prompts, design = tool.build_identity_plan(
        spec_payload=spec,
        source_payload=sources,
        clone_template=clone_template,
        spec_path=spec_path,
        sources_path=sources_path,
        clone_template_path=CLONE_TEMPLATE,
        required_targets=target_count,
        required_utterances=utterance_count,
        impostors_per_target=impostors_per_target,
    )
    return plan, prompts, design, spec, sources, spec_path, sources_path


def _node(prompt: dict, class_type: str) -> dict:
    matches = [node for node in prompt.values() if node.get("class_type") == class_type]
    assert len(matches) == 1
    return matches[0]


def test_formal_identity_grid_and_same_label_abx_are_balanced(tmp_path):
    tool = _load_tool()
    plan, prompts, design, _, _, _, _ = _build(tool, tmp_path)

    assert plan["plan_kind"] == "voice_clone_identity_formal"
    assert plan["case_count"] == 300
    assert len(prompts) == 300
    assert design["generation_case_count"] == 300
    assert design["abx_case_count"] == 90
    assert plan["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    assert design["decision_state"]["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    assert len({case["case_id"] for case in plan["cases"]}) == 300
    assert len({case["output_prefix"] for case in plan["cases"]}) == 300

    cases_by_target: dict[str, list[dict]] = defaultdict(list)
    for case in plan["cases"]:
        cases_by_target[case["speaker_id"]].append(case)
    assert len(cases_by_target) == 10
    for cases in cases_by_target.values():
        assert len(cases) == 30
        assert {case["utterance_id"] for case in cases} == {
            f"u{index:02d}" for index in range(1, 11)
        }
        for utterance_id in {case["utterance_id"] for case in cases}:
            assert {
                case["seed"] for case in cases if case["utterance_id"] == utterance_id
            } == {201, 202, 203}

    target_rows = {row["speaker_id"]: row for row in design["target_speakers"]}
    schedule_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in design["abx_schedule"]:
        schedule_by_target[row["target_speaker_id"]].append(row)
        assert target_rows[row["target_speaker_id"]]["gender_label"] == target_rows[
            row["impostor_speaker_id"]
        ]["gender_label"]
    assert len({row["candidate_generation_case_id"] for row in design["abx_schedule"]}) == 90
    for target_id, rows in schedule_by_target.items():
        assert len(rows) == 9
        assert len({row["impostor_speaker_id"] for row in rows}) == 3
        assert {row["seed"] for row in rows} == {201, 202, 203}
        assert len({row["utterance_id"] for row in rows}) == 9
        assert "u10" not in {row["utterance_id"] for row in rows}
        assert {row["metadata_block_label"] for row in rows} == {
            target_rows[target_id]["gender_label"]
        }

    for label in ("F", "M"):
        label_targets = {
            speaker_id
            for speaker_id, row in target_rows.items()
            if row["gender_label"] == label
        }
        assignment_uses = Counter(
            row["impostor_speaker_id"]
            for target_id, rows in schedule_by_target.items()
            if target_id in label_targets
            for row in rows[::3]
        )
        assert set(assignment_uses) == label_targets
        assert len(set(assignment_uses.values())) == 1

    first_case = plan["cases"][0]
    first_prompt = prompts[first_case["prompt_path"]]
    assert _node(first_prompt, "MiniMaxH3VoiceProfileT8")["inputs"]["voice_mode"] == (
        "reference_voice"
    )
    assert _node(first_prompt, "MiniMaxH3VoiceProfileT8")["inputs"]["rights_confirmed"] is True
    assert _node(first_prompt, "MiniMaxH3SpeechStudioT8")["inputs"]["release_policy"] == (
        "unload_all_models"
    )
    assert _node(first_prompt, "LoadAudio")["inputs"]["audio"].startswith("identity_refs/")
    for case in plan["cases"]:
        prompt_bytes = tool.matrix._json_bytes(prompts[case["prompt_path"]])
        assert hashlib.sha256(prompt_bytes).hexdigest().upper() == case["prompt_sha256"]


def test_identity_plan_write_is_idempotent_and_batch_executor_compatible(tmp_path):
    tool = _load_tool()
    plan, prompts, design, _, _, _, _ = _build(tool, tmp_path / "source")
    output = tmp_path / "plan"

    plan_path, design_path = tool.write_identity_plan(output, plan, prompts, design)
    before = (plan_path.read_bytes(), design_path.read_bytes())
    assert tool.write_identity_plan(output, plan, prompts, design) == (plan_path, design_path)
    assert (plan_path.read_bytes(), design_path.read_bytes()) == before

    batch = _load_path("h3_voice_clone_identity_batch_compat_test_tool", BATCH_TOOL_PATH)
    loaded_plan, loaded_prompts, loaded_sha = batch.load_plan(output)
    assert loaded_plan["plan_kind"] == tool.PLAN_KIND
    assert len(loaded_prompts) == 300
    assert loaded_sha == hashlib.sha256(plan_path.read_bytes()).hexdigest().upper()
    assert batch._collection_manifest_path(output, loaded_plan).name == (
        "identity_generation_manifest.raw.json"
    )

    drifted = json.loads(json.dumps(design))
    drifted["decision_state"]["generation"] = "CLAIMED_WITHOUT_EVIDENCE"
    with pytest.raises(ValueError, match="identity design differs"):
        tool.write_identity_plan(output, plan, prompts, drifted)


def test_collection_stays_unclaimed_until_all_outputs_are_unique(tmp_path):
    tool = _load_tool()
    plan, _, design, _, _, _, _ = _build(tool, tmp_path / "source")
    report, manifest, jobs = tool.collect_identity_outputs(
        plan=plan, design=design, comfy_output=tmp_path / "empty_output"
    )
    assert report["planned_case_count"] == 300
    assert report["collected_unique_case_count"] == 0
    assert report["status_counts"] == {"PENDING_MISSING_OUTPUT": 300}
    assert report["high_fidelity_clone_claim"] == "NOT_ESTABLISHED"
    assert manifest is None
    assert jobs is None


def test_complete_small_collection_emits_jobs_not_a_blind_package(tmp_path):
    tool = _load_tool()
    plan, _, design, _, _, _, _ = _build(
        tool,
        tmp_path / "source",
        target_count=2,
        utterance_count=3,
        impostors_per_target=1,
    )
    output_root = tmp_path / "comfy_output"
    for index, case in enumerate(plan["cases"]):
        prefix = Path(*tool.PurePosixPath(case["output_prefix"]).parts)
        output = output_root / prefix.parent / f"{prefix.name}_00001_.wav"
        _write_wav(output, identity=100 + index)

    report, manifest, jobs = tool.collect_identity_outputs(
        plan=plan, design=design, comfy_output=output_root
    )
    assert report["all_outputs_collected"] is True
    assert manifest is not None and len(manifest["cases"]) == 18
    assert jobs is not None and jobs["job_count"] == 6
    assert jobs["execution_started"] is False
    assert jobs["abx_manifest_written"] is False
    assert jobs["contract"]["sample_rate"] == 32000
    assert len(jobs["identity_design_sha256"]) == 64
    assert all(len(row["inputs"]["candidate"]["sha256"]) == 64 for row in jobs["jobs"])
    assert all(row["outputs"]["candidate"].endswith(".flac") for row in jobs["jobs"])


def test_identity_spec_and_source_contracts_fail_closed(tmp_path):
    tool = _load_tool()
    plan, _, _, spec, sources, spec_path, sources_path = _build(tool, tmp_path / "source")
    assert plan["case_count"] == 300
    clone_template = json.loads(CLONE_TEMPLATE.read_text(encoding="utf-8"))

    def build(candidate_spec=spec, candidate_sources=sources):
        return tool.build_identity_plan(
            spec_payload=candidate_spec,
            source_payload=candidate_sources,
            clone_template=clone_template,
            spec_path=spec_path,
            sources_path=sources_path,
            clone_template_path=CLONE_TEMPLATE,
        )

    invalid = json.loads(json.dumps(spec))
    invalid["seeds"] = [1, 1, 2]
    with pytest.raises(ValueError, match="exactly three distinct"):
        build(candidate_spec=invalid)

    invalid = json.loads(json.dumps(spec))
    invalid["speaker_metadata"]["sha256"] = "not-a-fixed-hash"
    with pytest.raises(ValueError, match="fixed 64-character SHA-256"):
        build(candidate_spec=invalid)

    invalid = json.loads(json.dumps(spec))
    for row in invalid["target_speakers"]:
        row["gender_label"] = "M"
    invalid["target_speakers"][0]["gender_label"] = "F"
    invalid["target_speakers"][1]["gender_label"] = "F"
    invalid["target_speakers"][2]["gender_label"] = "F"
    with pytest.raises(ValueError, match="insufficient"):
        build(candidate_spec=invalid)

    invalid_sources = json.loads(json.dumps(sources))
    invalid_sources["speakers"] = invalid_sources["speakers"][:-1]
    with pytest.raises(ValueError, match="at least 10 speakers"):
        build(candidate_sources=invalid_sources)

    tampered_design = {
        "schema": tool.DESIGN_SCHEMA,
        "plan_id": plan["plan_id"],
        "generation_plan_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="generation_plan_sha256 differs"):
        tool.collect_identity_outputs(
            plan=plan, design=tampered_design, comfy_output=tmp_path / "output"
        )
