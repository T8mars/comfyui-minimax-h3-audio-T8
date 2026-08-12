from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.tools.run_hybrid_model_matrix as matrix_tool
from h3_audio_t8_pkg.tools.run_hybrid_model_matrix import (
    BASE_CONTROL,
    REF_CONTROL,
    ValidationError,
    build_manifest,
    build_matrix_summary,
    build_prompt_matrix,
    completed_record_is_valid,
    control_fingerprint,
    load_or_create_manifest,
    make_treatment_prompt,
    output_descriptors,
    resolve_output_files,
    treatment_specs,
    validate_hybrid_template,
)
from h3_audio_t8_pkg.tools.validate_h3_vram import load_api_prompt


def template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "hybrid_model_advanced_api.json"


def load_template():
    return load_api_prompt(template_path())


def test_matrix_builds_controls_and_profiles_without_changing_scenario_controls():
    template = load_template()
    contract = validate_hybrid_template(template)
    profiles = [
        "blocks_25_49_video_audio_exp",
        "blocks_25_49_all_modalities_exp",
        "blocks_0_49_video_audio_exp",
    ]
    records = build_prompt_matrix(
        template,
        profiles,
        [101, 202],
        output_prefix="MiniMaxH3_T8/test-matrix",
    )

    assert len(records) == 10
    assert {record["treatment"]["id"] for record in records} == {
        BASE_CONTROL,
        REF_CONTROL,
        *profiles,
    }
    assert {record["control_fingerprint"] for record in records} == {
        control_fingerprint(template, contract)
    }
    assert {record["prompt"]["10"]["inputs"]["noise_seed"] for record in records} == {
        101,
        202,
    }


def test_ref_control_reuses_loader_node_id_so_downstream_model_links_do_not_change():
    template = load_template()
    prompt = make_treatment_prompt(
        template,
        {"id": REF_CONTROL, "kind": "ref_control"},
        seed=123,
        filename_prefix="MiniMaxH3_T8/ref-control",
    )

    assert prompt["3"] == {
        "inputs": {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Matrix control: stock Ref2VA"},
    }
    assert prompt["9"]["inputs"]["model"] == ["3", 0]
    assert prompt["11"]["inputs"]["conditioning"] == ["8", 0]
    assert prompt["12"]["inputs"]["latent_image"] == ["8", 1]


def test_template_validation_rejects_lora_no_reference_and_non_stock20():
    template = load_template()
    with_lora = copy.deepcopy(template)
    with_lora["99"] = {"class_type": "LoraLoader", "inputs": {}}
    with pytest.raises(ValidationError, match="must not contain a LoRA"):
        validate_hybrid_template(with_lora)

    no_reference = copy.deepcopy(template)
    no_reference["8"]["inputs"].pop("ref_images.ref_image_0")
    with pytest.raises(ValidationError, match="at least one connected"):
        validate_hybrid_template(no_reference)

    four_step = copy.deepcopy(template)
    four_step["9"]["inputs"]["steps"] = 4
    with pytest.raises(ValidationError, match="exactly Stock20"):
        validate_hybrid_template(four_step)


def test_treatment_specs_fail_closed_for_duplicate_or_unknown_profiles():
    with pytest.raises(ValidationError, match="Duplicate"):
        treatment_specs(["blocks_25_49_video_exp", "blocks_25_49_video_exp"])
    with pytest.raises(ValidationError, match="Unknown"):
        treatment_specs(["best_magic_profile"])


def test_history_output_resolution_hashes_files_and_rejects_escape(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    output = comfy_root / "output" / "MiniMaxH3_T8"
    output.mkdir(parents=True)
    video = output / "candidate.mp4"
    video.write_bytes(b"video-evidence")
    history = {
        "prompt-1": {
            "outputs": {
                "14": {
                    "gifs": [
                        {
                            "filename": video.name,
                            "subfolder": "MiniMaxH3_T8",
                            "type": "output",
                        }
                    ]
                }
            }
        }
    }
    descriptors = output_descriptors(history, "prompt-1")
    resolved = resolve_output_files(descriptors, comfy_root)

    assert len(resolved) == 1
    assert resolved[0]["path"] == str(video.resolve())
    assert len(resolved[0]["sha256"]) == 64

    escaping = [{**descriptors[0], "subfolder": "..\\.."}]
    with pytest.raises(ValidationError, match="escapes"):
        resolve_output_files(escaping, comfy_root)


def test_completed_record_revalidates_output_hash(tmp_path):
    output = tmp_path / "candidate.mp4"
    output.write_bytes(b"first")
    import hashlib

    digest = hashlib.sha256(b"first").hexdigest()
    record = {
        "status": "success",
        "outputs": [{"path": str(output), "sha256": digest}],
    }
    assert completed_record_is_valid(record) is True
    output.write_bytes(b"changed")
    assert completed_record_is_valid(record) is False


def test_manifest_resume_requires_identical_template_and_matrix(tmp_path):
    template = load_template()
    records = build_prompt_matrix(
        template,
        ["blocks_25_49_video_audio_exp"],
        [123],
        output_prefix="MiniMaxH3_T8/test",
    )
    output_dir = tmp_path / "matrix"
    created = build_manifest(template_path(), template, records, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(created), encoding="utf-8")

    resumed = load_or_create_manifest(
        template_path(), template, records, output_dir, resume=True
    )
    assert resumed["template"]["sha256"] == created["template"]["sha256"]

    changed = copy.deepcopy(records)
    changed[0]["prompt"]["8"]["inputs"]["prompt"] = "changed control"
    with pytest.raises(ValidationError, match="changed workflow_sha256"):
        load_or_create_manifest(
            template_path(), template, changed, output_dir, resume=True
        )


def test_speaker_metric_requires_reference_and_local_model_together(tmp_path):
    reference = tmp_path / "reference.flac"
    reference.write_bytes(b"placeholder")

    assert matrix_tool.main([
        str(template_path()),
        "--dry-run",
        "--speaker-reference",
        str(reference),
    ]) == 2


def test_speaker_identity_metric_is_explicit_single_reference_signal(tmp_path, monkeypatch):
    import hashlib
    import sys
    import types

    reference = tmp_path / "reference.flac"
    reference.write_bytes(b"reference")
    output = tmp_path / "candidate.mp4"
    output.write_bytes(b"candidate")
    model_dir = tmp_path / "wavlm"
    model_dir.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (model_dir / name).write_bytes(b"{}")

    class FakeExtractor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, samples, **_kwargs):
            return {"input_values": torch.from_numpy(samples)}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, input_values):
            vector = input_values[:2].to(dtype=torch.float32)
            return types.SimpleNamespace(embeddings=vector.unsqueeze(0))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoFeatureExtractor=FakeExtractor,
            WavLMForXVector=FakeModel,
        ),
    )
    monkeypatch.setattr(
        matrix_tool,
        "_decode_audio_mono_16k",
        lambda path, _ffmpeg: (
            __import__("numpy").array([1.0, 0.0], dtype="float32"),
            {"path": str(path), "sample_rate": 16000, "sample_count": 2},
        ),
    )
    manifest = {
        "runs": {
            "run": {
                "run_id": "run",
                "status": "success",
                "outputs": [{
                    "path": str(output),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }],
            }
        }
    }

    matrix_tool.add_speaker_identity_metrics(
        manifest,
        reference,
        model_directory=model_dir,
        ffmpeg="ffmpeg",
    )

    metric = manifest["runs"]["run"]["metrics"]["speaker_identity"]
    assert metric["cosine_similarity"] == pytest.approx(1.0)
    assert metric["threshold"] is None
    assert "cannot prove high-fidelity" in metric["scope"]


def test_matrix_summary_applies_whole_device_headroom_gate(tmp_path):
    manifest = {
        "runs": {
            "run": {
                "seed": 9,
                "treatment": {"id": "candidate", "kind": "hybrid"},
                "status": "success",
                "runtime_summary": {"peak_vram_used_bytes": 15_900 * 2**20},
                "release_before": {
                    "after": {"devices": [{"vram_total": 16_000 * 2**20}]}
                },
                "metrics": {
                    "video": {"frame_count": 124, "duration_seconds": 124 / 24},
                    "audio": {"duration_seconds": 5.152, "clipping_fraction": 0.0},
                    "asr": {"segments": [{"text": "expected sentence"}]},
                    "speaker_identity": {"cosine_similarity": 0.9},
                },
            }
        }
    }

    report = build_matrix_summary(manifest, tmp_path)

    assert report["rows"][0]["headroom_mib"] == 100.0
    assert report["rows"][0]["headroom_512mib_gate"] is False
    assert report["resource_gate"]["all_measured_runs_pass"] is False
    assert report["quality_decision"] == "not_ranked_requires_blind_review_and_broader_matrix"
    assert (tmp_path / "matrix_summary.json").is_file()
    assert (tmp_path / "matrix_summary.csv").is_file()
