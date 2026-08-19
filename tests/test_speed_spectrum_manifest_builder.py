from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.tools.build_h3_speed_spectrum_manifest import (
    _source_set_sha256,
    build_manifest,
)


def _report(path: Path, *, status: str = "provisional_candidate"):
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {
        "schema": "minimax_h3_speed_source_curation_v2",
        "mode": "signature",
        "policy": {"target_width": 736, "target_height": 416, "required_frames": 124},
        "items": [
            {
                "path": str(path),
                "status": status,
                "file_sha256": file_hash,
                "decoded_window_sha256": "D" * 64,
                "strict_decode": {"passed": True},
            }
        ],
    }


def _build(report, report_path: Path, input_root: Path):
    return build_manifest(
        report,
        report_path=report_path,
        input_root=input_root,
        dataset_name="proxy",
        task_family="T2VA",
        video_vae_name="minimax_h3_video_vae_fp16.safetensors",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
    )


def test_manifest_builder_selects_only_strict_provisional_candidates(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    manifest = _build(_report(media), report_path, tmp_path)
    assert manifest["entries"][0]["file"] == "clip.mp4"
    assert manifest["provenance"]["formal_dataset_authorized"] is False
    assert manifest["provenance"]["selected_provisional_candidates"] == 1
    assert manifest["provenance"]["selection_policy"] == "sha256_rank"
    assert manifest["entries"][0]["decoded_window_sha256"] == "D" * 64
    assert manifest["entries"][0]["source_entry"]["batch_id"] == manifest["entries"][0]["batch_id"]


def test_manifest_builder_rejects_non_signature_or_outside_sources(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    report = _report(media)
    with pytest.raises(ValueError, match="signature-mode"):
        _build({**report, "mode": "metadata"}, report_path, tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="outside"):
        _build(report, report_path, other)


def test_manifest_builder_never_promotes_fewer_than_100(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be lower than 100"):
        build_manifest(
            _report(media),
            report_path=report_path,
            input_root=tmp_path,
            dataset_name="unsafe",
            task_family="T2VA",
            video_vae_name="video.safetensors",
            checkpoint_fingerprint="sha256:model",
            vae_fingerprint="sha256:vae",
            minimum_formal_clips=99,
        )


def test_manifest_builder_binds_fixed_fetch_reports_and_reviewed_selection(tmp_path):
    items = []
    files = []
    for index in range(100):
        path = tmp_path / f"clip_{index:03d}.mp4"
        path.write_bytes(f"clip-{index}".encode())
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        decoded_hash = hashlib.sha256(f"decoded-{index}".encode()).hexdigest().upper()
        items.append(
            {
                "path": str(path),
                "status": "provisional_candidate",
                "file_sha256": file_hash,
                "decoded_signature": {"raw_sha256": decoded_hash},
                "strict_decode": {"passed": True},
            }
        )
        files.append({"sha256": file_hash})
    report = {
        "schema": "minimax_h3_speed_source_curation_v2",
        "mode": "signature",
        "policy": {"target_width": 736, "target_height": 416, "required_frames": 124},
        "items": list(reversed(items)),
    }
    report_path = tmp_path / "curation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    fetch = {
        "schema": "minimax_h3_speed_external_corpus_fetch_v1",
        "dataset": {
            "id": "example/natural-videos",
            "revision": "fixed-revision",
            "license": "apache-2.0",
            "shard": "00000/000000.tar",
            "shard_lfs_oid": "A" * 64,
        },
        "files": files,
    }
    fetch_path = tmp_path / "fetch.json"
    fetch_path.write_text(json.dumps(fetch), encoding="utf-8")
    selected_rows = [
        {
            "source_file_sha256": item["file_sha256"],
            "decoded_window_sha256": item["decoded_signature"]["raw_sha256"],
        }
        for item in items
    ]
    review = {
        "schema": "minimax_h3_speed_corpus_visual_review_v1",
        "selected_source_count": 100,
        "selected_source_set_sha256": _source_set_sha256(selected_rows),
        "independence_reviewed": True,
        "content_diversity_reviewed": True,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    manifest = build_manifest(
        report,
        report_path=report_path,
        input_root=tmp_path,
        dataset_name="formal",
        task_family="T2VA",
        video_vae_name="video.safetensors",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        maximum_entries=100,
        fetch_reports=[(fetch_path, fetch)],
        independence_reviewed=True,
        content_diversity_reviewed=True,
        review_report=(review_path, review),
    )
    assert manifest["provenance"]["formal_dataset_authorized"] is True
    assert manifest["dataset_provenance"]["selected_source_count"] == 100
    assert manifest["dataset_provenance"]["raw_media_redistributed"] is False
    assert len(manifest["dataset_provenance"]["review_report_sha256"]) == 64
    selected = [entry["source_file_sha256"] for entry in manifest["entries"]]
    assert selected == sorted(selected)


def test_manifest_builder_refuses_formal_claim_without_fetch_coverage(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"clip")
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    fetch = {
        "schema": "minimax_h3_speed_external_corpus_fetch_v1",
        "dataset": {
            "id": "example/natural-videos",
            "revision": "fixed",
            "license": "apache-2.0",
            "shard": "00000/000000.tar",
            "shard_lfs_oid": "A" * 64,
        },
        "files": [{"sha256": "B" * 64}],
    }
    fetch_path = tmp_path / "fetch.json"
    fetch_path.write_text(json.dumps(fetch), encoding="utf-8")
    with pytest.raises(ValueError, match="not fully covered"):
        build_manifest(
            _report(media),
            report_path=report_path,
            input_root=tmp_path,
            dataset_name="unsafe",
            task_family="T2VA",
            video_vae_name="video.safetensors",
            checkpoint_fingerprint="sha256:model",
            vae_fingerprint="sha256:vae",
            fetch_reports=[(fetch_path, fetch)],
            independence_reviewed=True,
            content_diversity_reviewed=True,
        )
