from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import pytest
import torch

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.long_video_delivery import (
    accept_long_video_candidate,
    long_video_chain_root,
    load_delivery_manifest,
    save_long_video_candidate,
)
from h3_audio_t8_pkg.repair_compose_advanced import compose_repair_overlay
from h3_audio_t8_pkg.repair_execution_advanced import (
    REPAIR_EXECUTION_SCHEMA,
    accept_staged_repair,
    bind_repair_execution,
    stage_repair_candidate,
)
from h3_audio_t8_pkg.studio_advanced import (
    build_selective_repair_plan,
    build_studio_timeline,
)
from helpers import make_audio


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    chain_id: str,
    index: int,
    start_frame: int,
    candidate_id: str,
    *,
    parent_id: str = "",
    parent_revision: int = 0,
    final: bool = False,
    value: float = 0.25,
):
    frame_count = 22
    frames = torch.full((frame_count, 128, 128, 3), value)
    audio = make_audio(frame_count / 24, sample_rate=32000, value=value, channels=2)
    latent, _ = empty_av_latent(128, 128, frame_count)
    return save_long_video_candidate(
        frames,
        audio,
        latent,
        chain_id,
        index,
        start_frame / 24,
        not final,
        parent_id,
        parent_revision,
        candidate_id,
        "test-model",
        "1-step-test",
        f"segment {index}",
        index + 100,
        24,
        8,
        28,
    )


def _repair_plan():
    timeline = build_studio_timeline(
        "repair-demo",
        json.dumps(
            [
                {"id": "a", "prompt": "first shot", "duration_seconds": 0.9},
                {"id": "b", "prompt": "second shot", "duration_seconds": 0.9},
            ]
        ),
        "minimax_h3",
        0.9,
        "16:9",
        100,
        "increment",
        True,
        True,
    )
    return build_selective_repair_plan(
        timeline,
        "",
        "manual",
        "1",
        "{}",
        "seed_retry",
        "preserve continuity",
        1009,
        22,
        22,
    )


def _accepted_two_segment_chain(chain_id: str):
    candidate0, _, _ = _candidate(chain_id, 0, 0, "base-a")
    accept_long_video_candidate(candidate0, True)
    candidate1, _, _ = _candidate(
        chain_id,
        1,
        22,
        "base-b",
        parent_id="base-a",
        parent_revision=1,
        final=True,
        value=-0.25,
    )
    accept_long_video_candidate(candidate1, True)


def test_selective_repair_overlay_preserves_base_and_streams_composition(
    monkeypatch,
    tmp_path,
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(
        delivery.folder_paths, "get_output_directory", lambda: str(tmp_path)
    )
    chain_id = "repair-overlay"
    _accepted_two_segment_chain(chain_id)
    root = long_video_chain_root(chain_id)
    base_manifest_path = root / "manifest.json"
    base_before = base_manifest_path.read_bytes()
    base_manifest, _ = load_delivery_manifest(chain_id)
    unselected_path = root / base_manifest["segments"][0]["video_path"]
    unselected_hash = _sha(unselected_path)

    execution, bind_report = bind_repair_execution(_repair_plan(), chain_id, 0)
    assert execution["schema"] == REPAIR_EXECUTION_SCHEMA
    assert execution["status"] == "bound"
    assert execution["source_segment"]["candidate_id"] == "base-b"
    assert bind_report["base_manifest_mutated"] is False

    repair_candidate, repair_video, _ = _candidate(
        chain_id,
        1,
        22,
        "repair-b",
        parent_id="base-a",
        parent_revision=2,
        final=True,
        value=0.75,
    )
    staged, staged_video, stage_report = stage_repair_candidate(
        execution,
        repair_candidate,
    )
    assert staged["status"] == "staged"
    assert staged_video == repair_video
    assert stage_report["exact_frame_and_sample_boundaries"] is True

    preview_path, accepted, preview_report = accept_staged_repair(
        staged,
        False,
    )
    assert preview_path == ""
    assert accepted is False
    assert preview_report["repair_overlay_mutated"] is False
    assert base_manifest_path.read_bytes() == base_before

    repair_manifest_path, accepted, accept_report = accept_staged_repair(
        staged,
        True,
    )
    assert accepted is True
    assert Path(repair_manifest_path).is_file()
    assert accept_report["base_manifest_mutated"] is False
    assert base_manifest_path.read_bytes() == base_before
    assert _sha(unselected_path) == unselected_hash

    same_path, accepted, idempotent = accept_staged_repair(staged, True)
    assert accepted is True
    assert same_path == repair_manifest_path
    assert idempotent["idempotent"] is True

    overlay_path, overlay_report = compose_repair_overlay(
        chain_id,
        repair_manifest_path,
        "repair_overlay",
        "repair-result",
        True,
        "none",
        0.0,
        28,
    )
    assert Path(overlay_path).is_file()
    assert overlay_report["replacement_indices"] == [1]
    assert overlay_report["unselected_indices"] == [0]
    assert overlay_report["unselected_source_files_sha256_unchanged"] is True
    assert overlay_report["base_manifest_mutated"] is False
    assert overlay_report["frame_count"] == 44
    assert overlay_report["audio_samples"] == round(44 / 24 * 32000)
    assert _sha(unselected_path) == unselected_hash
    assert base_manifest_path.read_bytes() == base_before
    with av.open(overlay_path) as container:
        assert sum(1 for _ in container.decode(video=0)) == 44

    rollback_path, rollback_report = compose_repair_overlay(
        chain_id,
        repair_manifest_path,
        "base_rollback",
        "repair-rollback",
        True,
        "none",
        0.0,
        28,
    )
    assert Path(rollback_path).is_file()
    assert rollback_report["rollback_used"] is True
    assert rollback_report["repair_overlay_used"] is False


def test_repair_stage_rejects_stale_base_and_media_contract_mismatch(
    monkeypatch,
    tmp_path,
):
    import h3_audio_t8_pkg.long_video_delivery as delivery

    monkeypatch.setattr(
        delivery.folder_paths, "get_output_directory", lambda: str(tmp_path)
    )
    chain_id = "repair-stale"
    _accepted_two_segment_chain(chain_id)
    execution, _ = bind_repair_execution(_repair_plan(), chain_id, 0)

    wrong, _, _ = _candidate(
        chain_id,
        1,
        22,
        "wrong-contract",
        parent_id="base-a",
        parent_revision=2,
        final=True,
        value=0.5,
    )
    descriptor = Path(wrong)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["sampling_summary"] = "different"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timeline/media identity fields"):
        stage_repair_candidate(execution, wrong)
