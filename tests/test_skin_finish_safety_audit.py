from __future__ import annotations

import json

import torch

from h3_audio_t8_pkg.multiface_refine_advanced import (
    _hash_json,
    _json_safe,
    _source_contract,
)
from h3_audio_t8_pkg.nodes_skin_finish_safety_audit import (
    MiniMaxH3SkinFinishSafetyAuditT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_safety_audit import (
    SKIN_FINISH_SAFETY_AUDIT_SCHEMA,
    audit_skin_finish_candidate,
)


def _source(frame_count: int = 4, channels: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(250825)
    frame = 0.2 + torch.rand((1, 64, 96, 3), generator=generator) * 0.55
    frames = frame.expand(frame_count, -1, -1, -1).clone()
    if channels == 3:
        return frames
    alpha = torch.linspace(0.0, 1.0, 64 * 96).view(1, 64, 96, 1)
    return torch.cat([frames, alpha.expand(frame_count, -1, -1, -1)], dim=-1)


def _mask(frame_count: int = 4) -> torch.Tensor:
    mask = torch.zeros((frame_count, 64, 96), dtype=torch.float32)
    mask[:, 16:52, 18:78] = 1.0
    return mask


def _candidate(source: torch.Tensor, mask: torch.Tensor, value: float = 0.015) -> torch.Tensor:
    candidate = source.clone()
    active = mask > 0
    candidate[..., :3][active] = (
        candidate[..., :3][active] + float(value)
    ).clamp(0.0, 1.0)
    return candidate


def _audio(offset: float = 0.0) -> dict:
    return {
        "waveform": (torch.linspace(-0.2, 0.2, 1600) + offset).view(1, 1, -1),
        "sample_rate": 32000,
    }


def _track_plan(source: torch.Tensor, *, overlap: bool = False) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    frame_count = int(source.shape[0])
    masks = torch.zeros((frame_count, 2, 16, 24), dtype=torch.bool)
    if overlap:
        masks[:, 0, 3:14, 2:15] = True
        masks[:, 1, 3:14, 9:22] = True
    else:
        masks[:, 0, 3:14, 2:11] = True
        masks[:, 1, 3:14, 13:22] = True
    packed = pack_masks(masks)
    source_contract = dict(_source_contract(source))
    source_contract["fps"] = 24.0
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": frame_count - 1,
        "frame_count": frame_count,
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": "skin-finish-safety-test",
        "mask_size": [16, 24],
    }
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source_contract,
        "analysis": {"height": 16, "width": 24},
        "sam31": {"track_identity_scope": "shot_local_only"},
        "shots": [shot],
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 0,
        "max_scene_delta": 0.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    return plan


def test_benign_candidate_passes_hard_gates_but_source_stays_selected_by_default():
    source = _source(channels=4)
    mask = _mask()
    candidate = _candidate(source, mask)
    audio = _audio()
    selected, returned, original, audio_out, passed, failed, preview, report = (
        audit_skin_finish_candidate(
            source,
            candidate,
            mask,
            temporal_policy="hard_gate",
            audio_source=audio,
            audio_passthrough=audio,
        )
    )
    parsed = json.loads(report)
    assert parsed["schema"] == SKIN_FINISH_SAFETY_AUDIT_SCHEMA
    assert parsed["status"] == "PASS_HARD_GATES"
    assert parsed["summary"]["automatic_accept"] is False
    assert parsed["summary"]["human_review_required"] is True
    assert selected is source
    assert returned is candidate
    assert original is source
    assert audio_out is audio
    assert passed is True
    assert failed == 0
    assert preview.shape == (1, 64, 96, 3)
    assert torch.equal(candidate[..., 3:], source[..., 3:])


def test_candidate_change_outside_skin_mask_fails_closed_to_source():
    source = _source()
    mask = _mask()
    candidate = _candidate(source, mask)
    candidate[1, 2, 2, 0] += 0.10
    selected, _, _, _, passed, failed, _, report = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        accept_candidate=True,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_HARD_GATE_FAILED"
    assert passed is False
    assert failed == 1
    assert selected is source
    assert "outside_skin_mask_pixels_changed" in parsed["frame_reports"][1]["reasons"]


def test_temporal_effect_jump_is_reported_or_hard_rejected_by_policy():
    source = _source()
    mask = _mask()
    candidate = _candidate(source, mask, value=0.01)
    active = mask[2] > 0
    candidate[2, ..., :3][active] = (
        source[2, ..., :3][active] + 0.12
    ).clamp(0.0, 1.0)
    report_only = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        temporal_policy="report_only",
        maximum_mean_abs_change=0.20,
        maximum_peak_abs_change=0.30,
        maximum_temporal_effect_jump=0.04,
    )
    hard = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        temporal_policy="hard_gate",
        maximum_mean_abs_change=0.20,
        maximum_peak_abs_change=0.30,
        maximum_temporal_effect_jump=0.04,
        accept_candidate=True,
    )
    report_only_json = json.loads(report_only[-1])
    hard_json = json.loads(hard[-1])
    assert report_only_json["status"] == "PASS_HARD_GATES"
    assert report_only_json["summary"]["maximum_observed_temporal_effect_jump"] > 0.04
    assert hard_json["status"] == "ABSTAIN_HARD_GATE_FAILED"
    assert hard[0] is source
    assert any(
        reason.startswith("temporal_effect_jump")
        for reason in hard_json["frame_reports"][2]["reasons"]
    )


def test_track_union_rejects_mask_pixels_outside_all_person_tracks():
    source = _source()
    plan = _track_plan(source)
    mask = torch.zeros(source.shape[:3])
    mask[:, 20:42, 10:36] = 1.0
    mask[:, 0:4, 0:4] = 1.0
    candidate = _candidate(source, mask, value=0.01)
    result = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        audit_scope="track_union",
        track_plan=plan,
        maximum_track_leak_fraction=0.0,
        accept_candidate=True,
    )
    parsed = json.loads(result[-1])
    assert result[4] is False
    assert result[0] is source
    assert result[1] is source
    assert parsed["summary"]["track_leak_pixels"] > 0
    assert all(
        "skin_mask_outside_person_track" in frame["reasons"]
        for frame in parsed["frame_reports"]
    )


def test_unique_owner_scope_rejects_skin_edits_on_overlapping_person_masks():
    source = _source()
    plan = _track_plan(source, overlap=True)
    mask = torch.zeros(source.shape[:3])
    mask[:, 20:46, 38:58] = 1.0
    candidate = _candidate(source, mask, value=0.01)
    result = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        audit_scope="unique_track_owner",
        track_plan=plan,
        maximum_track_leak_fraction=0.0,
    )
    parsed = json.loads(result[-1])
    assert result[4] is False
    assert parsed["summary"]["ambiguous_owner_pixels"] > 0
    assert any(
        "skin_mask_on_ambiguous_person_overlap" in frame["reasons"]
        for frame in parsed["frame_reports"]
    )


def test_required_invalid_track_plan_and_audio_mismatch_fail_closed():
    source = _source()
    mask = _mask()
    candidate = _candidate(source, mask)
    broken_plan = _track_plan(source)
    broken_plan["source"]["proxy_sha256"] = "0" * 64
    result = audit_skin_finish_candidate(
        source,
        candidate,
        mask,
        audit_scope="track_union",
        track_plan=broken_plan,
        audio_source=_audio(),
        audio_passthrough=_audio(0.01),
        accept_candidate=True,
    )
    parsed = json.loads(result[-1])
    assert result[4] is False
    assert result[5] == source.shape[0]
    assert result[0] is source
    assert parsed["status"] == "ABSTAIN_TRACK_PLAN_INVALID"
    assert parsed["audio"]["status"] == "pcm_mismatch"


def test_safety_audit_schema_is_append_only_and_source_safe_by_default():
    schema = MiniMaxH3SkinFinishSafetyAuditT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3SkinFinishSafetyAuditT8Advanced"
    assert schema.is_experimental is True
    assert inputs["audit_scope"].default == "mask_only"
    assert inputs["temporal_policy"].default == "report_only"
    assert inputs["accept_candidate"].default is False
    assert inputs["track_plan"].optional is True
