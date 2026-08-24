from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re

import torch

from .studio_advanced import validate_timeline


CREATOR_EDIT_SCHEMA = "t8.minimax_h3.creator_edit_plan.v1"
CREATOR_WORKSPACE_SCHEMA = "t8.minimax_h3.creator_workspace.v1"
CREATOR_REVIEW_SCHEMA = "t8.minimax_h3.creator_review.v1"
MAX_UINT64 = 0xFFFFFFFFFFFFFFFF
_MEDIA_SLOT = re.compile(r"^(picture|video|audio)_[1-9][0-9]*$")


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_object(value: str, name: str, default) -> dict:
    text = str(value or "").strip()
    if not text:
        return deepcopy(default)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(payload)


def _validate_edit_plan(plan, timeline_hash: str) -> dict:
    if plan is None:
        return {
            "schema": CREATOR_EDIT_SCHEMA,
            "timeline_hash": timeline_hash,
            "edits": [],
        }
    if not isinstance(plan, Mapping) or plan.get("schema") != CREATOR_EDIT_SCHEMA:
        raise ValueError("previous_edits must be a Creator Shot Override plan")
    if plan.get("timeline_hash") != timeline_hash:
        raise ValueError("previous_edits belongs to a different Studio Timeline")
    return deepcopy(dict(plan))


def add_creator_shot_override(
    timeline: Mapping,
    shot_index: int,
    enabled: bool,
    compiled_prompt_override: str,
    use_seed_override: bool,
    seed_override: int,
    variant_count: int,
    variant_seed_stride: int,
    media_roles_json: str,
    retention_policy: str,
    hold_policy: str,
    hold_frames: int,
    previous_edits=None,
) -> tuple[dict, str]:
    timeline = validate_timeline(timeline)
    index = int(shot_index)
    if not 0 <= index < len(timeline["shots"]):
        raise ValueError(f"shot_index must be between 0 and {len(timeline['shots']) - 1}")
    if not 1 <= int(variant_count) <= 64:
        raise ValueError("variant_count must be between 1 and 64")
    if not 1 <= int(variant_seed_stride) <= MAX_UINT64:
        raise ValueError("variant_seed_stride must be between 1 and 2^64-1")
    if not 0 <= int(seed_override) <= MAX_UINT64:
        raise ValueError("seed_override must be between 0 and 2^64-1")
    if retention_policy not in {
        "keep_all",
        "keep_winner_and_metadata",
        "keep_accepted_only",
        "metadata_only",
    }:
        raise ValueError("unsupported retention_policy")
    if hold_policy not in {"none", "hold_first", "hold_last", "hold_both", "custom_metadata"}:
        raise ValueError("unsupported hold_policy")
    if int(hold_frames) < 0:
        raise ValueError("hold_frames cannot be negative")
    media_roles = _json_object(media_roles_json, "media_roles_json", {})
    normalized_roles = {}
    for slot, role in media_roles.items():
        slot = str(slot).strip().lower()
        if not _MEDIA_SLOT.fullmatch(slot):
            raise ValueError(
                "media role keys must use picture_N, video_N or audio_N"
            )
        role = str(role or "").strip()
        if not role or len(role) > 128:
            raise ValueError(f"media role {slot} must contain 1-128 characters")
        normalized_roles[slot] = role

    plan = _validate_edit_plan(previous_edits, timeline["timeline_hash"])
    if any(int(item["shot_index"]) == index for item in plan["edits"]):
        raise ValueError(
            f"shot_index {index} already has an override; edit the existing node instead of "
            "creating order-dependent duplicates"
        )
    shot = timeline["shots"][index]
    base_seed = int(seed_override) if use_seed_override else int(shot["seed"])
    variant_seeds = [
        (base_seed + variant * int(variant_seed_stride)) & MAX_UINT64
        for variant in range(int(variant_count))
    ]
    override = {
        "shot_index": index,
        "shot_id": shot["id"],
        "enabled": bool(enabled),
        "compiled_prompt_override": str(compiled_prompt_override or "").strip(),
        "seed_source": "override" if use_seed_override else "timeline",
        "base_seed": base_seed,
        "variant_count": int(variant_count),
        "variant_seed_stride": int(variant_seed_stride),
        "variant_seeds": variant_seeds,
        "media_roles": normalized_roles,
        "retention_policy": retention_policy,
        "hold_policy": hold_policy,
        "hold_frames": int(hold_frames),
    }
    override["override_hash"] = _hash(override)
    plan["edits"].append(override)
    plan["edit_count"] = len(plan["edits"])
    plan["edit_hash"] = _hash(
        {
            "schema": plan["schema"],
            "timeline_hash": plan["timeline_hash"],
            "edits": plan["edits"],
        }
    )
    return plan, canonical_json(plan)


def compile_creator_workspace(
    timeline: Mapping,
    run_from_index: int,
    run_to_index: int,
    include_disabled_shots: bool,
    workspace_notes: str,
    edit_plan=None,
) -> tuple[dict, str, str, str]:
    timeline = validate_timeline(timeline)
    last_index = len(timeline["shots"]) - 1
    start = int(run_from_index)
    end = last_index if int(run_to_index) < 0 else int(run_to_index)
    if not 0 <= start <= last_index:
        raise ValueError(f"run_from_index must be between 0 and {last_index}")
    if not start <= end <= last_index:
        raise ValueError(f"run_to_index must be -1 or between {start} and {last_index}")
    edits = _validate_edit_plan(edit_plan, timeline["timeline_hash"])
    edit_map = {int(item["shot_index"]): item for item in edits["edits"]}
    effective_shots = []
    skipped = []
    for index in range(start, end + 1):
        source = timeline["shots"][index]
        edit = edit_map.get(index)
        enabled = bool(edit.get("enabled", True)) if edit else True
        if not enabled and not include_disabled_shots:
            skipped.append(index)
            continue
        packet = source["prompt_packet"]
        effective_prompt = (
            edit["compiled_prompt_override"]
            if edit and edit["compiled_prompt_override"]
            else packet["compiled_prompt"]
        )
        base_seed = int(edit["base_seed"]) if edit else int(source["seed"])
        variant_seeds = list(edit["variant_seeds"]) if edit else [base_seed]
        effective_shots.append(
            {
                "run_position": len(effective_shots),
                "shot_index": index,
                "shot_id": source["id"],
                "enabled": enabled,
                "frame_count": int(source["frame_count"]),
                "render_duration_seconds": float(source["render_duration_seconds"]),
                "effective_compiled_prompt": effective_prompt,
                "prompt_source": "override" if edit and edit["compiled_prompt_override"] else "timeline",
                "negative_prompt": packet.get("negative_prompt", ""),
                "base_seed": base_seed,
                "variant_seeds": variant_seeds,
                "media_roles": deepcopy(edit.get("media_roles", {})) if edit else {},
                "retention_policy": (
                    edit.get("retention_policy", "keep_winner_and_metadata")
                    if edit
                    else "keep_winner_and_metadata"
                ),
                "hold_policy": edit.get("hold_policy", "none") if edit else "none",
                "hold_frames": int(edit.get("hold_frames", 0)) if edit else 0,
                "source_packet_hash": packet.get("packet_hash"),
                "override_hash": edit.get("override_hash") if edit else None,
            }
        )
    if not effective_shots:
        raise ValueError("selected run window contains no enabled shots")
    hold_map = {
        shot["shot_id"]: {
            "policy": shot["hold_policy"],
            "frames": shot["hold_frames"],
        }
        for shot in effective_shots
        if shot["hold_policy"] != "none" or shot["hold_frames"]
    }
    workspace = {
        "schema": CREATOR_WORKSPACE_SCHEMA,
        "project_id": timeline["project_id"],
        "source_timeline_hash": timeline["timeline_hash"],
        "edit_hash": edits.get("edit_hash"),
        "run_from_index": start,
        "run_to_index": end,
        "include_disabled_shots": bool(include_disabled_shots),
        "skipped_disabled_indices": skipped,
        "shots": effective_shots,
        "run_count": len(effective_shots),
        "hold_map": hold_map,
        "workspace_notes": str(workspace_notes or "").strip(),
        "execution_authority": False,
        "queue_mutated": False,
        "compiler_only": True,
    }
    workspace["workspace_hash"] = _hash(workspace)
    sidecar = {
        "schema": CREATOR_WORKSPACE_SCHEMA,
        "project_id": workspace["project_id"],
        "workspace_hash": workspace["workspace_hash"],
        "source_timeline_hash": workspace["source_timeline_hash"],
        "run_window": [start, end],
        "shot_reproducibility": [
            {
                "shot_id": shot["shot_id"],
                "shot_index": shot["shot_index"],
                "frame_count": shot["frame_count"],
                "base_seed": shot["base_seed"],
                "variant_seeds": shot["variant_seeds"],
                "source_packet_hash": shot["source_packet_hash"],
                "override_hash": shot["override_hash"],
                "retention_policy": shot["retention_policy"],
            }
            for shot in effective_shots
        ],
        "hold_map": hold_map,
        "file_written": False,
    }
    summary = "\n".join(
        f"{item['run_position']}: shot {item['shot_index']} / {item['shot_id']} / "
        f"{len(item['variant_seeds'])} variant(s) / {item['retention_policy']}"
        for item in effective_shots
    )
    return workspace, summary, canonical_json(sidecar), canonical_json(workspace)


def validate_creator_workspace(workspace: Mapping) -> dict:
    if not isinstance(workspace, Mapping) or workspace.get("schema") != CREATOR_WORKSPACE_SCHEMA:
        raise ValueError("workspace must be a Creator Workspace plan")
    if not isinstance(workspace.get("shots"), list) or not workspace["shots"]:
        raise ValueError("Creator Workspace contains no shots")
    expected = _hash({key: value for key, value in workspace.items() if key != "workspace_hash"})
    if workspace.get("workspace_hash") != expected:
        raise ValueError("Creator Workspace hash mismatch")
    return deepcopy(dict(workspace))


def select_creator_workspace_shot(
    workspace: Mapping,
    run_position: int,
    variant_index: int,
) -> tuple[dict, str, str, int, int, str, str]:
    workspace = validate_creator_workspace(workspace)
    position = int(run_position)
    if not 0 <= position < len(workspace["shots"]):
        raise ValueError(f"run_position must be between 0 and {len(workspace['shots']) - 1}")
    shot = deepcopy(workspace["shots"][position])
    variant = int(variant_index)
    if not 0 <= variant < len(shot["variant_seeds"]):
        raise ValueError(
            f"variant_index must be between 0 and {len(shot['variant_seeds']) - 1} for this shot"
        )
    source_packet = {
        "schema": "t8.video_prompt_packet.v1",
        "backend": "minimax_h3",
        "compiled_prompt": shot["effective_compiled_prompt"],
        "visual_prompt": shot["effective_compiled_prompt"],
        "audio_prompt": "",
        "negative_prompt": shot["negative_prompt"],
        "duration_seconds": shot["render_duration_seconds"],
        "aspect_ratio": "inherited_from_source_timeline",
        "dialogue": "",
        "strict_exact_dialogue": True,
        "cast_ids": [],
        "backend_note": "Creator Workspace overlay; source packet hash is retained in report.",
        "compiler_only": True,
        "quality_guarantee": False,
        "creator_workspace_hash": workspace["workspace_hash"],
        "source_packet_hash": shot["source_packet_hash"],
    }
    source_packet["packet_hash"] = _hash(source_packet)
    report = {
        "schema": CREATOR_WORKSPACE_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "run_position": position,
        "shot_index": shot["shot_index"],
        "shot_id": shot["shot_id"],
        "variant_index": variant,
        "selected_seed": shot["variant_seeds"][variant],
        "retention_policy": shot["retention_policy"],
        "media_roles": shot["media_roles"],
        "hold_policy": shot["hold_policy"],
        "hold_frames": shot["hold_frames"],
    }
    return (
        source_packet,
        shot["effective_compiled_prompt"],
        shot["negative_prompt"],
        shot["frame_count"],
        shot["variant_seeds"][variant],
        canonical_json(shot),
        canonical_json(report),
    )


def _pad_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] not in {3, 4}:
        raise ValueError("comparison inputs must be ComfyUI IMAGE batches")
    source = frames.detach().to(device="cpu", dtype=torch.float32)[..., :3].clamp(0, 1)
    output = torch.zeros((source.shape[0], height, width, 3), dtype=torch.float32)
    top = (height - source.shape[1]) // 2
    left = (width - source.shape[2]) // 2
    output[:, top : top + source.shape[1], left : left + source.shape[2], :] = source
    return output


def _header(width: int, label_a: str, label_b: str, height: int = 32) -> torch.Tensor:
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (width, height), (24, 24, 24))
        draw = ImageDraw.Draw(image)
        midpoint = width // 2
        draw.rectangle((0, 0, midpoint - 1, height - 1), fill=(31, 78, 121))
        draw.rectangle((midpoint, 0, width - 1, height - 1), fill=(126, 72, 22))
        try:
            draw.text((8, 8), f"A: {label_a}", fill="white")
            draw.text((midpoint + 8, 8), f"B: {label_b}", fill="white")
        except UnicodeEncodeError:
            draw.text((8, 8), "A", fill="white")
            draw.text((midpoint + 8, 8), "B", fill="white")
        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone()
        return raw.reshape(height, width, 3).float().div(255.0)
    except Exception:
        header = torch.full((height, width, 3), 0.1, dtype=torch.float32)
        header[:, : width // 2, 2] = 0.5
        header[:, width // 2 :, 0] = 0.5
        return header


def build_synchronized_comparison(
    frames_a: torch.Tensor,
    frames_b: torch.Tensor,
    label_a: str,
    label_b: str,
    seed_a: int,
    seed_b: int,
    winner: str,
    reviewer_notes: str,
    require_equal_geometry: bool,
) -> tuple[torch.Tensor, str, int, str]:
    if not isinstance(frames_a, torch.Tensor) or not isinstance(frames_b, torch.Tensor):
        raise ValueError("frames_a and frames_b must be IMAGE batches")
    if frames_a.ndim != 4 or frames_b.ndim != 4:
        raise ValueError("frames_a and frames_b must be IMAGE batches")
    if winner not in {"ABSTAIN", "TIE", "A", "B"}:
        raise ValueError("winner must be ABSTAIN, TIE, A or B")
    frame_count = min(int(frames_a.shape[0]), int(frames_b.shape[0]))
    if frame_count <= 0:
        raise ValueError("comparison inputs contain no frames")
    geometry_equal = tuple(frames_a.shape[:3]) == tuple(frames_b.shape[:3])
    effective_winner = "ABSTAIN" if require_equal_geometry and not geometry_equal else winner
    height = max(int(frames_a.shape[1]), int(frames_b.shape[1]))
    width = max(int(frames_a.shape[2]), int(frames_b.shape[2]))
    left = _pad_frames(frames_a[:frame_count], height, width)
    right = _pad_frames(frames_b[:frame_count], height, width)
    divider = torch.full((frame_count, height, 8, 3), 0.05, dtype=torch.float32)
    body = torch.cat((left, divider, right), dim=2)
    header = _header(int(body.shape[2]), str(label_a), str(label_b))
    output = torch.cat((header.unsqueeze(0).expand(frame_count, -1, -1, -1), body), dim=1)
    selected_seed = int(seed_a) if effective_winner == "A" else int(seed_b) if effective_winner == "B" else 0
    report = {
        "schema": CREATOR_REVIEW_SCHEMA,
        "labels": {"A": str(label_a), "B": str(label_b)},
        "seeds": {"A": int(seed_a), "B": int(seed_b)},
        "requested_winner": winner,
        "winner": effective_winner,
        "selected_seed": selected_seed or None,
        "reviewer_notes": str(reviewer_notes or "").strip(),
        "source_shapes": {"A": list(frames_a.shape), "B": list(frames_b.shape)},
        "compared_frame_count": frame_count,
        "geometry_equal": geometry_equal,
        "require_equal_geometry": bool(require_equal_geometry),
        "spatial_alignment": "center_pad_without_resize",
        "temporal_alignment": "same_zero_origin_trim_to_shorter",
        "audio_compared": False,
        "source_frames_mutated": False,
        "comparison_hash": _hash(
            {
                "labels": [str(label_a), str(label_b)],
                "seeds": [int(seed_a), int(seed_b)],
                "winner": effective_winner,
                "source_shapes": [list(frames_a.shape), list(frames_b.shape)],
                "reviewer_notes": str(reviewer_notes or "").strip(),
            }
        ),
    }
    return output, effective_winner, selected_seed, canonical_json(report)
