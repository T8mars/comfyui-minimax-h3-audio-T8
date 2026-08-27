from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from .creator_workspace_advanced import validate_creator_workspace


CREATOR_SEGMENT_CACHE_PLAN_TYPE = "H3_T8_CREATOR_SEGMENT_CACHE_PLAN"
CREATOR_SEGMENT_CACHE_SCHEMA = "t8.minimax_h3.creator_segment_cache.v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_object(value: str, name: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _cache_entries(value: str) -> list[dict[str, Any]]:
    payload = _json_object(value, "cache_index_json")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("cache_index_json.entries must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"cache entry {index} must be an object")
        entry = deepcopy(dict(raw))
        cache_key = str(entry.get("cache_key", "") or "").strip()
        semantic_hash = str(entry.get("semantic_hash", "") or "").strip()
        if not cache_key or not semantic_hash:
            raise ValueError(f"cache entry {index} needs cache_key and semantic_hash")
        if cache_key in seen:
            raise ValueError(f"duplicate cache_key: {cache_key}")
        seen.add(cache_key)
        byte_size = int(entry.get("byte_size", 0))
        last_access = float(entry.get("last_access_unix", 0.0))
        if byte_size < 0 or not math.isfinite(last_access) or last_access < 0:
            raise ValueError(f"cache entry {cache_key} has invalid size/time")
        entry.update(
            {
                "cache_key": cache_key,
                "semantic_hash": semantic_hash,
                "byte_size": byte_size,
                "last_access_unix": last_access,
                "accepted": bool(entry.get("accepted", False)),
                "artifact_manifest": deepcopy(entry.get("artifact_manifest", {})),
            }
        )
        result.append(entry)
    return result


def _shot_contract(
    workspace: Mapping[str, Any],
    shot: Mapping[str, Any],
    variant_index: int,
    model_contract: Mapping[str, Any],
    lora_contract: Mapping[str, Any],
    sampling_contract: Mapping[str, Any],
    effect_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "project_id": workspace.get("project_id"),
        "source_timeline_hash": workspace.get("source_timeline_hash"),
        "shot_id": shot.get("shot_id"),
        "shot_index": int(shot.get("shot_index", -1)),
        "frame_count": int(shot.get("frame_count", 0)),
        "prompt": str(shot.get("effective_compiled_prompt", "")),
        "negative_prompt": str(shot.get("negative_prompt", "")),
        "seed": int(shot.get("variant_seeds", [])[variant_index]),
        "media_roles": deepcopy(shot.get("media_roles", {})),
        "hold_policy": shot.get("hold_policy"),
        "hold_frames": int(shot.get("hold_frames", 0)),
        "source_packet_hash": shot.get("source_packet_hash"),
        "override_hash": shot.get("override_hash"),
        "model_contract": deepcopy(dict(model_contract)),
        "lora_contract": deepcopy(dict(lora_contract)),
        "sampling_contract": deepcopy(dict(sampling_contract)),
        "effect_plan": deepcopy(dict(effect_plan)),
    }


def compile_creator_segment_cache_plan(
    workspace: Mapping[str, Any],
    model_contract_json: str,
    lora_contract_json: str,
    sampling_contract_json: str,
    effect_plan_json: str,
    cache_index_json: str = "",
    maximum_cache_gib: float = 20.0,
    maximum_entries: int = 100,
) -> tuple[dict[str, Any], str, int, int, int, str]:
    workspace = validate_creator_workspace(workspace)
    model_contract = _json_object(model_contract_json, "model_contract_json")
    lora_contract = _json_object(lora_contract_json, "lora_contract_json")
    sampling_contract = _json_object(sampling_contract_json, "sampling_contract_json")
    effect_plan = _json_object(effect_plan_json, "effect_plan_json")
    entries = _cache_entries(cache_index_json)
    maximum_cache_gib = float(maximum_cache_gib)
    maximum_entries = int(maximum_entries)
    if not math.isfinite(maximum_cache_gib) or maximum_cache_gib <= 0:
        raise ValueError("maximum_cache_gib must be finite and positive")
    if maximum_entries < 1:
        raise ValueError("maximum_entries must be positive")
    maximum_bytes = round(maximum_cache_gib * 1024**3)

    desired: dict[str, dict[str, Any]] = {}
    for shot in workspace["shots"]:
        for variant_index in range(len(shot["variant_seeds"])):
            contract = _shot_contract(
                workspace,
                shot,
                variant_index,
                model_contract,
                lora_contract,
                sampling_contract,
                effect_plan,
            )
            cache_key = f"{shot['shot_id']}:{variant_index}"
            desired[cache_key] = {
                "cache_key": cache_key,
                "shot_id": str(shot["shot_id"]),
                "shot_index": int(shot["shot_index"]),
                "variant_index": variant_index,
                "semantic_hash": _hash(contract),
                "semantic_contract": contract,
            }

    by_key = {entry["cache_key"]: entry for entry in entries}
    desired_rows: list[dict[str, Any]] = []
    hit_count = 0
    stale_count = 0
    miss_count = 0
    for cache_key, wanted in desired.items():
        existing = by_key.get(cache_key)
        if existing is None:
            status = "miss"
            miss_count += 1
        elif existing["semantic_hash"] == wanted["semantic_hash"]:
            status = "hit"
            hit_count += 1
        else:
            status = "stale_contract_changed"
            stale_count += 1
        desired_rows.append(
            {
                **wanted,
                "status": status,
                "existing": None if existing is None else deepcopy(existing),
            }
        )

    orphan_entries = [entry for entry in entries if entry["cache_key"] not in desired]
    stale_entries = [
        row["existing"]
        for row in desired_rows
        if row["status"] == "stale_contract_changed" and row["existing"] is not None
    ]
    mandatory_quarantine = [*orphan_entries, *stale_entries]
    protected = [entry for entry in entries if entry["accepted"]]
    candidates = [
        entry for entry in entries
        if not entry["accepted"] and entry not in mandatory_quarantine
    ]
    quarantine_by_key = {entry["cache_key"]: entry for entry in mandatory_quarantine if not entry["accepted"]}
    retained = [entry for entry in entries if entry["cache_key"] not in quarantine_by_key]
    retained_bytes = sum(entry["byte_size"] for entry in retained)
    retained_count = len(retained)
    for entry in sorted(candidates, key=lambda item: (item["last_access_unix"], item["cache_key"])):
        if retained_count <= maximum_entries and retained_bytes <= maximum_bytes:
            break
        if entry["cache_key"] in quarantine_by_key:
            continue
        quarantine_by_key[entry["cache_key"]] = entry
        retained_count -= 1
        retained_bytes -= entry["byte_size"]

    accepted_over_limit = (
        len(protected) > maximum_entries
        or sum(entry["byte_size"] for entry in protected) > maximum_bytes
    )
    status = "ABSTAIN_ACCEPTED_ARTIFACTS_EXCEED_LIMIT" if accepted_over_limit else "PLAN_READY"
    quarantine = [deepcopy(quarantine_by_key[key]) for key in sorted(quarantine_by_key)]
    plan = {
        "schema": CREATOR_SEGMENT_CACHE_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "status": status,
        "limits": {"maximum_entries": maximum_entries, "maximum_bytes": maximum_bytes},
        "contracts": {
            "model": model_contract,
            "lora": lora_contract,
            "sampling": sampling_contract,
            "effect_plan": effect_plan,
        },
        "desired_entries": desired_rows,
        "existing_entry_count": len(entries),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "stale_count": stale_count,
        "orphan_count": len(orphan_entries),
        "protected_accepted_count": len(protected),
        "proposed_quarantine": quarantine,
        "post_plan_retained_count": retained_count,
        "post_plan_retained_bytes": retained_bytes,
        "files_opened": False,
        "files_mutated": False,
        "files_deleted": False,
        "quarantine_executor_called": False,
        "execution_contract": (
            "review this plan, then route proposed entries through the existing hash-bound "
            "Creator retention/quarantine workflow; never delete accepted artifacts or receipts"
        ),
    }
    plan["plan_hash"] = _hash(plan)
    report = {
        "schema": CREATOR_SEGMENT_CACHE_SCHEMA,
        "status": status,
        "workspace_hash": workspace["workspace_hash"],
        "plan_hash": plan["plan_hash"],
        "hit_count": hit_count,
        "miss_count": miss_count,
        "stale_count": stale_count,
        "orphan_count": len(orphan_entries),
        "proposed_quarantine_count": len(quarantine),
        "accepted_artifacts_protected": True,
        "semantic_scope": [
            "source",
            "model",
            "lora",
            "prompt",
            "sampling",
            "effect_plan",
        ],
        "side_effects": False,
    }
    return (
        plan,
        status,
        hit_count,
        stale_count + len(orphan_entries),
        len(quarantine),
        json.dumps(report, ensure_ascii=False, indent=2),
    )
