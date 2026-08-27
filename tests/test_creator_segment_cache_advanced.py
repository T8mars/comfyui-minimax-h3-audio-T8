from __future__ import annotations

import json

from h3_audio_t8_pkg.creator_segment_cache_advanced import (
    compile_creator_segment_cache_plan,
)
from h3_audio_t8_pkg.creator_workspace_advanced import _hash


def _workspace():
    workspace = {
        "schema": "t8.minimax_h3.creator_workspace.v1",
        "project_id": "p",
        "source_timeline_hash": "timeline",
        "shots": [
            {
                "run_position": 0,
                "shot_index": 0,
                "shot_id": "shot-a",
                "frame_count": 124,
                "effective_compiled_prompt": "A",
                "negative_prompt": "",
                "variant_seeds": [1, 2],
                "media_roles": {},
                "hold_policy": "none",
                "hold_frames": 0,
                "source_packet_hash": "source-a",
                "override_hash": None,
            },
            {
                "run_position": 1,
                "shot_index": 1,
                "shot_id": "shot-b",
                "frame_count": 124,
                "effective_compiled_prompt": "B",
                "negative_prompt": "",
                "variant_seeds": [3],
                "media_roles": {},
                "hold_policy": "none",
                "hold_frames": 0,
                "source_packet_hash": "source-b",
                "override_hash": None,
            },
        ],
    }
    workspace["workspace_hash"] = _hash(workspace)
    return workspace


def _compile(workspace, index="", **kwargs):
    return compile_creator_segment_cache_plan(
        workspace,
        '{"model":"h3"}',
        '{"lora":"turbo8"}',
        '{"steps":8}',
        '{"relay":false}',
        index,
        **kwargs,
    )


def test_per_shot_semantic_hash_hits_and_only_changed_shot_invalidates():
    workspace = _workspace()
    plan, *_ = _compile(workspace)
    entries = []
    for row in plan["desired_entries"]:
        entries.append(
            {
                "cache_key": row["cache_key"],
                "semantic_hash": row["semantic_hash"],
                "byte_size": 100,
                "last_access_unix": 1,
            }
        )
    _plan, status, hits, invalid, _quarantine, _report = _compile(
        workspace, json.dumps({"entries": entries})
    )
    assert status == "PLAN_READY" and hits == 3 and invalid == 0

    changed = _workspace()
    changed["shots"][1]["effective_compiled_prompt"] = "B changed"
    changed["workspace_hash"] = _hash({k: v for k, v in changed.items() if k != "workspace_hash"})
    _plan, _status, hits, invalid, quarantine, _report = _compile(
        changed, json.dumps({"entries": entries})
    )
    assert hits == 2 and invalid == 1 and quarantine == 1


def test_orphan_and_lru_propose_quarantine_but_accepted_is_never_deleted():
    workspace = _workspace()
    seed_plan, *_ = _compile(workspace)
    entries = [
        {
            "cache_key": row["cache_key"],
            "semantic_hash": row["semantic_hash"],
            "byte_size": 100,
            "last_access_unix": index + 1,
            "accepted": index == 0,
        }
        for index, row in enumerate(seed_plan["desired_entries"])
    ]
    entries.append(
        {
            "cache_key": "deleted-shot:0",
            "semantic_hash": "old",
            "byte_size": 100,
            "last_access_unix": 0,
        }
    )
    plan, _status, _hits, invalid, quarantine, _report = _compile(
        workspace,
        json.dumps({"entries": entries}),
        maximum_cache_gib=0.0000002,
        maximum_entries=2,
    )
    keys = {item["cache_key"] for item in plan["proposed_quarantine"]}
    assert invalid == 1 and quarantine >= 2
    assert "shot-a:0" not in keys
    assert "deleted-shot:0" in keys
    assert plan["files_deleted"] is False
