from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.nodes_studio_advanced import STUDIO_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.nodes_studio_advanced import (
    MiniMaxH3StudioTimelineT8Advanced,
)
from h3_audio_t8_pkg.studio_advanced import (
    CAST_SCHEMA,
    PROMPT_PACKET_SCHEMA,
    REPAIR_PLAN_SCHEMA,
    SOUND_CANVAS_SCHEMA,
    STUDIO_TIMELINE_SCHEMA,
    build_selective_repair_plan,
    build_sound_canvas,
    build_studio_timeline,
    build_unified_cast,
    compile_prompt_packet,
    select_repair_segment,
    select_studio_shot,
)


def _cast():
    return build_unified_cast(
        json.dumps([
            {
                "id": "hero",
                "name": "Lin",
                "visual_identity": "oval face, short black hair, small scar above left eyebrow",
                "wardrobe_default": "dark blue field jacket",
                "forbidden_changes": ["face geometry", "scar side"],
            }
        ]),
        True,
    )


def _canvas(duration=12.0):
    return build_sound_canvas(
        json.dumps([
            {
                "id": "line_1",
                "role": "dialogue",
                "start_seconds": 0.5,
                "end_seconds": 2.5,
                "speaker_id": "hero",
                "exact_text": "We leave now.",
            },
            {
                "id": "rain",
                "role": "ambience",
                "start_seconds": 0.0,
                "end_seconds": duration,
                "description": "steady rain on a metal roof",
            },
        ]),
        duration,
        True,
        False,
    )


def _timeline(shots=None, **overrides):
    values = {
        "project_id": "demo",
        "shots_json": json.dumps(shots or [
            {"id": "a", "prompt": "Lin walks through the station", "duration_seconds": 5.0, "cast_ids": ["hero"]},
            {"id": "b", "prompt": "A train arrives", "duration_seconds": 4.0},
        ]),
        "default_backend": "minimax_h3",
        "default_duration_seconds": 5.0,
        "default_aspect_ratio": "16:9",
        "base_seed": 100,
        "seed_policy": "increment",
        "split_long_shots": True,
        "strict_exact_dialogue": True,
        "cast": _cast(),
        "sound_canvas": None,
    }
    values.update(overrides)
    return build_studio_timeline(**values)


def test_unified_cast_is_deterministic_and_rejects_duplicates():
    first = _cast()
    second = _cast()
    assert first["schema"] == CAST_SCHEMA
    assert first["cast_hash"] == second["cast_hash"]
    assert "scar side" in first["prompt_fragment"]
    with pytest.raises(ValueError, match="duplicate character id"):
        build_unified_cast(
            '[{"id":"a","visual_identity":"x"},{"id":"a","visual_identity":"y"}]'
        )


def test_sound_canvas_keeps_non_speech_bed_after_exact_dialogue():
    canvas = _canvas()
    assert canvas["schema"] == SOUND_CANVAS_SCHEMA
    assert "say exactly: We leave now." in canvas["h3_audio_prompt"]
    assert "continue only the requested music, ambience, and sound effects" in canvas["h3_audio_prompt"]
    assert canvas["source_separation_claim"] is False
    assert canvas["exact_timing_claim"] is False


def test_sound_canvas_dialogue_overlap_is_fail_closed_by_default():
    events = [
        {"id": "a", "role": "dialogue", "start_seconds": 0, "end_seconds": 2, "exact_text": "A"},
        {"id": "b", "role": "dialogue", "start_seconds": 1, "end_seconds": 3, "exact_text": "B"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        build_sound_canvas(json.dumps(events), 4, True, False)
    allowed = build_sound_canvas(json.dumps(events), 4, True, True)
    assert allowed["warnings"]


def test_prompt_compiler_keeps_h3_audio_but_wan_uses_sidecar_only():
    h3 = compile_prompt_packet(
        "A restrained close-up",
        "minimax_h3",
        5.167,
        "16:9",
        "We leave now.",
        "plastic skin",
        True,
        _cast(),
        ["hero"],
        _canvas(),
        {"camera": "slow push-in"},
    )
    assert h3["schema"] == PROMPT_PACKET_SCHEMA
    assert "Audio direction" in h3["compiled_prompt"]
    assert "Do not add" in h3["audio_prompt"]
    assert "slow push-in" in h3["compiled_prompt"]
    wan = compile_prompt_packet("A restrained close-up", "wan_2_2", 5, "16:9", sound_canvas=_canvas())
    assert "Audio direction" not in wan["compiled_prompt"]
    assert wan["audio_prompt"]
    assert "sidecar" in wan["backend_note"]


def test_studio_timeline_quantizes_h3_grid_and_keeps_deterministic_order():
    timeline = _timeline()
    assert timeline["schema"] == STUDIO_TIMELINE_SCHEMA
    assert [shot["seed"] for shot in timeline["shots"]] == [100, 101]
    assert all((shot["frame_count"] - 5) % 17 == 0 for shot in timeline["shots"])
    assert timeline["shots"][1]["start_frame"] == timeline["shots"][0]["frame_count"]
    assert timeline["timeline_hash"] == _timeline()["timeline_hash"]


def test_long_visual_shot_splits_but_long_dialogue_requires_author_boundaries():
    visual = _timeline(
        shots=[{"id": "long", "prompt": "A continuous journey", "duration_seconds": 30.0}],
        cast=None,
    )
    assert visual["shot_count"] == 2
    assert all(shot["part_count"] == 2 for shot in visual["shots"])
    assert all(shot["frame_count"] <= 362 for shot in visual["shots"])
    with pytest.raises(ValueError, match="split it explicitly"):
        _timeline(
            shots=[{"id": "long", "prompt": "A speech", "dialogue": "A long exact line", "duration_seconds": 30}],
            cast=None,
        )


def test_shot_select_exposes_existing_conditioning_inputs():
    shot, report = select_studio_shot(_timeline(), 1)
    assert shot["id"] == "b"
    assert shot["frame_count"] == report["frame_count"]
    assert report["is_last"] is True
    with pytest.raises(ValueError, match="shot_index"):
        select_studio_shot(_timeline(), 4)


def test_selective_repair_is_non_destructive_and_auto_routes_identity():
    timeline = _timeline()
    before = deepcopy(timeline)
    quality = {
        "segments": [
            {"index": 0, "status": "pass", "scores": {"identity": 0.9}},
            {"index": 1, "status": "failed", "scores": {"identity": 0.4}, "issues": ["identity drift"]},
        ]
    }
    plan = build_selective_repair_plan(
        timeline,
        json.dumps(quality),
        "failed_or_score",
        "",
        '{"identity":{"min":0.75}}',
        "auto",
        "restore the accepted character identity",
        1009,
        22,
        22,
    )
    assert plan["schema"] == REPAIR_PLAN_SCHEMA
    assert plan["selected_indices"] == [1]
    assert plan["unchanged_indices"] == [0]
    assert plan["repairs"][0]["mode"] == "reference_refresh"
    assert plan["accepted_media_mutated"] is False
    assert timeline == before
    repair, report = select_repair_segment(plan, 0)
    assert repair["shot_id"] == "b"
    assert report["is_last"] is True


def test_manual_repair_ranges_and_empty_evidence_failures_are_explicit():
    timeline = _timeline()
    plan = build_selective_repair_plan(
        timeline, "", "manual", "0-1", "{}", "seed_retry", "", 17, 0, 0
    )
    assert plan["repair_count"] == 2
    with pytest.raises(ValueError, match="no segment evidence"):
        build_selective_repair_plan(
            timeline, "", "failed_status", "", "{}", "auto", "", 17, 0, 0
        )


def test_studio_timeline_emits_bounded_read_only_ui_preview():
    result = MiniMaxH3StudioTimelineT8Advanced.execute(
        project_id="ui_preview",
        shots_json='[{"id":"a","prompt":"A short shot","duration_seconds":1}]',
        default_backend="minimax_h3",
        default_duration_seconds=5.0,
        default_aspect_ratio="16:9",
        base_seed=11,
        seed_policy="increment",
        split_long_shots=True,
        strict_exact_dialogue=True,
    )
    payload = json.loads(result.ui["t8_studio_timeline"][0])
    assert payload["project_id"] == "ui_preview"
    assert payload["shot_count"] == 1
    assert payload["shots"][0]["id"] == "a"
    assert set(payload["shots"][0]) == {
        "index",
        "id",
        "start_seconds",
        "end_seconds",
        "frame_count",
        "seed",
        "backend",
        "status",
        "prompt",
    }


def test_studio_timeline_frontend_uses_text_nodes_for_untrusted_fields():
    source = (
        Path(__file__).resolve().parents[1] / "web" / "studio_timeline_v2.js"
    ).read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent = String(shot.id" in source


@pytest.mark.parametrize(
    ("api_name", "workflow_name"),
    [
        ("environment_audit_advanced_api.json", "H3_Environment_Audit_Advanced.json"),
        ("activation_chunk_advanced_api.json", "H3_Activation_Chunk_Advanced.json"),
        ("qwen_prefix_cache_advanced_api.json", "H3_Qwen_Prefix_Cache_Advanced.json"),
        ("studio_timeline_advanced_api.json", "H3_Studio_Timeline_Advanced.json"),
        ("context_ir_provider_advanced_api.json", "H3_Context_IR_Provider_Advanced.json"),
        ("reel_delivery_advanced_api.json", "H3_Reel_Delivery_Advanced.json"),
        ("av_decode_safety_advanced_api.json", "H3_AV_Decode_Safety_Advanced.json"),
        ("scheduled_audio_injection_advanced_api.json", "H3_Scheduled_Audio_Injection_Advanced_EXP.json"),
        ("trajectory_probe_advanced_api.json", "H3_Trajectory_Probe_Advanced_EXP.json"),
        ("selective_repair_execution_advanced_api.json", "H3_Selective_Repair_Execution_Advanced.json"),
    ],
)
def test_new_api_and_frontend_examples_are_link_consistent(api_name, workflow_name):
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (root / "tests" / "fixtures" / "api" / api_name).read_text(encoding="utf-8")
    )
    workflow = json.loads((root / "examples" / "workflows" / workflow_name).read_text(encoding="utf-8"))
    assert api
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == len(workflow["links"])
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_new_api_examples_keep_safe_defaults_and_explicit_sidecar_boundary():
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api"
    environment = json.loads((root / "environment_audit_advanced_api.json").read_text(encoding="utf-8"))
    activation = json.loads((root / "activation_chunk_advanced_api.json").read_text(encoding="utf-8"))
    qwen = json.loads((root / "qwen_prefix_cache_advanced_api.json").read_text(encoding="utf-8"))
    studio = json.loads((root / "studio_timeline_advanced_api.json").read_text(encoding="utf-8"))
    assert environment["1"]["inputs"]["enforcement"] == "report_only"
    assert activation["2"]["inputs"]["mode"] == "report_only"
    assert qwen["2"]["inputs"]["mode"] == "report_only"
    assert studio["5"]["inputs"]["selection_policy"] == "manual"
    assert studio["7"]["inputs"]["backend"] == "wan_2_2"
    assert "audio remains sidecar" in studio["7"]["_meta"]["title"]
    context_ir = json.loads((root / "context_ir_provider_advanced_api.json").read_text(encoding="utf-8"))
    reel = json.loads((root / "reel_delivery_advanced_api.json").read_text(encoding="utf-8"))
    decode = json.loads((root / "av_decode_safety_advanced_api.json").read_text(encoding="utf-8"))
    injection = json.loads((root / "scheduled_audio_injection_advanced_api.json").read_text(encoding="utf-8"))
    trajectory = json.loads((root / "trajectory_probe_advanced_api.json").read_text(encoding="utf-8"))
    repair = json.loads((root / "selective_repair_execution_advanced_api.json").read_text(encoding="utf-8"))
    assert context_ir["1"]["inputs"]["provider_mode"] == "validate_local"
    assert context_ir["1"]["inputs"]["confirm_external_upload"] is False
    assert reel["2"]["inputs"]["confirm_compose"] is False
    assert decode["4"]["inputs"]["mode"] == "preflight_only"
    assert injection["7"]["inputs"]["mode"] == "report_only"
    assert trajectory["11"]["inputs"]["confirm_save"] is False
    assert "MiniMaxH3TrajectoryCheckpointLoadT8Advanced" not in {
        node["class_type"] for node in trajectory.values()
    }
    assert repair["3"]["inputs"]["accept_repair"] is False


def test_all_studio_node_schemas_are_append_only_experimental_contracts():
    schemas = [node.define_schema() for node in STUDIO_ADVANCED_NODE_CLASSES]
    assert len(schemas) == 7
    assert len({schema.node_id for schema in schemas}) == 7
    assert all(schema.is_experimental for schema in schemas)
    assert all(schema.category == "T8/MiniMax H3/Studio/Experimental" for schema in schemas)
    timeline = schemas[3]
    timeline_inputs = {item.id: item for item in timeline.inputs}
    assert timeline_inputs["split_long_shots"].default is True
    repair = schemas[5]
    repair_inputs = {item.id: item for item in repair.inputs}
    assert repair_inputs["selection_policy"].default == "manual"
