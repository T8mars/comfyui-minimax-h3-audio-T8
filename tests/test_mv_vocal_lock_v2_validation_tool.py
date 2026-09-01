from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "run_mv_vocal_lock_v2_validation.py"
V3_BUILDER_PATH = PROJECT_ROOT / "tools" / "build_mv_vocal_lock_v3_long32_validation.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "run_mv_vocal_lock_v2_validation", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _load_v3_builder():
    spec = importlib.util.spec_from_file_location(
        "build_mv_vocal_lock_v3_long32_validation", V3_BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3_builder = _load_v3_builder()


def _record(event: str, *, completed: bool, status_str: str) -> dict:
    return {
        "status": {
            "status_str": status_str,
            "completed": completed,
            "messages": [[event, {"node_id": "11"}]],
        }
    }


def test_execution_outcome_accepts_only_completed_success():
    assert tool._execution_outcome(
        _record("execution_success", completed=True, status_str="success")
    ) == ("completed_waiting_media_audit", "execution_success", "")


def test_execution_outcome_reports_interruption_as_non_success():
    status, event, detail = tool._execution_outcome(
        _record("execution_interrupted", completed=False, status_str="error")
    )
    assert status == "interrupted"
    assert event == "execution_interrupted"
    assert '"node_id": "11"' in detail


def test_execution_outcome_reports_execution_error_and_incomplete_history():
    status, event, detail = tool._execution_outcome(
        _record("execution_error", completed=False, status_str="error")
    )
    assert status == "failed"
    assert event == "execution_error"
    assert '"node_id": "11"' in detail

    status, event, detail = tool._execution_outcome(
        {"status": {"status_str": "error", "completed": False, "messages": []}}
    )
    assert status == "failed"
    assert event == "history_incomplete"
    assert '"completed": false' in detail


def test_required_node_contracts_accept_v2_or_v3_but_not_unrelated_nodes():
    v2 = {
        "9": {"class_type": "MiniMaxH3MVVocalLockScenePlannerV2T8Advanced"},
        "10": {"class_type": "MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced"},
        "11": {"class_type": "MiniMaxH3LocalMVVocalLockRendererV2T8Advanced"},
    }
    assert tool._required_node_contracts(v2) == tuple(
        v2[str(index)]["class_type"] for index in (9, 10, 11)
    )
    v3 = {
        **v2,
        "10": {"class_type": "MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced"},
        "11": {"class_type": "MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced"},
    }
    assert tool._required_node_contracts(v3) == tuple(
        v3[str(index)]["class_type"] for index in (9, 10, 11)
    )
    v3["11"] = {"class_type": "SaveVideo"}
    try:
        tool._required_node_contracts(v3)
    except ValueError as error:
        assert "validation workflow node 11" in str(error)
    else:
        raise AssertionError("unrelated renderer must be rejected")


def test_v3_validation_builder_uses_exact_official_ref2v_recipe_and_new_chain():
    workflow = v3_builder.build()
    assert workflow["2"]["inputs"] == {
        "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "strength_model": 1.0,
        "model": ["1", 0],
    }
    renderer = workflow["11"]["inputs"]
    assert renderer["chain_id"].endswith("r4_official_ref2v")
    assert (renderer["width"], renderer["height"]) == (1024, 768)
    assert (renderer["steps"], renderer["shift_video"], renderer["shift_audio"]) == (
        4,
        12.0,
        3.0,
    )
    assert (renderer["sampler_name"], renderer["scheduler"]) == ("euler", "simple")


def test_v3_scene02_probe_keeps_failed_scene_seed_and_isolates_one_audio_window():
    workflow = v3_builder.build_scene02_probe()
    assert workflow["7"]["inputs"]["audio"] == "mv_vocal_lock_scene02_full_mix_zh.wav"
    assert workflow["8"]["inputs"]["audio"] == "mv_vocal_lock_scene02_vocal_zh.wav"
    assert workflow["9"]["inputs"]["manual_boundaries_json"] == ""
    directions = __import__("json").loads(
        workflow["10"]["inputs"]["scene_directions_json"]
    )
    assert len(directions) == 1
    assert "left three-quarter" in directions[0]["camera"]
    renderer = workflow["11"]["inputs"]
    assert renderer["base_seed"] == 2609013202
    assert renderer["chain_id"].endswith("official_ref2v_same_seed")
