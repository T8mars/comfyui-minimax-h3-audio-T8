from __future__ import annotations

import json
from pathlib import Path

from tools.run_face_refine_window_memory_matrix import (
    build_case_prompt,
    matrix_groups,
    parse_args,
    summarize_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def _source_prompt():
    return json.loads(
        (ROOT / "tests" / "fixtures" / "api" / "face_refine_window_advanced_api.json").read_text(
            encoding="utf-8"
        )
    )


def test_memory_matrix_schedule_is_fifteen_strictly_serial_runs():
    groups = matrix_groups()
    flattened = [run for _name, runs in groups for run in runs]
    assert len(flattened) == 15
    for case in ("90", "124"):
        runs = [run for run in flattened if run["case"] == case]
        assert [run["phase"] for run in runs].count("cold") == 3
        assert [run["phase"] for run in runs].count("warm") == 3
    consecutive = [run for run in flattened if run["case"] == "consecutive"]
    assert [run["window_index"] for run in consecutive] == [0, 1, 2]


def test_memory_matrix_prompts_lock_90_124_and_three_distinct_windows():
    source = _source_prompt()
    prompt90 = build_case_prompt(
        source, case="90", seed=42, output_label="p90"
    )
    assert prompt90["4"]["inputs"]["min_render_frames"] == 90
    assert prompt90["5"]["inputs"]["window_index"] == 0

    prompt124 = build_case_prompt(
        source, case="124", seed=42, output_label="p124"
    )
    assert prompt124["1"]["inputs"]["file"] == (
        "face_refine_validation_dance_124_736x416.mp4"
    )
    assert prompt124["4"]["inputs"]["min_render_frames"] == 124
    assert prompt124["4"]["inputs"]["max_render_frames"] == 124
    assert prompt124["5"]["inputs"]["pad_policy"] == "reject"

    consecutive = [
        build_case_prompt(
            source,
            case="consecutive",
            seed=42,
            output_label=f"c{index}",
            window_index=index,
        )
        for index in range(3)
    ]
    assert {
        item["5"]["inputs"]["window_index"] for item in consecutive
    } == {0, 1, 2}
    assert all(
        item["4"]["inputs"]["repair_ranges"] == "0-23,124-147,248-271"
        for item in consecutive
    )
    assert all(item["25"]["inputs"]["audio"] == ["2", 1] for item in consecutive)


def test_memory_tool_defaults_to_preflight_only_and_two_gib_reserve():
    args = parse_args([])
    assert args.confirm_run is False
    assert args.mode == "single"
    assert args.reserve_vram_gib == 2.0
    assert args.post_run_seconds == 15.0
    assert args.sample_interval_seconds == 0.1
    assert args.port == 8197


def _memory_run(label: str, private_mib: float, free_mib: float = 700.0):
    return {
        "label": label,
        "terminal": {"type": "execution_success"},
        "memory": {
            "min_gpu_free_mib": free_mib,
            "observed_sample_hz": 10.0,
            "final_process_private_mib": private_mib,
        },
        "media": {"strict_decode_exit": {"video": 0, "audio": 0, "joint": 0}},
    }


def test_memory_summary_closes_full_matrix_and_staircase_gate():
    labels = [
        "01_90_cold_1",
        "02_90_cold_2",
        "03_90_cold_3",
        "04_90_warm_1",
        "05_90_warm_2",
        "06_90_warm_3",
        "07_124_cold_1",
        "08_124_cold_2",
        "09_124_cold_3",
        "10_124_warm_1",
        "11_124_warm_2",
        "12_124_warm_3",
        "13_consecutive_consecutive_1",
        "14_consecutive_consecutive_2",
        "15_consecutive_consecutive_3",
    ]
    runs = [_memory_run(label, 5000.0 + index * 2.0) for index, label in enumerate(labels)]
    summary = summarize_runs(runs, expected_run_count=15)
    assert summary["run_count_complete"] is True
    assert summary["no_staircase_growth"] is True
    assert summary["staircase_observations_complete"] is True
    assert summary["gate_pass"] is True


def test_memory_summary_rejects_missing_or_growing_matrix():
    runs = [
        _memory_run("03_90_cold_3", 5000.0),
        _memory_run("06_90_warm_3", 5300.1),
    ]
    summary = summarize_runs(runs, expected_run_count=15)
    assert summary["run_count_complete"] is False
    assert summary["staircase_observations_complete"] is False
    assert summary["no_staircase_growth"] is False
    assert summary["gate_pass"] is False
