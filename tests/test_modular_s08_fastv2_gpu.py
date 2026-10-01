"""Guard the isolated real-weight S08 probe without invoking a GPU."""

from copy import deepcopy
import json

from tools import run_modular_s08_fastv2_gpu as probe


def test_probe_graph_only_shrinks_private_execution_fixture():
    original = json.loads(probe.CANDIDATE.read_text(encoding="utf-8"))
    graph, source_sha = probe.build_probe_graph()
    expected = deepcopy(original)
    expected["9"]["inputs"].update(width=128, height=64)
    expected["10"]["inputs"]["min_tokens"] = 0
    expected["26"]["inputs"]["min_tokens"] = 0
    expected["40"]["inputs"]["length"] = 22
    expected["47"]["inputs"]["length"] = 22
    expected["16"]["inputs"]["filename_prefix"] = (
        "MiniMaxH3/S08_FastV2_RealSplit/full_selected"
    )
    assert graph == expected
    assert source_sha == probe.shared._sha256_file(probe.CANDIDATE)
    assert original == json.loads(probe.CANDIDATE.read_text(encoding="utf-8"))


def test_probe_graph_rejects_invalid_canvas():
    for size in ((0, 64, 22), (128, 63, 22), (128, 64, 0)):
        try:
            probe.build_probe_graph(width=size[0], height=size[1], frames=size[2])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid canvas {size}")


def test_stage_checks_require_true_order_and_all_eight_progress_events():
    phase = {
        "terminal": {"type": "execution_success"},
        "events": [
            *({"type": "executing", "node": node} for node in ("13", "50", "22", "23", "29", "51")),
            *({"type": "progress", "node": node} for node in ("13",) * 4 + ("29",) * 4),
        ],
    }
    assert all(probe._stage_checks(phase).values())
    bad = deepcopy(phase)
    bad["events"] = [event for event in bad["events"] if not (
        event["type"] == "executing" and event["node"] == "22"
    )]
    assert not probe._stage_checks(bad)["verified_low_before_independent_high_loader_and_trained_lift"]
    bad = deepcopy(phase)
    bad["events"][-1]["node"] = "13"
    assert not probe._stage_checks(bad)["low_four_then_high_four"]
