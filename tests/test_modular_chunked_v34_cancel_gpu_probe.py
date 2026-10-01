"""CPU contracts for owned Chunked v3/v4 cancellation and HIGH-only retry."""

from copy import deepcopy

import pytest

from tools import run_modular_chunked_v34_cancel_gpu as probe
from tools import run_modular_chunked_v34_gpu as base


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_cancel_graphs_change_only_private_output_name(variant):
    control = base.build_graph(variant, "resume", width=128, height=128)
    original = deepcopy(control)
    for name in ("cancel_attempt", "cancel_retry"):
        changed = probe.output_graph(variant, control, prefix=name)
        expected = deepcopy(control)
        expected[base.CONFIG[variant]["video"]]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/Chunked_{variant}_Real/{name}")
        assert changed == expected
        assert base.CONFIG[variant]["low"] not in changed
    assert control == original
    with pytest.raises(ValueError):
        probe.output_graph(variant, control, prefix="resume")


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_cancel_audit_requires_real_interrupt_and_fresh_high_three_steps(variant):
    spec = base.CONFIG[variant]
    graph = base.build_graph(variant, "resume", width=128, height=128)
    interrupted = {
        "interrupt_response": {}, "progress_at_interrupt": {"value": 1, "max": 3},
        "terminal": {"type": "execution_interrupted"},
        "events": [{"type": "executing", "node": spec["high"]},
                   {"type": "progress", "node": spec["high"]}],
    }
    retried = {"terminal": {"type": "execution_success"},
               "events": [{"type": "progress", "node": spec["high"]}] * 3}
    assert all(probe.phase_checks(variant, interrupted, retried, graph, graph).values())
    for value in (
        {"interrupt_response": None},
        {"terminal": {"type": "execution_success"}},
        {"progress_at_interrupt": {"value": 3, "max": 3}},
        {"events": [{"type": "progress", "node": spec["low"]}]},
    ):
        bad = deepcopy(interrupted)
        bad.update(value)
        assert not all(probe.phase_checks(variant, bad, retried, graph, graph).values())
    bad_retry = deepcopy(retried)
    bad_retry["events"].pop()
    assert not all(probe.phase_checks(
        variant, interrupted, bad_retry, graph, graph).values())


def test_audio_decode_rejects_unsupported_stream():
    with pytest.raises(ValueError):
        probe.decoded_md5(None, "other")
