"""CPU contracts for v3/v4 HIGH-only noise-edit cache probe."""

from copy import deepcopy

import pytest

from tools import run_modular_chunked_v34_cache_gpu as probe
from tools import run_modular_chunked_v34_gpu as base


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_cache_graphs_only_edit_high_noise_and_owned_media_prefix(variant):
    control = base.build_graph(variant, "resume", width=128, height=128)
    original = deepcopy(control)
    graphs = probe.cache_graphs(variant, control)
    assert tuple(graphs) == probe.CASES
    assert control == original
    for case, graph in graphs.items():
        expected = deepcopy(control)
        expected[base.CONFIG[variant]["video"]]["inputs"]["filename_prefix"] = (
            f"MiniMaxH3/Chunked_{variant}_Real/cache_v1_{case}")
        if case in {"seed_edit", "seed_repeat"}:
            expected["16"]["inputs"]["noise_seed"] += 1
        assert graph == expected
        assert base.CONFIG[variant]["low"] not in graph


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_phase_gate_requires_correct_reuse_and_invalidation(variant):
    high = base.CONFIG[variant]["high"]
    phases = {
        case: {"terminal": {"type": "execution_success"},
               "events": [{"type": "progress", "node": high}] * count}
        for case, count in probe.EXPECTED_HIGH_STEPS.items()
    }
    assert all(probe.phase_checks(variant, phases).values())
    bad = deepcopy(phases)
    bad["repeat"]["events"] = [{"type": "progress", "node": high}]
    assert not all(probe.phase_checks(variant, bad).values())
    bad = deepcopy(phases)
    bad["seed_edit"]["events"] = []
    assert not all(probe.phase_checks(variant, bad).values())
    bad = deepcopy(phases)
    bad["seed_repeat"]["events"] = [
        {"type": "progress", "node": base.CONFIG[variant]["low"]}]
    assert not all(probe.phase_checks(variant, bad).values())
