"""CPU contracts for owned Chunked v3/v4 bad-receipt GPU probes."""

from copy import deepcopy
import json

import pytest

from tools import run_modular_chunked_v34_gpu as base
from tools import run_modular_chunked_v34_negative_gpu as probe


@pytest.mark.parametrize("variant", ["v3", "v4"])
def test_negative_copies_change_exactly_one_load_receipt_field(variant):
    control = base.build_graph(variant, "resume", width=128, height=128)
    spec = base.CONFIG[variant]
    manifest = {"schema": "example", "checkpoint_id": "owned_first_pass"}
    receipt = {"path": "owned.safetensors",
               "manifest_json": json.dumps(manifest, sort_keys=True, separators=(",", ":")),
               "file_sha256": "A" * 64}
    control[spec["load"]]["inputs"].update(
        checkpoint_path=receipt["path"],
        expected_manifest_json=receipt["manifest_json"],
        expected_file_sha256=receipt["file_sha256"],
    )
    original = deepcopy(control)
    graphs = probe.negative_graphs(variant, control, receipt)
    assert control == original
    assert set(graphs) == set(probe.CASES)
    for case, graph in graphs.items():
        assert spec["low"] not in graph
        expected = deepcopy(original)
        if case == "bad_file_sha":
            expected[spec["load"]]["inputs"]["expected_file_sha256"] = "0" * 64
        elif case == "bad_external_manifest":
            changed = json.loads(receipt["manifest_json"])
            changed["checkpoint_id"] = "intentionally_wrong_checkpoint_id"
            expected[spec["load"]]["inputs"]["expected_manifest_json"] = (
                json.dumps(changed, sort_keys=True, separators=(",", ":")))
        else:
            expected[spec["load"]]["inputs"]["checkpoint_path"] = (
                "../not_an_owned_checkpoint.safetensors")
        assert graph == expected


@pytest.mark.parametrize("variant", ["v3", "v4"])
@pytest.mark.parametrize("case", list(probe.CASES))
def test_rejection_gate_requires_precise_load_error_and_no_sampler(variant, case):
    spec = base.CONFIG[variant]
    graph = base.build_graph(variant, "resume", width=128, height=128)
    phase = {"terminal": {"type": "execution_error", "data": {
        "node_id": spec["load"], "exception_type": "ValueError",
        "exception_message": probe.CASES[case],
    }}, "events": [{"type": "executing", "node": spec["load"]}]}
    assert all(probe.audit_rejection(variant, case, graph, phase).values())
    for changed in (
        {"terminal": {"type": "execution_success", "data": phase["terminal"]["data"]}},
        {"events": [{"type": "executing", "node": spec["high"]}]},
        {"events": [{"type": "progress", "node": spec["high"]}]},
    ):
        invalid = deepcopy(phase)
        invalid.update(changed)
        assert not all(probe.audit_rejection(variant, case, graph, invalid).values())
