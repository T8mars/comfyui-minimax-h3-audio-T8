"""Actual new V3 schemas; an ordering edge never invents execution evidence."""
import asyncio
import json
import math

import h3_audio_t8_pkg as package

from h3_audio_t8_pkg import nodes_projection_chunks as nodes
from test_fast_h3_v2_core_sampler import model
from test_h21_projection_model import sample


def test_new_apply_schema_has_no_arbitrary_row_ceiling_or_model_name_gate():
    schema = nodes.MiniMaxH3ProjectionChunksEXPT8.define_schema()
    fields = {item.id: item for item in schema.inputs}
    assert fields["rows"].min == fields["query_rows"].min == 1
    assert fields["rows"].max is None and fields["query_rows"].max is None
    assert schema.is_experimental and len(schema.outputs) == 2


def test_audit_accepts_ordering_dependency_but_does_not_infer_completion():
    selected, text = nodes.MiniMaxH3ProjectionChunksEXPT8.execute(model(), rows=5).result
    report = json.loads(text)
    assert report["status"] == "selected_not_executed"
    audit = nodes.MiniMaxH3ProjectionChunksAuditEXPT8.execute(selected, object())
    observed = json.loads(audit.result[0])
    assert observed["completed_diffusion_calls"] == 0
    assert not observed["quality_accepted"] and not observed["portable_cache_identity"]
    assert math.isnan(nodes.MiniMaxH3ProjectionChunksAuditEXPT8.fingerprint_inputs())


def test_apply_audit_actual_core_sampling_and_same_original_model_next_request():
    bare = model()
    selected, _ = nodes.MiniMaxH3ProjectionChunksEXPT8.execute(bare, rows=5, phase="HIGH").result
    output = sample(selected)
    audited = nodes.MiniMaxH3ProjectionChunksAuditEXPT8.execute(selected, output)
    report = json.loads(audited.result[0])
    assert report["completed_diffusion_calls"] == 2 and report["phase"] == "HIGH"
    assert report["calls"]["norm_qkv.fused"]["max_rows"] <= 5
    assert not bare.wrappers


def test_disabled_apply_retains_original_object_and_ordinary_no_selection_audit():
    bare = model()
    result, text = nodes.MiniMaxH3ProjectionChunksEXPT8.execute(bare, enabled=False).result
    assert result is bare and json.loads(text)["status"] == "disabled"
    audit = nodes.MiniMaxH3ProjectionChunksAuditEXPT8.execute(bare, {})
    assert json.loads(audit.result[0])["status"] == "not_selected"


def test_actual_registry_preserves_complete_previous_743_prefix_and_appends_only_two():
    previous = asyncio.run(package._ModularSamplingExtension().get_node_list())
    current = asyncio.run(package._ProjectionChunkExtension().get_node_list())
    assert len(previous) == 743 and current[:-2] == previous
    assert current[-2:] == nodes.NODES
    ids = [value.define_schema().node_id for value in current]
    assert len(ids) == len(set(ids)) == 745
