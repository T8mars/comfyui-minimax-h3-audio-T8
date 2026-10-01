"""Explicit I/O only: original sampling, audio, effects, approvals and old files remain exact."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools import build_formal_face_anime_isolated_delivery_workflows as builder
from tools import build_formal_face_refine_storage_workflows as storage
from h3_audio_t8_pkg.modular_sampling.video_io_nodes import MiniMaxH3SaveVideoIsolatedEXPT8


@pytest.mark.parametrize('mode', storage.MODES)
def test_explicit_anime_pair_changes_only_writer_and_appends_its_report(mode):
    original_files = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in storage.face.DESTINATION.glob('*.json')}
    assert len(original_files) == 84
    old_frontend, old_api = storage.graph_for('anime', 'combined', mode)
    frozen_frontend, frozen_api = deepcopy(old_frontend), deepcopy(old_api)
    frontend, api = builder.graph_for(mode)
    key = next(key for key, node in old_api.items() if node['class_type'] == 'SaveVideo')
    added = set(api) - set(old_api)
    assert len(added) == 1
    report = added.pop()
    assert api[report] == {'class_type': 'PreviewAny', 'inputs': {'source': [key, 2]}}
    assert api[key] == {'class_type': builder.WRITER, 'inputs': {
        'video': old_api[key]['inputs']['video'],
        'filename_prefix': old_api[key]['inputs']['filename_prefix'],
        'timeout_seconds': 600, 'max_staging_gib': 16., 'min_free_disk_gib': 2.}}
    assert {k: v for k, v in api.items() if k not in (key, report)} == {
        k: v for k, v in old_api.items() if k != key}
    old_nodes = {node['id']: node for node in old_frontend['nodes']}
    new_nodes = {node['id']: node for node in frontend['nodes']}
    assert all(new_nodes[node] == value for node, value in old_nodes.items() if str(node) != key)
    assert frontend['links'][:-1] == old_frontend['links']
    assert frontend['links'][-1][1:5] == [int(key), 2, int(report), 0]
    assert frontend['id'] != old_frontend.get('id')
    assert frontend['extra']['t8_split_example']['delivery'] == 'explicit_isolated_h264'
    assert sum(node['class_type'] == 'MiniMaxH3StageSamplerEXPT8' for node in api.values()) == (
        1 if mode == 'full_save' else 0)
    assert sum(node['class_type'] == 'MiniMaxH3StageLoadEXPT8' for node in api.values()) == (
        0 if mode == 'full_save' else 1)
    assert old_frontend == frozen_frontend and old_api == frozen_api
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in original_files.items())


def test_pair_uses_existing_public_writer_defaults_and_has_no_matrix():
    schema = MiniMaxH3SaveVideoIsolatedEXPT8.define_schema()
    assert schema.node_id == builder.WRITER
    assert list(builder.FILES) == ['full_save', 'cold_delivery']
    assert [item.id for item in schema.inputs] == [
        'video', 'filename_prefix', 'timeout_seconds', 'max_staging_gib', 'min_free_disk_gib']
    assert [item.id for item in schema.outputs] == ['video', 'video_path', 'report_json']
    with pytest.raises(ValueError, match='Only the explicit anime'):
        builder.graph_for('unknown')


@pytest.mark.parametrize('mode', storage.MODES)
def test_saved_additive_examples_match_deterministic_builder(mode):
    graph, _api = builder.graph_for(mode)
    saved = builder.DESTINATION / builder.FILES[mode]
    assert json.loads(saved.read_text(encoding='utf8')) == graph
