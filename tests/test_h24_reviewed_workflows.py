"""Protect the shipped recipes' resource boundaries and accepted execution choices."""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'examples/workflows/91-h24-reviewed'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def graph(case_id):
    row, = [row for row in read(DIRECTORY / 'index.json')['workflows']
            if row['case_id'] == case_id]
    return read(ROOT / row['api'])


def test_reviewed_graph_bytes_and_public_resource_boundaries():
    index = read(DIRECTORY / 'index.json')
    assert {row['case_id'] for row in index['workflows']} == {
        'r2-compact', 'r3-loop', 'r4-structure'}
    for row in index['workflows']:
        ui, api = DIRECTORY / row['workflow'], ROOT / row['api']
        assert api.is_relative_to(ROOT / 'workflow_api/h24-reviewed')
        assert hashlib.sha256(ui.read_bytes()).hexdigest() == row['workflow_sha256']
        assert hashlib.sha256(api.read_bytes()).hexdigest() == row['api_sha256']
        workflow, prompt = read(ui), read(api)
        assert not re.search(r'(?i)[a-z]:[/\\]', json.dumps([workflow, prompt]))
        assert len(prompt) == row['execution_nodes']
        assert len(workflow['links']) == row['links']
        assert workflow['extra']['t8_reviewed_source']['accepted_media_sha256'] == row['source_media_sha256']
        assert row['resource_adaptations_GPU_reexecuted'] is False


def test_compact_uses_the_checkpoint_sampler_and_full_grid():
    api = graph('r2-compact')
    setup_id, setup = next((nid, n) for nid, n in api.items()
                          if n['class_type'] == 'MiniMaxH3PTDSetupEXPT8')
    sampler, = [n for n in api.values() if n['class_type'] == 'SamplerCustomAdvanced']
    assert setup['inputs']['checkpoint'] == 'ptd_h3_ref2va_4step_int8.safetensors'
    assert sampler['inputs']['sampler'] == [setup_id, 1]
    assert sampler['inputs']['sigmas'] == [setup_id, 2]


def test_solar_retains_shared_seed_ordered_pair_and_explicit_delivery_volume():
    api = graph('r4-structure')
    assert api['6']['inputs']['model'] == ['5', 0]
    assert api['5']['inputs']['lora_name'] == 'solarrenderer_4step_lora.safetensors'
    assert api['6']['inputs']['lora_name'] == 'solarrenderer_control_lora.safetensors'
    assert api['13']['inputs']['seed'] == api['14']['inputs']['noise_seed'] == ['20', 0]
    assert api['21']['inputs'] == {'audio': ['11', 3], 'volume': -1.0}
    assert api['18']['inputs']['audio'] == ['21', 0]


def test_loop_keeps_periodic_decode_connected_to_its_plan():
    api = graph('r3-loop')
    setup_id, setup = next((nid, n) for nid, n in api.items()
                          if n['class_type'] == 'MiniMaxH3PeriodicAVSetupEXPT8')
    decode, = [n for n in api.values() if n['class_type'] == 'MiniMaxH3PeriodicAVDecodeEXPT8']
    assert setup['inputs']['enabled'] is True
    assert decode['inputs']['loop_plan'] == [setup_id, 1]
