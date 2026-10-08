"""Public blank-path decoder templates, real registered schemas; no inference."""
import asyncio
import json
from pathlib import Path

import nodes
import pytest
import h3_audio_t8_pkg
from comfy_extras.nodes_audio import AudioExtension

from tools.prepare_freevideo_decoder_workflows import template
from tools.validate_freevideo_quality_canvas import check_serialized


@pytest.mark.parametrize('family,name,stage_type', [
    ('quality', 'FVQ_Decode_EXP.json', 'H3_T8_FREEVIDEO_QUALITY_STAGE'),
    ('legacy', 'FV_Decode_EXP.json', 'H3_T8_FREEVIDEO_STAGE')])
def test_public_typed_decoder_templates_keep_audio_and_no_sampler(family, name, stage_type):
    root = Path(__file__).parents[1]
    path = root / 'examples/workflows/78-freevideo-decode' / name
    graph = json.loads(path.read_text(encoding='utf8'))
    types = {row['type'] for row in graph['nodes'] if row['type'] != 'Note'}
    classes = dict(nodes.NODE_CLASS_MAPPINGS)
    classes.update({cls.GET_SCHEMA().node_id: cls for cls in asyncio.run(AudioExtension().get_node_list())})
    classes.update({cls.GET_SCHEMA().node_id: cls for cls in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())})
    metadata = {kind: classes[kind].GET_NODE_INFO_V1()
        if hasattr(classes[kind], 'GET_NODE_INFO_V1') else {
            'input': classes[kind].INPUT_TYPES(),
            'output': classes[kind].RETURN_TYPES,
            'output_name': getattr(classes[kind], 'RETURN_NAMES', classes[kind].RETURN_TYPES)}
        for kind in types}
    _, api, serial = template(metadata, family)
    assert check_serialized(graph, api) == serial
    assert api['1']['inputs']['manifest_path'] == api['1']['inputs']['manifest_sha256'] == ''
    assert api['3']['inputs']['decoder_config'] == '' and api['3']['inputs']['decode_mode'] == 'eager'
    assert api['4']['inputs'] == dict(samples=['3', 1], vae=['2', 0])
    assert api['5']['inputs']['audio'] == ['4', 0] and api['6']['inputs']['audio'] == ['5', 1]
    assert not any('Sampler' in kind or 'Conditioning' in kind for kind in types)
    assert next(row for row in graph['links'] if row[3] == 3)[5] == stage_type
    if family == 'legacy':
        assert api['1']['inputs']['role'] == 'HIGH'
    assert len(path.name.encode('utf-16-le')) // 2 <= 64
    text = path.read_text(encoding='utf8')
    assert 'G:\\\\T8-' not in text and 'artifacts' not in text
