"""Fourteen additive graphs, exact old math/confirmation and native schema edges."""
import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from tools import build_visible_face_mask_workflows as visible
from tools import build_formal_face_refine_storage_workflows as storage


@pytest.fixture(scope='module')
def current_core():
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    load_live_info()
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise
    from comfy_extras.nodes_video import LoadVideo, GetVideoComponents, CreateVideo, SaveVideo
    from comfy_extras.nodes_mask import ImageToMask
    for cls in (BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise,
                LoadVideo, GetVideoComponents, CreateVideo, SaveVideo, ImageToMask):
        key = cls.define_schema().node_id
        nodes.NODE_CLASS_MAPPINGS[key] = cls


@pytest.mark.parametrize('key', tuple(visible.FILES))
def test_visible_mask_graph_keeps_every_old_named_input_except_explicit_delivery_edges(key):
    frontend, api = visible.graph_for(*key)
    old_frontend, old_api = storage.graph_for(key[0], 'none', key[1])
    nodes = {n['id']: n for n in frontend['nodes']}
    for _, source, slot, target, target_slot, _ in frontend['links']:
        name = nodes[target]['inputs'][target_slot]['name']
        assert api[str(target)]['inputs'][name] == [str(source), slot]
    assert len(api) == len(old_api) + 4
    for name, old in old_api.items():
        actual = deepcopy(api[name])
        if old['class_type'] == 'CreateVideo':
            finish = api[actual['inputs']['images'][0]]
            assert finish['class_type'] == 'MiniMaxH3VisibleFaceCompositeEXPT8'
            assert actual['inputs']['audio'] == [actual['inputs']['images'][0], 1]
            assert finish['inputs']['candidate_images'] == old['inputs']['images']
            assert finish['inputs']['audio'] == old['inputs']['audio']
            actual['inputs'].update(images=old['inputs']['images'], audio=old['inputs']['audio'])
        elif old['class_type'] == 'SaveVideo':
            assert actual['inputs']['filename_prefix'].startswith('MiniMaxH3/R07_VisibleFace/')
            actual['inputs']['filename_prefix'] = old['inputs']['filename_prefix']
        assert actual == old
    sampler_ids = [n for n, v in api.items() if v['class_type'] == 'MiniMaxH3StageSamplerEXPT8']
    if key[1] == 'cold_delivery':
        assert not sampler_ids
        for value in api.values():
            if value['class_type'] == 'MiniMaxH3StageLoadEXPT8':
                assert value['inputs']['artifact_path'] == value['inputs']['artifact_sha256'] == ''
    assert frontend['extra']['t8_visible_face_mask']['original_accept_confirmation_unchanged'] is True
    assert old_frontend.get('extra', {}).get('t8_visible_face_mask') is None


@pytest.mark.parametrize('key', tuple(visible.FILES))
def test_new_graph_actual_Core_schema_and_typed_optional_multiface_edges(key, current_core):
    import execution
    _, graph = visible.graph_for(*key)
    for node in graph.values():
        if node['class_type'] == 'LoadVideo':
            node['inputs']['file'] = '0.6.mp4'
        elif node['class_type'] == 'LoadImage':
            node['inputs']['image'] = '0 (1).png'
    valid, error, _, failures = asyncio.run(execution.validate_prompt('visible-face-new-only', deepcopy(graph), None))
    assert valid and not failures, (error, failures)


def test_all_fourteen_files_exist_exactly_and_all_old_56_source_files_remain_unchanged():
    sources = [storage.face.DESTINATION / name for name in storage.FILES.values()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert len(visible.FILES) == 14
    for key, name in visible.FILES.items():
        expected, _ = visible.graph_for(*key)
        assert json.loads((visible.DESTINATION / name).read_text(encoding='utf8')) == expected
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
