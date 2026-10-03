"""Opt-in light recipe/per-frame source mask; old defaults and14 files unchanged."""
import asyncio
from copy import deepcopy
import json

import pytest

from tools import build_visible_face_temporal_workflows as temporal
from tools import build_visible_face_mask_workflows as static
import test_visible_face_mask_workflows as mask_tests


@pytest.fixture(scope='module')
def current_core():
    return mask_tests.current_core.__wrapped__()


@pytest.mark.parametrize('mode', tuple(temporal.FILES))
def test_explicit_delta_only_mask_media_and_light_recipe_preserves_external_relay_audio_cold(mode):
    frontend, api = temporal.graph_for(mode)
    _, old = static.graph_for('standard', mode, effect='relay')
    for key, node in old.items():
        if node['class_type'] == 'LoadImage':
            assert key not in api
            continue
        actual = deepcopy(api[key])
        if node['class_type'] == 'ImageToMask':
            component = api[actual['inputs']['image'][0]]
            assert component['class_type'] == 'GetVideoComponents'
            assert api[component['inputs']['video'][0]]['class_type'] == 'LoadVideo'
            actual['inputs']['image'] = node['inputs']['image']
        elif node['class_type'] == 'MiniMaxH3VisibleFaceMaskBindEXPT8':
            assert actual['inputs']['broadcast_single_mask'] is False
            actual['inputs']['broadcast_single_mask'] = True
        elif node['class_type'] == 'MiniMaxH3FaceRefineSamplerT8Advanced':
            assert actual['inputs']['steps'] == 12 and actual['inputs']['denoise'] == .05
            actual['inputs']['denoise'] = node['inputs']['denoise']
        elif node['class_type'] == 'SaveVideo':
            actual['inputs']['filename_prefix'] = node['inputs']['filename_prefix']
        assert actual == node
    ids = {n['id']: n for n in frontend['nodes']}
    for _, source, slot, target, target_slot, _ in frontend['links']:
        name = ids[target]['inputs'][target_slot]['name']
        assert api[str(target)]['inputs'][name] == [str(source), slot]
    assert frontend['extra']['t8_visible_face_mask']['original_accept_confirmation_unchanged'] is True
    if mode == 'cold_delivery':
        assert not any(n['class_type'] in ('MiniMaxH3StageSamplerEXPT8', 'MiniMaxH3FaceRefineSamplerT8Advanced') for n in api.values())
        # The inherited source/effect audit rebuilds its exact AV/Relay identity.
        # Cold means zero diffusion, not permission to remove original loaders.
        for kind in ('CLIPLoader', 'UNETLoader'):
            assert [n for n in api.values() if n['class_type'] == kind] == [n for n in old.values() if n['class_type'] == kind]
        loaders = [n for n in api.values() if n['class_type'] == 'MiniMaxH3StageLoadEXPT8']
        assert loaders and all(n['inputs']['artifact_path'] == n['inputs']['artifact_sha256'] == '' for n in loaders)


@pytest.mark.parametrize('mode', tuple(temporal.FILES))
def test_live_Core_edges_and_exact_new_saved_files(mode, current_core):
    import execution
    frontend, api = temporal.graph_for(mode)
    assert json.loads((temporal.DESTINATION / temporal.FILES[mode]).read_text(encoding='utf8')) == frontend
    for node in api.values():
        if node['class_type'] == 'LoadVideo':
            node['inputs']['file'] = '0.6.mp4'
    valid, error, _, failures = asyncio.run(execution.validate_prompt('temporal-visible-light', api, None))
    assert valid and not failures, (error, failures)
