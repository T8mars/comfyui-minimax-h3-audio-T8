"""Additive generic templates; exact resume semantics and no private inputs."""
import json

import pytest

from tools.build_radar_recipe_workflows import DESTINATION, FILES, graph_for, load_info, build_candidate


@pytest.mark.parametrize('variant', FILES)
def test_public_recipe_files_preserve_execution_types_inputs_and_roundtrip(variant):
    graph = graph_for(variant)
    workflow = json.loads((DESTINATION / FILES[variant]).read_text(encoding='utf8'))
    expected, rebuilt, report = build_candidate(variant, load_info())
    assert expected == graph
    assert report['nodes'] == len(graph)
    # Generated workflow identity can vary; all execution nodes/edges/widgets
    # and explicit notes are exact and independent of UUID metadata.
    assert workflow['nodes'] == rebuilt['nodes'] and workflow['links'] == rebuilt['links']
    assert workflow['extra']['t8_radar_recipe']['human_quality_accepted'] is False
    assert workflow['extra']['t8_radar_recipe']['public_inputs_are_placeholders'] is True
    raw = json.dumps(workflow, ensure_ascii=False)
    assert 'G:\\' not in raw and 'F:\\' not in raw and 'reference_a.png' not in raw
    if variant == 'union_full':
        assert graph['90']['inputs']['steps'] == 40 and graph['111']['inputs']['strength'] == 1.
        assert not any(any(token in n['class_type'] for token in ('LoRA', 'Relay', 'EAV', 'Upscale')) for n in graph.values())
        assert graph['100']['inputs']['file'] == 'SELECT_SOURCE_VIDEO.mp4'
        assert graph['50']['inputs']['stage_result'] == ['13', 2]
        assert graph['14']['inputs']['av_latent'] == ['50', 0]
    elif variant == 'union_cold':
        assert '60' in graph and graph['60']['inputs']['expected_stage'] == 'native_low'
        assert not any(any(token in n['class_type'] for token in ('MODEL', 'UNET', 'CLIP', 'Sampler', 'Union', 'Conditioning')) for n in graph.values())
    else:
        assert graph['6']['inputs']['image'] == 'SELECT_PERFORMER_A.png'
        assignments = json.loads(graph['11']['inputs']['assignments_json'])
        assert [entry['exact_vocal_text'] for entry in assignments] == ['', '']
        assert graph['12']['inputs']['resume_existing'] is True


def test_cast_resume_retains_exact_full_producer_settings_and_no_fake_decode_only():
    assert graph_for('cast_full') == graph_for('cast_resume')
    assert graph_for('cast_resume')['12']['inputs']['chain_id'] == 'SELECT_NEW_CHAIN_FOR_THIS_AB_REQUEST'
    with pytest.raises(ValueError):
        graph_for('turbo4plus4')
