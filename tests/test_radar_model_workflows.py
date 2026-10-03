"""Per-domain existing loader graphs, only one mathematical comparison delta."""
from copy import deepcopy
import json

import pytest

from tools.build_radar_model_workflows import DESTINATION, FILES, WEIGHTS, graph_for, build_candidate, model_info, model_weight_choice
from pathlib import Path


@pytest.mark.parametrize('variant', FILES)
def test_saved_model_templates_actual_schema_serialization_and_domain_boundary(variant):
    graph, rebuilt, report = build_candidate(variant, model_info())
    saved = json.loads((DESTINATION / FILES[variant]).read_text(encoding='utf8'))
    assert saved['nodes'] == rebuilt['nodes'] and saved['links'] == rebuilt['links']
    assert report['nodes'] == len(graph)
    assert saved['extra']['t8_radar_model']['human_quality_accepted'] is False
    assert saved['extra']['t8_radar_model']['author_parity_claimed'] is False
    if variant.startswith('orbit_'):
        assert graph['9']['inputs']['first_frame'] == graph['9']['inputs']['last_frame'] == ['201', 0]
        assert graph['9']['inputs']['task_type'] == 'FL2VA'
        assert graph['9']['inputs']['length'] == 73 and graph['90']['inputs']['steps'] == 28
        assert 'audio' not in graph['15']['inputs']
        assert not any('turbo' in json.dumps(node).lower() for node in graph.values())
    elif variant.startswith('wallpaper_'):
        assert graph['9']['inputs']['task_type'] == 'Ref2VA'
        assert graph['9']['inputs']['prompt'].startswith('live_wallpaper:')
        assert graph['9']['inputs']['ref_images.ref_image_0'] == ['201', 0]
        assert '<Picture 2>' not in graph['9']['inputs']['prompt']
        assert 'ref2v_turbo_4step' in graph['3']['inputs']['lora_name']
    else:
        assert not any('Sampler' in node['class_type'] or node['class_type'] in {'UNETLoader', 'CLIPLoader'} for node in graph.values())
        assert graph['2']['inputs']['scale_by'] == 2.


@pytest.mark.parametrize('family', ['orbit', 'wallpaper', 'lms'])
def test_baseline_candidate_differs_only_selected_weight_and_delivery_label(family):
    baseline = deepcopy(graph_for(family + '_baseline'))
    candidate = deepcopy(graph_for(family + '_candidate'))
    field = 'model_name' if family == 'lms' else 'lora_name'
    assert baseline['2']['inputs'][field] != candidate['2']['inputs'][field]
    candidate['2']['inputs'][field] = baseline['2']['inputs'][field]
    candidate['16']['inputs']['filename_prefix'] = baseline['16']['inputs']['filename_prefix']
    assert baseline == candidate


@pytest.mark.parametrize('family', WEIGHTS)
def test_new_candidate_subfolder_is_explicit_and_original_baseline_unchanged(family):
    field = 'model_name' if family == 'lms' else 'lora_name'
    assert model_weight_choice(family) == str(Path('zz_radar') / WEIGHTS[family])
    assert graph_for(family + '_candidate')['2']['inputs'][field] == model_weight_choice(family)
    expected = 'minimax_h3_latent_upscaler_3d_fp16.safetensors' if family == 'lms' else 'disabled'
    assert graph_for(family + '_baseline')['2']['inputs'][field] == expected
