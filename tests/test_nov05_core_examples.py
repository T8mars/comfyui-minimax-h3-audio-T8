"""Real Core validation and independent serialization checks; no model loading."""
import asyncio

import pytest

from tools import build_nov05_core_examples as builder
from tools.build_modular_fast_h3_v2_workflow import split_graph_with_results


@pytest.fixture(scope='module')
def live_info():
    return builder.load_live_info()


# The source-video recipe has its own actual-file fixture and validation test.
@pytest.mark.parametrize('name', tuple(name for name in builder.NOTES if name != 'H3_SFX_Generate_EXP'))
def test_candidates_keep_actual_schema_edges_and_every_output(live_info, name):
    import execution
    import torch
    graph, workflow, audit, selected = builder.build(name, live_info)
    expected_outputs = {key for key, node in graph.items() if selected[node['class_type']].get('output_node')}
    valid = asyncio.run(execution.validate_prompt('nov05-example-'+name, graph, None))
    assert valid[0] and not valid[3], valid
    assert set(valid[2]) == expected_outputs
    assert audit['nodes'] == len(graph)
    assert len(workflow['nodes']) == len(graph) + 1
    assert workflow['nodes'][-1]['type'] == 'MarkdownNote'
    assert '未发布' in workflow['nodes'][-1]['widgets_values'][0]
    if name == 'Stage_Reuse_EXP':
        assert graph['16']['inputs']['format.codec'] == 'h264'
        assert graph['16']['inputs'].get('codec', 'auto') == 'auto'
    assert not torch.cuda.is_initialized()


def test_examples_preserve_independent_model_routes_defaults_and_recipe():
    examples = builder.recipes()
    tao = examples['TaoMate_HD_2plus1_EXP']
    assert tao['9']['inputs']['steps'] == 3 and tao['9']['inputs']['low_scale'] == .625
    assert not any('Upscale' in node['class_type'] for node in tao.values())
    assert tao['2']['inputs']['lora_name'] == 'TaoMate-HD-preview-comfy.safetensors'
    assert tao['14']['inputs']['frame_count'] == 120
    nag = examples['NAG_CFG1_EXP']
    assert nag['17']['inputs']['enabled'] is False
    assert nag['18']['class_type'] == 'BasicGuider'
    assert nag['9']['class_type'] == 'MiniMaxH3PikaFlexSetupEXPT8'
    assert nag['16']['inputs']['text']
    original = split_graph_with_results()
    stage = examples['Stage_Reuse_EXP']
    for key in ('13', '29'):
        assert stage[key]['class_type'] == 'MiniMaxH3StageReuseEXPT8'
        assert stage[key]['inputs'] == {**original[key]['inputs'], 'mode':'disabled'}
    for key in ('10', '22', '23', '24', '25', '26', '62'):
        assert stage[key] == original[key]


def test_noise_comparison_changes_only_mode_and_output_destination():
    from copy import deepcopy
    examples = builder.recipes()
    independent = deepcopy(examples['Paired_Noise_Independent_EXP'])
    haar = deepcopy(examples['Paired_Noise_Haar_EXP'])
    assert independent['11']['inputs']['mode'] == 'independent'
    assert haar['11']['inputs']['mode'] == 'haar'
    haar['11']['inputs']['mode'] = 'independent'
    haar['32']['inputs']['filename_prefix'] = independent['32']['inputs']['filename_prefix']
    assert haar == independent
    assert independent['9']['inputs']['steps'] == 8
    assert independent['10']['inputs']['low_evaluations'] == 6
    assert independent['21']['class_type'] == 'MiniMaxH3LearnedLatentUpscaleT8Advanced'


def test_endpoint_ab_shares_low_result_and_audio_but_has_distinct_high_branches():
    graph = builder.recipes()['Refine_Endpoints_AB_EXP']
    assert graph['30']['inputs']['av_latent'] == graph['40']['inputs']['av_latent'] == ['22', 0]
    assert graph['30']['inputs']['enabled'] is False and graph['40']['inputs']['enabled'] is True
    assert graph['22']['inputs']['second_pass_audio_source'] == 'first_pass'
    assert graph['22']['inputs']['second_pass_audio_strength'] == 0
    for field in ('noise', 'guider', 'sampler', 'sigmas'):
        assert graph['31']['inputs'][field] == graph['41']['inputs'][field]
    for key in ('32', '42'):
        assert graph[key]['inputs']['fail_on_locked_mismatch'] is True
    assert graph['50']['inputs']['av_latent'] == graph['22']['inputs']['learned_latent'] == ['20', 0]
    for key in ('34', '44', '51'):
        assert graph[key]['inputs']['frame_count'] == 120
