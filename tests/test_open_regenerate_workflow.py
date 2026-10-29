"""Protect the example's source soundtrack and one-pass sigma ownership."""
import json
from pathlib import Path


def test_refiner_example_routes_original_audio_and_one_raw_schedule():
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / 'examples/workflows/92-open-regenerate/RefinerOnly_2K_EXP.json').read_text(encoding='utf-8'))
    by_id = {node['id']: node for node in graph['nodes']}
    by_type = {node['type']: node for node in graph['nodes']}
    links = {link[0]: link for link in graph['links']}

    def origin(node, input_name):
        entry = next(item for item in node['inputs'] if item['name'] == input_name)
        link = links[entry['link']]
        return by_id[link[1]], link[2]

    setup = by_type['MiniMaxH3OpenRegenerateSetupEXPT8']
    sampler = by_type['SamplerCustomAdvanced']
    assert origin(sampler, 'sampler') == (setup, 1)
    assert origin(sampler, 'sigmas') == (setup, 2)
    assert origin(by_type['BasicGuider'], 'model') == (setup, 0)
    assert setup['widgets_values'][0] == 'published_3step'
    assert origin(sampler, 'latent_image') == (by_type['MiniMaxH3LearnedLatentUpscaleT8Advanced'], 0)
    trim = by_type['MiniMaxH3OutputTrimT8']
    assert origin(trim, 'audio') == (by_type['GetVideoComponents'], 1)
    assert origin(by_type['MiniMaxH3SafeAVSaveT8Advanced'], 'audio') == (trim, 1)
    loader = by_type['MiniMaxH3LoRACompatibilityLoaderT8Advanced']
    assert loader['widgets_values'][1] == 1.0
    assert 'CFGGuider' not in by_type and 'MiniMaxH3DualClockSamplerT8' not in by_type


def test_existing_node_example_needs_no_new_setup_and_keeps_raw_sigma_and_original_audio():
    root = Path(__file__).resolve().parents[1]
    path = root / 'examples/workflows/92-open-regenerate/RefinerOnly_ExistingNodes_2K.json'
    graph = json.loads(path.read_text(encoding='utf-8'))
    by_id = {node['id']: node for node in graph['nodes']}
    by_type = {node['type']: node for node in graph['nodes']}
    links = {link[0]: link for link in graph['links']}

    def origin(node, name):
        item = next(item for item in node['inputs'] if item['name'] == name)
        link = links[item['link']]
        return by_id[link[1]], link[2]

    assert 'MiniMaxH3OpenRegenerateSetupEXPT8' not in by_type
    assert by_type['LoraLoaderModelOnly']['widgets_values'][1] == 1.0
    setup = by_type['MiniMaxH3DualClockSamplerT8']
    assert setup['widgets_values'][:5] == [3, 1.0, 1.0, 'euler', 'native_flow']
    sampler = by_type['SamplerCustomAdvanced']
    assert origin(sampler, 'sigmas') == (by_type['ManualSigmas'], 0)
    assert origin(sampler, 'sampler') == (setup, 1)
    assert origin(by_type['BasicGuider'], 'model') == (setup, 0)
    trim = by_type['MiniMaxH3OutputTrimT8']
    assert origin(trim, 'audio') == (by_type['GetVideoComponents'], 1)
    assert origin(by_type['MiniMaxH3SafeAVSaveT8Advanced'], 'audio') == (trim, 1)
    audio_encoder = by_type['VAEEncodeAudio']
    assert origin(audio_encoder, 'audio') == (by_type['MiniMaxH3SourceMediaWindowT8'], 1)


def _complete_graph(filename):
    path = Path(__file__).resolve().parents[1] / 'examples/workflows/92-open-regenerate' / filename
    workflow = json.loads(path.read_text(encoding='utf-8'))
    by_id = {node['id']: node for node in workflow['nodes']}
    links = {link[0]: link for link in workflow['links']}

    def origin(node, name):
        item = next(item for item in node['inputs'] if item['name'] == name)
        link = links[item['link']]
        return by_id[link[1]], link[2]

    def selected(kind):
        return [node for node in by_id.values() if node['type'] == kind]

    return origin, selected


def test_complete_video_workflow_observes_actual_denoised_result_and_keeps_original_audio():
    origin, selected = _complete_graph('CompleteRecipe_ExistingVideo_2K_EXP.json')
    audit, = selected('MiniMaxH3OpenRegenerateAuditEXPT8')
    sample, = selected('SamplerCustomAdvanced')
    setup, = selected('MiniMaxH3OpenRegenerateSetupEXPT8')
    loader, = selected('MiniMaxH3OpenRegenerateRecipeEXPT8')
    unet, = selected('UNETLoader')
    assert origin(loader, 'model') == (unet, 0)
    assert loader['widgets_values'][-2:] == [1., 1.]
    assert setup['widgets_values'][0] == 'published_3step'
    assert origin(audit, 'after_sampling') == (sample, 1)
    assert origin(audit, 'model') == (setup, 0)
    assert origin(sample, 'sigmas') == (setup, 2)
    window, = selected('MiniMaxH3SourceMediaWindowT8')
    components, = selected('GetVideoComponents')
    trim, = selected('MiniMaxH3OutputTrimT8')
    assert origin(window, 'source_audio') == (components, 1)
    assert origin(trim, 'audio') == (window, 1)


def test_complete_hyperflow_workflow_has_common_base_independent_branches_and_no_video_reencode():
    origin, selected = _complete_graph('CompleteRecipe_HyperFlow8_Refine3_2K_EXP.json')
    unet, = selected('UNETLoader')
    complete, = selected('MiniMaxH3OpenRegenerateRecipeEXPT8')
    hyperflow, = selected('MiniMaxH3HyperFlowLoaderT8Advanced')
    setup, = selected('MiniMaxH3OpenRegenerateSetupEXPT8')
    assert origin(complete, 'model') == origin(hyperflow, 'model') == (unet, 0)
    first, slot = origin(setup, 'video_latent')
    assert first['type'] == 'SamplerCustomAdvanced' and slot == 1
    assert not selected('VAEEncode')
    upscaler, = selected('MiniMaxH3LearnedLatentUpscaleT8Advanced')
    assert origin(upscaler, 'av_latent') == (setup, 3)
    audit, = selected('MiniMaxH3OpenRegenerateAuditEXPT8')
    second, = [node for node in selected('SamplerCustomAdvanced') if node != first]
    assert origin(audit, 'after_sampling') == (second, 1)
    assert origin(second, 'latent_image') == (upscaler, 0)
    trims = selected('MiniMaxH3OutputTrimT8')
    final_trim, = [node for node in trims if origin(node, 'audio')[0]['type'] == 'MiniMaxH3OutputTrimT8']
    draft_trim, = [node for node in trims if node != final_trim]
    assert origin(final_trim, 'audio') == (draft_trim, 1)
