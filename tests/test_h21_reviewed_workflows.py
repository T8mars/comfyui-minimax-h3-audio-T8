"""Release source graphs and resource boundaries, without queue or GPU."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'examples/workflows/90-h21-reviewed'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def test_all_80_workflows_and_separate_API_files_match_their_recorded_bytes():
    index = read(DIRECTORY / 'index.json')
    assert index['source_workflows'] == len(index['workflows']) == 80
    assert index['ordinary_companions'] == 6
    assert len({(row['category'], row['name']) for row in index['workflows']}) == 80
    for row in index['workflows']:
        frontend = DIRECTORY / row['workflow']
        api = ROOT / row['api']
        assert api.is_relative_to(ROOT / 'workflow_api/h21-reviewed')
        assert hashlib.sha256(frontend.read_bytes()).hexdigest() == row['workflow_sha256']
        assert hashlib.sha256(api.read_bytes()).hexdigest() == row['api_sha256']
        workflow, prompt = read(frontend), read(api)
        assert isinstance(workflow['nodes'], list) and isinstance(prompt, dict)
        assert workflow['extra']['t8_reviewed_source']['accepted_media_sha256'] == row['accepted_source_media_sha256']
        assert row['resource_adaptations_reexecuted'] is False
        assert not re.search(r'(?i)[a-z]:[/\\]', json.dumps([workflow, prompt], ensure_ascii=False))
        assert not any('Private' in n['class_type'] for n in prompt.values())


def test_8second_portrait_sources_keep_accepted_LOW_and_learned_upscale():
    for name in ('real8_Standard4plus4_full', 'real8_HF_Upscale8plus4_full', 'real8_HF_Upscale4plus4_full'):
        graph = read(ROOT / 'workflow_api/h21-reviewed/01-control' / (name + '.api.json'))
        source = [n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3DirectorControlSourceEXPT8'
                  and n['inputs']['stage'] == 'LOW']
        assert source and all((n['width'], n['height'], n['length']) == (448, 800, 192) for n in source)
        upscale, = [n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3LearnedLatentUpscaleT8Advanced']
        assert (upscale['target_width'], upscale['target_height']) == (576, 1024)
        assert upscale['size_mode'] == 'target_dimensions' and upscale['aspect_policy'] == 'honor_dimensions_exp'


def test_cold_HEAD_needs_user_owned_file_and_SHA_not_the_authors_receipt():
    graph = read(ROOT / 'workflow_api/h21-reviewed/01-control/cold_middle_hf_tail4_RegionCrop.api.json')
    loader, = [n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3HyperFlowHeadLoadEXPT8']
    assert loader['artifact_path'] == loader['artifact_sha256'] == ''


def test_ref_audio_high_is_new_approved_cache_history_source_not_Max():
    index = read(DIRECTORY / 'index.json')
    row, = [r for r in index['workflows'] if r['name'] == 'FreeVideo_ref2va_audio_high']
    assert row['accepted_source_media_sha256'] == 'c2dd0eff7468a110bc02f0224002de2b37bc7021092792c7148340be9d9bc8cc'
    graph = read(ROOT / row['api'])
    sampler, = [n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3FreeVideoH21SamplerEXPT8']
    assert sampler['tier'] == 'high' and sampler['role'] == 'FULL'
    loader, = [n['inputs'] for n in graph.values() if n['class_type'] == 'MiniMaxH3FreeVideoH21LoaderEXPT8']
    assert loader['runtime_config'] == ''
