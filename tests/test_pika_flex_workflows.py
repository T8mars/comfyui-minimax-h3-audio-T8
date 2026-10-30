import json
from pathlib import Path
import hashlib

import pytest


ROOT=Path(__file__).resolve().parents[1]/'examples/workflows/94-pika-flex'


@pytest.mark.parametrize('task', ['t2va','fl2va','ref2va'])
def test_saved_pika_workflow_keeps_adapter_task_audio_and_recipe_wiring(task):
    graph=json.loads((ROOT/f'PikaFlex_{task}_6step_EXP.json').read_text('utf-8'))
    nodes={node['id']:node for node in graph['nodes']}
    links={link[0]:link for link in graph['links']}
    def one(kind):
        found=[node for node in nodes.values() if node['type']==kind]
        assert len(found)==1
        return found[0]
    def origin(node,name):
        entry=next(item for item in node['inputs'] if item['name']==name)
        link=links[entry['link']]
        return nodes[link[1]],link[2]
    setup=one('MiniMaxH3PikaFlexSetupEXPT8')
    source=one('MiniMaxH3AudioConditioningT8')
    sampler=one('SamplerCustomAdvanced')
    assert origin(sampler,'sampler')==(setup,1)
    assert origin(sampler,'sigmas')==(setup,2)
    assert origin(sampler,'latent_image')==(source,1)
    assert origin(one('BasicGuider'),'model')==(setup,0)
    assert origin(setup,'av_latent')==(source,1)
    loader=one('MiniMaxH3LoRACompatibilityLoaderT8Advanced')
    flavor='ref2va' if task=='ref2va' else 'fl2va'
    assert loader['widgets_values'][:2]==[f'pika_h3_flex_{flavor}_v1_bf16.safetensors',1.]
    assert one('UNETLoader')['widgets_values'][0]==f'minimax_h3_{flavor}_pruned_int8_convrot.safetensors'
    trim=one('MiniMaxH3OutputTrimT8')
    assert origin(trim,'audio')==(one('MiniMaxH3AVDecodeT8'),1)
    assert origin(one('MiniMaxH3SafeAVSaveT8Advanced'),'audio')==(trim,1)
    if task=='fl2va':
        assert trim['widgets_values'][-1]==124
        assert abs(trim['widgets_values'][1]-124/24)<1e-9
        assert origin(source,'first_frame')[0]['type']=='LoadImage'
        assert origin(source,'last_frame')[0]['type']=='LoadImage'


def test_reviewed_index_keeps_original_workflow_bytes_and_unreviewed_sound_status():
    index=json.loads((ROOT/'index.json').read_text('utf-8'))
    assert index['schema']=='t8.pika_flex.reviewed_workflows.v1'
    assert index['human_feedback_sha256']=='9029e6927b5471d39341f084c66f20d40e0b0f28b4364461f3df7b1983506d13'
    assert {row['case_id'] for row in index['workflows']}=={'t2va','fl2va','ref2va'}
    for row in index['workflows']:
        path=Path(__file__).resolve().parents[1]/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['workflow_sha256']
        assert row['picture']=='pass' and row['sound']=='unreviewed'
