"""Actual registered media consumers and native export; no trained encoder/queue."""
from copy import deepcopy
import json
import math
import uuid
import wave

from PIL import Image
import numpy as np
import torch

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import ProjectStore, new_project, compile_project
from h3_audio_t8_pkg.director_workflow_export import _live_object_info
from tools.api_to_frontend_workflow import convert
from test_director_generation import _patch_director_models
from test_director_workflow_export import _registry


def test_ordered_refs_literal_dialogue_actual_loaders_and_native_ui_export(tmp_path, monkeypatch):
    import folder_paths
    import nodes
    from comfy_extras.nodes_audio import LoadAudio, TrimAudioDuration
    _patch_director_models(monkeypatch)  # Loader names only; actual media below is not mocked.
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    monkeypatch.setattr(folder_paths, 'get_input_directory', lambda: str(store.input_root))
    assets=[]
    for name, color in [('red',(255,0,0)),('blue',(0,0,255))]:
        aid=str(uuid.uuid4())
        path=store.input_root/'t8_director'/aid/'source.png'
        path.parent.mkdir(parents=True)
        Image.new('RGB',(64,32),color).save(path)
        assets.append(store.register_asset(path,aid,name+'.png'))
    aid=str(uuid.uuid4())
    path=store.input_root/'t8_director'/aid/'source.wav'
    path.parent.mkdir(parents=True)
    pcm=(np.sin(np.arange(32000)*2*math.pi*400/16000)*10000).astype('<i2')
    with wave.open(str(path),'wb') as stream:
        stream.setparams((1,2,16000,0,'NONE','not compressed'))
        stream.writeframes(pcm.tobytes())
    assets.append(store.register_asset(path,aid,'source.wav'))
    project=new_project()
    project['assets']=assets
    shot=project['doc']['shots'][0]
    literal='<d>[Chinese]你好，别改这句话！  两个空格。</d>'
    shot.update(mode='refs',sound='voice',audio=assets[2]['id'],start=.25,end=1.25,
        refs=[assets[1]['id'],assets[0]['id']],tray=[a['id'] for a in assets],
        simplePrompt='@image1 hands a cup to @image2.\r\n'+literal)
    original=deepcopy(project)
    compiled=compile_project(project,store)
    assert compiled['ready']
    report=compiled['shots'][0]
    assert report['prompt']=='<Picture 2> hands a cup to <Picture 1>.\r\n'+literal
    built=build_director_generation_prompt(project,shot['id'],store)
    graph=built['prompt']
    condition=graph['5']['inputs']
    assert literal in condition['prompt'] and condition['prompt']==report['prompt']
    images=[]
    for field in ('ref_images.ref_image_0','ref_images.ref_image_1'):
        parent=graph[condition[field][0]]
        images.append(nodes.LoadImage().load_image(parent['inputs']['image'])[0])
    assert torch.equal(images[0][...,2],torch.ones_like(images[0][...,2]))
    assert torch.count_nonzero(images[0][...,:2])==0
    assert torch.equal(images[1][...,0],torch.ones_like(images[1][...,0]))
    selected_node=graph[condition['ref_audios.ref_audio_0'][0]]
    assert selected_node['class_type']=='TrimAudioDuration'
    loader=graph[selected_node['inputs']['audio'][0]]
    audio=LoadAudio.execute(loader['inputs']['audio']).result[0]
    cropped=TrimAudioDuration.execute(audio,start_index=.25,duration=1.).result[0]
    assert torch.equal(cropped['waveform'],audio['waveform'][...,4000:20000])
    registry=_registry(monkeypatch)
    registry.update({'LoadImage':nodes.LoadImage,'LoadAudio':LoadAudio,'TrimAudioDuration':TrimAudioDuration})
    workflow=convert(graph,_live_object_info(graph,registry),'Nov05 ordered references')
    ui_nodes={node['id']:node for node in workflow['nodes']}
    current=next(node for node in workflow['nodes'] if node['type']=='MiniMaxH3AudioConditioningT8')
    links={link[0]:link for link in workflow['links']}
    inputs={row['name']:row for row in current['inputs']}
    for field in ('ref_images.ref_image_0','ref_images.ref_image_1','ref_audios.ref_audio_0'):
        link=links[inputs[field]['link']]
        origin=ui_nodes[link[1]]
        api_origin=graph[condition[field][0]]
        assert origin['type']==api_origin['class_type'] and link[2]==condition[field][1]
        if origin['type']=='LoadImage':
            assert api_origin['inputs']['image'] in origin['widgets_values']
        else:
            assert origin['type']=='TrimAudioDuration'
            assert .25 in origin['widgets_values'] and 1. in origin['widgets_values']
    assert condition['prompt'] in current['widgets_values']
    assert project==original
    (tmp_path/'ordered-reference.ui.json').write_text(json.dumps(workflow,ensure_ascii=False,indent=2),encoding='utf-8')
    (tmp_path/'ordered-reference.api.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
