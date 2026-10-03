"""Exact six new graphs and real Core scheduler, not just switch-method tests."""
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools import build_no_face_lazy_workflows as lazy
from tools import build_formal_face_refine_storage_workflows as storage
from h3_audio_t8_pkg import face_observations as runtime
from h3_audio_t8_pkg.nodes_face_observations import NODES,MiniMaxH3StageSaveEXPT8


@pytest.fixture(scope='module')
def current_core():
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    load_live_info()
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider,BasicScheduler,KSamplerSelect,RandomNoise
    from comfy_extras.nodes_video import LoadVideo,GetVideoComponents,CreateVideo,SaveVideo
    from comfy_extras.nodes_logic import SwitchNode
    for cls in (BasicGuider,BasicScheduler,KSamplerSelect,RandomNoise,
                LoadVideo,GetVideoComponents,CreateVideo,SaveVideo,SwitchNode):
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id]=cls


@pytest.mark.parametrize('key',tuple(lazy.FILES))
def test_six_graphs_keep_all_original_sampling_audio_and_independent_manual_confirmation(key,current_core):
    frontend,api=lazy.graph_for(*key)
    _,old_api=storage.graph_for(key[0],'none',key[1])
    nodes_by_id={node['id']:node for node in frontend['nodes']}
    for _,source,slot,target,target_slot,_ in frontend['links']:
        name=nodes_by_id[target]['inputs'][target_slot]['name']
        assert api[str(target)]['inputs'][name]==[str(source),slot]
    assert len(api)==len(old_api)+6
    observer=next((k,n) for k,n in api.items() if n['class_type']=='MiniMaxH3CompleteFaceObserveEXPT8')
    decision=next((k,n) for k,n in api.items() if n['class_type']=='MiniMaxH3NoFaceDecisionEXPT8')
    assert decision[1]['inputs']=={'observation':[observer[0],0],
                                 'confirm_no_face':False,'confirmation_sha256':''}
    for key_id,old in old_api.items():
        actual=deepcopy(api[key_id])
        if old['class_type'] in lazy.PLANNERS:
            assert actual['class_type']==lazy.PLANNERS[old['class_type']]
            assert actual['inputs'].pop('observation')==[observer[0],0]
            assert actual['inputs']=={k:v for k,v in old['inputs'].items() if k not in lazy.BOUND_INPUTS}
            for name in ('fps','detector_mode','detector_model','detector_device','confidence'):
                assert observer[1]['inputs'][name]==old['inputs'][name]
            actual=deepcopy(old)
        elif old['class_type']=='MiniMaxH3StageSaveEXPT8':
            assert actual['class_type']=='MiniMaxH3FaceBranchStageSaveEXPT8'
            assert actual['inputs'].pop('face_branch')==[decision[0],0]
            actual['class_type']=old['class_type']
        elif old['class_type'] in lazy.DEPENDENT_OPERATIONS:
            assert actual['class_type']==lazy.DEPENDENT_OPERATIONS[old['class_type']]
            actual['class_type']=old['class_type']
        elif old['class_type']=='CreateVideo':
            for name,slot in (('images',0),('audio',1)):
                switch=api[actual['inputs'][name][0]]
                assert switch['class_type']=='ComfySwitchNode'
                assert switch['inputs']['switch']==[decision[0],0]
                assert switch['inputs']['on_false']==[observer[1]['inputs']['source_images'][0],slot]
                assert switch['inputs']['on_true']==old['inputs'][name]
                actual['inputs'][name]=old['inputs'][name]
        elif old['class_type']=='SaveVideo':
            assert actual['inputs']['filename_prefix'].startswith('MiniMaxH3/R08_NoFaceLazy/')
            actual['inputs']['filename_prefix']=old['inputs']['filename_prefix']
        assert actual==old
    assert frontend['extra']['t8_no_face_lazy']['full_save_output_roots_lazy'] is True
    if key[1]=='cold_delivery':
        assert not any(n['class_type'] in ('MiniMaxH3StageSamplerEXPT8','SamplerCustomAdvanced',
                                          'MiniMaxH3FaceRefineSamplerT8Advanced') for n in api.values())
        loads=[n for n in api.values() if n['class_type']=='MiniMaxH3StageLoadEXPT8']
        assert loads and all(n['inputs']['artifact_path']==n['inputs']['artifact_sha256']=='' for n in loads)
    import execution
    import nodes
    output_roots=[n['class_type'] for n in api.values()
                  if getattr(nodes.NODE_CLASS_MAPPINGS[n['class_type']],'OUTPUT_NODE',False)]
    assert set(output_roots)<= {'SaveVideo','PreviewAny','MiniMaxH3FaceBranchStageSaveEXPT8'}
    for node in api.values():
        if node['class_type']=='LoadVideo':
            node['inputs']['file']='0.6.mp4'
        elif node['class_type']=='LoadImage':
            node['inputs']['image']='0 (1).png'
    valid,error,_,failures=asyncio.run(execution.validate_prompt('no-face-additive-graphs',api,None))
    assert valid and not failures,(error,failures)


def test_six_files_exact_and_old_face_JSON_bytes_preserved():
    paths=[storage.face.DESTINATION/name for name in storage.FILES.values()]
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert len(lazy.FILES)==6
    for key,filename in lazy.FILES.items():
        assert json.loads((lazy.DESTINATION/filename).read_text(encoding='utf8'))==lazy.graph_for(*key)[0]
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with pytest.raises(ValueError,match='SAM'):
        lazy.graph_for('multiface2','full_save')


@pytest.mark.parametrize('dependent',NODES[5:])
def test_dependent_adapters_preserve_every_original_input_output_and_return_object(dependent,monkeypatch):
    from dataclasses import asdict
    from h3_audio_t8_pkg import nodes_face_observations as module
    originals={
        NODES[5]:module.MiniMaxH3FaceStageAuditEXPT8,
        NODES[6]:module.MiniMaxH3FaceParityStageAuditEXPT8,
        NODES[7]:module.MiniMaxH3FaceRefineStitchAuditT8Advanced,
        NODES[8]:module.MiniMaxH3FaceRefineParityStitchT8Advanced}
    original=originals[dependent]
    old=original.define_schema()
    new=dependent.define_schema()
    assert old.is_output_node is True and new.is_output_node is False
    before=asdict(old.get_v1_info(original))
    after=asdict(new.get_v1_info(dependent))
    ignored={'name','display_name','description','category','output_node','python_module'}
    assert {k:v for k,v in after.items() if k not in ignored}=={
        k:v for k,v in before.items() if k not in ignored}
    assert original.define_schema().is_output_node is True
    calls=[]
    output=object()
    monkeypatch.setattr(original,'execute',lambda **inputs:calls.append(inputs) or output)
    assert dependent.execute(exact_marker='preserved') is output
    assert calls==[{'exact_marker':'preserved'}]


@pytest.mark.parametrize('state',['confirmed_negative','unconfirmed_negative','unknown','face_found'])
def test_actual_Core_PromptExecutor_does_not_execute_unselected_heavy_or_full_save_root(state,tmp_path,monkeypatch):
    """Tiny fixture is a scheduler assertion, NOT native media/model qualification."""
    from comfy_api.latest import io
    from comfy_extras.nodes_logic import SwitchNode
    import nodes
    import execution
    path=tmp_path/'task_unit_detector.onnx'
    path.write_bytes(b'unit-only-not-real-model')
    monkeypatch.setattr(runtime.standard,'_resolve_detector_path',lambda _name:path)
    source=torch.full((5,64,96,3),.2)
    audio={'waveform':torch.zeros(1,2,1000),'sample_rate':48000}
    calls=[]
    received=[]

    def detector(images,*_args):
        calls.append('detector')
        if state=='unknown':
            raise RuntimeError('unit unknown runtime failure')
        rows=[[{'box':[12.,8.,38.,48.],'confidence':.9}] for _ in images] if state=='face_found' else [[] for _ in images]
        return rows,{'scope':'unit_scheduler_fixture_no_native_claim'}
    monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',detector)
    observation=runtime.observe(source)
    calls.clear()
    stage=object()

    class LazyFixtureSource(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__,inputs=[],outputs=[io.Image.Output(),io.Audio.Output()])
        @classmethod
        def execute(cls):
            return io.NodeOutput(source,audio)

    class LazyFixtureHeavy(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__,inputs=[io.Custom(runtime.OBSERVATION_TYPE).Input('observation')],
                outputs=[io.Image.Output(),io.Audio.Output(),io.Custom('T8_STAGE_RESULT').Output()])
        @classmethod
        def execute(cls,observation):
            calls.append('heavy-test-fixture')
            # Reuse the ACTUAL original planner from the already captured OBS.
            runtime.plan_from_observation(observation,'standard',canvas_size='384',
                manual_roi_x=.3,manual_roi_y=.1,manual_roi_width=.3,manual_roi_height=.45,
                scene_cut_threshold=.28,max_track_jump=.18,max_gap_frames=4,smoothing_radius=2,
                crop_context_scale=3.,require_h3_grid=True,analysis_chunk_frames=2)
            return io.NodeOutput(source+.1,audio,stage)

    class LazyFixtureSink(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__,is_output_node=True,
                inputs=[io.Image.Input('images'),io.Audio.Input('audio')],outputs=[])
        @classmethod
        def execute(cls,images,audio):
            received.append((images,audio))
            return io.NodeOutput()

    def saver(stage_result,prefix):
        calls.append('original-saver-test-double')
        assert stage_result is stage and prefix=='task-only'
        return io.NodeOutput(source,source,'unit-path','unit-SHA','unit-report')
    monkeypatch.setattr(MiniMaxH3StageSaveEXPT8,'execute',saver)
    for cls in (*NODES,SwitchNode,LazyFixtureSource,LazyFixtureHeavy,LazyFixtureSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS,cls.define_schema().node_id,cls)
    graph={
        '1':{'class_type':'LazyFixtureSource','inputs':{}},
        '2':{'class_type':'MiniMaxH3CompleteFaceObserveEXPT8','inputs':{'source_images':['1',0],
             'route':'standard','fps':24.,'source_start_frame':0,'detector_mode':'local_opencv_yunet',
             'detector_model':runtime.standard.YUNET_2023MAR_RELATIVE,'detector_device':'cpu','confidence':.35}},
        '3':{'class_type':'MiniMaxH3NoFaceDecisionEXPT8','inputs':{'observation':['2',0],
             'confirm_no_face':state=='confirmed_negative','confirmation_sha256':observation.contract_sha256}},
        '4':{'class_type':'LazyFixtureHeavy','inputs':{'observation':['2',0]}},
        '5':{'class_type':'ComfySwitchNode','inputs':{'switch':['3',0],'on_false':['1',0],'on_true':['4',0]}},
        '6':{'class_type':'ComfySwitchNode','inputs':{'switch':['3',0],'on_false':['1',1],'on_true':['4',1]}},
        '7':{'class_type':'LazyFixtureSink','inputs':{'images':['5',0],'audio':['6',0]}},
        '8':{'class_type':'MiniMaxH3FaceBranchStageSaveEXPT8','inputs':{'face_branch':['3',0],
             'stage_result':['4',2],'prefix':'task-only'}}}
    valid,error,_,failures=asyncio.run(execution.validate_prompt('actual-lazy-unit',graph,None))
    assert valid and not failures,(error,failures)
    server=SimpleNamespace(client_id=None,last_node_id=None,sockets_metadata={},send_sync=lambda *a,**k:None)
    executor=execution.PromptExecutor(server,cache_args={'ram':0.,'ram_inactive':0.},
                                      asset_manager=SimpleNamespace(enabled=False))
    executor.execute(deepcopy(graph),'actual-lazy-'+state,execute_outputs=['7','8'])
    assert executor.success,executor.status_messages
    assert calls.count('detector')==1
    if state=='face_found':
        assert calls.count('heavy-test-fixture')==calls.count('original-saver-test-double')==1
        assert len(received)==1 and torch.equal(received[0][0],source+.1) and received[0][1] is audio
    else:
        assert calls==['detector']
        if state=='confirmed_negative':
            assert len(received)==1 and received[0][0] is source and received[0][1] is audio
        else:
            assert received==[]
