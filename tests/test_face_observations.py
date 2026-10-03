"""One raw-detector call, exact unchanged original plans and explicit lazy decisions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math

import pytest
import torch

from comfy_execution.graph import ExecutionBlocker
from h3_audio_t8_pkg import face_observations as runtime
from h3_audio_t8_pkg.nodes_face_observations import NODES


def options(route):
    common = dict(manual_roi_x=.3,manual_roi_y=.1,manual_roi_width=.3,manual_roi_height=.45,
                  scene_cut_threshold=.28,max_track_jump=.18,max_gap_frames=4,
                  require_h3_grid=True,analysis_chunk_frames=2)
    if route=='standard':
        return common | dict(smoothing_radius=2,crop_context_scale=3.,canvas_size='384')
    return common | dict(center_smooth_window=21,size_smooth_window=51,crop_factor=3.,canvas_mode='manual_384')


@pytest.fixture
def raw_detector(tmp_path,monkeypatch):
    path = tmp_path/'local_detector.onnx'
    path.write_bytes(b'task-owned-unit-model-not-real-inference')
    monkeypatch.setattr(runtime.standard,'_resolve_detector_path',lambda _name:path)
    calls = []
    def detector(images,model,confidence,device):
        calls.append(images.clone())
        return [[{'box':[12.,8.,38.,48.],'confidence':.9}] for _ in images], {'backend':'unit_fixture_no_native_claim'}
    for module in (runtime.standard,runtime.parity):
        for mode in runtime.MODES:
            monkeypatch.setattr(module,'_detect_'+mode,detector)
    return path,calls,detector


def source(value=.2):
    frames = torch.full((5,64,96,3),value)
    frames[...,2] = value+.1
    return frames


@pytest.mark.parametrize('route',runtime.ROUTES)
@pytest.mark.parametrize('mode',runtime.MODES)
def test_reuse_actual_original_planner_code_is_byte_exact_and_no_second_detector(route,mode,raw_detector):
    path,calls,detector = raw_detector
    frames = source()
    before_globals = dict(runtime._planner(route).__globals__)
    observation = runtime.observe(frames,route=route,detector_mode=mode,detector_model=path.name)
    assert observation.verify()['status']=='face_found' and len(calls)==1
    actual = runtime.plan_from_observation(observation,route,**options(route))
    assert len(calls)==1
    expected = runtime._planner(route)(frames=frames,fps=24.,detector_mode=mode,detector_model=path.name,
                                      detector_device='cpu',confidence=.35,**options(route))
    assert len(calls)==2
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected,strict=True):
        assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
    assert all(runtime._planner(route).__globals__[key] is value for key,value in before_globals.items())
    assert runtime._detector(json.loads(observation.request_json)) is detector
    expected_received = frames[..., [2,1,0]] if route=='parity' and mode=='local_ultralytics' else frames
    assert torch.equal(calls[0],expected_received)


def test_scoped_namespace_two_concurrent_sources_no_global_detector_patch(raw_detector):
    path,calls,detector = raw_detector
    observations = [runtime.observe(source(value),detector_model=path.name) for value in (.1,.6)]
    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(lambda obs:runtime.plan_from_observation(obs,'standard',**options('standard')),observations))
    assert len(calls)==2 and runtime.standard._detect_local_opencv_yunet is detector
    assert outputs[0][0]['source']['proxy_sha256']!=outputs[1][0]['source']['proxy_sha256']
    assert not torch.equal(outputs[0][1],outputs[1][1])


def test_found_branch_and_negative_requires_manual_current_observation_SHA(raw_detector,monkeypatch):
    path,_,_ = raw_detector
    found = runtime.observe(source(),detector_model=path.name)
    selected,report = runtime.decision(found)
    assert selected is True and json.loads(report)['confirmed_no_face'] is False
    monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',
                        lambda images,*_args:([[] for _ in images],{'backend':'unit_negative_not_human_truth'}))
    negative = runtime.observe(source(),detector_model=path.name)
    assert negative.verify()['status']=='no_face_detected'
    for confirm,sha in [(False,''),(True,''),(True,found.contract_sha256),(False,negative.contract_sha256)]:
        selected,report = runtime.decision(negative,confirm_no_face=confirm,confirmation_sha256=sha)
        assert isinstance(selected,ExecutionBlocker) and json.loads(report)['delivery_blocked'] is True
    selected,report = runtime.decision(negative,confirm_no_face=True,confirmation_sha256=negative.contract_sha256)
    assert selected is False and json.loads(report)['confirmed_no_face'] is True
    assert json.loads(report)['automatic_accept'] is False
    with pytest.raises(ValueError,match='manufacture'):
        runtime.plan_from_observation(negative,'standard',**options('standard'))


@pytest.mark.parametrize('kind',['missing_model','exception','short','missing_frame','bad_box','nan','missing_report'])
def test_unknown_is_never_false_even_with_manual_confirmation(kind,raw_detector,monkeypatch):
    path,_,_ = raw_detector
    def bad(images,*_args):
        if kind=='exception':
            raise RuntimeError('unit detector failed')
        if kind=='short':
            return [[] for _ in images[:-1]],{}
        if kind=='missing_frame':
            return [None for _ in images],{}
        if kind=='bad_box':
            return [[{'box':[1,2,999,10],'confidence':.9}] for _ in images],{}
        if kind=='nan':
            return [[{'box':[1,2,10,12],'confidence':float('nan')}] for _ in images],{}
        return [[] for _ in images],None
    if kind=='missing_model':
        path.unlink()
        def missing(_name):
            raise ValueError('missing unit model')
        monkeypatch.setattr(runtime.standard,'_resolve_detector_path',missing)
    else:
        monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',bad)
    observation = runtime.observe(source(),detector_model=path.name)
    assert observation.verify()['status']=='unknown' and observation.verify()['detections'] is None
    result,report = runtime.decision(observation,confirm_no_face=True,confirmation_sha256=observation.contract_sha256)
    assert isinstance(result,ExecutionBlocker) and json.loads(report)['confirmed_no_face'] is False


@pytest.mark.parametrize('kind',['RGB_unsampled','model','receipt','request','seal','implementation','callable'])
def test_complete_content_binding_no_stale_no_face_or_plan(kind,raw_detector,monkeypatch):
    path,_,_ = raw_detector
    frames = source()
    observation = runtime.observe(frames,detector_model=path.name)
    if kind=='RGB_unsampled':
        frames[-1,-1,-1,0]+=.01
    elif kind=='model':
        path.write_bytes(b'changed-full-model-content')
    elif kind=='receipt':
        data = json.loads(observation.contract_json)
        data['automatic_accept']=True
        observation=replace(observation,contract_json=runtime.canonical(data))
    elif kind=='request':
        data=json.loads(observation.request_json)
        data['source_start_frame']=124
        observation=replace(observation,request_json=runtime.canonical(data))
    elif kind=='seal':
        observation=replace(observation,contract_sha256='0'*64)
    elif kind=='implementation':
        old=runtime._implementation()
        monkeypatch.setattr(runtime,'_implementation',lambda:old|{'another.py':'1'*64})
    else:
        monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',lambda *_args:([[]]*5,{}))
    with pytest.raises(ValueError,match='observe the actual source again'):
        runtime.decision(observation)


def test_late_source_mutation_and_repeated_or_overridden_detector_inputs_refused(raw_detector,monkeypatch):
    path,_,detector = raw_detector
    frames=source()
    def mutation(images,*args):
        result=detector(images,*args)
        frames[-1,-1,-1,0]+=.01
        return result
    monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',mutation)
    with pytest.raises(ValueError,match='observe the actual source again'):
        runtime.observe(frames,detector_model=path.name)
    monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',detector)
    observation=runtime.observe(frames,detector_model=path.name)
    with pytest.raises(ValueError,match='overridden'):
        runtime.plan_from_observation(observation,'standard',frames=frames,**options('standard'))
    with pytest.raises(ValueError,match='matching|manufacture'):
        runtime.plan_from_observation(observation,'parity',**options('parity'))


def test_mixed_boolean_blocker_keeps_report_visible_and_Core_lazy_switch_original_objects(raw_detector,monkeypatch):
    from comfy_extras.nodes_logic import SwitchNode
    import execution
    path,_,_=raw_detector
    monkeypatch.setattr(runtime.standard,'_detect_local_opencv_yunet',lambda images,*_:([[] for _ in images],{}))
    observation=runtime.observe(source(),detector_model=path.name)
    output=NODES[1].execute(observation)
    merged,ui,expand=execution.get_output_from_returns([output],NODES[1])
    assert isinstance(merged[0][0],ExecutionBlocker)
    assert isinstance(merged[1][0],str) and not expand
    original_RGB,original_AUDIO=observation.source,{'waveform':torch.zeros(1,2,1000),'sample_rate':48000}
    assert SwitchNode.check_lazy_status(False,on_false=None,on_true=None)==['on_false']
    assert SwitchNode.execute(False,on_false=original_RGB).result[0] is original_RGB
    assert SwitchNode.execute(False,on_false=original_AUDIO).result[0] is original_AUDIO
    # This is actual Core methods, not yet a native graph scheduling proof.


def test_appended_schema_adapters_keep_old_output_order_and_original_editable_crop_defaults():
    assert [node.define_schema().node_id for node in NODES]==[
        'MiniMaxH3CompleteFaceObserveEXPT8','MiniMaxH3NoFaceDecisionEXPT8',
        'MiniMaxH3ObservedStandardPlanEXPT8','MiniMaxH3ObservedParityPlanEXPT8',
        'MiniMaxH3FaceBranchStageSaveEXPT8','MiniMaxH3FaceDependentStageAuditEXPT8',
        'MiniMaxH3FaceDependentParityStageAuditEXPT8','MiniMaxH3FaceDependentStitchEXPT8',
        'MiniMaxH3FaceDependentParityStitchEXPT8']
    assert NODES[1].define_schema().inputs[1].default is False
    assert NODES[1].define_schema().inputs[2].default==''
    for current,old in [(NODES[2],runtime.standard),(NODES[3],runtime.parity)]:
        schema=current.define_schema()
        assert schema.inputs[0].id=='observation'
        assert not {'frames','fps','confidence','detector_model','detector_mode','detector_device'}.intersection(
            item.id for item in schema.inputs)
    assert all(math.isnan(node.fingerprint_inputs()) for node in NODES[:5])


def test_complete_executable_identity_stable_after_first_call_and_scoped_clone():
    from types import FunctionType
    def function(value):
        return [value, .35, b'constant', (3,4), frozenset({'a','b'}), Ellipsis]
    before = runtime._function_identity(function)
    for _ in range(10):
        function('first-call-interning')
        scoped = FunctionType(function.__code__,dict(function.__globals__),function.__name__)
        scoped(7)
        assert runtime._function_identity(function)==before
    def changed(value):
        return [value, .36, b'constant', (3,4), frozenset({'a','b'}), Ellipsis]
    assert runtime._function_identity(changed)['code_sha256']!=before['code_sha256']


def test_full_save_output_root_is_itself_lazy_and_never_invents_false_Stage(monkeypatch):
    from h3_audio_t8_pkg.nodes_face_observations import (
        MiniMaxH3FaceBranchStageSaveEXPT8 as gate,MiniMaxH3StageSaveEXPT8 as original)
    from comfy_api.latest import io
    calls=[]
    actual=object()
    expected=io.NodeOutput('real-output','real-denoised','real-path','real-SHA','real-report')
    monkeypatch.setattr(original,'execute',lambda stage_result,prefix:
                        calls.append((stage_result,prefix)) or expected)
    assert gate.define_schema().is_output_node is True
    assert gate.define_schema().inputs[1].lazy is True
    assert gate.check_lazy_status(False,stage_result=None)==[]
    assert gate.check_lazy_status(True,stage_result=None)==['stage_result']
    assert gate.check_lazy_status(True,stage_result=actual)==[]
    output=gate.execute(False)
    assert all(isinstance(value,ExecutionBlocker) for value in output.result[:4])
    assert json.loads(output.result[4])['stage_saved'] is False and not calls
    assert gate.execute(True,stage_result=actual,prefix='unchanged-prefix') is expected
    assert calls==[(actual,'unchanged-prefix')]
    with pytest.raises(ValueError,match='actual completed'):
        gate.execute(True)
    with pytest.raises(ValueError,match='Boolean'):
        gate.execute(0)
