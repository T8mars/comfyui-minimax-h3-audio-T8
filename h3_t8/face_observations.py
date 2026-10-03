"""One complete detector observation + explicit no-face decision + old plan reuse.

No second detector run, module-global patch, sampler, automatic approval or
negative-detection-as-ground-truth. Only new opt-in graphs use this route.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType

from . import face_refine_advanced as standard
from . import face_refine_parity_advanced as parity
from .visible_face_mask import canonical, identity, rgb
from .source_code_identity import _code_key

OBSERVATION_TYPE = 'T8_COMPLETE_FACE_OBSERVATION'
SCHEMA = 't8.complete-face-observation.v1'
MODES = ('local_opencv_yunet', 'local_anime_onnx_exp', 'local_ultralytics')
ROUTES = ('standard', 'parity')


def _implementation():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
            ('face_observations.py', 'nodes_face_observations.py', 'face_refine_advanced.py',
             'face_refine_parity_advanced.py', 'visible_face_mask.py', 'nodes_visible_face_mask.py',
             'source_code_identity.py')}


def _json_code_key(value):
    # Reuse complete typed executable fields. marshal embeds sharing/intern
    # flags that can legitimately change on first call without code mutation.
    if type(value) is bytes:
        return {'bytes':value.hex()}
    if type(value) is tuple:
        return [_json_code_key(item) for item in value]
    if type(value) is frozenset:
        return {'frozenset':sorted((_json_code_key(item) for item in value),key=canonical)}
    if value is Ellipsis:
        return {'literal':'Ellipsis'}
    if value is None or type(value) in (str,int,bool):
        return value
    raise ValueError('Unknown executable key; do not invent portable function identity')


def _function_identity(function):
    code = getattr(function, '__code__', None)
    if type(code) is not CodeType:
        return {'kind': 'opaque_process_local_not_portable', 'object_id': id(function)}
    return {'kind': 'python_code_in_current_process_not_persistent_cache',
            'module': function.__module__, 'qualname': function.__qualname__,
            'code_sha256': hashlib.sha256(canonical(_json_code_key(_code_key(code))).encode()).hexdigest()}


def _module(route):
    return standard if route == 'standard' else parity


def _detector(request):
    return getattr(_module(request['route']), '_detect_' + request['detector_mode'])


def _planner(route):
    return standard.build_face_refine_plan if route == 'standard' else parity.build_face_refine_parity_plan


def _model(request):
    try:
        path = standard._resolve_detector_path(request['detector_model'])
        return {'status': 'actual_local_file', 'path': str(path), 'size': path.stat().st_size,
                'sha256': standard._file_sha256(path)}
    except (ValueError, OSError) as exc:
        return {'status': 'unavailable_unknown_not_no_face', 'error': type(exc).__name__ + ': ' + str(exc)[:2048]}


def _request(request):
    if set(request) != {'route', 'fps', 'source_start_frame', 'detector_mode', 'detector_model',
                        'detector_device', 'confidence'}:
        raise ValueError('Exact full-source detector/clock request required')
    if request['route'] not in ROUTES or request['detector_mode'] not in MODES:
        raise ValueError('Select an explicit existing automatic Standard/Parity detector; ROI is not a no-face proof')
    if (type(request['fps']) not in (int, float) or request['fps'] != 24 or
            type(request['source_start_frame']) is not int or not 0 <= request['source_start_frame'] <= 5_000_000):
        raise ValueError('Explicit declared 24fps/integer whole-source interval required; no retiming')
    if (request['detector_device'] not in ('cpu', 'cuda_auto') or type(request['detector_model']) is not str or
            type(request['confidence']) not in (int, float) or not 0 < request['confidence'] <= 1):
        raise ValueError('Invalid explicit local detector/device/confidence')


def _current(source, request):
    _request(request)
    frames, height, width = rgb(source)
    return {'source': identity(source), 'request': request, 'implementation': _implementation(),
            'model': _model(request), 'detector_callable': _function_identity(_detector(request)),
            'planner_callable': _function_identity(_planner(request['route'])),
            'frame_interval': [request['source_start_frame'], request['source_start_frame']+frames],
            'geometry': [frames, height, width], 'fps': {'num': 24, 'den': 1},
            'clock_scope': 'caller_declared_complete_RGB_buffer_not_measured_media_PTS',
            'colour_policy': 'old_parity_BGR_input' if request['route']=='parity' and request['detector_mode']=='local_ultralytics'
                else 'old_standard_RGB_input'}


def _validate_detections(detections, shape):
    count, height, width = shape
    if type(detections) is not list or len(detections) != count:
        raise ValueError('Partial detector coverage is unknown, not no-face')
    for items in detections:
        if type(items) is not list:
            raise ValueError('Missing frame observation is unknown, not no-face')
        for item in items:
            if type(item) is not dict or 'box' not in item or 'confidence' not in item:
                raise ValueError('Incomplete detection record')
            box, confidence = item['box'], item['confidence']
            if type(box) is not list or len(box) != 4 or any(type(v) not in (int,float) for v in box):
                raise ValueError('Expected finite source-coordinate box')
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                    and type(confidence) in (int,float) and 0 <= confidence <= 1):
                raise ValueError('Invalid detector box/confidence is unknown')
    canonical(detections)  # Reject nonfinite/opaque hidden fields too.


@dataclass(frozen=True)
class FaceObservation:
    source: object
    request_json: str
    contract_json: str
    contract_sha256: str

    def verify(self):
        try:
            contract = json.loads(self.contract_json)
            request = json.loads(self.request_json)
            if hashlib.sha256(self.contract_json.encode()).hexdigest() != self.contract_sha256:
                raise ValueError('Observation receipt changed')
            actual_binding = _current(self.source, request)
            if (contract['schema'] != SCHEMA or contract['binding'] != actual_binding
                    or contract['automatic_accept'] is not False or contract['persistent_cache'] is not False):
                changed = {key:{'expected':value,'actual':actual_binding.get(key)}
                           for key,value in contract['binding'].items() if value!=actual_binding.get(key)}
                raise ValueError('Observation source/request/model/code changed: '+canonical(changed))
            if contract['status'] in ('face_found', 'no_face_detected'):
                _validate_detections(contract['detections'], contract['binding']['geometry'])
                expected = 'face_found' if any(contract['detections']) else 'no_face_detected'
                if contract['status'] != expected or contract['observed_frames'] != contract['binding']['geometry'][0]:
                    raise ValueError('No-face/coverage status contradicts actual observations')
            elif contract['status'] != 'unknown' or contract['detections'] is not None:
                raise ValueError('Unknown must not carry guessed empty observations')
            return contract
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError('Complete face observation is stale/invalid; observe the actual source again: '+str(exc)) from exc


def observe(source, *, route='standard', fps=24., source_start_frame=0, detector_mode='local_opencv_yunet',
            detector_model=standard.YUNET_2023MAR_RELATIVE, detector_device='cpu', confidence=.35):
    request = {'route':route, 'fps':fps, 'source_start_frame':source_start_frame, 'detector_mode':detector_mode,
               'detector_model':detector_model, 'detector_device':detector_device, 'confidence':confidence}
    before = _current(source, request)
    contract = {'schema':SCHEMA,'binding':before, 'status':'unknown','detections':None,'observed_frames':0,
                'detector_report':None,'automatic_accept':False,'persistent_cache':False,
                'negative_detection_is_human_truth':False,'sampling_nfe':0}
    try:
        if before['model']['status'] != 'actual_local_file':
            raise ValueError('Local detector is unavailable; no-face cannot be confirmed')
        received = source[..., [2,1,0]] if before['colour_policy']=='old_parity_BGR_input' else source
        detections, detector_report = _detector(request)(received, detector_model, confidence, detector_device)
        _validate_detections(detections, before['geometry'])
        if type(detector_report) is not dict:
            raise ValueError('A complete detector report is required')
        canonical(detector_report)
        if _current(source, request) != before:
            raise ValueError('Original RGB/model/implementation changed during detector run')
        contract.update(status='face_found' if any(detections) else 'no_face_detected',
                        detections=detections, observed_frames=len(detections), detector_report=detector_report)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        contract['failure'] = type(exc).__name__+': '+str(exc)[:2048]
    content = canonical(contract)
    observation = FaceObservation(source,canonical(request),content,hashlib.sha256(content.encode()).hexdigest())
    observation.verify()  # Late mutation must not even become a reusable unknown.
    return observation


def decision(observation, *, confirm_no_face=False, confirmation_sha256=''):
    from comfy_execution.graph import ExecutionBlocker
    if type(observation) is not FaceObservation or type(confirm_no_face) is not bool or type(confirmation_sha256) is not str:
        raise ValueError('Connect the explicit full-source observation and boolean confirmation')
    contract = observation.verify()
    status = contract['status']
    confirmed = status == 'no_face_detected' and confirm_no_face and confirmation_sha256 == observation.contract_sha256
    if status == 'face_found':
        selected = True
    elif confirmed:
        selected = False
    else:
        selected = ExecutionBlocker('No-face bypass blocked: unknown or not explicitly confirmed for this exact observation SHA')
    report = {'schema': SCHEMA+'.decision','status':status,'observation_sha256':observation.contract_sha256,
              'confirmed_no_face':confirmed,'face_branch': True if status=='face_found' else False if confirmed else None,
              'delivery_blocked':status!='face_found' and not confirmed, 'automatic_accept':False,
              'negative_detection_is_human_truth':False, 'source':contract['binding']['source']}
    return selected, canonical(report)


def plan_from_observation(observation, route, **options):
    if type(observation) is not FaceObservation or route not in ROUTES:
        raise ValueError('Connect the complete observation to its matching original plan family')
    contract = observation.verify()
    request = contract['binding']['request']
    if contract['status'] != 'face_found' or request['route'] != route:
        raise ValueError('No-face/unknown does not manufacture a Face plan; leave this branch unselected')
    reserved = {'frames','fps','detector_mode','detector_model','confidence','detector_device'}
    if reserved.intersection(options):
        raise ValueError('Source/detector parameters are bound by observation, not overridden')
    original = _planner(route)
    if type(original) is not FunctionType or original.__code__.co_freevars:
        raise ValueError('Original planner ABI changed; explicit adapter update required')
    expected_input = observation.source[..., [2,1,0]] if contract['binding']['colour_policy']=='old_parity_BGR_input' else observation.source
    expected_identity, calls = identity(expected_input), []
    def reuse(received, detector_model, confidence, detector_device):
        observation.verify()
        if (calls or identity(received) != expected_identity or detector_model != request['detector_model']
                or confidence != request['confidence'] or detector_device != request['detector_device']):
            raise ValueError('Original planner asked a different/repeated detector input; no hidden re-detection')
        calls.append(True)
        return deepcopy(contract['detections']),deepcopy(contract['detector_report'])
    namespace = dict(original.__globals__)
    namespace['_detect_'+request['detector_mode']] = reuse
    scoped = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)
    scoped.__kwdefaults__ = deepcopy(original.__kwdefaults__)
    result = scoped(frames=observation.source, fps=request['fps'], detector_mode=request['detector_mode'],
                    detector_model=request['detector_model'], detector_device=request['detector_device'],
                    confidence=request['confidence'], **options)
    if len(calls) != 1:
        raise ValueError('Original planner no longer consumes exactly one detector observation')
    observation.verify()
    return result
