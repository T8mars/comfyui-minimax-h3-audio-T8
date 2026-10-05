"""Explicit literal v5 restart capsule; no re-encoding/lift/noise draw on load.

Only inert native values are stored. Foreign live callbacks still run in Full
but cannot be serialized as executable objects or portable provider identities.
"""
from dataclasses import asdict, fields
import os
from pathlib import Path
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

from .. import temporal_dialogue_bank as banks, temporal_dialogue_scope as scope
from ..temporal_dialogue_identity import LiveMetadataIdentity
from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from . import chunked_v5 as base, chunked_v5_storage
from . import temporal_chunked_storage as scoped
from .temporal_chunked_v5 import ScopedWindowResult
from .results import canonical, _input_identity
from .storage import MAX_JSON, MAX_FILE, _root, _path, _lease, _digest, _json_file, file_sha, fingerprint_stage


SCHEMA = 't8.temporal-dialogue.literal-v5-resume.v1'


def _typed(cls, raw, **overrides):
    if type(raw) is not dict or set(raw) != {f.name for f in fields(cls)}:
        raise ValueError('Unknown Temporal resume typed fields')
    return cls(**{**raw, **overrides})


def _dialogue(raw):
    return _typed(scope.DialoguePlan, raw,
        events=tuple(_typed(scope.DialogueEvent, v) for v in raw['events']),
        performance_events=tuple(_typed(scope.PerformanceEvent, v) for v in raw['performance_events']))


def _compiled(raw):
    return _typed(scope.CompiledWindowText, raw, window=_typed(scope.WindowDescriptor, raw['window']),
                  dialogue=tuple(_typed(scope.ProjectedDialogue, v) for v in raw['dialogue']))


def _bank_state(bank):
    if not bank.identity_context.portable:
        raise ValueError('Live foreign metadata remains supported in Full, but cannot be frozen as a native resume capsule')
    return dict(dialogue_plan=asdict(bank.dialogue_plan), source_identity=bank.source_identity,
        chunk_plan_identity=bank.chunk_plan_identity, width=bank.width, height=bank.height,
        executor_contract=bank.executor_contract, sha256=bank.sha256,
        encoded=[dict(compiled=asdict(v.compiled), positive=v.positive, prepared_prompt=v.prepared_prompt,
                      tokens=v.tokens, report=v.report, content_identity=v.content_identity) for v in bank.encoded])


def _bank(raw):
    names = {f.name for f in fields(banks.WindowConditioningBank)}-{'identity_context'}
    if type(raw) is not dict or set(raw) != names:
        raise ValueError('Unknown native resume bank fields')
    encoded = tuple(_typed(banks.EncodedWindow, v, compiled=_compiled(v['compiled'])) for v in raw['encoded'])
    return banks.WindowConditioningBank(**{**raw, 'dialogue_plan':_dialogue(raw['dialogue_plan']),
        'encoded':encoded, 'identity_context':LiveMetadataIdentity()})


def _implementation():
    return dict(resume=file_sha(Path(__file__)), scope=scoped._implementation(),
                base=chunked_v5_storage._implementation_identity())


def _state(result, source, lifted, prepared, plan, bank):
    binding = scoped.verify_scoped_window(result, source, lifted, prepared, plan, bank)
    lift = prepared.lift
    return dict(source=source, lifted=lifted, plan=plan, bank=_bank_state(bank),
        lift=dict(plan_sha256=lift.plan_sha256, source_identity=lift.source_identity,
            lifted_identity=lift.lifted_identity, segments=lift.segments, audio_bounds=lift.audio_bounds,
            upscale_report=lift.upscale_report),
        prepared=dict(video_noise=prepared.video_noise, audio_noise=prepared.audio_noise,
                      noise_report=prepared.noise_report, noise_seed=prepared.noise_seed),
        output=result.base.output_latent, binding=binding)


def _rebuild(values):
    required = {'source','lifted','plan','bank','lift','prepared','output','binding'}
    if type(values) is not dict or set(values) != required:
        raise ValueError('Unknown native resume state inventory')
    source, lifted, plan = values['source'], values['lifted'], values['plan']
    raw = values['lift']
    lift_fields = {f.name for f in fields(base.StandardLift)}-{'source_latent','lifted_latent'}
    if type(raw) is not dict or set(raw) != lift_fields:
        raise ValueError('Unknown native resume lift fields')
    lift = base.StandardLift(**raw, source_latent=source, lifted_latent=lifted)
    noise = values['prepared']
    if type(noise) is not dict or set(noise) != {f.name for f in fields(base.StandardPrepared)}-{'lift'}:
        raise ValueError('Unknown native resume noise fields')
    prepared = base.StandardPrepared(lift=lift, **noise)
    bank = _bank(values['bank'])
    binding = values['binding']
    if type(binding) is not dict or set(binding) != {'base_binding','bank_sha256','receipt'}:
        raise ValueError('Unknown native resume window binding')
    raw = binding['base_binding']
    window = base.StandardWindowResult(raw['plan_sha256'], raw['source_identity'], raw['lifted_identity'],
        raw['index'], raw['count'], values['output'], raw['output_identity'])
    result = ScopedWindowResult(window, binding['bank_sha256'], scoped._typed_receipt(binding['receipt']))
    if scoped.verify_scoped_window(result, source, lifted, prepared, plan, bank) != binding:
        raise ValueError('Native resume literal values or window receipt differ')
    return source, lifted, prepared, plan, bank, result


def save_resume(result, source, lifted, prepared, plan, bank, storage_root):
    values = _state(result, source, lifted, prepared, plan, bank)
    tensors = {}
    descriptor = _encode_metadata_value(values, tensors, path='temporal.resume')
    # The native serializer copies tensor bytes to owned CPU buffers.
    rebuilt = _rebuild(_decode_metadata_value(descriptor, tensors, set(), path='temporal.resume'))
    if _input_identity(_state(rebuilt[-1], *rebuilt[:5])) != _input_identity(values):
        raise ValueError('Native resume CPU copy changed literal values')
    if sum(t.numel()*t.element_size() for t in tensors.values()) > MAX_FILE-MAX_JSON:
        raise ValueError('Native resume tensors exceed size limit')
    payload = canonical(descriptor)
    if len(payload.encode('utf8')) > MAX_JSON-4096:
        raise ValueError('Native resume metadata exceeds size limit')
    root = _root(storage_root, create=True)
    directory = root/('resume-'+uuid.uuid4().hex)
    directory.mkdir()
    with _lease(directory):
        state, pending = directory/'state.safetensors', directory/'state.safetensors.partial'
        save_file(tensors, str(pending), metadata={'resume_state':payload})
        with pending.open('rb+') as stream:
            header = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < header <= MAX_JSON:
                raise ValueError('Native resume header exceeds size limit')
            os.fsync(stream.fileno())
        os.replace(pending, state)
        scoped.verify_scoped_window(result, source, lifted, prepared, plan, bank)
        manifest = dict(schema=SCHEMA, state_file=state.name, state_sha256=file_sha(state), state_bytes=state.stat().st_size,
            values_identity=_input_identity(values), bank_sha256=bank.sha256, index=result.base.index,
            implementation=_implementation(), automatic_cache_reuse=False, provider_equivalence_certified=False,
            quality_accepted=False)
        data = canonical(manifest).encode('utf8')
        if len(data) > MAX_JSON:
            raise ValueError('Native resume manifest exceeds size limit')
        pending, path = directory/'manifest.json.partial', directory/'manifest.json'
        with pending.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    return path.relative_to(root).as_posix(), file_sha(path), canonical(manifest)


def load_resume(storage_root, artifact_path, artifact_sha256, expected_index):
    if type(expected_index) is not int or expected_index < 0:
        raise ValueError('Native resume expected index must be an integer')
    root, digest = _root(storage_root), _digest(artifact_sha256)
    path = _path(root, artifact_path)
    if path.name != 'manifest.json':
        raise ValueError('Select a completed native resume manifest.json')
    with _lease(path.parent):
        manifest = _json_file(path)
        names = {'schema','state_file','state_sha256','state_bytes','values_identity','bank_sha256','index',
                 'implementation','automatic_cache_reuse','provider_equivalence_certified','quality_accepted'}
        if file_sha(path) != digest:
            raise ValueError('Native resume manifest SHA mismatch')
        if (set(manifest) != names or manifest['schema'] != SCHEMA or manifest['state_file'] != 'state.safetensors'
                or type(manifest['index']) is not int or manifest['index'] != expected_index
                or manifest['implementation'] != _implementation()
                or any(manifest[k] is not False for k in ('automatic_cache_reuse','provider_equivalence_certified','quality_accepted'))):
            raise ValueError('Native resume implementation/contract/expected window differs')
        state = _path(root, (path.parent/'state.safetensors').relative_to(root).as_posix())
        if (type(manifest['state_bytes']) is not int or not 8 < manifest['state_bytes'] <= MAX_FILE
                or state.stat().st_size != manifest['state_bytes'] or file_sha(state) != _digest(manifest['state_sha256'])):
            raise ValueError('Native resume tensor size/SHA mismatch')
        with state.open('rb') as stream:
            header = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header <= MAX_JSON or header > state.stat().st_size-8:
            raise ValueError('Native resume tensor header exceeds size limit')
        with safe_open(str(state), framework='pt', device='cpu') as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {'resume_state'}:
                raise ValueError('Native resume metadata inventory differs')
            import json
            descriptor = json.loads(metadata['resume_state'])
            tensors = {key:handle.get_tensor(key) for key in handle.keys()}
        used = set()
        values = _decode_metadata_value(descriptor, tensors, used, path='temporal.resume')
        if used != set(tensors) or _input_identity(values) != manifest['values_identity']:
            raise ValueError('Native resume literal value identity differs')
        outputs = _rebuild(values)
        if outputs[4].sha256 != manifest['bank_sha256'] or outputs[5].base.index != expected_index:
            raise ValueError('Native resume bank or actual window differs')
        if file_sha(path) != digest or file_sha(state) != manifest['state_sha256']:
            raise ValueError('Native resume changed during load')
    return (*outputs, canonical(dict(status='literal_resume_loaded_no_sampling_no_encoding_no_lift',
        expected_index=expected_index, automatic_cache_reuse=False, provider_equivalence_certified=False, quality_accepted=False)))


def fingerprint_resume(storage_root, artifact_path):
    return fingerprint_stage(storage_root, artifact_path)
