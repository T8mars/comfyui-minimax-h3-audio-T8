"""Independent literal H16 bundle: native conditions, global noise and each lift.

No earlier sampling/encoding/lifting is performed when loading a selected bundle.
This is not a MODEL identity certificate, or a conversion of released H16 caches.
"""
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import struct
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..native_latent_checkpoint_advanced import _encode_metadata_value, _decode_metadata_value
from ..temporal_dialogue_bank import validate_window_bank
from .chunked_source import ChunkedSourceSegment, slice_chunked_source
from .chunked_stages import ChunkedPass2Context, _check_segment, lift_chunked_segment
from .results import _input_identity, canonical
from .storage import MAX_FILE, MAX_JSON, _root, _path, _lease, _digest, _json_file, file_sha
from .temporal_resume import _bank, _bank_state, _typed
from . import temporal_h16_storage as storage
from .temporal_h16 import ScopedH16Result
from .temporal_chunked_storage import _typed_receipt


SCHEMA = 't8.temporal-h16.literal-resume-bundle.v1'


@dataclass(frozen=True)
class H16ResumeBundle:
    context: ChunkedPass2Context
    plan: dict
    bank: object
    pieces: tuple
    identity: dict


def _payload(bundle):
    return dict(context={f.name:getattr(bundle.context, f.name) for f in fields(ChunkedPass2Context)},
        plan=bundle.plan, bank=_bank_state(bundle.bank), pieces=[dict(source=source,
        spec={f.name:getattr(spec, f.name) for f in fields(ChunkedSourceSegment)}, lifted=lifted)
        for source,spec,lifted in bundle.pieces])


def verify_bundle(bundle):
    if type(bundle) is not H16ResumeBundle or _input_identity(_payload(bundle)) != bundle.identity:
        raise ValueError('H16 literal bundle changed')
    validate_window_bank(bundle.bank, bundle.context.source_latent, bundle.plan)
    if len(bundle.pieces) != len(bundle.bank.encoded):
        raise ValueError('H16 literal bundle window inventory differs')
    for index,(source,spec,lifted) in enumerate(bundle.pieces):
        _check_segment(source, spec, bundle.context, bundle.plan)
        if spec.index != index or spec.count != len(bundle.pieces):
            raise ValueError('H16 literal bundle window order differs')
        _, actual_spec, _ = slice_chunked_source(bundle.context.source_latent, bundle.plan, index)
        if actual_spec != spec:
            raise ValueError('H16 literal bundle source slice differs')
        video,audio = lifted['samples'].unbind()
        if (tuple(video.shape) != (1,24,spec.end_token-spec.start_token,spec.target_height//16,spec.target_width//16)
                or not torch.equal(audio,source['samples'].tensors[1]) or not torch.isfinite(video).all()):
            raise ValueError('H16 literal lift time/audio/geometry differs')
    return bundle


def prepare_bundle(context, plan, bank):
    validate_window_bank(bank, context.source_latent, plan)
    # Serialization is explicitly optional; foreign live providers keep the old
    # separated Full path rather than being banned from native sampling.
    _bank_state(bank)
    pieces = []
    for index in range(len(bank.encoded)):
        source,spec,_ = slice_chunked_source(context.source_latent, plan, index)
        lifted,_ = lift_chunked_segment(source,spec,context,plan)
        pieces.append((source,spec,lifted))
    draft = H16ResumeBundle(context,plan,bank,tuple(pieces),{})
    return verify_bundle(H16ResumeBundle(context,plan,bank,draft.pieces,_input_identity(_payload(draft))))


def select_bundle_window(bundle, index):
    verify_bundle(bundle)
    if type(index) is not int or not 0 <= index < len(bundle.pieces):
        raise ValueError('H16 literal bundle window index invalid')
    source,spec,lifted = bundle.pieces[index]
    return source,lifted,spec,bundle.context,bundle.plan,bundle.bank


def _values(bundle, result):
    args = select_bundle_window(bundle,result.index)
    binding = storage.verify_h16_scoped_window(result,*args)
    return dict(bundle=_payload(bundle),bundle_identity=bundle.identity,output=result.output_latent,binding=binding)


def _rebuild(raw):
    if type(raw) is not dict or set(raw) != {'bundle','bundle_identity','output','binding'}:
        raise ValueError('Unknown H16 literal resume inventory')
    payload = raw['bundle']
    if type(payload) is not dict or set(payload) != {'context','plan','bank','pieces'}:
        raise ValueError('Unknown H16 literal bundle inventory')
    context = _typed(ChunkedPass2Context,payload['context'])
    pieces = []
    for piece in payload['pieces']:
        if type(piece) is not dict or set(piece) != {'source','spec','lifted'}:
            raise ValueError('Unknown H16 literal piece inventory')
        pieces.append((piece['source'],_typed(ChunkedSourceSegment,piece['spec']),piece['lifted']))
    bundle = verify_bundle(H16ResumeBundle(context,payload['plan'],_bank(payload['bank']),tuple(pieces),raw['bundle_identity']))
    b = raw['binding']
    result = ScopedH16Result(b['plan_sha256'],b['source_identity'],b['bank_sha256'],b['index'],b['count'],
        raw['output'],b['output_identity'],_typed_receipt(b['receipt']))
    if storage.verify_h16_scoped_window(result,*select_bundle_window(bundle,result.index)) != b:
        raise ValueError('H16 literal actual prefix differs')
    return bundle,result


def _implementation():
    from . import temporal_resume
    return dict(bundle=file_sha(Path(__file__)),bank_serializer=file_sha(Path(temporal_resume.__file__)),
        H16=storage._implementation())


def save_h16_resume(bundle, result, storage_root):
    raw = _values(bundle,result)
    tensors = {}
    descriptor = _encode_metadata_value(raw,tensors,path='h16.resume')
    restored = _rebuild(_decode_metadata_value(descriptor,tensors,set(),path='h16.resume'))
    if _input_identity(_values(*restored)) != _input_identity(raw):
        raise ValueError('H16 literal CPU copy differs')
    payload = canonical(descriptor)
    if len(payload.encode('utf8')) > MAX_JSON-4096 or sum(t.numel()*t.element_size() for t in tensors.values()) > MAX_FILE-MAX_JSON:
        raise ValueError('H16 literal bundle exceeds size limits')
    root = _root(storage_root,create=True)
    directory = root/('h16-resume-'+uuid.uuid4().hex)
    directory.mkdir()
    with _lease(directory):
        state,pending = directory/'state.safetensors',directory/'state.safetensors.partial'
        save_file(tensors,str(pending),metadata={'h16_resume':payload})
        with pending.open('rb+') as stream:
            header = struct.unpack('<Q',stream.read(8))[0]
            if not 0 < header <= MAX_JSON:
                raise ValueError('H16 literal header exceeds limit')
            os.fsync(stream.fileno())
        os.replace(pending,state)
        _values(bundle,result)
        manifest = dict(schema=SCHEMA,state_file=state.name,state_sha256=file_sha(state),state_bytes=state.stat().st_size,
            values_identity=_input_identity(raw),index=result.index,implementation=_implementation(),
            automatic_cache_reuse=False,provider_equivalence_certified=False,quality_accepted=False)
        data = canonical(manifest).encode('utf8')
        if len(data) > MAX_JSON:
            raise ValueError('H16 literal manifest exceeds limit')
        pending,path = directory/'manifest.json.partial',directory/'manifest.json'
        with pending.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending,path)
    return path.relative_to(root).as_posix(),file_sha(path),canonical(manifest)


def load_h16_resume(storage_root,path,digest,expected_index):
    if type(expected_index) is not int or expected_index < 0:
        raise ValueError('H16 literal expected window invalid')
    root,digest = _root(storage_root),_digest(digest)
    path = _path(root,path)
    if path.name != 'manifest.json':
        raise ValueError('Select H16 literal manifest.json')
    with _lease(path.parent):
        manifest = _json_file(path)
        names = {'schema','state_file','state_sha256','state_bytes','values_identity','index','implementation',
                 'automatic_cache_reuse','provider_equivalence_certified','quality_accepted'}
        if file_sha(path) != digest:
            raise ValueError('H16 literal manifest SHA mismatch')
        if (set(manifest) != names or manifest['schema'] != SCHEMA or manifest['state_file'] != 'state.safetensors'
                or type(manifest['index']) is not int or manifest['index'] != expected_index
                or manifest['implementation'] != _implementation()
                or any(manifest[k] is not False for k in ('automatic_cache_reuse','provider_equivalence_certified','quality_accepted'))):
            raise ValueError('H16 literal implementation/expected window differs')
        state = _path(root,(path.parent/'state.safetensors').relative_to(root).as_posix())
        if (type(manifest['state_bytes']) is not int or not 8 < manifest['state_bytes'] <= MAX_FILE
                or state.stat().st_size != manifest['state_bytes'] or file_sha(state) != _digest(manifest['state_sha256'])):
            raise ValueError('H16 literal state size/SHA mismatch')
        with state.open('rb') as stream:
            header = struct.unpack('<Q',stream.read(8))[0]
        if not 0 < header <= MAX_JSON or header > state.stat().st_size-8:
            raise ValueError('H16 literal header exceeds limit')
        with safe_open(str(state),framework='pt',device='cpu') as handle:
            metadata = handle.metadata()
            if type(metadata) is not dict or set(metadata) != {'h16_resume'}:
                raise ValueError('H16 literal metadata inventory differs')
            descriptor = json.loads(metadata['h16_resume'])
            tensors = {key:handle.get_tensor(key) for key in handle.keys()}
        used = set()
        raw = _decode_metadata_value(descriptor,tensors,used,path='h16.resume')
        if used != set(tensors) or _input_identity(raw) != manifest['values_identity']:
            raise ValueError('H16 literal values differ')
        bundle,result = _rebuild(raw)
        if result.index != expected_index or file_sha(path) != digest or file_sha(state) != manifest['state_sha256']:
            raise ValueError('H16 literal changed during load')
    return bundle,result,canonical(dict(status='literal_H16_loaded_no_sampling_encoding_lift_or_noise',
        index=result.index,quality_accepted=False,provider_equivalence_certified=False,automatic_cache_reuse=False))
