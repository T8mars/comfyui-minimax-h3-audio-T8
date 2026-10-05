"""Literal scoped H16 AV windows; never reinterpret old crossfade caches."""
from dataclasses import asdict
import os
from pathlib import Path
import struct
import uuid

import comfy.nested_tensor
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .. import temporal_dialogue_scope as scope, temporal_dialogue_bank as banks
from .. import temporal_dialogue_encoding as encoding, temporal_dialogue_identity as identity
from . import temporal_h16 as runtime, temporal_h16_effects as effects
from . import chunked_source, chunked_stages
from .chunked_stages import _check_segment
from .results import _input_identity, canonical
from .storage import MAX_FILE, MAX_JSON, _root, _path, _lease, _digest, _json_file, file_sha, fingerprint_stage
from .temporal_chunked_storage import _typed_receipt


SCHEMA = 't8.temporal-h16.frozen-accepted-av.v1'


def _implementation():
    from .. import chunked_two_pass_upscale_advanced as legacy
    from .. import sampling
    from . import temporal_chunked_storage, temporal_chunked_relay
    modules = (runtime, effects, scope, banks, encoding, identity, chunked_source, chunked_stages,
               legacy, sampling, temporal_chunked_storage, temporal_chunked_relay)
    paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    return {path.name: file_sha(path) for path in paths}


def _context_identity(context):
    return _input_identity(dict(source=context.source_latent, audio=context.original_audio,
        video_noise=context.global_video_noise, audio_noise=context.global_audio_noise,
        inherited_video_mask=context.target_inherited_video_mask, noise_seed=context.noise_seed))


def verify_h16_scoped_window(result, source_segment, lifted_segment, spec, context, plan, bank):
    _check_segment(source_segment, spec, context, plan)
    encoded = banks.select_window_conditioning(bank, context.source_latent, plan, spec.index)
    if (type(result) is not runtime.ScopedH16Result or result.plan_sha256 != spec.plan_sha256
            or result.source_identity != spec.source_identity or result.bank_sha256 != bank.sha256
            or result.index != spec.index or result.count != spec.count):
        raise ValueError('Scoped H16 freeze source/plan/bank/window differs')
    scope._validate_receipt(result.dialogue_receipt)
    receipt = result.dialogue_receipt
    if (receipt.window != encoded.compiled.window or receipt.plan_sha256 != bank.dialogue_plan.sha256
            or receipt.text_sha256 != encoded.compiled.sha256
            or receipt.published_audio_sha256 != runtime._audio_sha(result)
            or ((spec.index == 0) != (receipt.previous_receipt_sha256 is None))):
        raise ValueError('Scoped H16 actual audio receipt or ownership differs')
    if type(result.output_latent) is not dict or set(result.output_latent) != {'samples'}:
        raise ValueError('Scoped H16 frozen output has unsupported metadata')
    video, audio = result.output_latent['samples'].unbind()
    end_audio = context.original_audio.shape[-1] if spec.index+1 == spec.count else spec.audio_end
    if (tuple(video.shape) != (1, 24, spec.end_token, spec.target_height//16, spec.target_width//16)
            or tuple(audio.shape) != (1, 32, 2, end_audio)
            or not torch.isfinite(video).all() or not torch.isfinite(audio).all()):
        raise ValueError('Scoped H16 frozen AV coverage is invalid')
    return dict(contract=runtime.CONTRACT, plan_sha256=spec.plan_sha256, source_identity=spec.source_identity,
        bank_sha256=bank.sha256, index=spec.index, count=spec.count, output_identity=result.output_identity,
        receipt=asdict(receipt), context_identity=_context_identity(context), lifted_identity=_input_identity(lifted_segment))


def _rebuild(tensors, binding, context):
    if set(tensors) != {'video', 'audio'}:
        raise ValueError('Scoped H16 frozen tensor inventory differs')
    source_video, source_audio = context.source_latent['samples'].unbind()
    output = {'samples': comfy.nested_tensor.NestedTensor((
        tensors['video'].to(device=source_video.device), tensors['audio'].to(device=source_audio.device)))}
    return runtime.ScopedH16Result(binding['plan_sha256'], binding['source_identity'], binding['bank_sha256'],
        binding['index'], binding['count'], output, binding['output_identity'], _typed_receipt(binding['receipt']))


def save_h16_scoped_window(result, source_segment, lifted_segment, spec, context, plan, bank, storage_root):
    args = source_segment, lifted_segment, spec, context, plan, bank
    binding = verify_h16_scoped_window(result, *args)
    root = _root(storage_root, create=True)
    directory = root / ('scoped-h16-'+uuid.uuid4().hex)
    directory.mkdir()
    with _lease(directory):
        video, audio = result.output_latent['samples'].unbind()
        tensors = {'video': video.detach().cpu().contiguous().clone(), 'audio': audio.detach().cpu().contiguous().clone()}
        if sum(v.numel()*v.element_size() for v in tensors.values()) > MAX_FILE-MAX_JSON:
            raise ValueError('Scoped H16 frozen tensors exceed size limit')
        if verify_h16_scoped_window(_rebuild(tensors, binding, context), *args) != binding:
            raise ValueError('Scoped H16 literal tensor copy differs')
        pending, state = directory/'state.safetensors.partial', directory/'state.safetensors'
        save_file(tensors, str(pending))
        with pending.open('rb+') as stream:
            header = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < header <= MAX_JSON:
                raise ValueError('Scoped H16 tensor header exceeds limit')
            os.fsync(stream.fileno())
        os.replace(pending, state)
        verify_h16_scoped_window(result, *args)
        manifest = dict(schema=SCHEMA, state_file=state.name, state_sha256=file_sha(state), state_bytes=state.stat().st_size,
            binding=binding, implementation=_implementation(), automatic_cache_reuse=False,
            execution_identity_certified=False, quality_accepted=False)
        payload = canonical(manifest).encode('utf8')
        if len(payload) > MAX_JSON:
            raise ValueError('Scoped H16 manifest exceeds size limit')
        pending, path = directory/'manifest.json.partial', directory/'manifest.json'
        with pending.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    return result.output_latent, result, path.relative_to(root).as_posix(), file_sha(path), canonical(manifest)


def load_h16_scoped_window(source_segment, lifted_segment, spec, context, plan, bank, storage_root,
                            artifact_path, artifact_sha256):
    args = source_segment, lifted_segment, spec, context, plan, bank
    _check_segment(source_segment, spec, context, plan)
    banks.select_window_conditioning(bank, context.source_latent, plan, spec.index)
    root, digest = _root(storage_root), _digest(artifact_sha256)
    path = _path(root, artifact_path)
    if path.name != 'manifest.json':
        raise ValueError('Select a completed scoped H16 manifest.json')
    with _lease(path.parent):
        manifest = _json_file(path)
        required = {'schema', 'state_file', 'state_sha256', 'state_bytes', 'binding', 'implementation',
                    'automatic_cache_reuse', 'execution_identity_certified', 'quality_accepted'}
        if file_sha(path) != digest:
            raise ValueError('Scoped H16 manifest SHA mismatch')
        if (set(manifest) != required or manifest['schema'] != SCHEMA or manifest['state_file'] != 'state.safetensors'
                or manifest['implementation'] != _implementation()
                or any(manifest[k] is not False for k in ('automatic_cache_reuse','execution_identity_certified','quality_accepted'))):
            raise ValueError('Unknown/stale scoped H16 implementation or contract')
        binding = manifest['binding']
        fields = {'contract','plan_sha256','source_identity','bank_sha256','index','count','output_identity',
                  'receipt','context_identity','lifted_identity'}
        if (type(binding) is not dict or set(binding) != fields or binding['contract'] != runtime.CONTRACT
                or binding['plan_sha256'] != spec.plan_sha256 or binding['source_identity'] != spec.source_identity
                or binding['bank_sha256'] != bank.sha256 or binding['index'] != spec.index or binding['count'] != spec.count
                or binding['context_identity'] != _context_identity(context)
                or binding['lifted_identity'] != _input_identity(lifted_segment)):
            raise ValueError('Scoped H16 frozen bank/source/noise/lift/window differs')
        state = _path(root, (path.parent/'state.safetensors').relative_to(root).as_posix())
        if (type(manifest['state_bytes']) is not int or not 8 < manifest['state_bytes'] <= MAX_FILE
                or state.stat().st_size != manifest['state_bytes'] or file_sha(state) != _digest(manifest['state_sha256'])):
            raise ValueError('Scoped H16 tensor size/SHA mismatch')
        with state.open('rb') as stream:
            header = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header <= MAX_JSON or header > state.stat().st_size-8:
            raise ValueError('Scoped H16 tensor header exceeds limit')
        with safe_open(str(state), framework='pt', device='cpu') as handle:
            if handle.metadata() not in (None, {}):
                raise ValueError('Scoped H16 frozen tensors have unexpected metadata')
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        result = _rebuild(tensors, binding, context)
        if verify_h16_scoped_window(result, *args) != binding:
            raise ValueError('Scoped H16 frozen output/receipt binding differs')
        if file_sha(state) != manifest['state_sha256'] or file_sha(path) != digest:
            raise ValueError('Scoped H16 frozen artifact changed during load')
    return result.output_latent, result, canonical(dict(status='explicit_scoped_h16_window_loaded',
        index=spec.index, automatic_cache_reuse=False, execution_identity_certified=False, quality_accepted=False))


def fingerprint_h16_scoped_window(storage_root, artifact_path):
    return fingerprint_stage(storage_root, artifact_path)
