"""Explicit saved/adopted external RGB/PCM context, never native ancestry."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import director_external as takes, director_project as projects, long_video, core
from .director_media import run_media
from .patch_stack_policy import _execution_selection, model_identity_matches, nonportable_component_identity

SOURCE_TYPE = 'T8_EXTERNAL_CONTINUATION_SOURCE'
CONTEXT_TYPE = 'T8_EXTERNAL_REENCODED_CONTEXT'
SCHEMA = 't8.external-continuation-source.v1'
CONTEXT_SCHEMA = 't8.external-reencoded-context.v1'
AUDIO_POLICIES = {'video_only', 'reencode_stereo_pcm_context'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def tensor_identity(value):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError('External context tensor must be finite')
    data = value.detach().cpu().contiguous()
    return {'shape': list(data.shape), 'dtype': str(data.dtype),
            'sha256': hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()}


def implementation():
    from . import director_media, director_media_worker, conditioning, progressive_producers
    paths = [Path(module.__file__).resolve() for module in
             (takes, projects, director_media, director_media_worker, long_video, core, conditioning, progressive_producers)]
    paths.append(Path(__file__).resolve())
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def current_binding(store, output_root, request):
    if set(request) != {'project_id', 'shot_id', 'take_id', 'context_frames', 'audio_policy'}:
        raise ValueError('External continuation request contains unknown generation/ancestor fields')
    project_id, shot_id, take_id = (projects.identity(request[key]) for key in ('project_id', 'shot_id', 'take_id'))
    count, policy = request['context_frames'], request['audio_policy']
    if type(count) is not int or count not in long_video.CONTEXT_FRAME_STEPS or policy not in AUDIO_POLICIES:
        raise ValueError('Select an explicit 5/22/39-frame tail and audio-context policy')
    with projects._LOCK:
        project = store.load(project_id)
        if project['revision'] < 1:
            raise ValueError('Save and manually adopt the external take before continuing')
        shot = next((s for s in project['doc']['shots'] if s['id'] == shot_id), None)
        if shot is None or shot.get('adoptedResultId') != 'external:' + take_id:
            raise ValueError('Selected external take is not this saved shot\'s manually adopted version')
        receipt = takes._read_receipt(takes._receipt_path(store, project_id, take_id), project_id, take_id)
        if receipt['shot_id'] != shot_id:
            raise ValueError('External take belongs to another shot')
        record = takes._record(receipt, output_root)  # Complete immutable copied-file SHA, no asset fallback.
        clock = record['decoded_clock']
        if clock['video']['fps'] != {'num': 24, 'den': 1} or not clock['zero_origin_film_compatible']:
            raise ValueError('First bridge supports actual zero-origin 24fps CFR only; explicitly convert a new asset')
        trim = deepcopy(shot.get('filmTrim', {'in_frame': 0, 'out_frame': record['media']['frames']}))
        if (set(trim) != {'in_frame', 'out_frame'} or any(type(v) is not int for v in trim.values())
                or not 0 <= trim['in_frame'] < trim['out_frame'] <= record['media']['frames']
                or trim['out_frame'] - trim['in_frame'] < count):
            raise ValueError('Saved adopted trim does not contain the selected complete motion tail')
        if policy == 'reencode_stereo_pcm_context':
            audio = clock['audio']
            if not audio or audio['channels'] != 2:
                raise ValueError('Audio continuation requires an actual stereo track; no silent channel duplication')
            if any((frame * audio['sample_rate']) % 24 for frame in (trim['out_frame'] - count, trim['out_frame'])):
                raise ValueError('Audio window boundaries are not exact source samples; no implicit rounding')
            if audio['samples'] < trim['out_frame'] * audio['sample_rate'] // 24:
                raise ValueError('Actual source audio does not cover the adopted motion tail')
        return {'schema': SCHEMA, 'request': deepcopy(request), 'project_revision': project['revision'],
                'project_sha256': projects.sha(project), 'shot_revision': shot['rev'], 'film_trim': trim,
                'frame_interval': [trim['out_frame'] - count, trim['out_frame']],
                'receipt_sha256': record['receipt_sha256'], 'media_sha256': record['media_sha256'],
                'output_file': receipt['output_file'], 'decoded_clock': clock,
                'origin': 'external', 'original_native_latent_available': False,
                'has_sampling_ancestor': False, 'can_resample_original': False,
                'implementation': implementation()}


@dataclass(frozen=True)
class ExternalSource:
    store: object
    output_root: Path
    contract_json: str

    @property
    def binding(self):
        return json.loads(self.contract_json)

    def verify(self):
        binding = self.binding
        current = current_binding(self.store, self.output_root, binding['request'])
        if current != binding:
            raise ValueError('External take/adoption/trim/project/code changed; capture a new explicit source')
        return binding

    @property
    def media(self):
        return projects.contained(self.output_root, self.binding['output_file'])


def capture_source(store, output_root, *, project_id, shot_id, take_id, context_frames=22, audio_policy='video_only'):
    request = dict(project_id=project_id, shot_id=shot_id, take_id=take_id,
                   context_frames=context_frames, audio_policy=audio_policy)
    binding = current_binding(store, Path(output_root).resolve(), request)
    actual = run_media('external', projects.contained(Path(output_root), binding['output_file']))
    if actual['decoded_clock'] != binding['decoded_clock']:
        raise ValueError('Recorded external clock differs from complete actual decode')
    source = ExternalSource(store, Path(output_root).resolve(), canonical(binding))
    source.verify()
    return source


def producer(component, role):
    import comfy.sd
    from .progressive_producers import native_producer_identity
    if type(component) is comfy.sd.VAE:
        return native_producer_identity(component, role)
    if hasattr(component, 'patcher'):
        return nonportable_component_identity(component, 'External context VAE wrapper has no portable identity',
                                             schema='t8.external-context.nonportable-vae.v1')
    # User wrappers still execute. This process-local selection is not a content
    # certificate, persistent cache key or permission to load saved native tails.
    return {'portable_cache_reuse': False, 'component': _execution_selection(component),
            'encode': _execution_selection(getattr(component, 'encode', None))}


@dataclass(frozen=True)
class ExternalContext:
    source: ExternalSource
    video_vae: object
    audio_vae: object
    context: dict
    last_frame: torch.Tensor
    contract_json: str

    def descriptor(self):
        binding = self.source.verify()
        value = {'schema': CONTEXT_SCHEMA, 'source': binding,
                 'context': {'metadata': deepcopy(self.context['metadata']),
                             'video_tail': tensor_identity(self.context['video_tail']),
                             'audio_tail': tensor_identity(self.context['audio_tail'])},
                 'last_frame': tensor_identity(self.last_frame),
                 'video_vae': producer(self.video_vae, 'video_vae'),
                 'audio_vae': producer(self.audio_vae, 'audio_vae') if self.audio_vae is not None else None,
                 'native_latent_reconstructed_from_RGB_PCM_not_original_state': True,
                 'additional_sampling_nfe': 0, 'automatic_accept': False,
                 'audio_mux_policy': 'no_original_movie_mutation_no_automatic_mux'}
        return value

    def verify(self, video_vae=None, audio_vae=None):
        if video_vae is not None and video_vae is not self.video_vae:
            raise ValueError('Use the same actual VAE that prepared this external context')
        if self.audio_vae is not None and audio_vae is not None and audio_vae is not self.audio_vae:
            raise ValueError('External PCM context belongs to a different audio VAE')
        expected = json.loads(self.contract_json)
        if not model_identity_matches(expected, self.descriptor()):
            raise ValueError('External reencoded context/tail/VAE was changed')
        return expected


def prepare_context(source, video_vae, width, height, audio_vae=None):
    if type(source) is not ExternalSource:
        raise ValueError('Capture an adopted saved external source; native accepted parents use their own route')
    binding = source.verify()
    count = binding['request']['context_frames']
    policy = binding['request']['audio_policy']
    if any(type(x) is not int or not 32 <= x <= 16384 or x % 32 for x in (width, height)):
        raise ValueError('External context canvas must be positive multiples of32')
    if policy != 'video_only' and audio_vae is None:
        raise ValueError('Connect audio VAE explicitly for PCM reencoding')
    with tempfile.TemporaryDirectory(prefix='t8-external-tail-') as root:
        output = Path(root)/'tail.npz'
        evidence = run_media('external_tail', source.media, start_frame=binding['frame_interval'][0],
            end_frame=binding['frame_interval'][1], include_audio=policy!='video_only', output_npz=output)
        if evidence['media_sha256'] != binding['media_sha256'] or evidence['evidence']['decoded_clock'] != binding['decoded_clock']:
            raise ValueError('External decoder used a different complete source/clock')
        if evidence['frame_interval'] != binding['frame_interval'] or evidence['output_sha256'] != projects.file_sha(output):
            raise ValueError('External tail output/interval differs from its worker receipt')
        with np.load(output, allow_pickle=False) as archive:
            if set(archive.files) != {'frames', 'audio'}:
                raise ValueError('External tail archive has unexpected data')
            pixels, pcm = archive['frames'], archive['audio']
            if (pixels.dtype != np.uint8 or list(pixels.shape) != evidence['frames_shape']
                    or pcm.dtype != np.float32 or list(pcm.shape) != evidence['audio_shape'] or not np.isfinite(pcm).all()):
                raise ValueError('External tail RGB/PCM shapes or values are invalid')
            frames = torch.from_numpy(pixels.copy()).float()/255
            if policy == 'video_only':
                if pcm.size:
                    raise ValueError('Video-only policy cannot silently use external audio')
                audio_tail = torch.zeros(1, 32, 2, round(count/24*core.AUDIO_LATENT_FPS))
            else:
                clock = binding['decoded_clock']['audio']
                samples = count * clock['sample_rate'] // 24
                if pcm.shape != (2, samples):
                    raise ValueError('External audio slice is not the exact selected stereo sample range')
                audio_tail = core.encode_audio_once(audio_vae,
                    {'waveform': torch.from_numpy(pcm.copy()).unsqueeze(0), 'sample_rate': clock['sample_rate']})
    source.verify()
    with torch.inference_mode():
        video_tail = video_vae.encode(core.resize_image(frames, width, height)).contiguous()
    if tuple(video_tail.shape) != (1, 24, long_video.CONTEXT_FRAME_STEPS[count], height//16, width//16):
        raise ValueError('External RGB reencode does not return the native H3 video geometry')
    tensor_identity(video_tail)
    tensor_identity(audio_tail)
    delta = int(audio_tail.shape[-1]) - long_video.FRAME_RESCALE * count
    overhang = delta if policy != 'video_only' and 0 <= delta < 1 else 0.
    metadata = {'source_segment_index': 0, 'max_context_frames': count, 'audio_overhang': float(overhang),
        'audio_reencoded': policy != 'video_only', 'audio_context_pcm_frame_interval': binding['frame_interval'] if policy!='video_only' else None,
        'native_audio_latent_end_delta_tokens': float(delta) if policy!='video_only' else None,
        'audio_overhang_policy': 'original_H3_context_grid_rule_not_PCM_time_normalization',
        'origin': 'external_reencoded', 'local_conditioning_slot_only_not_sampling_ancestor': True,
        'original_native_latent_available': False, 'external_source_sha256': projects.sha(binding),
        'width': width, 'height': height, 'context_audio': 'video_only' if policy=='video_only' else 'video_and_audio'}
    context = {'schema': long_video.LONG_VIDEO_SCHEMA, 'empty': False, 'video_tail': video_tail,
               'audio_tail': audio_tail, 'metadata': metadata}
    long_video._validate_context(context, 1, count, width, height)
    provisional = ExternalContext(source, video_vae, audio_vae if policy!='video_only' else None,
                                  context, frames[-1:].clone(), '')
    descriptor = provisional.descriptor()
    result = ExternalContext(source, provisional.video_vae, provisional.audio_vae, context, provisional.last_frame,
                             canonical(descriptor))
    result.verify()
    return result


def condition(contexts, *, clip, video_vae, audio_vae, prompt, length, **options):
    if type(contexts) is not ExternalContext:
        raise ValueError('Connect the explicit external reencoded context')
    contexts.verify(video_vae, audio_vae)
    metadata = contexts.context['metadata']
    if type(length) is not int or length <= metadata['max_context_frames']:
        raise ValueError('Continuation must generate new frames beyond its motion tail')
    forbidden = {'model', 'context', 'segment_index', 'context_frames', 'context_audio', 'width', 'height', 'return_details'}
    if set(options) & forbidden:
        raise ValueError('External context geometry/origin cannot be overridden by native ancestor fields')
    result = long_video.build_long_video_conditioning(clip, video_vae, audio_vae, contexts.context,
        1, metadata['max_context_frames'], metadata['context_audio'], prompt,
        metadata['width'], metadata['height'], length, **options)
    contexts.verify(video_vae, audio_vae)
    positive = [[embedding, {**settings, 't8_external_continuation': json.loads(contexts.contract_json)}]
                for embedding, settings in result[0]]
    report = json.loads(result[5])
    report['external_continuation'] = {'source': contexts.source.binding,
        'RGB_PCM_reencode_not_original_latent': True, 'automatic_accept': False, 'additional_sampling_nfe': 0}
    return (positive, *result[1:5], canonical(report))
