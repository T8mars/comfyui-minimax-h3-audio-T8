"""Explicit external RGB/PCM motion effects; never an accepted native ancestor."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import external_continuation as bridge, long_video
from . import prompt_relay_long_video_advanced as window
from .director_project import sha
from .patch_stack_policy import model_identity_matches, UnverifiedModelStack

RELAY_TYPE = 'T8_EXTERNAL_RELAY_WINDOW'
SCOPE_TYPE = 'T8_EXTERNAL_MOTION_EFFECT_SCOPE'
KEY = 't8_external_motion_effect_scope_v1'
SCHEMA = 't8.external-motion-effects.v1'
SOURCE_MARKER = 't8_external_continuation'


def _identity(value):
    from .modular_sampling.results import _input_identity
    return _input_identity(value)


def _condition_identity(value):
    from .modular_sampling.results import _conditions
    return _conditions(value)


def _implementation():
    from .modular_sampling.progressive_effect_identity import function_identity
    from .modular_sampling.results import _conditions, _input_identity
    functions = (project_relay, _project, relay_condition, bind_scope, project_scope, _guides, _identity,
                 _condition_identity, _conditions, _input_identity, _implementation,
                 ExternalRelayWindow.descriptor,
                 ExternalRelayWindow.verify, ExternalEffectScope.verify,
                 ExternalEffectScope.validate_payload, ExternalEffectScope.descriptor)
    return {'files': {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for module in (bridge, long_video, window)},
            'self_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'functions': [function_identity(function) for function in functions]}


def _project(context, global_plan, length, generated_start_frame):
    if type(context) is not bridge.ExternalContext:
        raise ValueError('Use the explicit external reencoded context, not a native ancestor')
    context.verify()
    count = context.context['metadata']['max_context_frames']
    if type(length) is not int or length <= count:
        raise ValueError('External Relay render length must generate frames beyond the motion tail')
    if type(generated_start_frame) is not int or generated_start_frame < count:
        raise ValueError('External Relay generated_start_frame must be an integer at or beyond context_frames')
    # This is an independently declared Relay clock. The numerical "accepted"
    # fields in the old window projector are NOT an adoption/parent certificate.
    return window.project_prompt_relay_plan_to_long_video_window(global_plan, 1, length, count,
        generated_start_frame / 24, (generated_start_frame + length - count) / 24)[0]


@dataclass(frozen=True)
class ExternalRelayWindow:
    context: bridge.ExternalContext
    global_plan: dict
    projected: dict
    length: int
    generated_start_frame: int
    contract_json: str

    def descriptor(self):
        projected = _project(self.context, self.global_plan, self.length, self.generated_start_frame)
        if projected != self.projected:
            raise ValueError('External Relay window or global Plan changed')
        return {'schema': SCHEMA, 'context': self.context.verify(), 'global_plan': self.global_plan,
                'projected': projected, 'length': self.length,
                'generated_start_frame': self.generated_start_frame,
                'independent_relay_clock_not_external_movie_or_native_parent_clock': True,
                'automatic_accept': False, 'implementation': _implementation()}

    def verify(self):
        current = self.descriptor()
        current['sha256'] = sha(current)
        if not model_identity_matches(json.loads(self.contract_json), current):
            raise ValueError('External Relay window source/context/implementation changed')
        return current


def project_relay(context, global_plan, length=124, generated_start_frame=22):
    global_plan = deepcopy(global_plan)
    projected = _project(context, global_plan, length, generated_start_frame)
    initial = ExternalRelayWindow(context, global_plan, projected, length, generated_start_frame, '')
    descriptor = initial.descriptor()
    descriptor['sha256'] = sha(descriptor)
    result = ExternalRelayWindow(context, global_plan, projected, length, generated_start_frame,
                                 bridge.canonical(descriptor))
    result.verify()
    return result


def relay_condition(context, projected, *, model, clip, video_vae, audio_vae, length=124, **options):
    if type(projected) is not ExternalRelayWindow or projected.context is not context:
        raise ValueError('Pair External Relay with the very same encoded context')
    projected.verify()
    context.verify(video_vae, audio_vae)
    if length != projected.length or type(length) is not int:
        raise ValueError('External Relay conditioning length differs from its projected window')
    forbidden = {'context', 'segment_index', 'context_frames', 'context_audio', 'width', 'height',
                 'prompt_relay_plan', 'prompt', 'return_details'}
    if set(options) & forbidden:
        raise ValueError('External Relay geometry, prompt and origin cannot be overridden')
    metadata = context.context['metadata']
    original = window.build_prompt_relay_long_video_conditioning(model, clip, video_vae, audio_vae,
        context.context, projected.projected, 1, metadata['max_context_frames'], metadata['context_audio'],
        metadata['width'], metadata['height'], length, **options)
    projected.verify()
    context.verify(video_vae, audio_vae)
    # Old report_only Relay intentionally returns the original MODEL. This new
    # external route still needs the unchanged native motion payload patch.
    selected = long_video.patch_long_video_model(original[0])
    positive = [[value, {**settings, SOURCE_MARKER: context.verify()}] for value, settings in original[1]]
    report = json.loads(original[-1])
    report['external_motion_effects'] = {'schema': SCHEMA, 'source': context.source.binding,
        'projection_sha256': projected.verify()['sha256'],
        'independent_relay_clock_not_native_ancestry': True,
        'RGB_PCM_reencode_not_original_latent': True, 'automatic_accept': False, 'sampling_calls': 0}
    return selected, positive, *original[2:-1], bridge.canonical(report)


def _guides(context, positive):
    if not isinstance(positive, list) or not positive:
        raise ValueError('External effects need the actual external positive conditioning')
    expected = context.verify()
    count = context.context['metadata']['max_context_frames']
    motion, audio = long_video._motion_context_blocks(context.context, count,
        context.context['metadata']['context_audio'] == 'video_and_audio')
    guides = []
    for entry in positive:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
            raise ValueError('External conditioning entries must be native embedding/settings pairs')
        settings = entry[1]
        if (settings.get(long_video.LONG_VIDEO_CONDITIONING_KEY) != long_video.LONG_VIDEO_SCHEMA
                or not model_identity_matches(settings.get(SOURCE_MARKER), expected)):
            raise ValueError('External effects positive is not paired to this source and encoded context')
        keyframes = settings.get('minimax_keyframes', [])
        refs = settings.get('minimax_refs', [])
        actual_motion = [item for item in keyframes if long_video.MOTION_FRAME_INDEX in item]
        actual_audio = [item for item in refs if long_video.MOTION_AUDIO_END_FRAME in item]
        if (_identity(actual_motion) != _identity(motion)
                or _identity(actual_audio) != _identity([] if audio is None else [audio])):
            raise ValueError('External effects actual motion guides differ from the reencoded RGB/PCM tail')
        frame_count = settings.get('minimax_frame_count')
        if type(frame_count) is not int or frame_count <= count:
            raise ValueError('External effects target frame count is invalid')
        guides.append({'keyframes': _identity(keyframes), 'refs': _identity(refs), 'frame_count': frame_count})
    if any(item != guides[0] for item in guides):
        raise ValueError('External effects scheduled conditions disagree on native motion guides')
    return guides[0]


@dataclass(frozen=True)
class ExternalEffectScope:
    context: bridge.ExternalContext
    base: object
    positive: list
    av_latent: dict
    contract_json: str

    def descriptor(self):
        return {'schema': SCHEMA, 'context': self.context.verify(), 'guides': _guides(self.context, self.positive),
                'positive': _condition_identity(self.positive), 'stage_source': _identity(self.av_latent),
                'has_sampling_ancestor': False, 'RGB_PCM_reencode_not_original_latent': True,
                'automatic_accept': False, 'implementation': _implementation()}

    def verify(self, model=None, av_latent=None):
        if type(self.context) is not bridge.ExternalContext:
            raise ValueError('External effect scope cannot be a native ancestor or prepared P7 phase')
        if model is not None and (model.model is not self.base or model.get_attachment(KEY) is not self):
            raise ValueError('External effect scope belongs to a different MODEL/context')
        current = self.descriptor()
        current['sha256'] = sha(current)
        if not model_identity_matches(json.loads(self.contract_json), current):
            raise ValueError('External effect source/conditions/AV/context/implementation changed')
        if av_latent is not None and _identity(av_latent) != current['stage_source']:
            raise ValueError('External effect stage AV differs from its independently bound source')
        return current

    def validate_payload(self, payload):
        current = self.verify()
        actual = {'keyframes': _identity(list(payload.get('keyframes') or [])),
                  'refs': _identity(list(payload.get('refs') or [])), 'frame_count': payload.get('frame_count')}
        if actual != current['guides']:
            raise RuntimeError('External effects actual runtime guides/frame clock differ from the bound condition')


def bind_scope(context, model, positive, av_latent):
    from . import enhance_a_video_advanced as feta
    if type(context) is not bridge.ExternalContext:
        raise ValueError('Bind external RGB/PCM effects separately from native accepted-parent effects')
    count = context.context['metadata']['max_context_frames']
    feta._assert_long_video_contract(model, segment_index=1, context_frames=count)
    if model.get_attachment(KEY) is not None:
        raise ValueError('External effect MODEL already has a bound scope; use an independent branch')
    video, _audio = bridge.core.nested_av_parts(av_latent)
    guide = _guides(context, positive)
    if (tuple(video.shape)[-2:] != (context.context['metadata']['height']//16,
                                   context.context['metadata']['width']//16)
            or long_video.pixel_frames_from_latent_t(video.shape[2]) != guide['frame_count']):
        raise ValueError('External effect AV canvas/frame count differs from actual conditioning')
    initial = ExternalEffectScope(context, model.model, positive, av_latent, '')
    descriptor = initial.descriptor()
    descriptor['sha256'] = sha(descriptor)
    scope = ExternalEffectScope(context, model.model, positive, av_latent, bridge.canonical(descriptor))
    selected = model.clone()
    selected.set_attachments(KEY, scope)
    scope.verify(selected, av_latent)
    return selected, scope, scope.contract_json


def project_scope(model):
    """Inspection clone only. Opaque selected encoders stay usable/nonportable."""
    scope = model.get_attachment(KEY)
    if scope is None:
        return model, None
    if type(scope) is not ExternalEffectScope or set(vars(scope)) != {
            'context', 'base', 'positive', 'av_latent', 'contract_json'}:
        raise UnverifiedModelStack('External effect scope has an unknown owner')
    report = scope.verify(model)
    for name in ('video_vae', 'audio_vae'):
        encoder = report['context'][name]
        if encoder is not None and encoder.get('portable_cache_reuse') is False:
            raise UnverifiedModelStack('External effect encoder is usable but lacks portable content identity')
    clone = model.clone()
    clone.remove_attachments(KEY)
    return clone, report
