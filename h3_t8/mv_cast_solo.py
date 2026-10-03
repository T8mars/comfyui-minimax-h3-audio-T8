"""Explicit independent A/B solo MV route; no simultaneous duet or quality claim."""
from copy import copy, deepcopy
import json
import re

import torch

from . import mv_lipsync_advanced as mv

SCHEMA = 't8.minimax_h3.mv_cast_solo_plan.v1'
TYPE = 'H3_T8_MV_CAST_SOLO_PLAN'
ROUTE = 'cast_solo_v4'


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(field + ' must be nonempty bounded text')
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate assignment field: ' + key)
        result[key] = value
    return result


def _lyrics(value):
    if type(value) is not str or len(value) > 4000 or re.search(r'<[^>]*>', value):
        raise ValueError('Vocal text is bounded literal content, not native control tags')
    return value


def validate_plan(value):
    if not isinstance(value, dict):
        raise ValueError('Connect the explicit cast-solo planner')
    plan = deepcopy(value)
    claimed = plan.pop('prompt_plan_hash', None)
    if plan.get('schema') != SCHEMA or plan.get('type') != TYPE or claimed != mv._hash(plan):
        raise ValueError('Cast-solo plan type/schema/content hash mismatch')
    actors = plan.get('performers')
    if (type(actors) is not list or len(actors) != 2
            or any(type(row) is not dict for row in actors)
            or [row.get('id') for row in actors] != ['A', 'B']):
        raise ValueError('Exactly two explicitly bound A/B performer slots are required')
    for row in actors:
        if set(row) != {'id', 'description'}:
            raise ValueError('Unknown cast fields')
        _text(row['description'], 'performer description')
    template = deepcopy(plan['template'])
    mv.validate_mv_vocal_lock_visual_prompt_plan(template)
    scenes = template['scene_plan']['scenes']
    segments = plan.get('segments')
    if type(segments) is not list or len(segments) != len(scenes):
        raise ValueError('Every scene needs one explicit performer and cue')
    used = set()
    language = _text(plan.get('vocal_language'), 'vocal_language')
    if language != mv._single_line(language) or '[' in language or ']' in language or '<' in language:
        raise ValueError('Vocal language must be a plain native language label')
    for index, (scene, item) in enumerate(zip(scenes, segments, strict=True)):
        if (type(item) is not dict or item.get('index') != index or type(item.get('index')) is not int
                or item.get('performer_id') not in {'A', 'B'}
                or item.get('start_frame') != scene['start_frame']
                or item.get('end_frame') != scene['end_frame']
                or item.get('performance_state') != scene['performance_state']):
            raise ValueError('Cast scene index/performer/exact song-frame cue mismatch')
        raw = _lyrics(item.get('exact_vocal_text'))
        active = item['performance_state'] == 'vocal_active'
        if raw.strip() and not active:
            raise ValueError('Explicit lyrics cannot be assigned to a non-vocal cue')
        if item.get('cue_id') != 'scene-' + str(index + 1) or item.get('reset_policy') != 'independent_ref2va_no_previous_tail':
            raise ValueError('Explicit cue/reset contract is missing')
        used.add(item['performer_id'])
        actor = actors[0 if item['performer_id'] == 'A' else 1]
        binding = '<Subject 1> is the exact performer shown in <Picture 1>: ' + mv._single_line(actor['description']) + '.'
        if type(item.get('prompt')) is not str or item['prompt'].count(binding) != 1:
            raise ValueError('Prompt performer description disagrees with the assigned actual reference slot')
        spans = re.findall(r'<d>(.*?)</d>', item['prompt'], flags=re.DOTALL)
        expected_spans = ['[' + language + '] ' + raw] if raw.strip() and active else []
        if spans != expected_spans or item.get('exact_vocal_text_supplied') is not bool(raw.strip()):
            raise ValueError('Literal supplied lyrics disagree with the native dialogue span')
        mv._validate_official_ref2va_prompt(item['prompt'], vocal_active=item['performance_state'] == 'vocal_active')
        if mv._SINGLE_SUBJECT_VISUAL_CONTRACT not in item['prompt']:
            raise ValueError('Cast-solo scene lost the exact one-person contract')
    if used != {'A', 'B'}:
        raise ValueError('Cast-solo preview must actually assign at least one scene to each performer')
    if (plan.get('audio_contract') != template['audio_contract']
            or plan.get('scene_plan') != template['scene_plan']
            or plan.get('visual_contract') != template['visual_contract']
            or plan.get('external_api_used') is not False):
        raise ValueError('Cast-solo plan lost its original scene/audio/visual contracts')
    plan['prompt_plan_hash'] = claimed
    return plan


def build_plan(scene_plan, performer_a_description, performer_b_description, assignments_json,
               global_creative_prompt, visual_style, scene_directions_json='',
               vocal_content_type='singing', vocal_language='English'):
    scene_plan = mv.validate_mv_scene_plan(scene_plan)
    actors = [{'id': 'A', 'description': _text(performer_a_description, 'A')},
              {'id': 'B', 'description': _text(performer_b_description, 'B')}]
    if not isinstance(assignments_json, str) or len(assignments_json.encode('utf8')) > 512 * 1024:
        raise ValueError('Assignments must be a bounded explicit JSON list')
    assignments = json.loads(assignments_json, object_pairs_hook=_unique_pairs)
    if type(assignments) is not list or len(assignments) != scene_plan['scene_count']:
        raise ValueError('Assignments must cover every scene, without guessing or cycling')
    exact_text = []
    for index, row in enumerate(assignments):
        if (type(row) is not dict or set(row) != {'scene_index', 'performer_id', 'exact_vocal_text'}
                or type(row['scene_index']) is not int or row['scene_index'] != index
                or row['performer_id'] not in {'A', 'B'} or type(row['exact_vocal_text']) is not str):
            raise ValueError('Use explicit zero-based scene_index, A/B performer_id, exact_vocal_text')
        exact_text.append(row['exact_vocal_text'])
        _lyrics(row['exact_vocal_text'])
        if row['exact_vocal_text'].strip() and scene_plan['scenes'][index]['performance_state'] != 'vocal_active':
            raise ValueError('Explicit lyrics cannot be assigned to a non-vocal cue')
    language = mv._single_line(_text(vocal_language, 'vocal_language'))
    if '[' in language or ']' in language or '<' in language:
        raise ValueError('Vocal language must be a plain native language label')
    variants = {}
    for actor in actors:
        variants[actor['id']] = mv.build_mv_vocal_lock_visual_prompt_plan(
            scene_plan, global_creative_prompt, actor['description'], visual_style,
            scene_directions_json, vocal_content_type, language,
            json.dumps(exact_text, ensure_ascii=False))[0]
        # The legacy V3 compiler deliberately single-lines text. This opt-in
        # plan preserves the actual supplied words/whitespace within its one
        # native dialogue span without rewriting the old compiler.
        variant = variants[actor['id']]
        for index, segment in enumerate(variant['segments']):
            raw = exact_text[index]
            normalized = mv._single_line(raw)
            if normalized and segment['performance_state'] == 'vocal_active' and raw != normalized:
                token = '<d>[' + language + '] ' + normalized + '</d>'
                if segment['prompt'].count(token) != 1:
                    raise ValueError('Exact native dialogue span is ambiguous; no guessed replacement')
                segment['prompt'] = segment['prompt'].replace(token, '<d>[' + language + '] ' + raw + '</d>', 1)
        variant.pop('prompt_plan_hash')
        variant['prompt_plan_hash'] = mv._hash(variant)
        mv.validate_mv_vocal_lock_visual_prompt_plan(variant)
    template = variants['A']
    segments = []
    for index, assignment in enumerate(assignments):
        segment = deepcopy(variants[assignment['performer_id']]['segments'][index])
        segment.update(performer_id=assignment['performer_id'], cue_id='scene-' + str(index + 1),
                       exact_vocal_text=assignment['exact_vocal_text'],
                       reset_policy='independent_ref2va_no_previous_tail')
        segments.append(segment)
    plan = {'schema': SCHEMA, 'type': TYPE, 'performers': actors, 'segments': segments, 'vocal_language': language,
            'template': template, 'scene_plan': template['scene_plan'],
            'audio_contract': template['audio_contract'], 'visual_contract': template['visual_contract'],
            'external_api_used': False}
    plan['prompt_plan_hash'] = mv._hash(plan)
    validate_plan(plan)
    report = {'schema': 't8.mv.cast-solo.plan-report.v1', 'prompt_plan_hash': plan['prompt_plan_hash'],
              'scene_count': len(segments), 'assignments': [{key: row[key] for key in ('index', 'performer_id', 'cue_id', 'start_frame', 'end_frame', 'reset_policy')} for row in segments],
              'boundary': 'One solo performer per independent scene, not simultaneous duet or a lip-sync/identity quality guarantee. Exact supplied text is retained, never ASR-guessed.'}
    return plan, json.dumps(report, ensure_ascii=False, indent=2)


def references(reference_a, reference_b):
    result = {'A': reference_a, 'B': reference_b}
    for name, value in result.items():
        if (type(value) is not torch.Tensor or value.ndim != 4 or value.shape[0] != 1
                or value.shape[-1] not in (3, 4) or min(value.shape[1:3]) < 16
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or value.min() < 0 or value.max() > 1):
            raise ValueError(name + ' needs exactly one finite 0..1 IMAGE; no implicit image batch or resize')
    return result


def independent_clock_container(model):
    """Keep Core's sampler installation off the caller's root network.

    This is not a second diffusion model or a weight copy. Native clone
    semantics retain LoRA, hooks, callbacks, load bookkeeping and the actual
    shared diffusion tensors. Only the root module table and its object-patch
    backup table are private. Opaque patchers keep their own clone/execution
    semantics and remain nonportable; never guess their internals.
    """
    import comfy.model_patcher
    native = (comfy.model_patcher.ModelPatcher, comfy.model_patcher.ModelPatcherDynamic)
    if (type(model) not in native or 'clone' in vars(model)
            or not isinstance(model.model, torch.nn.Module)):
        return model
    network = copy(model.model)
    network._modules = dict(model.model._modules)
    _network, (weights, buffers, objects, pinned) = model.get_clone_model_override()
    return model.clone(model_override=(network, (weights, buffers, dict(objects), pinned)))


class CastRuntimeBinding:
    """Full input bytes and actual selected producers, never model_id labels.

    Unknown owners remain executable and nonce-bound/nonportable. Only within
    this call may their observed weight/selection snapshot match; no assertion
    about arbitrary hidden callable state or pretrained quality is made.
    """
    def __init__(self, model, clip, video_vae, audio_vae, refs, full_song, vocal_lock_audio):
        self.components = dict(model=model, clip=clip, video_vae=video_vae, audio_vae=audio_vae)
        self.media = dict(references=refs, full_song=full_song, vocal_lock_audio=vocal_lock_audio)
        self.sampling_model = independent_clock_container(model)
        self.identity = self._capture()
        self._diagnostic_model_state = self._model_snapshot()

    def _model_snapshot(self):
        import comfy.model_patcher
        from .long_video_dual_identity import content_identity, _original_state
        model = self.components['model']
        if type(model) not in (comfy.model_patcher.ModelPatcher, comfy.model_patcher.ModelPatcherDynamic):
            return None
        # Diagnostic only: do not replace the selection/owner/cache contract.
        return content_identity(_original_state(model, model.model_state_dict()))

    def _capture(self):
        import comfy.sd
        from .long_video_dual_identity import content_identity
        from .modular_sampling.results import selected_model_identity
        from .progressive_producers import native_producer_identity, _native_producer_description
        from .patch_stack_policy import nonportable_component_identity, UnverifiedModelStack

        producers = {'model': selected_model_identity(self.components['model'])}
        for role in ('clip', 'video_vae', 'audio_vae'):
            component = self.components[role]
            expected = comfy.sd.CLIP if role == 'clip' else comfy.sd.VAE
            if type(component) is expected:
                try:
                    description = _native_producer_description(component, role)
                except UnverifiedModelStack:
                    producers[role] = native_producer_identity(component, role)
                else:
                    # Keep the actual inspected fields, not only their digest:
                    # failures can name a location without exposing media bytes.
                    producers[role] = {'role': role, 'sha256': mv._hash(description),
                                       'native_description': description}
            else:
                producers[role] = nonportable_component_identity(component,
                    'Cast-solo component wrapper has no native producer adapter', schema='t8.mv.cast-solo.nonportable-producer.v1')
        from .long_video_dual_identity import _implementation
        from .conditioning import build_conditioning
        return {'schema': 't8.mv.cast-solo.runtime.v1', 'media': content_identity(self.media),
                'producers': producers, 'implementation': {
                    'cast': _implementation(CastRuntimeBinding),
                    'mv': _implementation(mv.run_local_mv_in_node_loop),
                    'conditioning': _implementation(build_conditioning)}}

    def verify(self):
        from .patch_stack_policy import model_identity_matches
        from .modular_sampling.results import _identity_difference_path
        current = self._capture()
        if not model_identity_matches(self.identity, current):
            location = str(_identity_difference_path(self.identity, current))
            if location.startswith('$.producers.model.'):
                state_location = _identity_difference_path(self._diagnostic_model_state, self._model_snapshot())
                if state_location is not None:
                    location += '; native MODEL state ' + state_location
            raise ValueError('Cast-solo reference/audio or actual producer content changed during rendering at '
                             + location)

    @property
    def portable(self):
        return all(value.get('portable_cache_reuse') is not False for value in self.identity['producers'].values())


def render(model, clip, video_vae, audio_vae, reference_a, reference_b, full_song,
           vocal_lock_audio, cast_solo_plan, **settings):
    plan = validate_plan(cast_solo_plan)
    refs = references(reference_a, reference_b)
    audio_contract = mv._validate_vocal_lock_audio_contract(full_song, vocal_lock_audio, plan['scene_plan']['total_frames'])
    # Bind the effective existing H3 encode policy, not its pre-shim default.
    # encode_audio_once already applies this exact compatibility shim. Keep the
    # field bound afterwards: an unexpected later crop change must still fail.
    from .core import ensure_h3_audio_vae_non_aligned_crop_compat
    ensure_h3_audio_vae_non_aligned_crop_compat(audio_vae)
    binding = CastRuntimeBinding(model, clip, video_vae, audio_vae, refs, full_song, vocal_lock_audio)
    result = mv.run_local_mv_in_node_loop(
        model, clip, video_vae, audio_vae, refs, full_song, plan,
        vocal_lock_audio=vocal_lock_audio, prompt_plan_validator=validate_plan,
        route_revision=ROUTE, loop_state_name='mv_cast_solo_state.json',
        state_schema='t8.minimax_h3.mv_cast_solo_loop.v1', cast_binding=binding, **settings)
    report = json.loads(result[4])
    report.update(route=ROUTE, assignments=[{'scene': row['index'], 'performer_id': row['performer_id'], 'cue_id': row['cue_id']} for row in plan['segments']],
                  vocal_lock_audio_validation=audio_contract, conditioning_audio='vocal_lock_audio',
                  delivery_audio='full_song_muxed_once',
                  runtime_inputs_portable=binding.portable, human_quality_accepted=False,
                  boundary='New explicit independent A/B solo route; no previous performer tail. Original audio content and cue mapping bound to resume contract; perceptual identity/lip-sync still require normal-speed human review.')
    return (*result[:4], json.dumps(report, ensure_ascii=False, indent=2))
