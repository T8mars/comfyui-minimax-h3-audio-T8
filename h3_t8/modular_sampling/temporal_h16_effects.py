"""External scoped H16 effects on the new real accepted-audio prefix contract.

Reuse the released authenticated native Relay and generic EAV machinery. Their
H16 validators can bind actual pieces independently of the old window driver;
new owners additionally bind this scope bank. No old source pins are changed.
"""
from dataclasses import dataclass

import comfy.patcher_extension as extension
import node_helpers
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from .. import enhance_a_video_advanced as feta
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..temporal_dialogue_bank import select_window_conditioning
from .contracts import StageContext
from .eav import EAVConfig, KEY as EAV_KEY, apply_stage_eav
from . import h16_effects, h16_relay
from .chunked_relay import _assert_paired_conditioning
from .results import _input_identity, canonical
from .temporal_chunked_relay import ScopedRelayConfig, _local_plan
from .temporal_h16 import CONTRACT, scoped_h16_piece


KEY = 't8_temporal_dialogue_h16_effects'
RUNTIME_TYPE = 'T8_TEMPORAL_H16_EFFECT_RUNTIME'


@dataclass(frozen=True)
class H16ScopedEffects:
    bank_sha256: str
    window_index: int
    piece_identity: dict
    sigma_identity: dict
    sampling_object: object
    relay_binding: dict | None
    relay_runtime: object
    eav_runtime: object


def bind_h16_scoped_effects(model, source_segment, lifted_segment, spec, context, plan,
                            sigmas, bank, previous=None, *, clip=None, relay_config=None, eav_config=None):
    window = scoped_h16_piece(source_segment, lifted_segment, spec, context, plan, bank, previous)
    encoded = select_window_conditioning(bank, context.source_latent, plan, spec.index)
    if (not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(sigmas) < 2
            or not sigmas.is_floating_point() or not torch.isfinite(sigmas).all()
            or torch.any(sigmas[:-1] < sigmas[1:]) or float(sigmas[-1]) != 0.):
        raise ValueError('Scoped H16 effects need the actual finite descending refine SIGMAS')
    if (model.get_attachment(KEY) is not None or model.get_attachment(h16_effects.KEY) is not None
            or model.get_attachment(h16_relay.KEY) is not None or model.get_attachment(EAV_KEY) is not None
            or model.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)):
        raise ValueError('Bind scoped H16 effects from the raw HIGH MODEL branch')
    selected, positive, binding, relay_runtime, eav_runtime = model, encoded.positive, None, None, None
    video, audio = window.piece['samples'].unbind()
    sigma_values = tuple(float(v) for v in sigmas.tolist())
    if relay_config is not None:
        if type(relay_config) is not ScopedRelayConfig:
            raise ValueError('Connect the external Scoped Relay Config')
        if len(encoded.compiled.dialogue) > 1:
            if clip is None:
                raise ValueError('Scoped H16 Relay requires the actual native CLIP tokenizer')
            local_plan = _local_plan(encoded, relay_config)
            binding = relay.build_prompt_relay_binding(clip, local_plan, encoded.prepared_prompt,
                                                       encoded.positive, encoded.tokens)
            local = legacy.reanchor_conditioning(encoded.positive, spec.start_frame, spec.end_frame,
                                                  tuple(video.shape[-2:]))
            meta = local[0][1]
            keyframes, refs = meta.get('minimax_keyframes', []), meta.get('minimax_refs', [])
            layout = build_packed_layout(binding['text_len'], *video.shape[2:], audio.shape[-1],
                keyframes=keyframes, refs=refs, frame_count=meta.get('minimax_frame_count'))
            task = feta._classify_visual_task(keyframes, latent_frames=video.shape[2], refs=refs)
            binding = relay._bind_layout_contract(binding, layout, resolved_task=task, keyframes=keyframes, refs=refs)
            positive = node_helpers.conditioning_set_values(positive, {relay.PROMPT_RELAY_BINDING_KEY: binding})
            positive = relay._attach_binding_model_cond(positive, binding['binding_hash'])
            selected, _ = relay.patch_prompt_relay_model(selected, binding, relay_config.query_chunk_rows)
            owner = h16_relay.H16RelayOwner(spec.plan_sha256, spec.source_identity,
                _input_identity(lifted_segment), _input_identity(window.piece),
                previous.output_identity if previous else None, spec.index, CONTRACT,
                binding['binding_hash'], sigma_values, model.get_model_object('model_sampling'))
            selected = selected.clone()
            selected.set_attachments(h16_relay.KEY, owner)
            relay_runtime = h16_relay.H16RelayRuntime(selected, owner,
                len(selected.get_model_object('diffusion_model').blocks))
            selected.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, h16_relay.KEY, relay_runtime.prepare)
            selected.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, h16_relay.KEY, relay_runtime.cleanup)
    if eav_config is not None:
        if type(eav_config) is not EAVConfig:
            raise ValueError('Connect the external Stage EAV Config')
        options = selected.model_options['transformer_options']
        shifts = options.get('minimax_h3_sigma_shift_video'), options.get('minimax_h3_sigma_shift_audio')
        profile = dict(contract=CONTRACT, bank_sha256=bank.sha256, plan_sha256=spec.plan_sha256,
                       window_index=spec.index, audio_output=CONTRACT, sigma_dtype=str(sigmas.dtype))
        stage = StageContext(recipe=h16_effects.RECIPE, stage=f'scoped_h16_window_{spec.index}',
            profile=canonical(profile), start=0, end=len(sigmas)-1, trajectory_sigmas=sigma_values,
            video_shift=shifts[0], audio_shift=shifts[1], video_shape=tuple(video.shape), audio_shape=tuple(audio.shape),
            input_semantics='scoped_real_accepted_av_prefix_read_only_with_inherited_video_mask',
            output_semantics='exact_append_refined_av_source_audio_padding_retained',
            denoised_semantics='terminal_scoped_h16_prediction')
        selected = selected.clone()
        selected.set_attachments(h16_effects.KEY, h16_effects.H16EAVOwner(stage, spec.plan_sha256,
            spec.source_identity, _input_identity(lifted_segment), _input_identity(window.piece),
            previous.output_identity if previous else None, spec.index, CONTRACT,
            model.get_model_object('model_sampling'), eav_config.mode))
        selected, eav_runtime, _ = apply_stage_eav(selected, sigmas, window.piece, stage, eav_config)
    runtime = H16ScopedEffects(bank.sha256, spec.index, _input_identity(window.piece), _input_identity(sigmas),
        model.get_model_object('model_sampling'), binding, relay_runtime, eav_runtime)
    selected = selected.clone()
    selected.set_attachments(KEY, runtime)
    report = dict(contract=CONTRACT, status='scoped_h16_effects_bound', sampled=False,
        relay=('apply_exp' if binding else ('passthrough_zero_or_single_event' if relay_config else 'not_requested')),
        eav=eav_config.mode if eav_config else 'not_requested', hard_time_isolation=False, quality_accepted=False)
    return selected, positive, runtime, canonical(report)


def assert_h16_scoped_effects(model, positive, window, spec, sigmas, bank):
    runtime = model.get_attachment(KEY)
    if runtime is None:
        if (model.get_attachment(h16_effects.KEY) is not None or model.get_attachment(h16_relay.KEY) is not None
                or model.get_attachment(EAV_KEY) is not None
                or model.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)):
            raise ValueError('Scoped H16 cannot consume old/full-text effect owners')
        return
    if (type(runtime) is not H16ScopedEffects or runtime.bank_sha256 != bank.sha256
            or runtime.window_index != spec.index or runtime.piece_identity != _input_identity(window.piece)
            or runtime.sigma_identity != _input_identity(sigmas)
            or model.get_model_object('model_sampling') is not runtime.sampling_object):
        raise ValueError('Scoped H16 effect MODEL differs from the actual bank/prefix/window')
    if runtime.relay_binding is not None:
        _assert_paired_conditioning(positive, runtime.relay_binding)
        if runtime.eav_runtime is None or runtime.eav_runtime.config.mode == 'disabled':
            if relay.prompt_relay_model_contract(model)['binding_hash'] != runtime.relay_binding['binding_hash']:
                raise ValueError('Scoped H16 Relay wrapper changed')
        elif not runtime.eav_runtime.relay_required:
            raise ValueError('Scoped H16 combined EAV lost Relay ownership')
    if runtime.eav_runtime is not None and runtime.eav_runtime.config.mode != 'disabled':
        if (model.get_attachment(EAV_KEY) is not runtime.eav_runtime
                or len(model.get_wrappers('diffusion_model', EAV_KEY)) != 1):
            raise ValueError('Scoped H16 EAV runtime/wrapper changed')


def audit_h16_scoped_effects(result, runtime):
    from .temporal_h16 import ScopedH16Result
    if (type(result) is not ScopedH16Result or type(runtime) is not H16ScopedEffects
            or result.bank_sha256 != runtime.bank_sha256 or result.index != runtime.window_index
            or _input_identity(result.output_latent) != result.output_identity):
        raise ValueError('Scoped H16 effect audit needs the matching actual window result')
    reports = {}
    if runtime.eav_runtime is not None:
        report = runtime.eav_runtime.snapshot()
        if report['status'] == 'aborted':
            raise RuntimeError('Scoped H16 EAV aborted: '+report['feta']['aborted'])
        reports['eav'] = report
    if runtime.relay_runtime is not None and (runtime.eav_runtime is None or runtime.eav_runtime.config.mode == 'disabled'):
        reports['relay'] = runtime.relay_runtime.snapshot()
    return result.output_latent, canonical(dict(contract=CONTRACT, window_index=result.index,
        effects=reports, hard_time_isolation=False, quality_accepted=False))
