"""Independent H16 scoped continuation: real accepted AV, not a text-only receipt.

This explicit EXP uses read-only accepted overlaps and exact append instead of
the released H16 post-loop crossfade/energy gate. Old H16 remains untouched.
Nominal sampled audio ends retain the source's unsampled padding at completion.
"""
from dataclasses import dataclass

import comfy.model_management
import comfy.nested_tensor
import torch

from .. import chunked_two_pass_parity as parity
from .. import chunked_two_pass_upscale_advanced as legacy
from ..sampling import rebind_dual_clock_sampler
from ..temporal_dialogue_bank import select_window_conditioning
from ..temporal_dialogue_scope import DialogueReceipt, bind_published_window, verify_previous_window
from .chunked_stages import _check_segment
from .results import _input_identity, canonical
from .temporal_chunked_v5 import _without_relay_pair


CONTRACT = 't8.temporal-h16.accepted-av-exact-prefix.v1'


@dataclass(frozen=True)
class ScopedH16Result:
    plan_sha256: str
    source_identity: dict
    bank_sha256: str
    index: int
    count: int
    output_latent: dict
    output_identity: dict
    dialogue_receipt: DialogueReceipt


@dataclass(frozen=True)
class ScopedH16Input:
    piece: dict
    video_noise: torch.Tensor
    audio_noise: torch.Tensor
    video_overlap: int
    audio_overlap: int


def _audio_sha(result):
    if type(result) is not ScopedH16Result or _input_identity(result.output_latent) != result.output_identity:
        raise ValueError('Scoped H16 accepted AV content changed')
    return _input_identity(result.output_latent['samples'].tensors[1])['tensor_sha256']


def scoped_h16_piece(source_segment, lifted_segment, spec, context, plan, bank, previous=None):
    _check_segment(source_segment, spec, context, plan)
    if (plan.get('schema') != legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4
            or plan.get('spatial_strategy') != 'full_frame_safe'
            or plan.get('second_pass_audio_policy') != 'joint_av_preserve_input'):
        raise ValueError('Scoped H16 requires the actual v4 full-frame joint AV plan')
    encoded = select_window_conditioning(bank, context.source_latent, plan, spec.index)
    window = encoded.compiled.window
    if (bank.executor_contract != 'legacy_chunked_audio_context'
            or (window.start_frame, window.end_frame, window.audio_start, window.audio_stop)
            != (spec.start_frame, spec.end_frame, spec.audio_start, spec.audio_end)
            or window.audio_padding_policy != 'retain_source_tail'):
        raise ValueError('Scoped H16 bank does not describe this actual nominal audio window')
    previous_audio_sha = None
    if previous is not None:
        if (type(previous) is not ScopedH16Result or previous.bank_sha256 != bank.sha256
                or previous.plan_sha256 != spec.plan_sha256 or previous.source_identity != spec.source_identity
                or previous.index != spec.index-1 or previous.count != spec.count
                or previous.index != previous.dialogue_receipt.window.index):
            raise ValueError('Scoped H16 previous result belongs to another source/bank/window')
        previous_audio_sha = _audio_sha(previous)
    verify_previous_window(bank.dialogue_plan, window,
                           previous.dialogue_receipt if previous else None, previous_audio_sha)
    source_video, source_audio = source_segment['samples'].unbind()
    video, lifted_audio = lifted_segment['samples'].unbind()
    target = (spec.target_height//16, spec.target_width//16)
    if (tuple(video.shape[:3]) != tuple(source_video.shape[:3]) or tuple(video.shape[-2:]) != target
            or not torch.equal(source_audio, lifted_audio)):
        raise ValueError('Scoped H16 learned lift changed time/audio or target geometry')
    if context.global_video_noise is None or context.global_audio_noise is None:
        raise ValueError('Scoped H16 requires the actual full-target joint noise preparation')
    video, audio = video.clone(), source_audio.clone()
    video_mask = torch.ones((1, 1, video.shape[2], *target), dtype=video.dtype, device=video.device)
    audio_mask = torch.ones_like(audio)
    inherited = context.target_inherited_video_mask
    if inherited is not None:
        expected = inherited[:, :, spec.start_token:spec.end_token]
        if 'noise_mask' not in lifted_segment or not torch.equal(lifted_segment['noise_mask'].tensors[0].to(expected), expected):
            raise ValueError('Scoped H16 lift lost or changed the inherited video mask')
        video_mask *= expected.to(video_mask)
    vo = ao = 0
    if previous is not None:
        pv, pa = previous.output_latent['samples'].unbind()
        expected_tokens = legacy.tokens_for_frames(window.owned_start_frame)
        if (tuple(pv.shape) != (*video.shape[:2], expected_tokens, *target)
                or tuple(pa.shape) != (*audio.shape[:-1], window.owned_audio_start)):
            raise ValueError('Scoped H16 previous video target geometry differs')
        vo, ao = pv.shape[2]-spec.start_token, pa.shape[-1]-spec.audio_start
        if not (0 <= vo <= video.shape[2] and 0 <= ao <= audio.shape[-1]):
            raise ValueError('Scoped H16 accepted prefix leaves a gap or invalid overlap')
        if vo:
            video[:, :, :vo] = pv[:, :, spec.start_token:spec.start_token+vo]
            video_mask[:, :, :vo] = 0
        if ao:
            audio[..., :ao] = pa[..., spec.audio_start:spec.audio_start+ao]
            audio_mask[..., :ao] = 0
    piece = dict(samples=comfy.nested_tensor.NestedTensor((video, audio)),
                 noise_mask=comfy.nested_tensor.NestedTensor((video_mask, audio_mask)))
    return ScopedH16Input(piece, context.global_video_noise[:, :, spec.start_token:spec.end_token].contiguous(),
                          context.global_audio_noise[..., spec.audio_start:spec.audio_end].contiguous(), vo, ao)


def sample_scoped_h16(model, source_segment, lifted_segment, spec, context, plan, noise,
                       sampler, sigmas, bank, previous=None, positive=None, negative=None, cfg=1.):
    window = scoped_h16_piece(source_segment, lifted_segment, spec, context, plan, bank, previous)
    encoded = select_window_conditioning(bank, context.source_latent, plan, spec.index)
    selected = encoded.positive if positive is None else positive
    if bank.identity_context.describe(_without_relay_pair(selected)) != bank.identity_context.describe(encoded.positive):
        raise ValueError('Scoped H16 CONDITIONING differs from the native precompiled window')
    if (not torch.is_tensor(sigmas) or sigmas.ndim != 1 or not sigmas.is_floating_point()
            or len(sigmas) < 2 or not torch.isfinite(sigmas).all()
            or torch.any(sigmas[:-1] < sigmas[1:]) or float(sigmas[-1]) != 0.):
        raise ValueError('Scoped H16 needs finite descending refine SIGMAS ending at zero')
    if context.noise_seed is not None and getattr(noise, 'seed', None) != context.noise_seed:
        raise ValueError('Scoped H16 noise seed differs from joint preparation')
    comfy.model_management.throw_exception_if_processing_interrupted()
    if hasattr(model, 'get_attachment'):
        from .temporal_h16_effects import assert_h16_scoped_effects
        assert_h16_scoped_effects(model, selected, window, spec, sigmas, bank)
    local = legacy.reanchor_conditioning(selected, spec.start_frame, spec.end_frame,
                                        tuple(window.piece['samples'].tensors[0].shape[-2:]))
    local_negative = None if negative is None else legacy.reanchor_conditioning(
        negative, spec.start_frame, spec.end_frame, tuple(window.piece['samples'].tensors[0].shape[-2:]))
    sampled = legacy.sample_piece(window.piece, local, model, noise,
        rebind_dual_clock_sampler(model, window.piece, sampler), sigmas, local_negative, cfg,
        prepared_noise=comfy.nested_tensor.NestedTensor((window.video_noise, window.audio_noise)))
    video, audio = parity._parts(sampled, 'scoped H16 actual joint refine output')
    original_video, original_audio = window.piece['samples'].unbind()
    if video.shape != original_video.shape or audio.shape != original_audio.shape:
        raise RuntimeError('Scoped H16 sampler changed AV geometry')
    vm, am = window.piece['noise_mask'].unbind()
    video, vr = parity._restore_zero_mask(video, original_video, vm)
    audio, ar = parity._restore_zero_mask(audio, original_audio, am)
    pv, pa = previous.output_latent['samples'].unbind() if previous else (None, None)
    video = parity._append_exact(pv, video, spec.start_token, 2, window.video_overlap)
    audio = parity._append_exact(pa, audio, spec.audio_start, -1, window.audio_overlap)
    if spec.index+1 == spec.count:
        # H16 never sampled this padding; retain it literally, do not call it generated.
        audio = torch.cat((audio, context.original_audio[..., spec.audio_end:]), dim=-1)
        source_video, source_audio = context.source_latent['samples'].unbind()
        if video.shape[2] != source_video.shape[2] or audio.shape != source_audio.shape:
            raise RuntimeError('Scoped H16 completed output does not cover the full AV timeline')
    output = {'samples': comfy.nested_tensor.NestedTensor((video, audio))}
    previous_receipt = previous.dialogue_receipt if previous else None
    previous_sha = _audio_sha(previous) if previous else None
    receipt = bind_published_window(bank.dialogue_plan, encoded.compiled,
        _input_identity(audio)['tensor_sha256'], previous_receipt, previous_sha)
    result = ScopedH16Result(spec.plan_sha256, spec.source_identity, bank.sha256, spec.index,
                             spec.count, output, _input_identity(output), receipt)
    return output, result, canonical(dict(contract=CONTRACT, window_index=spec.index,
        window_count=spec.count, completed=spec.index+1 == spec.count,
        audio_output='scoped_refined_prefix_exp', audio_merge='exact_append_source_padding_retained',
        locked_video_overlap=window.video_overlap, locked_audio_overlap=window.audio_overlap,
        video_read_only_roundoff=vr, audio_read_only_roundoff=ar,
        scoped_native_conditioning_consumed=True, hard_time_isolation=False, quality_accepted=False))
