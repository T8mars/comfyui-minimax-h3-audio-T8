"""Explicit native-mask contract for the initialized progressive EAV owner.

FETA keeps its target-video CFI/gain definition. Native H3, not this adapter,
owns known-token times, input injection and final known-region constraints.
The runtime must bind the HIGH clean-source anchor independently of its start
state. This contract audits the exact token masks consumed by the DiT; it does
not turn an arbitrary EAV wrapper into a continuation or inpainting sampler.
"""

from __future__ import annotations

import hashlib
import json

import torch


class NativeProgressiveMaskContract:
    def __init__(self, model, normalized_mask, video_shape, audio_shape, *, continuation=None, accepted_source=None):
        import comfy.model_base
        import comfy.utils

        base = model.model
        if type(base) is not comfy.model_base.MiniMaxH3:
            raise ValueError('Initialized EAV mask contract requires native H3')
        self._shapes = (tuple(video_shape), tuple(audio_shape))
        self._long_video = None
        if continuation is not None and accepted_source is not None:
            raise ValueError('Select paired continuation or one independently prepared accepted source')
        if continuation is not None:
            from .progressive_continuation_runtime import PreparedProgressiveContinuation
            from .enhance_a_video_advanced import _assert_long_video_contract
            if type(continuation) is not PreparedProgressiveContinuation:
                raise ValueError('Unknown EAV progressive continuation inputs')
            continuation.verify()
            request = continuation.source.binding['request']
            self._long_video = _assert_long_video_contract(model,
                segment_index=request['segment_index'], context_frames=request['context_frames'])
        elif accepted_source is not None:
            from .progressive_continuation import ProgressiveContinuationSource
            from .enhance_a_video_advanced import _assert_long_video_contract
            if type(accepted_source) is not ProgressiveContinuationSource:
                raise ValueError('Independent continuation EAV requires an authenticated accepted source')
            request = accepted_source.revalidate()['request']
            self._long_video = _assert_long_video_contract(model,
                segment_index=request['segment_index'], context_frames=request['context_frames'])
        if (len(video_shape) != 5 or tuple(video_shape[:2]) != (1, 24) or
                len(audio_shape) != 4 or tuple(audio_shape[:3]) != (1, 32, 2)):
            raise ValueError('Initialized EAV mask contract has invalid AV shapes')
        self._expected = {}
        raw_hash = None
        if normalized_mask is not None:
            if not getattr(normalized_mask, 'is_nested', False):
                raise ValueError('Initialized EAV requires normalized native nested masks')
            parts = normalized_mask.unbind()
            if tuple(tuple(p.shape) for p in parts) != self._shapes:
                raise ValueError('Initialized EAV mask shapes differ from the stage')
            for p in parts:
                if (not p.is_floating_point() or not bool(torch.isfinite(p).all()) or
                        bool(((p < 0) | (p > 1)).any()) or not torch.equal(p, p[:, :1].expand_as(p))):
                    raise ValueError('Initialized EAV requires finite shared-channel masks in [0,1]')
            packed = comfy.utils.pack_latents([p.detach().float().cpu() for p in parts])[0]
            raw_hash = hashlib.sha256(packed.contiguous().numpy().tobytes()).hexdigest()
            self._expected = {key: value.detach().float().cpu().clone() for key, value in
                              base._denoise_mask_values(packed, self._shapes).items()}
        mask_hashes = {key: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
                       for key, t in self._expected.items()}
        description = {'schema': 'native_progressive_eav_masks_v1',
                       'video_shape': list(video_shape), 'audio_shape': list(audio_shape),
                       'raw_mask_sha256': raw_hash, 'native_token_mask_sha256': mask_hashes,
                       'mask_present': normalized_mask is not None,
                       'policy': 'unchanged_target_video_feta_with_native_h3_known_region_constraints'}
        if self._long_video is not None:
            description['long_video_contract'] = self.long_video_contract
            description['accepted_source_sha256'] = (
                continuation.source.sha256 if continuation is not None else accepted_source.sha256)
        self._description = description | {'binding_sha256': hashlib.sha256(json.dumps(
            description, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        self._device_masks = {}

    @property
    def long_video_contract(self):
        return json.loads(json.dumps(self._long_video))

    def report(self):
        # Never expose mutable mask snapshots or device cache in attachments.
        return json.loads(json.dumps(self._description))

    def validate(self, x, video_mask, audio_mask):
        if len(x) != 2 or tuple(tuple(p.shape) for p in x) != self._shapes:
            raise RuntimeError('Initialized EAV masks bound to different actual AV shapes')
        observed = {'denoise_mask': video_mask, 'audio_denoise_mask': audio_mask}
        for key, actual in observed.items():
            expected = self._expected.get(key)
            if expected is None:
                if actual is not None:
                    raise RuntimeError(f'Initialized EAV received an unexpected {key}')
                continue
            if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(expected.shape):
                raise RuntimeError(f'Initialized EAV lost the bound native {key}')
            cache_key = (key, actual.device, actual.dtype)
            if cache_key not in self._device_masks:
                self._device_masks[cache_key] = expected.to(actual)
            if not torch.equal(actual, self._device_masks[cache_key]):
                raise RuntimeError(f'Initialized EAV native {key} changed after binding')
        return self.report()


def validate_progressive_eav_mask_scope(contract, *, profile, allowed_tasks, reference, long_video, stg):
    """Only the explicit initialized composer may enable these masks."""
    if contract is None:
        if profile == 'progressive_initialized_exp':
            raise ValueError('Initialized EAV profile requires a native stage mask contract')
        return
    if type(contract) is not NativeProgressiveMaskContract:
        raise ValueError('Unknown initialized EAV mask contract')
    if contract.long_video_contract is not None:
        if (profile != 'progressive_initialized_exp' or stg is not None or not reference
                or set(allowed_tasks) != {'LongVideoMotion'} or long_video != contract.long_video_contract):
            raise ValueError('Progressive continuation EAV requires its exact native long-video owner')
        return
    if (profile != 'progressive_initialized_exp' or reference or long_video is not None or
            stg is not None or not set(allowed_tasks).issubset({'T2VA', 'I2VA'})):
        raise ValueError('Initialized EAV mask composer cannot enable other legacy algorithm scopes')
