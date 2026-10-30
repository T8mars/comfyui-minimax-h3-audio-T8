"""Pika's native LCM ladder across two independently configured AV canvases.

Only the stage boundary is new: Core still owns conditioning, masks, model
patches and all six LCM evaluations. No Euler update or decoded-frame bridge.
"""
from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import torch

from .core import nested_av_parts
from .pika_flex import pika_sigmas, prepare_pika_flex


def _checked_parts(latent, label):
    video, audio = nested_av_parts(latent)
    for value in (video, audio):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f'{label} requires finite floating AV samples')
    if video.ndim != 5 or audio.ndim != 4 or video.shape[:2] != (1, 24) or audio.shape[:3] != (1, 32, 2):
        raise ValueError(f'{label} requires batch-1 native H3 AV shapes')
    return video, audio


class _NoiseStream:
    """One native-style RNG, retained across LCM calls and canvas changes."""
    def __init__(self, seed, split, low_shapes, high_shapes):
        self.seed, self.split = seed, split
        self.low_shapes, self.high_shapes = low_shapes, high_shapes
        self.generator = None
        self.draws = 0
        self.boundary_video_noise = None
        self.draw_shapes = []

    def draw(self, template, sigma, sigma_next):
        import comfy.utils
        if self.generator is None:
            self.generator = torch.Generator(device=template.device)
            self.generator.manual_seed((self.seed + int(template.device.type == 'cpu')) % 2**64)
        elif self.generator.device != template.device:
            raise ValueError('LCM RNG continuation requires both stages on the same sampling device')
        boundary = self.draws == self.split - 1 and self.low_shapes != self.high_shapes
        shapes = self.high_shapes if boundary else None
        size = (template.shape[0], 1, sum(math.prod(s[1:]) for s in shapes)) if boundary else template.size()
        noise = torch.randn(size, dtype=template.dtype, layout=template.layout,
                            device=template.device, generator=self.generator)
        self.draw_shapes.append(list(noise.shape))
        self.draws += 1
        if boundary:
            video, audio = comfy.utils.unpack_latents(noise, shapes)
            self.boundary_video_noise = video.detach().cpu()
            # This LOW video is discarded after the call. Audio uses the exact
            # same fresh draw carried into HIGH; do not draw boundary noise twice.
            small = video.flatten(1)[:, :math.prod(self.low_shapes[0][1:])].reshape(self.low_shapes[0])
            return comfy.utils.pack_latents([small, audio])[0]
        return noise


@torch.inference_mode()
def sample_pika_two_pass(model, positive_low, av_latent_low, positive_high, av_latent_high,
                         upscaler_model, seed, low_evaluations=4, precision='fp16',
                         model_high=None, callback=None):
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    from comfy.k_diffusion.sampling import sample_lcm
    from . import progressive_sampling_runtime as runtime
    from .progressive_masking import normalize_av_masks
    from .progressive_stage_models import stage_report

    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    if type(low_evaluations) is not int or not 1 <= low_evaluations <= 5:
        raise ValueError('The six-step ladder requires 1..5 LOW evaluations and at least one HIGH evaluation')
    low_video, low_audio = _checked_parts(av_latent_low, 'LOW')
    high_video, high_audio = _checked_parts(av_latent_high, 'HIGH')
    if low_video.shape[:3] != high_video.shape[:3] or low_audio.shape != high_audio.shape:
        raise ValueError('Both stages must use the same video timeline and audio shape')
    if any(a > b for a, b in zip(low_video.shape[-2:], high_video.shape[-2:])):
        raise ValueError('HIGH canvas cannot be smaller than LOW along either spatial axis')
    low_shapes = [tuple(low_video.shape), tuple(low_audio.shape)]
    high_shapes = [tuple(high_video.shape), tuple(high_audio.shape)]
    resizing = low_shapes != high_shapes
    identity = runtime._lifter_identity(upscaler_model, precision) if resizing else None
    low, _, sigmas, _ = prepare_pika_flex(model, av_latent_low)
    high, _, _, _ = prepare_pika_flex(model if model_high is None else model_high, av_latent_high)
    branches = {'low': low, 'high': high}
    stream = _NoiseStream(seed, low_evaluations, low_shapes, high_shapes)
    raw = {}
    counts = {'low': 0, 'high': 0}
    callbacks = {'low': 0, 'high': 0}
    times = {}
    observer_key = 't8_pika_two_pass_completed_forward'
    wrapper_kind = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    stage = 'low'
    started = time.perf_counter()

    def observe(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        counts[stage] += 1
        return result

    def progress(index, prediction, state, total):
        callbacks[stage] += 1
        if callback is not None:
            callback(index + (low_evaluations if stage == 'high' else 0), prediction, state, 6)

    def low_solver(inner, initial, ladder, *, extra_args, **kwargs):
        def notify(row):
            raw['clean'] = row['denoised'].detach().to(mm.intermediate_device(), copy=True)
            supplied = kwargs.get('callback')
            if supplied is not None:
                supplied(row)
        options = {**kwargs, 'callback': notify}
        result = sample_lcm(inner, initial, ladder, extra_args=extra_args,
                            noise_sampler=lambda s, n: stream.draw(initial, s, n), **options)
        # Capture BEFORE KSAMPLER.inverse_noise_scaling(nonzero terminal sigma).
        raw['state'] = result.detach().to(mm.intermediate_device(), copy=True)
        base = inner.inner_model.inner_model
        clean = base.process_latent_out(raw['clean'])
        raw['clean_vae'] = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(clean, low_shapes))
        return result

    def high_solver(inner, discarded_initial, ladder, *, extra_args, **kwargs):
        base = inner.inner_model.inner_model
        if resizing:
            zero_audio = torch.zeros_like(high_audio)
            clean = comfy.utils.pack_latents([raw.pop('lifted'), zero_audio])[0].to(discarded_initial)
            clean = base.process_latent_in(clean)
            noise = comfy.utils.pack_latents([stream.boundary_video_noise, zero_audio])[0].to(discarded_initial)
            state = base.model_sampling.noise_scaling(ladder[0], noise, clean)
            video = comfy.utils.unpack_latents(state, high_shapes)[0]
            audio = comfy.utils.unpack_latents(raw.pop('state').to(discarded_initial), low_shapes)[1]
            initial = comfy.utils.pack_latents([video, audio])[0]
            stream.boundary_video_noise = None
        else:
            initial = raw.pop('state').to(discarded_initial)
        if initial.shape != discarded_initial.shape:
            raise ValueError('Pika continuation state does not match the actual HIGH packed AV shape')
        # Core has already prepared the HIGH clean anchors, reference conditions,
        # masks and conditioning noise. Replace only the solver initial state.
        return sample_lcm(inner, initial, ladder, extra_args=extra_args,
                          noise_sampler=lambda s, n: stream.draw(initial, s, n), **kwargs)

    for branch in branches.values():
        branch.add_wrapper_with_key(wrapper_kind, observer_key, observe)
    try:
        low_noise = comfy.sample.prepare_noise(av_latent_low['samples'], seed)
        low_mask = normalize_av_masks(av_latent_low.get('noise_mask'), low_video, low_audio)
        began = time.perf_counter()
        runtime._native_stage(low, comfy.samplers.KSAMPLER(low_solver, {}, {}),
            sigmas[:low_evaluations + 1], av_latent_low['samples'], low_noise,
            positive_low, [], 1., seed, progress, denoise_mask=low_mask,
            preview_phase='low', preview_offset=0, preview_total=6)
        times['low'] = time.perf_counter() - began
        clean_low = {**av_latent_low, 'samples': raw.pop('clean_vae')}
        raw.pop('clean', None)
        lift_report = {'status': 'same_canvas_no_lift'}
        if resizing:
            began = time.perf_counter()
            plan = SimpleNamespace(target_width=high_video.shape[-1] * 16,
                                   target_height=high_video.shape[-2] * 16)
            raw['lifted'], lift_report = runtime._lift_video(
                clean_low['samples'].unbind()[0], torch.zeros_like(low_audio), plan, identity)
            if raw['lifted'].shape != high_video.shape:
                raise ValueError('Learned upscaler returned a video shape different from HIGH')
            times['learned_lift'] = time.perf_counter() - began
        stage = 'high'
        high_noise = comfy.sample.prepare_noise(av_latent_high['samples'], seed)
        # Keep the original audio conditioning-noise anchor when masks lock it.
        high_noise = comfy.nested_tensor.NestedTensor((high_noise.unbind()[0], low_noise.unbind()[1]))
        high_mask = normalize_av_masks(av_latent_high.get('noise_mask'), high_video, high_audio)
        began = time.perf_counter()
        result = runtime._native_stage(high, comfy.samplers.KSAMPLER(high_solver, {}, {}),
            sigmas[low_evaluations:], av_latent_high['samples'], high_noise,
            positive_high, [], 1., seed, progress, denoise_mask=high_mask,
            preview_phase='high', preview_offset=low_evaluations, preview_total=6)
        times['high'] = time.perf_counter() - began
        if [tuple(t.shape) for t in result.unbind()] != high_shapes:
            raise RuntimeError('Pika two-pass output AV shapes differ from HIGH')
        if not all(bool(torch.isfinite(t).all()) for t in result.unbind()):
            raise RuntimeError('Pika two-pass output contains NaN or Inf')
        report = {'schema': 't8.pika_two_pass.v1', 'status': 'sampled_quality_unverified',
            'solver': 'native_lcm', 'video_shift': 12., 'audio_shift': 3., 'cfg': 1.,
            'video_sigmas': pika_sigmas().tolist(), 'low_evaluations': low_evaluations,
            'high_evaluations': 6 - low_evaluations, 'callbacks': callbacks,
            'completed_apply_model_calls': counts, 'fresh_noise_draws': stream.draws,
            'noise_draw_shapes': stream.draw_shapes, 'low_shapes': low_shapes, 'high_shapes': high_shapes,
            'upscaler': lift_report, 'low_model': stage_report(model),
            'high_model': stage_report(model if model_high is None else model_high),
            'shared_base_object': low.model is high.model,
            'partial_low_is_final_clip': False, 'audio': 'unfinished_joint_AV_continues_into_HIGH',
            'seconds': time.perf_counter() - started, 'timings': times,
            'trained_GPU_quality_verified': False}
        output = {**av_latent_high, 'samples': result.to(mm.intermediate_device())}
        return output, clean_low, json.dumps(report, ensure_ascii=False, allow_nan=False)
    finally:
        for branch in branches.values():
            branch.remove_wrappers_with_key(wrapper_kind, observer_key)
        raw.clear()
        stream.boundary_video_noise = None
