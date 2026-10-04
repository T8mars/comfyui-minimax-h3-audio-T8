"""Explicit DMD8 range sampler. Imported only by an owned external worker.

The first range exports a partial x0 for learned lifting AND the real noisy AV
state. The second range restarts lifted video x0 at clock 4, but continues noisy
audio through clocks 4..7. This is not the official completed-audio tail restart.
No new four-step schedule, projection download, or additional model evaluation.
"""
from contextlib import contextmanager
import time
from types import SimpleNamespace

import torch

PROFILE = "dmd8_split4_lift_x0_continue_audio_exp_v1"


def data_prediction(sample, velocity, timestep):
    # H3 predicts data-ward velocity: PLUS, and sigma recovered from t, exactly
    # as the pinned scheduler (not the grid sigma's float32 round trip).
    return sample + (1 - timestep.to(device=sample.device, dtype=sample.dtype)) * velocity


def unpack_video(rows, frames, height, width, channels, patch):
    value = rows.reshape(-1, frames, height // patch[1], width // patch[2], channels, *patch)
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, channels, frames, height, width).contiguous()


@torch.no_grad()
def run(transformer, prompt_embeds, text_token_tags, num_frames, num_steps, seed, device,
        *, canvas, start, end, initial=None, carry_video=False, conditions=None,
        step_seconds=None, video_shift=12., audio_shift=3., runtime=None, capture=None):
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, MiniMaxH3Ref2VAPrepareLayoutStep, patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS as AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG as AUDIO_TAG,
        MINIMAX_H3_VIDEO_TAG as VIDEO_TAG, align_num_frames, audio_latent_num_frames,
        video_latent_num_frames)
    from src.models.sequence_layout import layout_from_indices
    from src.models.hybrid_transform import iter_hybrids, set_layout
    from src.inference.render import KEYFRAME_NOISE_AUG
    if (num_steps != 8 or type(start) is not int or type(end) is not int or not 0 <= start < end <= 8
            or bool(initial is not None) != bool(start) or carry_video and initial is None):
        raise ValueError("Split sampling requires an explicit range of the ORIGINAL eight-step schedule")
    num_frames = align_num_frames(num_frames, 17, 5)
    vf, af = video_latent_num_frames(num_frames, 17, 5), audio_latent_num_frames(num_frames)
    h, w = canvas["height"] // 16, canvas["width"] // 16
    patch, channels = tuple(transformer.config.patch_size), transformer.config.in_channels
    refs = bool(conditions and isinstance(conditions[0], dict))
    if refs:
        condition_latents = [r["latent"].to(device) for r in conditions if r["kind"] in ("image", "video")]
        audio_conditions = [r["audio_latent"].to(device) for r in conditions if r.get("audio_latent") is not None]
        reference_info = [SimpleNamespace(kind=r["kind"], has_audio=r.get("audio_latent") is not None) for r in conditions]
        layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            text_token_tags, reference_info, condition_latents, audio_conditions, vf, h, w, af,
            patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG)
    else:
        anchors, condition_latents = conditions if conditions else ((), [])
        audio_conditions = []
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags, vf, h, w, af, patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG,
            keyframe_anchors=anchors)
    positions, tags, vi, ai, ti, vc, ac = layout
    positions, tags, vi, ai, ti = (v.to(device) for v in (positions, tags, vi, ai, ti))
    if next(iter_hybrids(transformer), None) is not None:
        set_layout(transformer, layout_from_indices(vi[vc:], vf, (h // patch[1]) * (w // patch[2]),
            seq_len=positions.shape[0], frame_size=(h // patch[1], w // patch[2]), text_indices=ti))
    vs, aus = MiniMaxH3Scheduler(shift=video_shift), MiniMaxH3Scheduler(shift=audio_shift)
    vs.set_timesteps(8, device=device)
    aus.set_timesteps(8, device=device)
    if start:
        vs.set_begin_index(start)
        aus.set_begin_index(start)
    generator = torch.Generator(device).manual_seed(seed)
    fixed = []
    for condition in condition_latents:
        condition = condition.to(device)
        noise = torch.randn(condition.shape, generator=generator, device=device, dtype=torch.float32)
        fixed.append(patchify_video_latents(vs.scale_noise(condition, KEYFRAME_NOISE_AUG, noise), patch))
    shape = (1, channels, vf, h, w)
    if initial is None:
        video = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        generated_audio = torch.randn((af * AUDIO_CHANNELS, 32), generator=generator, device=device, dtype=torch.float32)
    else:
        video, audio = initial
        if tuple(video.shape) != shape or tuple(audio.shape) != (AUDIO_CHANNELS, 32, af):
            raise ValueError("Split resume AV shape does not match this canvas")
        video = video.to(device=device, dtype=torch.float32)
        if not carry_video:
            noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            video = vs.scale_noise(video, vs.timesteps[start], noise)
        generated_audio = audio.to(device=device, dtype=torch.float32).permute(0, 2, 1).reshape(-1, 32).contiguous()
    video_rows = patchify_video_latents(video, patch)
    if fixed:
        video_rows = torch.cat(fixed + [video_rows])
    audio_rows = torch.cat(audio_conditions + [generated_audio]) if audio_conditions else generated_audio
    if video_rows.shape[0] != vi.numel() or audio_rows.shape[0] != ai.numel():
        raise ValueError("Split reference rows disagree with packed layout")
    # Initial noise is consumed only at start=0. The schedulers locate the
    # original index on their first step; never construct a four-step grid.
    if runtime:
        runtime.barrier()
    last_x0 = None
    for index in range(start, end):
        started = time.perf_counter()
        t, at = vs.timesteps[index], aus.timesteps[index]
        clocks = torch.full((positions.shape[0],), float(t), dtype=torch.float32, device=device)
        if vc:
            clocks[vi[:vc]] = max(float(t), KEYFRAME_NOISE_AUG)
        clocks[ai[ac:]] = float(at)
        clocks[ai[:ac]] = 0.
        timestep, timestep_indices = torch.unique(clocks, sorted=True, return_inverse=True)
        pred, apred = transformer(hidden_states=video_rows[None], audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds[None], timestep=timestep, timestep_indices=timestep_indices,
            token_tags=tags, position_ids=positions, video_indices=vi, audio_indices=ai, text_indices=ti,
            return_dict=False)
        velocity = pred[0, vc:].float()
        if index == end - 1:
            last_x0 = data_prediction(video_rows[vc:], velocity, t)
        video_rows[vc:] = vs.step(velocity, t, video_rows[vc:], return_dict=False)[0]
        audio_rows[ac:] = aus.step(apred[0, ac:].float(), at, audio_rows[ac:], return_dict=False)[0]
        if step_seconds is not None:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started)
    if runtime:
        runtime.barrier()
    video_state = unpack_video(video_rows[vc:], vf, h, w, channels, patch)
    audio_state = audio_rows[ac:].reshape(AUDIO_CHANNELS, af, 32).permute(0, 2, 1).contiguous()
    x0 = unpack_video(last_x0, vf, h, w, channels, patch)
    if capture is not None:
        capture.update(video_state=video_state.detach().cpu(), video_x0=x0.detach().cpu(),
            range_start=start, range_end=end, video_sigmas=vs.sigmas.detach().cpu().tolist(),
            audio_sigmas=aus.sigmas.detach().cpu().tolist(), audio_policy="continue_noisy_audio_no_freeze")
    return (x0 if end < 8 else video_state), audio_state


@contextmanager
def install(engine, request, tensors):
    """Scope substitution to this owned worker; restore all originals on error."""
    import src.inference.render as render
    import freevideo_engine.reference_sampler as reference
    import freevideo_engine.sampling_progress as progress
    originals = render.generate_latents, reference.generate_latents, progress.SamplingProgress
    capture = {}
    first = request["role"] == "SPLIT_LOW"
    start, end = (0, 4) if first else (4, 8)
    initial = None if first else (tensors["initial_video"], tensors["initial_audio"])
    def generate(*args, **kwargs):
        engine.cursor.reset(start)
        return run(*args, **kwargs, canvas=request["geometry"], start=start, end=end,
                   initial=initial, capture=capture)
    def telemetry(steps, blocks, *args, **kwargs):
        return originals[2](end - start, blocks, *args, **kwargs)
    render.generate_latents = reference.generate_latents = generate
    progress.SamplingProgress = telemetry
    try:
        yield capture
    finally:
        render.generate_latents, reference.generate_latents, progress.SamplingProgress = originals
