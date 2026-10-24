"""Owned-engine orchestration: exact upstream passes, measured execution counts."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import os


@contextmanager
def controlled_cache_environment():
    """Worker-local only; host FREEVIDEO_STEP_CACHE may not rewrite a request.

    The supervisor also strips this variable before spawning. Restoration makes
    this scope safe for independent CPU tests and exceptions, not a global mode.
    """
    previous = os.environ.pop("FREEVIDEO_STEP_CACHE", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["FREEVIDEO_STEP_CACHE"] = previous
        else:
            os.environ.pop("FREEVIDEO_STEP_CACHE", None)


class Observations:
    def __init__(self, model):
        self.model = model
        self.forwards = 0
        self.block_calls = [0 for _ in model.transformer_blocks]
        self.handles = []

    def __enter__(self):
        def forward(*args):
            self.forwards += 1
        self.handles.append(self.model.register_forward_pre_hook(forward))
        for index, block in enumerate(self.model.transformer_blocks):
            def count(*args, index=index):
                self.block_calls[index] += 1
            self.handles.append(block.register_forward_pre_hook(count))
        return self

    def snapshot(self):
        # Hooks observe actual block work. A cached callback or a model-level
        # early return is not counted as a full DiT evaluation.
        complete = min(self.block_calls) if self.block_calls else None
        uniform = len(set(self.block_calls)) <= 1
        return dict(actual_transformer_calls=self.forwards, actual_block_calls=list(self.block_calls),
            actual_full_DiT_forwards=complete if uniform else None, block_coverage_uniform=uniform,
            partial=not uniform, counter_scope="actual_module_calls_not_sampler_callbacks")

    def __exit__(self, *error):
        for handle in reversed(self.handles):
            handle.remove()


def execute(engine, conditioning, seed, selected, *, upscaler=None, low=None, lifted=None, progress=None, pass_context=None):
    """No synthetic sampling, implicit interpolation, schedule or audio rewrite.

    `engine` is the real pinned Engine in the production owned process. CPU
    contract tests supply a separately labelled real-tensor sampler owner; they
    cannot yield a production worker receipt or trained/GPU qualification.
    """
    from .source import canonical, digest
    import hashlib
    from pathlib import Path
    import torch

    # The real Engine re-reads its own derived .pt file for each pass. Bind
    # that actual input too, not only the parent's original safetensors.
    condition_path = Path(conditioning).resolve(strict=True) if isinstance(conditioning, (str, os.PathLike)) else None
    condition_sha256 = digest(condition_path) if condition_path is not None else None
    def verify_condition():
        if condition_path is not None and digest(condition_path) != condition_sha256:
            raise ValueError("H21 actual conditioning file content changed during sampling; no completion")

    sampling = selected["sampling"]
    role = selected["role"]
    before = copy.deepcopy(engine.config)
    requested_cache = selected["step_cache"]
    engine.config["step_cache"] = requested_cache
    passes, callbacks = [], 0
    low_audio = None
    model = engine.transformer

    def run(canvas, pass_role, *, initial=None):
        nonlocal callbacks
        completed = 0
        def step(seconds):
            nonlocal callbacks, completed
            callbacks += 1
            completed += 1
            if progress:
                progress(callbacks, selected["expected_callbacks"], seconds)
        extra = {} if initial is None else dict(initial_latents=initial, refine_steps=sampling["refine_steps"],
                                                 refine_schedule=sampling["refine_schedule"])
        scope = pass_context(canvas, pass_role) if pass_context is not None else nullcontext()
        with scope, Observations(model) as observed:
            verify_condition()
            video, audio, actual = engine.sample(conditioning,
                seed if pass_role != "HIGH" else (seed + sampling["restart_seed_offset"]) % (1 << 64),
                **canvas, **extra, allow_smaller_canvas=pass_role == "LOW", step_callback=step)
            verify_condition()
            measured = observed.snapshot()
        expected = sampling["refine_steps"] if pass_role == "HIGH" else sampling["base_steps"]
        if completed != expected or len(actual.get("step_seconds", [])) != expected:
            raise ValueError("H21 actual sampler callback completion disagrees with its explicit pass")
        if measured["actual_transformer_calls"] != completed:
            measured["partial"] = True
            measured["reason"] = "actual_transformer_calls_differ_from_callbacks"
        passes.append(dict(role=pass_role, canvas=canvas, completed_callbacks=completed,
            measurements=measured, sample=actual))
        return video, audio

    try:
        with controlled_cache_environment(), torch.no_grad():
            if not sampling["enabled"]:
                video, audio = run(sampling["first"], "SINGLE")
            else:
                if role == "HIGH":
                    if low is None or lifted is None:
                        raise ValueError("HIGH requires completed LOW and its actual learned spatial lift")
                    _, low_audio = low
                    video = lifted
                    audio = low_audio
                else:
                    video, audio = run(sampling["first"], "LOW")
                    low_audio = audio.detach().clone()
                if role != "LOW":
                    if role != "HIGH" and sampling.get("upscaler_sha256"):
                        if upscaler is None:
                            raise ValueError("This exact plan requires its selected learned 3D upscaler")
                        video, upscale_report = upscaler(video, sampling["upscale_target"])
                    else:
                        upscale_report = dict(noop=True)
                    if role != "HIGH":
                        from .source import Source
                        source = Source(engine.h21_source_root)
                        video = source.module("two_pass").crop_latents(video, sampling)
                    video, audio = run(sampling["second"], "HIGH", initial=(video, audio))
                    if not torch.equal(audio, low_audio):
                        raise ValueError("H21 HIGH changed completed LOW audio")
                else:
                    upscale_report = None
            if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in (video, audio)):
                raise ValueError("H21 actual output contains non-finite or non-floating latents")
        if callbacks != selected["expected_callbacks"]:
            raise ValueError("H21 request did not complete all actual callbacks")
        verify_condition()
        return video, audio, dict(passes=passes, completed_callbacks=callbacks,
            plan_identity=selected["identity"], plan_sha256=hashlib.sha256(canonical(selected).encode()).hexdigest(),
            step_cache=requested_cache, high_audio_preserved=low_audio is not None and role != "LOW",
            upscale=upscale_report if sampling["enabled"] else None,
            actual_conditioning_file_sha256=condition_sha256,
            trained_GPU_qualified=False, human_quality_accepted=False)
    finally:
        engine.config.clear()
        engine.config.update(before)
        # The author's end_pass normally clears this. Exceptions before its
        # loop must not leave mutable cache state in a reusable owned engine.
        if hasattr(model, "_freevideo_step_cache"):
            model._freevideo_step_cache = None
