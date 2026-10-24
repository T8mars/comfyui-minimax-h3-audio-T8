"""Explicit real-VDN CPU contract probe, NOT the production Engine or trained weights."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


class TinyVDNCPU:
    """Actual Diffusers transformer + original author samplers, small random weights.

    Production Engine refuses CPU; this explicitly separate test owner neither
    lies about its backend nor emits production completed Stage receipts.
    """
    def __init__(self, source, selected, *, zero_heads=False):
        import torch
        from diffusers import MiniMaxH3Transformer3DModel
        self.source = source
        self.h21_source_root = str(source.root)
        self.steps = selected["sampling"]["base_steps"]
        self.config = dict(step_cache=selected["step_cache"], probe_only=True)
        torch.manual_seed(211026)
        self.transformer = MiniMaxH3Transformer3DModel(num_attention_heads=1, attention_head_dim=24,
            hidden_size=24, num_layers=2, num_refiner_layers=1, ffn_dim=48, in_channels=24,
            audio_in_channels=32, text_dim=8, freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=12,
            rope_freq_dim=4).eval().requires_grad_(False)
        if zero_heads:
            for module in (self.transformer.proj_out, self.transformer.audio_proj_out):
                module.weight.zero_()
                module.bias.zero_()
        source.module("packing").install_streamed_forward(self.transformer, chunk=128)

    def sample(self, condition, seed, *, width, height, frames, step_callback,
               allow_smaller_canvas=False, initial_latents=None, refine_steps=None, refine_schedule=None):
        from src.inference.render import generate_latents
        model = self.transformer
        source = self.source
        cache_module = source.module("step_cache")
        refining = initial_latents is not None
        if refining:
            generate_latents = source.module("refine").generate_latents
        elif condition.get("references"):
            generate_latents = source.module("reference_sampler").generate_latents
        sampler = source.module("geometry").sampler_for_canvas(generate_latents, width, height)
        seconds, trace = [], []
        start = [None]
        def before(module, args, kwargs):
            start[0] = time.perf_counter()
            trace.append(dict(timestep=kwargs["timestep"].tolist(),
                timestep_indices=kwargs["timestep_indices"].tolist(), token_tags=kwargs["token_tags"].tolist(),
                video_indices=kwargs["video_indices"].tolist(), audio_indices=kwargs["audio_indices"].tolist()))
        def after(*args):
            elapsed = time.perf_counter() - start[0]
            seconds.append(elapsed)
            step_callback(elapsed)
        handles = [model.register_forward_pre_hook(before, with_kwargs=True), model.register_forward_hook(after)]
        extra = {} if not refining else dict(initial_latents=initial_latents, refine_steps=refine_steps, refine_schedule=refine_schedule)
        conditions = condition.get("references") or ((condition["keyframe_anchors"], condition["condition_latents"])
                                                   if condition.get("keyframe_anchors") else None)
        cache = None
        from h3_t8.freevideo_h21.cache_state import snapshot_history
        history_scope = snapshot_history(cache_module, enabled=self.config.get("step_cache") == "easycache-av")
        try:
            with history_scope:
                if not refining:
                    cache_module.begin_pass(model, self.steps, self.config.get("step_cache"))
                # step_seconds=None deliberately avoids the old CUDA-only timing
                # call. Real module hooks measure actual CPU evaluations instead;
                # sampler code, schedules, original tensor math and RNG are intact.
                video, audio = sampler(model, condition["prompt_embeds"], condition["text_token_tags"],
                    frames, self.steps, seed, "cpu", conditions=conditions, step_seconds=None, **extra)
        finally:
            cache = cache_module.end_pass(model)
            for handle in reversed(handles):
                handle.remove()
        return video, audio, dict(step_seconds=seconds, actual_clock_trace=trace, step_cache=cache,
            probe_family="actual_random_weight_tiny_VDN_CPU_not_Engine", production_engine=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vdn-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("light", "medium", "high", "max"), default="light")
    parser.add_argument("--audio-reference", action="store_true")
    parser.add_argument("--zero-heads", action="store_true")
    parser.add_argument("--retry-after-cancel", action="store_true")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--role", choices=("FULL", "LOW"), default="FULL")
    parser.add_argument("--allow-real-CPU-contract", action="store_true")
    args = parser.parse_args()
    if not args.allow_real_CPU_contract:
        raise ValueError("Explicit CPU model execution opt-in required")
    if args.output.exists():
        raise FileExistsError("Keep each CPU probe output immutable")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU probe requires CUDA_VISIBLE_DEVICES=-1 before Torch imports")
    sys.path[:0] = [str(Path(__file__).parents[2]), str(args.vdn_root / "diffusers/src"), str(args.vdn_root)]
    from h3_t8.freevideo_h21.source import Source, canonical
    from h3_t8.freevideo_h21.plans import plan
    from h3_t8.freevideo_h21.execution import execute
    from h3_t8.freevideo_h21.effects import pass_canvas
    from h3_t8.freevideo_h21.stages import tensor_record
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU probe unexpectedly initialized CUDA")
    torch.set_num_threads(1)
    source = Source(args.source_root)
    task = "ref2va_audio" if args.audio_reference else "t2va"
    selected = plan(source, args.tier, args.width, args.height, 39, task=task, role=args.role)
    owner = TinyVDNCPU(source, selected, zero_heads=args.zero_heads)
    condition = dict(prompt_embeds=torch.randn(3, 8), text_token_tags=torch.ones(3, dtype=torch.long))
    if args.audio_reference:
        condition["references"] = [dict(kind="audio", audio_latent=torch.randn(4, 32))]
    # The actual shared effects body owns a plain `canvas` field. This observes
    # that exact geometry-scope contract around real packed forwards, without
    # pretending to execute its cuDNN-only effect kernels on a CPU facade.
    from types import SimpleNamespace
    from contextlib import contextmanager
    state = SimpleNamespace(canvas=dict(selected["actual_canvas"]))
    initial_canvas = state.canvas
    pass_trace = []
    @contextmanager
    def observed_canvas(canvas, role):
        entry = dict(role=role, actual_canvas=dict(canvas), actual_video_rows=[])
        with pass_canvas(state, canvas, role):
            def observe(module, values, kwargs):
                if state.canvas != canvas:
                    raise AssertionError("Effect canvas differs from the actual sampled stage")
                rows = len(kwargs["video_indices"])
                wanted = ((canvas["frames"] - 5) // 17 * 5 + 2) * (canvas["width"] // 32) * (canvas["height"] // 32)
                if rows != wanted:
                    raise AssertionError("Effect canvas differs from real packed target video rows")
                entry["actual_video_rows"].append(rows)
            handle = owner.transformer.register_forward_pre_hook(observe, with_kwargs=True)
            try:
                yield
            finally:
                handle.remove()
        if state.canvas is not initial_canvas:
            raise AssertionError("Pass-local effect canvas was not restored by identity")
        entry["original_canvas_restored"] = True
        pass_trace.append(entry)
    retry_report = None
    if args.retry_after_cancel:
        from h3_t8.freevideo_h21.execution import Observations
        original_config = dict(owner.config)
        original_state = {name: tensor_record(value) for name, value in owner.transformer.state_dict().items()}
        def cancel(completed, total, seconds):
            if completed == 2:
                raise InterruptedError("explicit_CPU_contract_cancel")
        with Observations(owner.transformer) as interrupted:
            try:
                execute(owner, condition, 211026, selected, progress=cancel, pass_context=observed_canvas)
            except InterruptedError as error:
                if str(error) != "explicit_CPU_contract_cancel":
                    raise
            else:
                raise AssertionError("Real interrupted sampler did not propagate cancellation")
            aborted = interrupted.snapshot()
        if (owner.config != original_config or getattr(owner.transformer, "_freevideo_step_cache", None) is not None
                or owner.transformer._forward_pre_hooks or owner.transformer._forward_hooks):
            raise AssertionError("Interrupted CPU sampler retained request state/hooks")
        if original_state != {name: tensor_record(value) for name, value in owner.transformer.state_dict().items()}:
            raise AssertionError("Interrupted CPU sampler changed actual model parameters")
    video, audio, actual = execute(owner, condition, 211026, selected, pass_context=observed_canvas)
    if args.retry_after_cancel:
        fresh = TinyVDNCPU(source, selected, zero_heads=args.zero_heads)
        fresh_video, fresh_audio, fresh_actual = execute(fresh, condition, 211026, selected)
        if not torch.equal(video, fresh_video) or not torch.equal(audio, fresh_audio):
            raise AssertionError("Retry after cancellation differs from an actual fresh sampler owner")
        retry_report = dict(aborted_request=aborted, retry_callbacks=actual["completed_callbacks"],
            fresh_reference_callbacks=fresh_actual["completed_callbacks"], model_bytes_preserved=True,
            AV_bit_exact_with_actual_fresh_owner=True, cancelled_request_not_completed=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU sampler unexpectedly initialized CUDA")
    source.verify()
    value = dict(schema="t8.freevideo.h21.tiny-CPU-probe.v1", plan=selected, actual=actual,
        video=tensor_record(video), audio=tensor_record(audio), CUDA_initialized=False,
        model="Actual Diffusers MiniMaxH3Transformer3DModel two blocks random weights text8 on CPU",
        cancellation_retry=retry_report,
        actual_pass_canvas_trace=pass_trace, effects_kernels_qualified=False,
        source_identity=source.identity, trained_model_qualification=False, Engine_qualification=False,
        upscaler_qualification=False, quality_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        stream.write(canonical(value))


if __name__ == "__main__":
    main()
