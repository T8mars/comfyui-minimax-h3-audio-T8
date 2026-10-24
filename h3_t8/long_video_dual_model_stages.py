"""Private stage primitives for the independent dual-MODEL long-video route.

Used by the independent EXP runner. Old single-model schemas are untouched.
The first stage must export denoised x0, not its nonzero-sigma noisy output.
"""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import time

import torch
import comfy.nested_tensor
import comfy.utils
from comfy.patcher_extension import WrappersMP

from .execution_timing import WallTimings
from .patch_stack_policy import warn_patch_stack

from .learned_latent_upscale_advanced import build_learned_two_pass_parity_plan
from .long_video_in_node_loop_effects_advanced import _sample_prepared_segment
from .relay_sol_backend import capture_composed_backend
from .h3_core_compat import plain_attention_backend, set_h3_attention_backend


class _NativePytorchObserver:
    """Count a verified native per-model selector without altering its math."""
    def __init__(self):
        from comfy.ldm.modules import attention
        self.function = attention.attention_pytorch
        self.completed = 0

    def attention(self, *args, **kwargs):
        result = self.function(*args, **{**kwargs, '_inside_attn_wrapper': True})
        self.completed += 1
        return result

    def report(self):
        return {'kind': 'native_core_pytorch_selector',
            'completed_calls': {'pytorch:completed': self.completed},
            'scope': 'completed calls to the authenticated native Core selector; no numerical replacement'}


def _stage_backend_call_deltas(after, before):
    """Read-only per-stage counts, including nested fallback delegates."""
    report = deepcopy(after)
    for key, value in report.items():
        previous = before.get(key, {})
        if key == 'completed_calls' and isinstance(value, dict) and isinstance(previous, dict):
            report[key] = {name: count - previous.get(name, 0) for name, count in value.items()}
        elif isinstance(value, dict) and isinstance(previous, dict):
            report[key] = _stage_backend_call_deltas(value, previous)
    return report


def bind_stage_conditioning(model, builder, **kwargs):
    """Bind spatial/Relay conditions before reattaching this stage's LoRA stack.

    Existing Relay's public guard remains strict. This private composer only
    removes weight-patch descriptors from an unloaded clone during binding;
    it does not mutate, load, or merge the shared base model's weights.
    """
    weight_patches = {key: list(value) for key, value in model.patches.items()}
    weight_uuid = model.patches_uuid
    unweighted = model.clone()
    unweighted.patches = {}
    result = builder(model=unweighted, **kwargs)
    if not isinstance(result, tuple) or not result:
        raise RuntimeError("Dual-stage conditioning builder must return a MODEL-first tuple")
    bound = result[0]
    if bound.model is not model.model or bound.patches:
        raise RuntimeError("Conditioning builder changed the base model or installed unrequested weights")
    if model.patches_uuid != weight_uuid or model.patches != weight_patches:
        raise RuntimeError("Stage weight configuration changed while binding conditioning")
    patched = bound.clone()
    patched.patches = {key: list(value) for key, value in weight_patches.items()}
    patched.patches_uuid = weight_uuid
    return (patched, *result[1:])


def dual_model_schedules(first_model, second_model, *, coarse_steps=4, refine_steps=4):
    """Each MODEL's own clock supplies its own schedule; no shared MODEL alias."""
    first, _, first_report = build_learned_two_pass_parity_plan(first_model, 8, coarse_steps, refine_steps)
    _, second, second_report = build_learned_two_pass_parity_plan(second_model, 8, coarse_steps, refine_steps)
    return first, second, {
        "mode": "independent_models_learned_upscale",
        "first_pass": json.loads(first_report), "second_pass": json.loads(second_report),
        "first_output": "denoised_x0", "second_input": "learned_upscaled_x0_with_fresh_highres_conditions",
        "second_output": "sampler_output_at_zero", "total_nfe": coarse_steps + refine_steps,
        "audio_clock": "each pass derives its audio sigma from its own model's video/base-flow clock",
    }


def _validate_av_samples(samples):
    if not getattr(samples, "is_nested", False):
        raise RuntimeError("Dual-stage H3 requires joint AV nested samples")
    parts = tuple(samples.unbind())
    if (len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4
            or tuple(parts[0].shape[:2]) != (1, 24) or tuple(parts[1].shape[:3]) != (1, 32, 2)):
        raise RuntimeError("Dual-stage H3 received incompatible AV channels or batch size")
    if any(not bool(torch.isfinite(value).all()) for value in parts):
        raise RuntimeError("Dual-stage H3 latent contains NaN or Inf")
    return tuple(tuple(value.shape) for value in parts)


def sample_model_stage(model, positive, av_latent, *, sampler, sigmas, seed,
                       segment_index=0, output_kind="denoised_x0"):
    # The successful runner releases residency after collecting its report.
    # Exceptions (including Core cancellation) never reach that success path.
    # Do not release a stage rejected before sampling has started.
    started = [False]
    try:
        return _sample_model_stage(model, positive, av_latent, sampler=sampler, sigmas=sigmas,
            seed=seed, segment_index=segment_index, output_kind=output_kind, stage_started=started)
    except BaseException as error:
        if started[0]:
            from .long_video_dual_residency import release_stage_residency
            try:
                release_stage_residency(model)
            except BaseException as cleanup_error:
                message = f"Failed-stage residency cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                if hasattr(error, 'add_note'):
                    error.add_note(message)
                logging.warning(message)
        raise


def _sample_model_stage(model, positive, av_latent, *, sampler, sigmas, seed,
                        segment_index, output_kind, stage_started):
    if output_kind not in {"denoised_x0", "zero_sigma_output"}:
        raise ValueError("Unknown dual-stage output kind")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2
            or not bool(torch.isfinite(sigmas).all())
            or not bool(torch.all(sigmas[:-1] > sigmas[1:])) or float(sigmas[-1]) < 0):
        raise ValueError("Dual-stage sigmas must be a finite strict descent")
    if output_kind == "zero_sigma_output" and float(sigmas[-1]) != 0:
        raise ValueError("Final high-resolution output must end at sigma zero")
    expected_shapes = _validate_av_samples(av_latent["samples"])
    observed = model.clone()
    from .fast_h3_v2_advanced import KEY as V2_KEY, capture_fast_h3_v2_owner
    v2_owner = capture_fast_h3_v2_owner(observed)
    from .h3_memory_advanced import inspect_t8_memory_composition
    from .prompt_relay_advanced import PROMPT_RELAY_WRAPPER_KEY, prompt_relay_model_contract

    diffusion_groups = {
        key
        for key, values in observed.wrappers.get(
            WrappersMP.DIFFUSION_MODEL, {}
        ).items()
        if values
    }
    allowed_wrapper_keys = tuple(key for key in (PROMPT_RELAY_WRAPPER_KEY, V2_KEY)
        if key in diffusion_groups and (key != V2_KEY or v2_owner is not None))
    memory_composition = inspect_t8_memory_composition(observed, allowed_wrapper_keys=allowed_wrapper_keys)
    memory_report = (
        None
        if memory_composition is None
        else {
            key: value
            for key, value in memory_composition.items()
            if key != "methods"
        }
    )
    relay_counts_before = None
    relay_contract = None
    if PROMPT_RELAY_WRAPPER_KEY in diffusion_groups:
        relay_contract = prompt_relay_model_contract(observed)
        relay_counts_before = relay_contract["execution_counts"]
    preserve_relay_owner = (v2_owner is None and relay_contract is not None
        and relay_contract["attention_owner_verified"])
    override = observed.model_options.get("transformer_options", {}).get("optimized_attention_override")
    if preserve_relay_owner:
        # This is the authenticated captured owner, not the callable's public
        # tag. Observe its existing delegate without wrapping the Relay router.
        backend = relay_contract["attention_backend"]
    else:
        # Later user-selected delegates and FastH3V2 keep their original path.
        backend = capture_composed_backend(override)
        if backend is None and plain_attention_backend(override) == 'pytorch':
            backend = _NativePytorchObserver()
        if relay_contract is not None and backend is None:
            backend = relay_contract["attention_backend"]
        if backend is not None and v2_owner is None:
            set_h3_attention_backend(observed, backend.attention)
    backend_counts_before = deepcopy(backend.report()) if preserve_relay_owner and backend is not None else None
    completed_forwards = 0
    timings = WallTimings()

    def count_forward(executor, *args, **kwargs):
        nonlocal completed_forwards
        result = timings.call('forward_including_dynamic_transfers', executor, *args, **kwargs)
        completed_forwards += 1
        return result

    observer_key = "t8_dual_stage_network_observer"
    prepare_key = 't8_dual_stage_prepare_timer'
    prepare_type = getattr(WrappersMP, 'PREPARE_SAMPLING', None)
    if observed.get_wrappers(WrappersMP.APPLY_MODEL, observer_key) or (
            prepare_type and observed.get_wrappers(prepare_type, prepare_key)):
        raise RuntimeError('Reserved dual-stage observation wrapper already exists')
    def observe_prepare(executor, *args, **kwargs):
        return timings.call('prepare_sampling_including_model_load', executor, *args, **kwargs)
    # Native H3 BaseModel._apply_model calls the DiT once. Observe its outer
    # boundary so Relay/EAV remain the sole owners of the inner DiT wrapper.
    observed.add_wrapper_with_key(WrappersMP.APPLY_MODEL, observer_key, count_forward)
    preview = {}
    started = time.perf_counter()
    stage_started[0] = True
    try:
        if prepare_type:
            observed.add_wrapper_with_key(prepare_type, prepare_key, observe_prepare)
        output = _sample_prepared_segment(observed, positive, av_latent, sampler=sampler, sigmas=sigmas,
                                          seed=seed, segment_index=segment_index, preview_state=preview)
    finally:
        observed.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, observer_key)
        if prepare_type:
            observed.remove_wrappers_with_key(prepare_type, prepare_key)
    if output_kind == "denoised_x0":
        if "x0" not in preview:
            raise RuntimeError("First stage did not return denoised x0; noisy output cannot be upscaled")
        x0 = preview.pop("x0")
        if not x0.is_nested:
            x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, expected_shapes))
        # Mirrors Core SamplerCustomAdvanced.denoised_output processing;
        # raw callback tensors are still in the model's latent normalization.
        output["samples"] = model.model.process_latent_out(x0.cpu())
    if _validate_av_samples(output["samples"]) != expected_shapes:
        raise RuntimeError("Stage sampler changed the AV latent geometry")
    if completed_forwards != sigmas.numel() - 1:
        warn_patch_stack('Dual stage forward observer coverage differs from CFG1 schedule; user stack unverified')
    relay_execution = None
    if relay_counts_before is not None:
        relay_counts_after = prompt_relay_model_contract(observed)["execution_counts"]
        relay_execution = {
            key: int(relay_counts_after.get(key, 0))
            - int(relay_counts_before.get(key, 0))
            for key in ("completed_forwards", "routed_attention_calls")
        }
        if relay_execution["completed_forwards"] != completed_forwards:
            warn_patch_stack(
                "Prompt Relay completed-forward count differs from the native stage observer"
            )
    backend_report = (
        backend.report()
        if backend is not None
        else {"status": "not_measured_by_plain_selector_observer"}
    )
    if backend_counts_before is not None:
        backend_report = _stage_backend_call_deltas(backend_report, backend_counts_before)
        backend_report['counter_scope'] = 'this stage; captured Relay delegate delta, not end-to-end speed'
    if memory_report is not None:
        backend_report["memory_composition"] = memory_report
    if v2_owner is not None:
        capture_fast_h3_v2_owner(observed)
        backend_report['fasth3_v2'] = v2_owner.runtime.snapshot()
    return output, {
        "output_kind": output_kind, "seed": int(seed), "nfe": int(sigmas.numel() - 1),
        "completed_network_forwards": completed_forwards,
        "forward_evidence_complete": completed_forwards == sigmas.numel() - 1,
        "forward_observation": "completed native BaseModel.apply_model calls; one DiT call per native H3 invocation",
        "backend": backend_report,
        "memory_composition": memory_report,
        "prompt_relay_execution": relay_execution,
        "sigmas": sigmas.detach().cpu().tolist(), "sample_seconds": time.perf_counter() - started,
        "shape": expected_shapes,
        "execution_timings": {**timings.report(),
            "prepare_observation": 'native_wrapper_available' if prepare_type else 'not_supported_by_this_core',
            "dynamic_transfer_note": 'Dynamic Core can transfer weights inside forward; those transfers remain included, not claimed as pure compute.'},
        "scope": "sampling call wall time, including model movement inside sampler; not end-to-end video time",
    }
