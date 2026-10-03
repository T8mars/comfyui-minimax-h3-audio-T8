"""Opt-in stage-local FETA routing; legacy EAV recipes are not reinterpreted.

This adapter retains the incoming attention delegate and any DiT producers. A
source-authenticated native V2 producer has its own bounded FETA adapter.
Unknown producers which bypass both routes are reported as uncovered, never
replaced with Dense and never certified by an empty audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from collections.abc import Mapping
import json
import math

import torch
import comfy.patcher_extension as extension

from .. import enhance_a_video_advanced as feta
from .. import fast_h3_v2_advanced as v2
from .. import prompt_relay_advanced as relay
from ..patch_stack_policy import warn_patch_stack
from ..progressive_eav_masks import NativeProgressiveMaskContract
from ..progressive_masking import normalize_av_masks
from ..sampling import nested_av_parts
from .contracts import StageContext
from .fast_h3_v2 import RECIPE

KEY = "t8_modular_stage_eav_v1"
RUNTIME_TYPE = "T8_STAGE_EAV_RUNTIME"
CONFIG_TYPE = "T8_STAGE_EAV_CONFIG"


@dataclass(frozen=True)
class EAVConfig:
    mode: str = "report_only"
    tau: float = 4.0
    start_video_progress: float = 0.15
    end_video_progress: float = 0.90
    max_workspace_mib: int = 32
    g_hard_limit: float = 1.5

    def __post_init__(self):
        if self.mode not in ("disabled", "report_only", "apply_exp"):
            raise ValueError("Unknown stage EAV mode")
        for name in ("tau", "start_video_progress", "end_video_progress", "g_hard_limit"):
            if type(getattr(self, name)) not in (float, int) or not math.isfinite(getattr(self, name)):
                raise ValueError(f"Stage EAV {name} must be finite")
        if not -32 <= self.tau <= 32 or not 1 <= self.g_hard_limit <= 3:
            raise ValueError("Stage EAV tau/g hard limit out of range")
        if not 0 <= self.start_video_progress < self.end_video_progress <= 1:
            raise ValueError("Stage EAV progress must satisfy 0 <= start < end <= 1")
        if type(self.max_workspace_mib) is not int or not 4 <= self.max_workspace_mib <= 512:
            raise ValueError("Stage EAV workspace must be an integer in [4,512] MiB")


def validate_stage(model, sigmas, av_latent, context):
    from . import hyperflow_curve_effects
    if type(context) is StageContext and context.recipe == hyperflow_curve_effects.RECIPE:
        return hyperflow_curve_effects.validate_stage(model, sigmas, av_latent, context)
    from . import audio_refine_effects
    if type(context) is StageContext and context.recipe == audio_refine_effects.RECIPE:
        return audio_refine_effects.validate_stage(model, sigmas, av_latent, context)
    from . import h16_effects
    if type(context) is StageContext and context.recipe == h16_effects.RECIPE:
        return h16_effects.validate_stage(model, sigmas, av_latent, context)
    from . import chunked_v5_effects
    if type(context) is StageContext and context.recipe == chunked_v5_effects.RECIPE:
        return chunked_v5_effects.validate_stage(model, sigmas, av_latent, context)
    from . import chunked_effects
    if type(context) is StageContext and context.recipe == chunked_effects.RECIPE:
        return chunked_effects.validate_stage(model, sigmas, av_latent, context)
    from . import speed_effects
    if type(context) is StageContext and context.recipe == speed_effects.RECIPE:
        return speed_effects.validate_stage(model, sigmas, av_latent, context)
    from . import hyperflow_fresh
    if type(context) is StageContext and context.recipe in hyperflow_fresh.RECIPES:
        return hyperflow_fresh.validate_stage(model, sigmas, av_latent, context)
    from . import hyperflow_effects
    if type(context) is StageContext and context.recipe == hyperflow_effects.RECIPE:
        return hyperflow_effects.validate_stage(model, sigmas, av_latent, context)
    from . import native_dual, manual_pass, rf_stages, native_explicit, pdd_stages, vdn_stages
    if type(context) is StageContext and context.recipe == vdn_stages.RECIPE:
        return vdn_stages.validate_stage(model, sigmas, av_latent, context)
    if type(context) is StageContext and context.recipe == pdd_stages.RECIPE:
        return pdd_stages.validate_stage(model, sigmas, av_latent, context)
    if type(context) is StageContext and context.recipe == native_explicit.RECIPE:
        return native_explicit.validate_stage(model, sigmas, av_latent, context)
    if type(context) is StageContext and context.recipe == rf_stages.RECIPE:
        return rf_stages.validate_stage(model, sigmas, av_latent, context)
    if type(context) is StageContext and context.recipe == manual_pass.RECIPE:
        return manual_pass.validate_stage(model, sigmas, av_latent, context)
    if type(context) is StageContext and context.recipe == native_dual.RECIPE:
        return native_dual.validate_stage(model, sigmas, av_latent, context)
    if type(context) is not StageContext or context.recipe != RECIPE:
        raise ValueError("Stage EAV needs an implemented recipe adapter, not an arbitrary sigma slice")
    expected_start, expected_end = {"low_0_4": (0, 4), "high_4_8": (4, 8)}.get(context.stage, (-1, -1))
    full = v2.dmd_sigmas()
    if (context.start, context.end) != (expected_start, expected_end) or tuple(full.tolist()) != context.trajectory_sigmas:
        raise ValueError("Stage context does not match the exact V2 absolute trajectory")
    if (context.video_shift, context.audio_shift) != (10., 3.):
        raise ValueError("Stage AV clock shifts do not match V2")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("Stage SIGMAS must be a floating one-dimensional tensor")
    expected = full[context.start:context.end + 1].to(sigmas)
    if sigmas.shape != expected.shape or not torch.allclose(sigmas, expected, rtol=0, atol=1e-7):
        raise ValueError("Stage SIGMAS differ from the bound absolute window")
    owner = v2.capture_fast_h3_v2_owner(model)
    if owner is None or owner.profile != context.profile:
        raise ValueError("Stage MODEL is not paired with this V2 context/profile")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != context.video_shape or tuple(audio.shape) != context.audio_shape:
        raise ValueError("Stage AV layout differs from context")
    return video, audio, owner


def expected_forward_plan(context, blocks):
    from . import rf_stages, native_explicit
    if context.recipe == native_explicit.RECIPE:
        return native_explicit.forward_plan(context)
    if context.recipe == rf_stages.RECIPE:
        return rf_stages.forward_plan(context)
    return {"known": True, "forwards": [{"sigma": sigma, "attention_blocks": blocks, "weak": False}
        for sigma in context.trajectory_sigmas[context.start:context.end]]}


class StageEAVRuntime:
    """Execution telemetry only. This token does not authorize persisted reuse."""
    def __init__(self, config, context, blocks):
        self.config, self.context, self.blocks = config, context, int(blocks)
        self.closed = True
        self.completed_forwards = 0
        self.selector_calls = 0
        self.sparse_calls = 0
        self.relay_calls = 0
        self.observed_sigmas = []
        self.relay_required = False
        self.v2_runtime = None
        self.telemetry = feta.EAVRuntime({"mode": config.mode, "stage": context.to_dict()})

    def prepare(self, patcher, timestep, model_options):
        if self.closed:
            self.completed_forwards = self.selector_calls = self.sparse_calls = self.relay_calls = 0
            self.observed_sigmas = []
            self.telemetry = feta.EAVRuntime({"mode": self.config.mode, "stage": self.context.to_dict()})
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        result = self.telemetry.snapshot(consume=False)
        plan = expected_forward_plan(self.context, self.blocks)
        planned = [item["sigma"] for item in plan["forwards"]]
        clock_match = len(planned) == len(self.observed_sigmas) and all(
            math.isclose(a, b, rel_tol=0, abs_tol=2e-6) for a, b in zip(planned, self.observed_sigmas)
        )
        active = [item for item in result["forwards"] if item["active"]]
        measured = len(result["forwards"]) == len(plan["forwards"]) and all(
            not item["active"] or item["attention_count"] >= expected["attention_blocks"]
            for item, expected in zip(result["forwards"], plan["forwards"]))
        attention_count = sum(item["attention_blocks"] for item in plan["forwards"])
        if self.config.mode == "disabled":
            status = "disabled_identity"
        elif self.context.recipe == "audio_refine.signed-tail.v1" and self.context.steps == 0:
            status = "abstain_no_sample"
        elif result["aborted"]:
            status = "aborted"
        elif (not plan["known"] or not clock_match or self.completed_forwards != len(planned)
              or self.selector_calls + self.sparse_calls < attention_count):
            status = "unverified_incomplete_stage_coverage"
        elif not measured:
            status = "unverified_incomplete_feta_coverage"
        elif self.relay_required and self.relay_calls < attention_count:
            status = "unverified_relay_coverage"
        elif not active:
            status = "observed_no_steps_in_effect_window"
        else:
            status = "observed_report_only" if self.config.mode == "report_only" else "observed_apply_exp"
        return {"schema": "t8.modular-sampling.eav-audit.v1", "status": status,
                "config": asdict(self.config), "stage_context": self.context.to_dict(),
                "completed_forwards": self.completed_forwards, "planned_forwards": len(planned),
                "planned_integrator_intervals": self.context.steps, "forward_plan": plan,
                "selector_calls": self.selector_calls, "sparse_producer_calls": self.sparse_calls,
                "relay_attention_calls": self.relay_calls,
                "relay_required": self.relay_required,
                "v2_dispatch": self.v2_runtime.snapshot() if self.v2_runtime is not None else None,
                "observed_absolute_sigmas": list(self.observed_sigmas), "clock_match": clock_match,
                "feta": result, "cache_reuse_authorized": False, "quality_accepted": False,
                "boundary": "Selector and authenticated V2/VDN producer calls are counted separately. "
                            "Sparse FETA uses bounded extra native target-video projections before quantization; "
                            "the original VSA kernel/plan/sinks/gate are retained. VSA Relay timeline bias is not yet "
                            "adapted. VDN Relay requires its separate window/linear adapter and audit; its linear "
                            "text weighting is experimental. Unknown producers can bypass effects; no silent Dense replacement."}


def apply_stage_eav(model, sigmas, av_latent, stage_context, config, *, p7_phase=None, external_scope=None):
    if type(config) is not EAVConfig:
        raise TypeError("Use the external Stage EAV Config node")
    video, audio, v2_owner = validate_stage(model, sigmas, av_latent, stage_context)
    long_video_contract = None
    native_relay_task = None
    if external_scope is not None:
        from ..external_continuation_effects import ExternalEffectScope
        if type(external_scope) is not ExternalEffectScope or p7_phase is not None:
            raise TypeError('External Stage EAV requires its own bound scope, not a native/P7 ancestor')
        external_scope.verify(model, av_latent)
        long_video_contract = feta._assert_long_video_contract(model, segment_index=1,
            context_frames=external_scope.context.context['metadata']['max_context_frames'])
        if model.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY):
            native_relay_task = relay.prompt_relay_model_contract(model)['binding']['task'].lower()
    from . import audio_refine_effects
    if stage_context.recipe == audio_refine_effects.RECIPE:
        tail_owner = audio_refine_effects.capture_owner(model)
        long_video_contract = tail_owner.long_contract
        tail_binding = audio_refine_effects._bound_relay(model)
        native_relay_task = tail_binding["task"].lower() if tail_binding is not None else None
    if p7_phase is not None:
        from . import hyperflow_p7 as p7

        if type(p7_phase) is not p7.P7Phase:
            raise TypeError("P7 Stage EAV requires an authenticated prepared phase")
        phase_receipt = p7_phase.verify()
        request = (p7_phase.contexts.parent.binding["request"]
                   if type(p7_phase.contexts) is p7.P7Contexts else p7_phase.contexts.request)
        segment_index = request["segment_index"]
        if p7_phase.phase == "low":
            marker = av_latent.get(p7.LOW_SOURCE_KEY)
            if marker != p7.low_stage_source(p7_phase)[p7.LOW_SOURCE_KEY]:
                raise ValueError("P7 Stage EAV LOW source is not the prepared phase")
        else:
            marker = av_latent.get(p7.HIGH_SOURCE_KEY)
            if not isinstance(marker, Mapping) or marker.get("high_phase_sha256") != phase_receipt["sha256"]:
                raise ValueError("P7 Stage EAV HIGH source is not the prepared phase")
        if marker.get("segment_index", segment_index) != segment_index:
            raise ValueError("P7 Stage EAV source belongs to a different segment")
        long_video_contract = feta._assert_long_video_contract(model,
            segment_index=segment_index, context_frames=request["context_frames"])
        native_relay_task = p7_phase.result[-1]["resolved_task"].lower()
        if (segment_index == 0 and native_relay_task != "t2va") or (
                segment_index > 0 and not native_relay_task.endswith("-motion")):
            raise ValueError("P7 Stage EAV prepared phase has an unexpected native task")
    runtime = StageEAVRuntime(config, stage_context, len(model.get_model_object("diffusion_model").blocks))
    runtime.v2_runtime = v2_owner.runtime
    if config.mode == "disabled" or (stage_context.recipe == audio_refine_effects.RECIPE and stage_context.steps == 0):
        return model, runtime, json.dumps(runtime.snapshot(), ensure_ascii=False)
    if model.get_wrappers("diffusion_model", KEY):
        raise ValueError("Stage EAV already installed; use one config per stage branch")
    normalized = normalize_av_masks(av_latent.get("noise_mask"), video, audio)
    mask_contract = NativeProgressiveMaskContract(model, normalized, video.shape, audio.shape) if normalized is not None else None
    relay_contract = None
    if model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        relay_contract = relay.prompt_relay_model_contract(model)
        runtime.relay_required = True
    previous = model.model_options["transformer_options"].get("optimized_attention_override")
    if previous is not None and not callable(previous):
        raise TypeError("Existing attention override must be callable")
    patched = model.clone()
    if relay_contract is not None:
        # Replace only the authenticated Relay wrapper. Its attention delegate,
        # MODEL-COND identity, LoRA, masks, and other hooks remain in the graph.
        patched.remove_wrappers_with_key("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
        patched.remove_attachments(relay.PROMPT_RELAY_WRAPPER_KEY)
    binding = deepcopy(relay_contract["binding"]) if relay_contract is not None else None

    def stage_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options if transformer_options is not None else {}
        if options.get("optimized_attention_override") is not stage_attention:
            warn_patch_stack("Stage EAV selector was replaced; effect coverage is unverified")
        if feta.EAV_RUNTIME_KEY in options or relay.PROMPT_RELAY_RUNTIME_KEY in options:
            raise RuntimeError("Nested stage EAV/Relay runtime is invalid")
        try:
            if binding is not None and kwargs.pop(relay.PROMPT_RELAY_PAYLOAD_KEY, None) != binding["binding_hash"]:
                raise RuntimeError("Stage Relay MODEL and CONDITIONING binding hashes differ")
            payload = kwargs.get("minimax_payload")
            if not isinstance(payload, Mapping):
                raise RuntimeError("Stage EAV requires the native H3 payload")
            if external_scope is not None:
                external_scope.verify(model, av_latent)
                external_scope.validate_payload(payload)
            if (len(x) != 2 or tuple(x[0].shape) != stage_context.video_shape
                    or tuple(x[1].shape) != stage_context.audio_shape):
                raise RuntimeError("Stage EAV actual AV layout differs from the bound stage")
            route = feta._runtime_route(
                x=x, timestep=timestep, context=context, payload=payload,
                denoise_mask=kwargs.get("denoise_mask"), audio_denoise_mask=kwargs.get("audio_denoise_mask"),
                start_progress=config.start_video_progress, end_progress=config.end_video_progress,
                allowed_tasks=("LongVideoSegment0", "LongVideoMotion") if long_video_contract is not None
                    else feta.EAV_ALL_TASKS, allow_reference_blocks=True,
                long_video_contract=long_video_contract,
                progressive_mask_contract=mask_contract,
            )
            actual_sigma = route["sigma_video"]
            from .manual_pass import allows_intermediate_evaluations
            forward_plan = expected_forward_plan(stage_context, runtime.blocks)
            discrete_match = any(math.isclose(actual_sigma, item["sigma"], rel_tol=0, abs_tol=2e-6)
                                 for item in forward_plan["forwards"])
            intermediate_match = (allows_intermediate_evaluations(stage_context)
                and stage_context.trajectory_sigmas[stage_context.end] - 2e-6 <= actual_sigma
                <= stage_context.trajectory_sigmas[stage_context.start] + 2e-6)
            unknown_clock = not forward_plan["known"] and math.isfinite(actual_sigma) and 0 <= actual_sigma <= 1
            if not discrete_match and not intermediate_match and not unknown_clock:
                raise RuntimeError("Stage EAV received a forward outside its absolute sigma window")
            runtime.observed_sigmas.append(actual_sigma)
            index = runtime.telemetry.begin_forward(sigma_video=actual_sigma,
                progress_video=route["progress_video"], route=route)
            route.update(mode=config.mode, tau=config.tau, max_workspace_mib=config.max_workspace_mib,
                         g_hard_limit=config.g_hard_limit, runtime=runtime.telemetry, forward_index=index)
            options[feta.EAV_RUNTIME_KEY] = route
            if binding is not None:
                relay_route = relay._runtime_route(payload["layout"], binding, x[0].device)
                expected_relay_task = native_relay_task if long_video_contract is not None else route["task"].lower()
                if expected_relay_task != binding["task"].lower():
                    raise RuntimeError("Stage Relay task differs from actual conditioning")
                options[relay.PROMPT_RELAY_RUNTIME_KEY] = relay_route
                from ..relay_kj_memory import bind_memory_runtime
                bind_memory_runtime(relay_contract.get("attention_backend"), relay_route)
            result = executor(x, timestep, context, options, **kwargs)
            runtime.completed_forwards += 1
            return result
        except BaseException as exc:
            runtime.telemetry.abort(exc)
            raise
        finally:
            options.pop(feta.EAV_RUNTIME_KEY, None)
            options.pop(relay.PROMPT_RELAY_RUNTIME_KEY, None)

    def stage_attention(func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        route = options.get(feta.EAV_RUNTIME_KEY)
        output = previous(func, q, k, v, heads, *args, **kwargs) if previous is not None else func(q, k, v, heads, *args, **kwargs)
        if route is None or q.shape[-2] != route["seq_len"]:
            return output
        if args or not kwargs.get("skip_reshape", False) or kwargs.get("skip_output_reshape", False):
            raise RuntimeError("Stage EAV requires native packed H3 attention layout")
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads:
            raise RuntimeError("Stage EAV requires batch-1 native packed Q/K")
        if kwargs.get("mask") is not None:
            warn_patch_stack("Stage EAV keeps the attention mask; FETA Q/K statistics do not include that mask")
        runtime.selector_calls += 1
        if binding is not None and relay_contract["attention_owner_verified"]:
            runtime.relay_calls += 1
        return feta._apply_eav_output_gain(q, k, output, route)

    patched.add_wrapper_with_key(extension.WrappersMP.DIFFUSION_MODEL, KEY, stage_wrapper)
    patched.model_options["transformer_options"]["optimized_attention_override"] = stage_attention
    patched.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    patched.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    patched.set_attachments(KEY, runtime)
    from . import vdn_stages, vdn_effects
    if stage_context.recipe == vdn_stages.RECIPE:
        vdn_effects.install(patched, runtime)
    if v2_owner.runtime is not None and v2_owner.profile != "dense_compat_exp":
        from .sparse_eav import install_sparse_eav
        install_sparse_eav(patched, v2_owner.runtime, runtime)
    report = runtime.snapshot()
    report["mask_contract"] = mask_contract.report() if mask_contract is not None else None
    report["relay_binding_hash"] = binding["binding_hash"] if binding else None
    if long_video_contract is not None:
        report["long_video_contract"] = long_video_contract
    if external_scope is not None:
        report['external_motion_scope_sha256'] = external_scope.verify(model, av_latent)['sha256']
    report["source_model_unchanged"] = True
    return patched, runtime, json.dumps(report, ensure_ascii=False, indent=2)


def audit_stage_eav(av_latent, runtime):
    if type(runtime) is not StageEAVRuntime:
        raise TypeError("Stage audit requires the runtime from the matching Stage EAV Apply")
    video, audio = nested_av_parts(av_latent)
    if tuple(video.shape) != runtime.context.video_shape or tuple(audio.shape) != runtime.context.audio_shape:
        raise ValueError("Stage EAV audit latent shape differs from its runtime")
    report = runtime.snapshot()
    if report["status"] == "aborted":
        raise RuntimeError("Stage EAV aborted: " + report["feta"]["aborted"])
    return av_latent, json.dumps(report, ensure_ascii=False, indent=2)
