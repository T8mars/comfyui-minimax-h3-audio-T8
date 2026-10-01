"""Original full8/partial4/fresh4 samplers with explicit per-phase ownership.

This is not the continuous x_sigma route. Core still initializes fresh NOISE
and the original HyperFlow sampler retains its native audio-start rebase.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import json
import math
import threading
from types import MethodType

import comfy.patcher_extension
import comfy.samplers
import torch

from .. import hyperflow_runtime_advanced as runtime, hyperflow_sampling_advanced as native
from .. import prompt_relay_advanced as relay, sampling
from ..long_video_dual_identity import _v2_sampling_identity
from ..patch_stack_policy import warn_patch_stack
from ..progressive_relay import strip_paired_conditioning
from ..vdn_attention_compat import _factory_closure
from .contracts import StageContext
from .results import canonical, sha

KEY = "t8_modular_hyperflow_fresh_stage_v1"
RECIPES = ("hyperflow.fresh_full8plus4.v1", "hyperflow.fresh_partial4plus4.v1")
STAGES = ("hyperflow_low_full8", "hyperflow_low_partial4", "hyperflow_high_after_full8", "hyperflow_high_after_partial4")
SPECS = {STAGES[0]: (RECIPES[0], 0, 8), STAGES[1]: (RECIPES[1], 0, 4),
         STAGES[2]: (RECIPES[0], 4, 8), STAGES[3]: (RECIPES[1], 4, 8)}
SCHEMA = "t8.modular-sampling.hyperflow-fresh-execution.v1"


def install(model, weights):
    """Original two-time math with an explicitly unpatched endpoint snapshot.

    A late graph branch may be loaded while its sibling's weights are resident.
    Recover the four original time tensors from Core backups without unpatching
    or unloading the shared model. Legacy Loader implementation is untouched.
    """
    from ..long_video_dual_identity import _original_state
    original = _original_state(model, model.model_state_dict())
    snapshot = tuple(original[f"diffusion_model.time_embedder.{part}.{kind}"].detach()
                     .to(device="cpu", dtype=torch.float32).clone()
                     for part in ("proj_in", "proj_out") for kind in ("weight", "bias"))
    if not all(bool(torch.isfinite(value).all()) for value in snapshot):
        raise ValueError("HyperFlow fresh loader original time snapshot is non-finite")
    selected, binding, report = runtime.install_hyperflow(model, weights)
    # Replace only this just-created MODEL's own descriptors. No installed
    # live method is touched; content LoRA order and selected backends remain.
    selected.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                     "t8_hyperflow_" + binding.owner)
    runtime._install_two_time(selected, selected.get_model_object("diffusion_model"), weights, snapshot, binding.owner)
    report = {**report, "stage_loader": "fresh_readonly_original_endpoint_v1",
              "endpoint_snapshot": "original_base_via_core_backups", "shared_model_unloaded": False}
    return selected, binding, report


@dataclass(frozen=True)
class FreshStageOwner:
    context: StageContext
    plan: native.HyperFlowPlan
    binding: runtime.HyperFlowBinding
    sampling_object: object
    sampling_json: str
    audio_velocity_is_raw: bool
    relay_binding_json: str | None
    relay_counts: dict | None
    relay_wrapper: object | None
    frozen: str
    lock: object = field(default_factory=threading.Lock, compare=False)

    @property
    def runtime(self):
        return None

    @property
    def profile(self):
        return self.context.profile

    def descriptor(self):
        return {"context": self.context.to_dict(), "plan": self.plan.as_report(),
                "model_sampling": json.loads(self.sampling_json), "audio_velocity_is_raw": self.audio_velocity_is_raw,
                "relay_binding_json": self.relay_binding_json}


def capture_owner(model):
    owner = model.get_attachment(KEY)
    if type(owner) is not FreshStageOwner or set(vars(owner)) != set(FreshStageOwner.__dataclass_fields__):
        raise ValueError("Use the dedicated HyperFlow fresh stage setup")
    if canonical(owner.descriptor()) != owner.frozen or type(owner.plan) is not native.HyperFlowPlan:
        raise ValueError("HyperFlow fresh stage descriptor changed")
    spec = SPECS.get(owner.context.stage)
    if spec != (owner.context.recipe, owner.context.start, owner.context.end):
        raise ValueError("HyperFlow fresh stage has another route or interval")
    if (model.get_attachment(runtime.ATTACHMENT_KEY) != owner.binding
            or owner.plan != native.build_hyperflow_plan(model, spec[1], spec[2])
            or owner.context.profile != canonical(owner.plan.as_report())
            or owner.context.trajectory_sigmas != owner.plan.video_sigmas
            or (owner.context.video_shift, owner.context.audio_shift) != (12., 3.)):
        raise ValueError("HyperFlow fresh stage differs from its real two-time owner/plan")
    if (model.object_patches.get("model_sampling") is not owner.sampling_object
            or canonical(_v2_sampling_identity(owner.sampling_object)) != owner.sampling_json
            or sampling.model_uses_raw_audio_velocity(model) != owner.audio_velocity_is_raw):
        raise ValueError("HyperFlow fresh stage native AV sampling object changed")
    options = model.model_options["transformer_options"]
    if (options.get("minimax_h3_sigma_shift_video"), options.get("minimax_h3_sigma_shift_audio")) != (12., 3.):
        raise ValueError("HyperFlow fresh stage AV clocks changed")
    if owner.relay_binding_json is not None:
        state = _factory_closure(owner.relay_wrapper, relay._install_prompt_relay_model, "_diffusion_wrapper")
        if (state is None or state.get("execution_counts") is not owner.relay_counts
                or canonical(state.get("binding")) != owner.relay_binding_json):
            raise ValueError("HyperFlow fresh Relay counter/binding owner changed")
    return owner


def build_stage(model, source, stage=STAGES[0]):
    from . import eav, hyperflow_effects
    if stage not in SPECS:
        raise ValueError("Unknown HyperFlow fresh stage")
    if any(model.get_attachment(key) is not None for key in (KEY, hyperflow_effects.KEY, eav.KEY)):
        raise ValueError("Use a fresh Loader/Relay branch before stage setup and EAV")
    recipe, start, end = SPECS[stage]
    plan = native.build_hyperflow_plan(model, start, end)
    # Exact original public setup flags. No x_sigma override or capture.
    prepared, sampler, sigmas = native.setup_hyperflow_sampler(model, source, plan,
        internal_continuation=(start, end) != (0, 8), new_noise_restart=start > 0)
    video, audio = sampling.nested_av_parts(source)
    context = StageContext(recipe, stage, canonical(plan.as_report()), start, end, plan.video_sigmas,
        12., 3., tuple(video.shape), tuple(audio.shape),
        "explicit_clean_av_fresh_joint_noise" if start else "native_av_template_and_initial_noise",
        "core_nonterminal_output_not_upscale_source" if end == 4 else "completed_av",
        "core_denoised_output_prediction_x0")
    relay_json = counts = wrapper = None
    if model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is not None:
        contract = relay.prompt_relay_model_contract(model)
        wrapper = model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)[0]
        state = _factory_closure(wrapper, relay._install_prompt_relay_model, "_diffusion_wrapper")
        if state is None or type(state.get("execution_counts")) is not dict:
            raise ValueError("HyperFlow fresh Relay needs its actual counter owner")
        relay_json, counts = canonical(contract["binding"]), state["execution_counts"]
    sampling_object = prepared.object_patches["model_sampling"]
    owner = FreshStageOwner(context, plan, model.get_attachment(runtime.ATTACHMENT_KEY), sampling_object,
        canonical(_v2_sampling_identity(sampling_object)), sampling.model_uses_raw_audio_velocity(prepared),
        relay_json, counts, wrapper, "")
    object.__setattr__(owner, "frozen", canonical(owner.descriptor()))
    prepared.set_attachments(KEY, owner)
    validate_stage(prepared, sigmas, source, context)
    return prepared, sampler, sigmas, context, canonical({"schema": SCHEMA, "stage_context": context.to_dict(),
        "sampled": False, "fresh_noise_restart": start > 0, "relay_bound": relay_json is not None,
        "boundary": "One original typed sampler only. HIGH clean input provenance is not inferred from LATENT."})


def validate_stage(model, sigmas, source, context):
    owner = capture_owner(model)
    video, audio = sampling.nested_av_parts(source)
    if context != owner.context or (tuple(video.shape), tuple(audio.shape)) != (context.video_shape, context.audio_shape):
        raise ValueError("HyperFlow fresh MODEL/context/AV geometry is not paired")
    if not all(part.dtype.is_floating_point and bool(torch.isfinite(part).all()) for part in (video, audio)):
        raise ValueError("HyperFlow fresh AV must be finite floating tensors")
    if (type(sigmas) is not torch.Tensor or sigmas.dtype != torch.float32
            or sigmas.shape != owner.plan.video_segment.shape or not torch.equal(sigmas.cpu(), owner.plan.video_segment)):
        raise ValueError("HyperFlow fresh SIGMAS differ from their original absolute grid")
    return video, audio, owner


def sampler_is_known(model, sampler, context):
    owner = capture_owner(model)
    if type(sampler) is not comfy.samplers.KSAMPLER or sampler.extra_options or sampler.inpaint_options:
        return False
    state = _factory_closure(sampler.sampler_function, native.setup_hyperflow_sampler, "sampler_function")
    expected = {"plan": owner.plan, "video_values": math.prod(context.video_shape[1:]),
        "packed_values": math.prod(context.video_shape[1:]) + math.prod(context.audio_shape[1:]),
        "raw_audio": owner.audio_velocity_is_raw, "internal_continuation": (context.start, context.end) != (0, 8),
        "new_noise_restart": context.start > 0, "endpoint_capture": None, "x_sigma_override": None}
    return state == expected


def project_owner(model, owner=None):
    from .native_dual import project_identity
    owner = capture_owner(model) if owner is None else owner
    projected, _ = project_identity(model, owner=owner, key=KEY)
    return projected, owner.descriptor()


def live_identity(model, reason):
    """Unknown stacks remain execution-local; normalize Core installation only."""
    from ..long_video_dual_identity import _original_state
    from ..patch_stack_policy import nonportable_model_identity, _execution_selection
    owner = capture_owner(model)
    state = {key: value for key, value in _original_state(model, model.model_state_dict()).items()
             if not key.startswith("model_sampling.")}
    state.update({"model_sampling." + key: value for key, value in owner.sampling_object.state_dict().items()})

    class ReadOnlySelection:
        backup = {}
        backup_buffers = {}

        def __getattr__(self, name):
            return getattr(model, name)

        def model_state_dict(self):
            return state

    identity = nonportable_model_identity(ReadOnlySelection(), reason,
        schema="t8.modular-sampling.hyperflow-fresh-live-model.v1")
    selected = identity["execution_selection"]
    network, classes = selected["network_execution"], {}
    for name, original in model.model.named_modules():
        module = owner.sampling_object if name == "model_sampling" else original
        native_forward = inspect.getattr_static(type(module), "forward")
        classes[name] = _execution_selection(native_forward)
        current = vars(module).get("forward")
        installed = model.object_patches.get(name + ".forward")
        # LOW and HIGH may be sibling clones of one Core network. Before Core
        # loads HIGH, LOW's genuine HyperFlow forward is still live; Core then
        # installs HIGH's selected object patch. Both must unwrap to the same
        # native callable and originate from the same T8 patch factory. Any
        # foreign/unmarked forward remains in the live execution identity.
        sibling_hyperflow = (callable(current) and callable(installed)
            and getattr(current, "_t8_hyperflow_owner", None) is not None
            and getattr(installed, "_t8_hyperflow_owner", None) is not None
            and getattr(current, "__code__", None) is getattr(installed, "__code__", None)
            and runtime._unpatch_own(current) == runtime._unpatch_own(installed))
        if current is None or current is installed or (type(current) is MethodType
                and current.__self__ is module and current.__func__ is native_forward) or sibling_hyperflow:
            # Exact selected callable is already in object_patches identity.
            # Its installation is not a new user selection. All hooks remain.
            if module._forward_pre_hooks or module._forward_hooks:
                network[name] = _execution_selection({"forward": None,
                    "pre_hooks": module._forward_pre_hooks, "hooks": module._forward_hooks})
            else:
                network.pop(name, None)
    selected["class_forward_dispatch"] = classes
    # Core lazily adds an empty apply_model wrapper slot while running a
    # long-video MODEL. An empty slot executes nothing; nonempty user wrappers
    # remain in this identity and must still match before/after sampling.
    selected["wrappers"] = [slot for slot in selected["wrappers"]
                            if not (len(slot) == 2 and slot[0] == "apply_model" and not slot[1])]
    protocol = {name: getattr(model.model, name, None) for name in
        ("audio_scale", "_scale_audio_slice", "process_latent_in", "process_latent_out", "extra_conds", "apply_model", "_apply_model")}
    # The selected T8 long-video extra_conds is likewise installed by Core.
    # Normalize only that proven native->selected transition; any unrelated
    # replacement is rejected and the chosen callable remains in the hash.
    chosen_extra = model.object_patches.get("extra_conds")
    original_extra = getattr(getattr(chosen_extra, "__func__", chosen_extra),
                             "_t8_long_video_original_extra_conds", None)
    if original_extra is not None:
        live_extra = protocol["extra_conds"]
        live_function = getattr(live_extra, "__func__", live_extra)
        chosen_function = getattr(chosen_extra, "__func__", chosen_extra)
        sibling_long_video = (getattr(chosen_function, "_t8_long_video_patch_version", None) is not None
            and getattr(live_function, "_t8_long_video_patch_version", None)
                == getattr(chosen_function, "_t8_long_video_patch_version", None)
            and getattr(live_function, "_t8_long_video_original_extra_conds", None) == original_extra
            and getattr(live_function, "__code__", None) is getattr(chosen_function, "__code__", None)
            and getattr(live_extra, "__self__", None) is getattr(chosen_extra, "__self__", None))
        if live_extra not in (original_extra, chosen_extra) and not sibling_long_video:
            raise ValueError("HyperFlow fresh long-video extra_conds owner changed")
        protocol["extra_conds"] = chosen_extra
    selected["base_protocol"] = _execution_selection(protocol)
    selected["latent_coordinates"] = _execution_selection({
        "attributes": vars(model.model.latent_format), "scale_factor": model.model.latent_format.scale_factor,
        "in": model.model.latent_format.process_in, "out": model.model.latent_format.process_out})
    selected["fresh_stage"] = owner.descriptor()
    identity["execution_selection"] = {"selection_sha256": sha(selected)}
    return identity


def _conditions(guider, owner):
    pairs = {key: [[entry.get("cross_attn"), entry] for entry in entries]
             for key, entries in guider.original_conds.items()}
    if owner.relay_binding_json is not None:
        binding = json.loads(owner.relay_binding_json)
        contract = {"binding": binding, "binding_hash": binding["binding_hash"]}
        strip_paired_conditioning(pairs.get("positive", []), contract, required=True)
        strip_paired_conditioning(pairs.get("negative", []), contract, required=False)
    elif any(relay.PROMPT_RELAY_BINDING_KEY in metadata
             or relay.PROMPT_RELAY_PAYLOAD_KEY in metadata.get("model_conds", {})
             for entries in pairs.values() for _, metadata in entries):
        raise ValueError("HyperFlow fresh Relay CONDITIONING is not paired with this MODEL")


@contextmanager
def execution(model, guider, context):
    """Observe the real Core invocation; preserve its sampler and both outputs."""
    if context.recipe not in RECIPES:
        yield None
        return
    from . import eav, hyperflow_effects
    owner = capture_owner(model)
    _conditions(guider, owner)
    telemetry = model.get_attachment(eav.KEY)
    if telemetry is not None and (type(telemetry) is not eav.StageEAVRuntime or telemetry.context != context):
        raise ValueError("HyperFlow fresh EAV belongs to another stage")
    if (telemetry is not None or owner.relay_binding_json is not None) and getattr(guider, "cfg", None) != 1.:
        raise ValueError("HyperFlow fresh stage effects require CFG1")
    locks = [owner.lock] + ([hyperflow_effects._relay_lock(owner.relay_wrapper)] if owner.relay_wrapper is not None else [])
    acquired, added = [], False
    report = {"schema": SCHEMA, "absolute_apply_intervals": [], "composition_verified": True,
              "stage_context": context.to_dict(), "fresh_noise_restart": context.start > 0,
              "continuous_x_sigma_override": False, "quality_accepted": False}
    key, kind = KEY + ".observer", comfy.patcher_extension.WrappersMP.APPLY_MODEL
    def observe(executor, *args, **kwargs):
        step = runtime.active_step()
        result = executor(*args, **kwargs)
        if step is not None and step.owner == owner.plan.owner:
            report["absolute_apply_intervals"].append(step.absolute_index)
        return result
    try:
        for lock in locks:
            if not lock.acquire(blocking=False):
                raise RuntimeError("HyperFlow fresh stage owner is busy; use independent branches")
            acquired.append(lock)
        if model.get_wrappers(kind, key):
            raise RuntimeError("HyperFlow fresh observer is already active")
        model.add_wrapper_with_key(kind, key, observe)
        added = True
        before = dict(owner.relay_counts or {})
        config = None
        if telemetry is not None:
            config = canonical(telemetry.snapshot()["config"])
            telemetry.closed = True
            telemetry.prepare(None, None, None)
        yield report
        capture_owner(model)
        # Core may restore native methods by clearing backups, but an active
        # selected patch must not have been replaced by a sibling or user hook.
        # This is actual in-run mutation detection, not an admission restriction.
        for path in model.object_patches_backup:
            if path.endswith(".forward") and path in model.object_patches:
                module = model.model.get_submodule(path.removesuffix(".forward"))
                if module.forward is not model.object_patches[path]:
                    raise ValueError("HyperFlow fresh installed forward changed during execution")
        if telemetry is not None:
            current = telemetry.snapshot()
            if canonical(current["config"]) != config:
                raise ValueError("HyperFlow fresh EAV configuration changed during execution")
            report["eav"] = deepcopy(current)
            report["composition_verified"] &= current["status"] in {
                "observed_report_only", "observed_apply_exp", "observed_no_steps_in_effect_window"}
        if owner.relay_counts is not None:
            counts = {key: value - before.get(key, 0) for key, value in owner.relay_counts.items()}
            forwards = telemetry.completed_forwards if telemetry is not None and telemetry.relay_required else counts["completed_forwards"]
            wanted = context.steps * len(model.model.diffusion_model.blocks)
            verified = forwards == context.steps and counts["routed_attention_calls"] == wanted
            report["relay"] = {"binding": json.loads(owner.relay_binding_json), "completed_forwards": forwards,
                "routed_attention_calls": counts["routed_attention_calls"], "planned_attention_calls": wanted,
                "composition_verified": verified}
            report["composition_verified"] &= verified
        if not report["composition_verified"]:
            warn_patch_stack("HyperFlow fresh stage actual effect coverage is incomplete")
    except BaseException as error:
        if telemetry is not None and acquired:
            telemetry.telemetry.abort(error)
        raise
    finally:
        if added:
            model.remove_wrappers_with_key(kind, key)
        for lock in reversed(acquired):
            lock.release()


def verify_receipt(result, receipt):
    context = StageContext.from_dict(receipt["request"]["stage_context"])
    if context.recipe not in RECIPES:
        return
    if SPECS.get(context.stage) != (context.recipe, context.start, context.end):
        raise ValueError("HyperFlow fresh result has another route/interval")
    actual = receipt["execution"].get("hyperflow_fresh")
    if (not isinstance(actual, dict) or actual.get("schema") != SCHEMA
            or actual.get("stage_context") != context.to_dict()
            or actual.get("fresh_noise_restart") != (context.start > 0)
            or actual.get("continuous_x_sigma_override") is not False):
        raise ValueError("HyperFlow fresh result is missing its actual phase execution")
    complete = (actual.get("sampler_verified") is True and actual.get("guider_verified") is True
        and set(actual["absolute_apply_intervals"]) == set(range(context.start, context.end))
        and receipt["execution"]["callbacks"] == list(range(context.steps))
        and receipt["execution"]["denoiser_evaluations"] == context.steps)
    if receipt["verified_recipe_completion"] != complete:
        raise ValueError("HyperFlow fresh completion differs from actual original sampler execution")
    effects_ok = True
    if "eav" in actual:
        value = actual["eav"]
        effects_ok &= (value["status"] in {"observed_report_only", "observed_apply_exp", "observed_no_steps_in_effect_window"}
            and value["stage_context"] == context.to_dict() and value["completed_forwards"] == context.steps
            and value["selector_calls"] + value["sparse_producer_calls"] >= context.steps * 50
            and value["clock_match"] is True)
    if "relay" in actual:
        value = actual["relay"]
        relay_ok = (value["completed_forwards"] == context.steps
            and value["planned_attention_calls"] == context.steps * 50
            and value["routed_attention_calls"] == context.steps * 50)
        if value["composition_verified"] != relay_ok:
            raise ValueError("HyperFlow fresh Relay coverage differs from actual counts")
        effects_ok &= relay_ok
    if actual.get("composition_verified") != effects_ok:
        raise ValueError("HyperFlow fresh effect coverage differs from actual counts")
    request = receipt["request"]
    identity_ok = (request["model"].get("portable_cache_reuse", True)
                   and request["noise_operator"]["portable"] and "execution_only" not in request["conditions"])
    if receipt["portable_identity"] and (not complete or not effects_ok or not identity_ok):
        raise ValueError("HyperFlow fresh unverified effects cannot claim portable completion")
    for latent in (result.output, result.denoised_output):
        if not all(bool(torch.isfinite(part).all()) for part in sampling.nested_av_parts(latent)):
            raise ValueError("HyperFlow fresh result is non-finite")


def lift_input(result):
    from .results import StageResult
    if type(result) is not StageResult:
        raise ValueError("Connect the real HyperFlow fresh LOW Stage Result")
    receipt = result.verify()
    stage = receipt["request"]["stage_context"]["stage"]
    if stage not in STAGES[:2] or not receipt["verified_recipe_completion"]:
        raise ValueError("Only a completed HyperFlow full8/partial4 LOW can supply the lift input")
    output = result.output if stage == STAGES[0] else result.denoised_output
    return output, canonical({"schema": SCHEMA, "source_receipt_sha256": receipt["receipt_sha256"],
        "selected_socket": "output" if stage == STAGES[0] else "denoised_output",
        "sampling_calls": 0, "upscale_calls": 0, "source_identity": sha(receipt["outputs"])})
