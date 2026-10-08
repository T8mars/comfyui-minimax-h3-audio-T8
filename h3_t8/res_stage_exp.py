"""One actual RES invocation -> completed frozen Stage evidence, never x0 history.

This adapter does not change the legacy Stage sampler, RES solver, MODEL or
history checkpoint format. A scoped result identity is not universal MODEL or
CUDA qualification. Unknown MODEL owners still run but cannot authorize reuse.
"""
from copy import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import CodeType

import comfy.samplers
import torch
from comfy_extras.nodes_custom_sampler import Guider_Basic, SamplerCustomAdvanced

from . import res_history_setup as setup
from . import res_history_exp as history
from .core import nested_av_parts
from .modular_sampling.contracts import StageContext
from .modular_sampling.core_sampler_identity import native_basic_guider_class
from .modular_sampling.noise import operator_identity
from .modular_sampling.results import StageResult, _input_identity, canonical, sha
from .patch_stack_policy import model_identity_matches
from .res_compiled_assets import freeze_compiled_assets, validate_bound_compiled_assets
from .res_history_exp import source_contract
from .res_recovery_identity import history_recovery_allowed
from .source_code_identity import executable_code_equal
from .vdn_attention_compat import _factory_closure

RECIPE = "h05_RES_history_complete_v1"
STAGES = ("res_complete", "res_remaining_complete")


def implementation_identity():
    compiled = compile(Path(setup.__file__).read_text(encoding="utf8"),
                       setup.setup_res_history_sampling.__code__.co_filename, "exec", dont_inherit=True)
    factory = next(code for code in compiled.co_consts if type(code) is CodeType
                   and code.co_name == "setup_res_history_sampling")
    if not executable_code_equal(factory, setup.setup_res_history_sampling.__code__):
        raise ValueError("RES Stage actual setup executable differs from its source")
    compiled_history = compile(Path(history.__file__).read_text(encoding="utf8"),
        inspect.unwrap(history.sample_res_history).__code__.co_filename, "exec", dont_inherit=True)
    history_code = next(code for code in compiled_history.co_consts if type(code) is CodeType
                        and code.co_name == "sample_res_history")
    if (setup.sample_res_history is not history.sample_res_history
            or not executable_code_equal(history_code, inspect.unwrap(history.sample_res_history).__code__)):
        raise ValueError("RES Stage actual history executable differs from its source")
    return {"res_stage": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "res_setup": hashlib.sha256(Path(setup.__file__).read_bytes()).hexdigest(),
            "actual_Core_RES": source_contract()}


def _selection(sampler, sigmas, source):
    if (type(sampler) is not comfy.samplers.KSAMPLER
            or set(vars(sampler)) != {"sampler_function", "extra_options", "inpaint_options"}
            or sampler.extra_options != {} or sampler.inpaint_options != {}):
        raise ValueError("RES Stage needs the unchanged RES History sampler, not another recipe")
    state = _factory_closure(sampler.sampler_function, setup.setup_res_history_sampling, "sampler_function")
    if state is None:
        raise ValueError("RES Stage sampler lacks its actual native setup factory")
    full, offset = state["full"], state["offset"]
    if not torch.equal(sigmas.detach().cpu(), full[offset:].detach().cpu()):
        raise ValueError("RES Stage selected remaining schedule differs from its complete trajectory")
    video, audio = nested_av_parts(source)
    if (video.shape[0] != 1 or audio.shape[0] != 1
            or video[0].numel() + audio[0].numel() != state["packed_values"]):
        raise ValueError("RES Stage source differs from the sampler's packed AV coordinates")
    declaration = state["declaration"]
    # disabled setup has no declaration. AV shifts are then obtained from the
    # selected guider MODEL, not guessed here; _context supplies those values.
    selection = dict(mode=state["mode"], offset=offset, full=_input_identity(full),
        selected_sigmas=_input_identity(sigmas), declaration=declaration,
        checkpoint_step=state["checkpoint_step"], checkpoint_path=state["relative"],
        saved_boundary_sha256=state["saved"]["file_sha256"] if state["saved"] else None,
        packed_values=state["packed_values"], chunk_bytes=state["chunk_bytes"],
        sampler_function_identity="actual_setup_factory", source=implementation_identity())
    return state, selection


def _context(state, model, source):
    video, audio = nested_av_parts(source)
    sampling = model.get_model_object("model_sampling")
    from .long_video_dual_identity import _v2_sampling_identity
    coordinates = _v2_sampling_identity(sampling)
    return StageContext(RECIPE, STAGES[bool(state["offset"])], "deterministic_eta0_cfgpp_false",
        state["offset"], len(state["full"]) - 1,
        tuple(float(value) for value in state["full"]),
        float(coordinates["configuration"]["shift"]), float(coordinates["configuration"]["audio_shift"]),
        tuple(video.shape), tuple(audio.shape), "original_AV_input_and_optional_RES_history",
        "terminal_native_AV_output", "terminal_native_AV_denoised_output")


def verify_receipt(receipt):
    """Check the RES-specific completion assertions when reading typed data."""
    request = receipt["request"]
    if request["stage_context"]["recipe"] != RECIPE:
        return
    context = StageContext.from_dict(request["stage_context"])
    execution = receipt["execution"]
    selection = request["res_selection"]
    if (context.stage not in STAGES or selection["offset"] != context.start
            or context.stage != STAGES[bool(context.start)]
            or context.end != len(context.trajectory_sigmas) - 1
            or context.trajectory_sigmas[-1] != 0.):
        raise ValueError("RES completed Stage descriptor differs from its actual interval")
    complete = (execution["callbacks"] == list(range(context.steps))
                and execution["global_post_denoiser_callbacks"] == list(range(context.start, context.end))
                and execution["denoiser_evaluations"] == context.steps
                and execution["native_sampler_returns"] == 1
                and execution["sampler_and_guider_verified"] is True)
    if receipt["verified_recipe_completion"] is not complete:
        raise ValueError("RES completion differs from its literal NFE/callback observations")
    qualification = request["res_result_qualification"]
    actual = request["res_runtime"]
    if (request["model"] != actual["actual_loaded_model"]
            or request["conditions"] != actual["actual_conditioning"]
            or request["actual_processed_position"] != actual["actual_processed_position_condition"]
            or request["seed"] != actual["seed"]):
        raise ValueError("RES frozen result lost its actual bound MODEL/condition/runtime identity")
    from .res_compiled_assets import validate_compiled_assets_data
    validate_compiled_assets_data(request["compiled_assets"])
    qualified = (request["noise_operator"]["portable"]
        and actual["actual_conditioning"]["portable_condition_identity"]
        and history_recovery_allowed(actual["actual_loaded_model"], actual["actual_processed_position_condition"],
                                     request["compiled_assets"], request["compiled_assets_check"]))
    if (qualification["generic_MODEL_portability_granted"] is not False
            or qualification["CUDA_numerical_qualification_granted"] is not False
            or qualification["frozen_result_reuse_verified"] != bool(qualified)
            or receipt["portable_identity"] != bool(complete and qualification["frozen_result_reuse_verified"])):
        raise ValueError("RES frozen result scope/identity differs from its completion evidence")


def sample_res_stage(noise, guider, sampler, sigmas, latent_image):
    model = guider.model_patcher
    state, selection = _selection(sampler, sigmas, latent_image)
    context = _context(state, model, latent_image)
    source_before = _input_identity(latent_image)
    noise_before = operator_identity(noise)
    implementation = implementation_identity()
    observations = {"callbacks": [], "global_post_denoiser_callbacks": [],
                    "denoiser_evaluations": 0, "native_sampler_returns": 0,
                    "sampler_and_guider_verified": False, "noise_identity": None}
    runtime = {}
    observed_sampler = copy(sampler)

    def observed_function(actual_model, x, actual_sigmas, extra_args=None, callback=None, disable=None):
        assets_before, assets_after = {}, {}
        runtime["before"] = setup._runtime_contract(actual_model, dict(extra_args or {}),
            state["declaration"], state["chunk_bytes"], observed_artifacts=assets_before)
        def completed_denoiser(item):
            # The authenticated RES body emits this immediately after its one
            # real model return per global interval. Keep the actual Core
            # model wrapper/type unchanged, including for legacy boundaries.
            if not isinstance(item["denoised"], torch.Tensor):
                raise ValueError("RES completed call has no actual denoised tensor")
            observations["denoiser_evaluations"] += 1
            observations["global_post_denoiser_callbacks"].append(int(item["global_i"]))
            if callback is not None:
                return callback(item)

        value = sampler.sampler_function(actual_model, x, actual_sigmas,
            extra_args=extra_args, callback=completed_denoiser, disable=disable)
        runtime["after"] = setup._runtime_contract(actual_model, dict(extra_args or {}),
            state["declaration"], state["chunk_bytes"], observed_artifacts=assets_after)
        runtime["compiled_assets"] = freeze_compiled_assets(assets_after)
        runtime["compiled_assets_check"] = validate_bound_compiled_assets(runtime["compiled_assets"], assets_after)
        observations["native_sampler_returns"] += 1
        return value

    observed_sampler.sampler_function = observed_function

    class ObservedGuider:
        model_patcher = model

        def sample(self, actual_noise, *args, callback=None, **kwargs):
            observations["noise_identity"] = _input_identity(actual_noise)

            def record(*items, **options):
                index = items[0] if items else options.get("step")
                observations["callbacks"].append(int(index))
                if callback is not None:
                    return callback(*items, **options)

            return guider.sample(actual_noise, *args, callback=record, **kwargs)

    output, denoised = SamplerCustomAdvanced.execute(noise, ObservedGuider(), observed_sampler,
                                                    sigmas, latent_image).result
    if source_before != _input_identity(latent_image):
        raise ValueError("RES Stage source input changed during sampling")
    if _selection(sampler, sigmas, latent_image)[1] != selection or implementation_identity() != implementation:
        raise ValueError("RES Stage controls/implementation changed during sampling")
    if operator_identity(noise) != noise_before:
        raise ValueError("RES Stage noise provider changed during sampling")
    if not model_identity_matches(runtime["before"], runtime["after"]):
        raise ValueError("RES Stage actual MODEL/condition/runtime identity changed during sampling")
    known_guider = (type(guider) in (Guider_Basic, native_basic_guider_class())
        and not any(callable(value) and not (key == "inner_model" and value is model.model)
                    for key, value in vars(guider).items()))
    observations["sampler_and_guider_verified"] = known_guider
    complete = (known_guider and observations["native_sampler_returns"] == 1
        and observations["callbacks"] == list(range(context.steps))
        and observations["global_post_denoiser_callbacks"] == list(range(context.start, context.end))
        and observations["denoiser_evaluations"] == context.steps)
    actual = runtime["after"]
    qualified = (noise_before["portable"] and actual["actual_conditioning"]["portable_condition_identity"]
        and history_recovery_allowed(actual["actual_loaded_model"], actual["actual_processed_position_condition"],
                                     runtime["compiled_assets"], runtime["compiled_assets_check"]))
    qualification = dict(frozen_result_reuse_verified=bool(qualified),
        generic_MODEL_portability_granted=False, CUDA_numerical_qualification_granted=False,
        scope="explicit completed deterministic RES result; not a solver-history or LOW/HIGH conversion")
    request = dict(stage_context=context.to_dict(), source=source_before,
        model=actual["actual_loaded_model"], conditions=actual["actual_conditioning"],
        actual_processed_position=actual["actual_processed_position_condition"],
        noise=observations["noise_identity"], noise_operator=noise_before, seed=int(noise.seed),
        cfg=getattr(guider, "cfg", None), sigmas=_input_identity(sigmas), implementation=implementation,
        res_selection=selection, res_runtime=actual,
        compiled_assets=runtime["compiled_assets"], compiled_assets_check=runtime["compiled_assets_check"],
        res_result_qualification=qualification)
    receipt = dict(schema="t8.modular-sampling.result.v1", request=request, request_sha256=sha(request),
        outputs=dict(output=_input_identity(output), denoised_output=_input_identity(denoised)),
        execution=observations, verified_recipe_completion=bool(complete),
        portable_identity=bool(complete and qualified), cache_reuse_authorized=False,
        boundary="One actual native Core RES sampler return; both complete AV outputs stay distinct. "
                 "Explicit frozen result only, not universal MODEL/CUDA or human quality qualification.")
    receipt["receipt_sha256"] = sha(receipt)
    result = StageResult(output, denoised, canonical(receipt))
    result.verify()
    return output, denoised, result, context, json.dumps(receipt, ensure_ascii=False, indent=2)
