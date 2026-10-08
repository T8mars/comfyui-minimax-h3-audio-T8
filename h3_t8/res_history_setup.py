"""User-wired deterministic RES setup; no queue, global patch, or Euler change."""
import hashlib
import json
import math
from pathlib import Path

import torch
import comfy.samplers

from .core import nested_av_parts
from .native_latent_checkpoint_advanced import _relative_parts, _sha256_file
from .nfe_resume_advanced import (
    _canonical_run_contract, _json, _runtime_signature, _validate_model_contract_id,
)
from .nfe_run_contract_advanced import _ConditioningDigester
from .patch_stack_policy import model_identity_matches, UnverifiedModelStack
from .res_layout_identity import inspect_processed_layouts
from .res_model_identity import loaded_model_identity
from .res_compiled_assets import freeze_compiled_assets, validate_bound_compiled_assets
from .res_recovery_identity import history_recovery_allowed
from .res_history_exp import (
    EXTENSION, read_checkpoint, resolve_path, sample_res_history,
    save_checkpoint, source_contract,
)
from .sampling import setup_dual_clock_sampling

MODES = ("disabled", "checkpoint", "resume")


def _conditioning_identity(model, chunk_bytes):
    guider = getattr(model, "inner_model", None)
    original = getattr(guider, "original_conds", None)
    if not isinstance(original, dict):
        raise ValueError("RES history requires the actual Core guider original_conds")
    # Core convert_cond creates fresh uuid.uuid4() per row. Only that known
    # top-level bookkeeping UUID is excluded; all semantic fields stay bound.
    projection = {name: [{key: value for key, value in row.items() if key != "uuid"}
                         for row in rows] for name, rows in original.items()}
    hasher = hashlib.sha256()
    digester = _ConditioningDigester(chunk_bytes)
    digester._digest(projection, hasher, "$.original_conds")
    return {"sha256": hasher.hexdigest(), "opaque_paths": digester.opaque_paths,
            "portable_condition_identity": not digester.opaque_paths}


def _runtime_contract(model, extra_args, declaration, chunk_bytes, *, observed_artifacts=None):
    actual_model = loaded_model_identity(model.inner_model.model_patcher, observed_artifacts=observed_artifacts)
    try:
        position_identity = inspect_processed_layouts(model.inner_model, chunk_bytes)
    except UnverifiedModelStack as error:
        position_identity = {"portable_condition_layout": False, "reason": str(error)}
    result = {**declaration, "seed": extra_args.get("seed"),
              "actual_loaded_model": actual_model,
              "loaded_weight_fingerprint_verified": actual_model["automatic_loaded_weights_verified"],
              "runtime_signature": _runtime_signature(model, extra_args),
              "actual_conditioning": _conditioning_identity(model, chunk_bytes),
              "actual_processed_position_condition": position_identity,
              "torch": str(torch.__version__), "CUDA": torch.version.cuda,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "setup_source_sha256": _sha256_file(Path(__file__))}
    if (isinstance(result["seed"], bool) or not isinstance(result["seed"], int)
            or not 0 <= result["seed"] < 2**64):
        raise ValueError("RES history needs the unchanged integer noise seed")
    return json.loads(_json(result))


def _same_tensor(actual, expected, name):
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or not torch.equal(actual.detach().cpu(), expected)):
        raise ValueError(f"RES current {name} differs from original saved sampler input")


def setup_res_history_sampling(model, av_latent, *, steps=8, shift_video=12., shift_audio=3.,
                               mode="disabled", checkpoint_step=4,
                               checkpoint_path="res_step4.h3res.safetensors",
                               model_contract_id="", run_contract_json="{}",
                               confirm_checkpoint_write=False, hash_chunk_megabytes=8,
                               storage_root):
    if mode not in MODES:
        raise ValueError("unknown RES history mode")
    if (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100
            or isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int)
            or not 1 <= checkpoint_step <= steps):
        raise ValueError("RES steps/checkpoint_step must be within the complete 1..100 schedule")
    if (not math.isfinite(shift_video) or shift_video <= 0
            or not math.isfinite(shift_audio) or shift_audio <= 0):
        raise ValueError("RES shifts must be positive finite numbers")
    if not 1 <= hash_chunk_megabytes <= 64:
        raise ValueError("RES hash_chunk_megabytes must be within 1..64")
    video, audio = nested_av_parts(dict(av_latent))
    branch, _, full = setup_dual_clock_sampling(model, dict(av_latent), steps,
        float(shift_video), float(shift_audio), "res_multistep", "native_flow")
    source_contract()
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    chunk_bytes = int(hash_chunk_megabytes) * 1024**2
    saved, declaration, relative = None, {}, ""
    if mode != "disabled":
        parts = _relative_parts(checkpoint_path, "RES checkpoint_path")
        if not parts[-1].endswith(EXTENSION):
            raise ValueError(f"RES checkpoint_path must end with {EXTENSION}")
        relative = "/".join(parts)
        canonical_run, run_sha = _canonical_run_contract(run_contract_json)
        declaration = {"model_contract_id": _validate_model_contract_id(model_contract_id),
                       "declared_run_contract": json.loads(canonical_run),
                       "declared_run_contract_sha256": run_sha,
                       "video_values": video_values, "packed_values": packed_values,
                       "video_shape": list(video.shape), "audio_shape": list(audio.shape),
                       "shift_video": float(shift_video), "shift_audio": float(shift_audio),
                       "model_identity_policy": "automatic_actual_storage_ordered_patches_and_coordinates_v1"}
        if mode == "checkpoint":
            if not confirm_checkpoint_write:
                raise ValueError("RES checkpoint mode requires explicit write confirmation")
            if Path(storage_root).exists():
                target, relative = resolve_path(storage_root, relative, require_file=False)
                if target.exists():
                    raise FileExistsError("RES boundary is create-only; choose a new filename")
        else:
            if confirm_checkpoint_write:
                raise ValueError("RES resume is read-only; do not request overwrite")
            saved = read_checkpoint(storage_root, relative, hash_chunk_bytes=chunk_bytes)
            if saved["history"].completed_steps >= steps:
                raise ValueError("RES checkpoint has no remaining steps")
            if not torch.equal(saved["tensors"]["full_sigmas"], full):
                raise ValueError("RES requested complete schedule changed")
            for key, value in declaration.items():
                if saved["payload"]["run_contract"].get(key) != value:
                    raise ValueError(f"RES declaration changed: {key}")
    offset = saved["history"].completed_steps if saved else 0
    output_sigmas = full[offset:]

    def sampler_function(model_wrap, x, sigmas, extra_args=None, callback=None, disable=None):
        args = dict(extra_args or {})
        if x.shape[-1] != packed_values:
            raise ValueError("RES current packed AV size changed")
        if mode == "disabled":
            return sample_res_history(model_wrap, x, sigmas, args, callback, disable,
                                      full_sigmas=full.to(sigmas.device))
        full_live = full.to(device=sigmas.device)
        if not torch.equal(sigmas, full_live[offset:]):
            raise ValueError("RES selected remaining sigma table changed")
        observed = {}
        contract = _runtime_contract(model_wrap, args, declaration, chunk_bytes, observed_artifacts=observed)
        original_noise = getattr(model_wrap, "noise", None)
        original_latent = getattr(model_wrap, "latent_image", None)
        for name, tensor in (("noise", original_noise), ("latent", original_latent)):
            if not isinstance(tensor, torch.Tensor) or tensor.shape != x.shape or tensor.dtype != x.dtype:
                raise ValueError(f"RES requires actual Core processed original {name}")
        mask = args.get("denoise_mask")
        history = None
        if saved:
            asset_check = None
            if "compiled_assets" in saved["payload"]:
                asset_check = validate_bound_compiled_assets(saved["payload"]["compiled_assets"], observed)
            if not history_recovery_allowed(contract["actual_loaded_model"],
                    contract["actual_processed_position_condition"],
                    saved["payload"].get("compiled_assets"), asset_check):
                raise ValueError("unknown MODEL execution has no portable RES identity; sample normally instead")
            if (contract["actual_loaded_model"].get("original_sol_structure") is not None
                    and not contract["actual_processed_position_condition"]["portable_condition_layout"]):
                raise ValueError("Sol RES resume requires the actual processed position/condition content identity")
            # Source file cannot be silently swapped between setup and execution.
            current = read_checkpoint(storage_root, relative, expected_contract=contract,
                                      hash_chunk_bytes=chunk_bytes)
            if current["file_sha256"] != saved["file_sha256"]:
                raise ValueError("RES boundary changed after setup")
            if not contract["actual_conditioning"]["portable_condition_identity"]:
                raise ValueError("opaque conditioning has no cross-run RES identity; rebuild normally")
            tensors = current["tensors"]
            _same_tensor(original_noise, tensors["original_noise"], "noise")
            _same_tensor(original_latent, tensors["original_latent_image"], "latent")
            if (mask is None) != ("denoise_mask" not in tensors):
                raise ValueError("RES current denoise mask presence changed")
            if mask is not None:
                _same_tensor(mask, tensors["denoise_mask"], "denoise mask")
            history = current["history"]
            model_wrap.sigmas = full_live
            options = dict(args.get("model_options", {}))
            transformer = dict(options.get("transformer_options", {}))
            transformer["sample_sigmas"] = full_live
            options["transformer_options"] = transformer
            args["model_options"] = options

        def post_step(state):
            if mode != "checkpoint" or state.completed_steps != checkpoint_step:
                return
            boundary_observed = {}
            now = _runtime_contract(model_wrap, args, declaration, chunk_bytes,
                                    observed_artifacts=boundary_observed)
            if contract["actual_conditioning"]["portable_condition_identity"] and not model_identity_matches(contract, now):
                raise ValueError("RES actual condition/runtime contract changed during sampling")
            save_checkpoint(storage_root, relative, state, full_live,
                original_noise=original_noise, original_latent_image=original_latent,
                denoise_mask=mask, run_contract=contract, hash_chunk_bytes=chunk_bytes,
                compiled_assets=freeze_compiled_assets(boundary_observed))

        return sample_res_history(model_wrap, x, sigmas, args, callback, disable,
            full_sigmas=full_live, resume=history, post_step=post_step)

    sampler_function.__name__ = "sample_minimax_h3_RES_history_EXPT8"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    status = {"disabled": "RES_FULL_NO_IO", "checkpoint": "RES_BOUNDARY_CREATE_ONLY",
              "resume": "RES_HISTORY_RESUME_READ_ONLY"}[mode]
    report = {"status": status, "mode": mode, "steps": steps, "remaining_steps": len(output_sigmas) - 1,
              "saved_boundary_step": saved["history"].completed_steps if saved else checkpoint_step,
              "checkpoint_path": relative, "setup_writes_files": False,
              "source_contract": source_contract(), "eta": 0., "cfg_pp": False,
              "model_sampling": "ModelSamplingAV_common_video_clock", "shift_video": shift_video,
              "shift_audio": shift_audio, "old_Euler_resume_changed": False,
              "checkpoint_write_once": mode == "checkpoint", "resume_writes_files": False,
              "weights_identity": ("not_checked_disabled_mode" if mode == "disabled" else
                                   "automatic_actual_loaded_storage_ordered_patches_and_coordinates"),
              "portable_Stage_qualified": False, "human_quality_qualified": False,
              "boundary": "post-step_x_and_old_denoised_and_old_sigma_down_not_x0_or_LOW_HIGH"}
    return branch, sampler, output_sigmas, status, relative, _json(report, indent=2)
