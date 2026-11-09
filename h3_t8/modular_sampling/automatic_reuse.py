"""Transparent reuse at an existing sampler boundary; never invent a recipe.

Only cache preparation can fall back. Once the selected sampler starts, errors
propagate and it is never retried behind the user's back.
"""
from copy import copy
import json

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from ..patch_stack_policy import UnverifiedModelStack
from . import native_explicit, results, reuse


def _execute_native(noise, guider, sampler, sigmas, latent_image, status, reason=None):
    output, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image).result
    report = dict(schema="t8.modular-sampling.reuse.v1", status=status,
                  bypass_reason=reason, current_denoiser_evaluations=None,
                  artifact_path=None, artifact_sha256=None)
    return output, denoised, None, results.canonical(report)


def creator_selection(workspace, run_position, variant_index):
    from ..creator_workspace_advanced import select_creator_workspace_shot
    selected = select_creator_workspace_shot(workspace, run_position, variant_index)
    shot = json.loads(selected[5])
    # Retention, display position and unused variants do not affect sampling.
    shot = {key: shot[key] for key in ("shot_id", "frame_count", "effective_compiled_prompt",
                                     "negative_prompt", "media_roles", "source_packet_hash")}
    return {"shot": shot, "variant_index": int(variant_index), "seed": selected[4]}


def sample(noise, guider, sampler, sigmas, latent_image, storage_root, *,
           mode="disabled", stage_context=None, stage_role="native_low",
           creator_workspace=None, run_position=0, variant_index=0):
    if mode not in reuse.MODES:
        raise ValueError("Unknown stage reuse mode")
    if mode == "disabled" and stage_context is None:
        return _execute_native(noise, guider, sampler, sigmas, latent_image, "disabled_executed")

    source = latent_image
    if creator_workspace is not None:
        selection = creator_selection(creator_workspace, run_position, variant_index)
        if int(noise.seed) != selection["seed"]:
            raise ValueError("Creator variant seed differs from the connected NOISE seed")
        # Per-shot contents, not the whole workspace hash: unrelated edits do
        # not evict this stage. Actual conditions/MODEL/tensors remain checked.
        source = {**latent_image, "t8_creator_selection": selection}
    if stage_context is not None:
        return reuse.sample_or_reuse(noise, guider, sampler, sigmas, source,
                                     stage_context, storage_root, mode)
    try:
        if not results.guider_is_known(guider):
            raise UnverifiedModelStack("Connected guider has no transparent cache adapter")
        bound, _, _, context, _ = native_explicit.bind_stage(
            guider.model_patcher, sampler, sigmas, source, stage_role)
        prepared = copy(guider)
        prepared.model_patcher = bound
        prepared.model_options = bound.model_options
        # Authenticate before entering the sampling call. Unknown stacks are
        # delegated using the original objects; genuine sampler errors remain.
        identity = reuse.request_identity(noise, prepared, sampler, sigmas, source, context)
    except (UnverifiedModelStack, ValueError, TypeError, AttributeError) as error:
        return _execute_native(noise, guider, sampler, sigmas, latent_image,
                               "uncertified_executed", str(error))
    return reuse.sample_or_reuse(noise, prepared, sampler, sigmas, source,
                                 context, storage_root, mode, _prepared_identity=identity)


def ui_report(raw):
    report = json.loads(raw)
    return {key: report.get(key) for key in (
        "status", "request_sha256", "current_denoiser_evaluations",
        "artifact_path", "artifact_sha256", "bypass_reason")}
