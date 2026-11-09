"""Opt-in actual Stage consumer, using the existing immutable artifact store.

The index only locates an artifact. Its manifest, tensors, receipt and today's
full request are checked before returning it. A declaration-only Creator plan
never enters this path. Uncertified user stacks execute without persisted reuse.
"""
from __future__ import annotations

import os

from ..patch_stack_policy import UnverifiedModelStack
from . import results, storage
from .noise import operator_identity

MODES = ("disabled", "reuse_completed", "refresh")
REQUEST_FIELDS = ("stage_context", "source", "model", "conditions", "noise_operator", "seed", "cfg",
                  "sigmas", "implementation")


def request_identity(noise, guider, sampler, sigmas, latent_image, stage_context):
    model = guider.model_patcher
    results.validate_stage(model, sigmas, latent_image, stage_context)
    if not results.sampler_is_known(model, sampler, stage_context) or not results.guider_is_known(guider):
        raise UnverifiedModelStack("The selected sampler/guider is executable but has no portable adapter")
    selected = results.selected_model_identity(model)
    noise_id = operator_identity(noise)
    if selected.get("portable_cache_reuse", True) is False or not noise_id["portable"]:
        raise UnverifiedModelStack("MODEL or NOISE selection has no portable identity")
    execution = results.execution_identity(model, model.model.model_sampling)
    if execution.get("unverified_memory_forwards"):
        raise UnverifiedModelStack("User memory forward executes without persistent reuse")
    return dict(stage_context=stage_context.to_dict(), source=results._input_identity(latent_image),
                model=selected, conditions=results.request_conditions(guider.original_conds),
                noise_operator=noise_id, seed=int(noise.seed), cfg=getattr(guider, "cfg", None),
                sigmas=results._input_identity(sigmas), implementation=results.implementation_identity())


def _projection(receipt):
    return {key: receipt["request"][key] for key in REQUEST_FIELDS}


def _reply(values, status, *, path=None, digest=None, reason=None):
    output, denoised, stage_result, original_report = values
    receipt = stage_result.verify()
    report = dict(schema="t8.modular-sampling.reuse.v1", status=status,
                  request_sha256=receipt["request_sha256"],
                  current_denoiser_evaluations=0 if status == "verified_hit" else receipt["execution"]["denoiser_evaluations"],
                  source_denoiser_evaluations=receipt["execution"]["denoiser_evaluations"],
                  artifact_path=path, artifact_sha256=digest, bypass_reason=reason,
                  stage_receipt=receipt,
                  note="A cache hit returns the exact completed AV state, not a new sampling claim. "
                       "Decode/export may be retried independently; old renders remain history.")
    return output, denoised, stage_result, results.canonical(report)


def sample_or_reuse(noise, guider, sampler, sigmas, latent_image, stage_context, storage_root, mode="disabled",
                    *, _prepared_identity=None):
    args = noise, guider, sampler, sigmas, latent_image, stage_context
    if mode not in MODES:
        raise ValueError("Unknown stage reuse mode")
    if mode == "disabled":
        return _reply(results.sample_stage(*args), "disabled_executed")
    try:
        # The transparent adapter has just authenticated the same inputs.
        # Avoid hashing tens of GB twice at the same entry boundary. This is
        # a lookup hint only: hit return and miss publication independently
        # recheck the full live request below; it cannot authorize a hit.
        identity = request_identity(*args) if _prepared_identity is None else _prepared_identity
    except UnverifiedModelStack as error:
        return _reply(results.sample_stage(*args), "uncertified_executed", reason=str(error))
    key = results.sha(identity)
    root = storage._root(storage_root, create=True)
    index_dir = storage._path(root, "reuse-index/" + key)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = storage._path(root, "reuse-index/" + key + "/entry.json")
    # The OS lease serializes the same computation. A killed writer releases
    # it; a partial index or orphan tensor is never a completion record.
    with storage._lease(index_dir):
        if mode == "reuse_completed" and index.exists():
            entry = storage._json_file(index)
            if (set(entry) != {"schema", "lookup_sha256", "artifact_path", "artifact_sha256"}
                    or entry["schema"] != "t8.stage-reuse-index.v1" or entry["lookup_sha256"] != key):
                raise ValueError("Stage reuse index is corrupt; use refresh to create a new completed artifact")
            output, denoised, _, restored, report = storage.load_stage(
                root, entry["artifact_path"], entry["artifact_sha256"], stage_context.stage)
            if _projection(restored.verify()) != identity or request_identity(*args) != identity:
                raise ValueError("Completed Stage does not match the current inputs; no cache was reused")
            return _reply((output, denoised, restored, report), "verified_hit",
                          path=entry["artifact_path"], digest=entry["artifact_sha256"])
        values = results.sample_stage(*args)
        receipt = values[2].verify()
        if not receipt["portable_identity"] or not receipt["verified_recipe_completion"]:
            return _reply(values, "uncertified_executed", reason="Actual execution did not certify portable completion")
        if _projection(receipt) != identity or request_identity(*args) != identity:
            raise ValueError("Stage inputs changed while preparing the cache; no reuse index was published")
        path, digest, _ = storage.save_stage(values[2], root, "reused/" + stage_context.stage)
        entry = dict(schema="t8.stage-reuse-index.v1", lookup_sha256=key,
                     artifact_path=path, artifact_sha256=digest)
        pending = storage._path(root, "reuse-index/" + key + "/entry.partial")
        with pending.open("wb") as handle:
            handle.write(results.canonical(entry).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, index)
        return _reply(values, "refreshed" if mode == "refresh" else "miss_saved", path=path, digest=digest)
