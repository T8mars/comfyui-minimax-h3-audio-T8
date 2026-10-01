"""Bind the current V2 execution to its direct accepted parent, if any.

The source node's previous_job_sha256 is a user-selected parent claim. This
gate compares it with the current two-stage recipe only after both real stages
and their handoff have been attested. It never writes or accepts a candidate.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .continuation import ContinuationContexts
from .fast_h3_v2_current_handoff import FastH3V2CurrentHandoff
from .fast_h3_v2_job import FastH3V2CurrentRecipe
from .results import canonical, implementation_identity


SCHEMA = "t8.modular-sampling.fasth3-v2-current-job-binding.v1"


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _parent(recipe, segment_index, handoff, contexts):
    if segment_index == 0:
        if contexts is not None or handoff["accepted_source_sha256"] is not None:
            raise ValueError("First V2 segment cannot declare an accepted parent")
        return None
    if type(contexts) is not ContinuationContexts:
        raise ValueError("Continuation job binding requires accepted-parent contexts")
    descriptor = contexts.verify()
    source = contexts.source.revalidate()
    request = source["request"]
    geometry = recipe["geometry"]
    expected = {"chain_id": recipe["chain_id"], "segment_index": segment_index,
                "job_sha256": recipe["sha256"],
                "context_frames": recipe["window"]["continuation_context_frames"],
                "width": geometry["width"], "height": geometry["height"],
                "low_width": geometry["low_width"], "low_height": geometry["low_height"]}
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Accepted parent belongs to another current V2 execution recipe")
    if (handoff["accepted_source_sha256"] != contexts.source.sha256
            or descriptor["source_sha256"] != contexts.source.sha256
            or source["accepted"]["timeline_end_frame"] != recipe["window"]["first_render_frames"]):
        raise ValueError("Current V2 handoff or timeline differs from its direct accepted parent")
    model_id = recipe["model_pass1"]["sha256"][:16] + ":" + recipe["model_pass2"]["sha256"][:16]
    if contexts.high["metadata"].get("model_id") != model_id:
        raise ValueError("Accepted parent model identity differs from current V2 recipe")
    return {"accepted_source_sha256": contexts.source.sha256,
            "parent_candidate_id": request["parent_candidate_id"],
            "parent_revision": request["parent_revision"],
            "accepted_video_sha256": source["accepted"]["video_sha256"],
            "accepted_context_sha256": source["accepted"]["context_sha256"],
            "timeline_start_frame": source["accepted"]["timeline_end_frame"]}


@dataclass(frozen=True)
class FastH3V2CurrentJobBinding:
    payload_json: str
    sha256: str
    recipe: FastH3V2CurrentRecipe
    handoff: FastH3V2CurrentHandoff
    contexts: ContinuationContexts | None

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical current V2 job binding")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("Current V2 job binding fingerprint changed")
        if payload.get("implementation_sha256") != implementation_sha256():
            raise ValueError("Current V2 job binding implementation changed")
        if type(self.recipe) is not FastH3V2CurrentRecipe or type(self.handoff) is not FastH3V2CurrentHandoff:
            raise ValueError("Current V2 job binding lost its typed execution witnesses")
        recipe, handoff = self.recipe.verify(), self.handoff.verify()
        if (recipe.get("implementation") != implementation_identity()
                or self.recipe.sha256 != payload["current_recipe_sha256"]
                or self.handoff.sha256 != payload["handoff_sha256"]
                or handoff["current_recipe_sha256"] != self.recipe.sha256
                or handoff["segment_index"] != payload["segment_index"]):
            raise ValueError("Current V2 job execution witnesses changed")
        checked_recipe = {**recipe, "sha256": self.recipe.sha256}
        if _parent(checked_recipe, payload["segment_index"], handoff, self.contexts) != payload["parent"]:
            raise ValueError("Current V2 accepted parent changed after job binding")
        return payload


def bind_current_job(current_recipe, handoff_attestation, segment_index, contexts=None):
    if type(current_recipe) is not FastH3V2CurrentRecipe or type(handoff_attestation) is not FastH3V2CurrentHandoff:
        raise ValueError("Current V2 job requires typed recipe and completed handoff")
    if type(segment_index) is not int or segment_index not in (0, 1):
        raise ValueError("Current V2 job supports only segment 0 or 1")
    recipe, handoff = current_recipe.verify(), handoff_attestation.verify()
    if (recipe.get("implementation") != implementation_identity()
            or handoff["current_recipe_sha256"] != current_recipe.sha256
            or handoff["segment_index"] != segment_index):
        raise ValueError("Current V2 job recipe and completed handoff differ")
    checked_recipe = {**recipe, "sha256": current_recipe.sha256}
    parent = _parent(checked_recipe, segment_index, handoff, contexts)
    model_id = recipe["model_pass1"]["sha256"][:16] + ":" + recipe["model_pass2"]["sha256"][:16]
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
               "current_recipe_sha256": current_recipe.sha256,
               "handoff_sha256": handoff_attestation.sha256,
               "chain_id": recipe["chain_id"], "segment_index": segment_index,
               "model_id": model_id, "sampling_summary": current_recipe.sha256,
               "seed": recipe["window"]["first_seed"] + segment_index,
               "parent": parent,
               "boundary": "Current two-stage execution and direct parent only; no candidate write, acceptance or composition"}
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    binding = FastH3V2CurrentJobBinding(encoded, digest, current_recipe, handoff_attestation, contexts)
    binding.verify()
    report = {"schema": SCHEMA, "status": "current_job_bound", "segment_index": segment_index,
              "job_sha256": current_recipe.sha256, "accepted_source_sha256":
              None if parent is None else parent["accepted_source_sha256"],
              "boundary": payload["boundary"]}
    return binding, current_recipe.sha256, model_id, json.dumps(report, ensure_ascii=False, sort_keys=True)
