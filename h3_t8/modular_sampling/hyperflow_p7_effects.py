"""External per-phase Prompt Relay for authenticated HyperFlow P7 windows.

Projection is independent of LOW/HIGH so its compiled prompt can be encoded
by either phase. Application preserves that phase's native motion guides and
binds the same Relay descriptor to both MODEL and CONDITIONING.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from .. import long_video_delivery as delivery
from .. import long_video
from .. import prompt_relay_advanced as relay
from ..conditioning import build_packed_layout
from ..prompt_relay_long_video_advanced import project_prompt_relay_plan_to_long_video_window
from . import hyperflow_p7 as p7
from .results import canonical, sha


SCHEMA = "t8.modular-sampling.hyperflow-p7-relay-window.v1"


def _request(contexts):
    if type(contexts) is p7.P7InitialContexts:
        return contexts.request
    if type(contexts) is p7.P7Contexts:
        return contexts.parent.binding["request"]
    raise ValueError("P7 Relay needs authenticated initial or accepted contexts")


def _accepted_start(contexts):
    if type(contexts) is p7.P7InitialContexts:
        contexts.verify()
        return 0
    binding = contexts.parent.revalidate()
    path = delivery._resolve_inside(contexts.parent.root,
                                    contexts.parent.root / delivery.MANIFEST_NAME)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["manifest_sha256"]:
        raise ValueError("P7 Relay accepted manifest changed")
    manifest = delivery._validate_manifest(json.loads(raw), binding["request"]["chain_id"])
    index = binding["request"]["segment_index"]
    entry = manifest["segments"][index - 1]
    if entry["candidate_id"] != binding["request"]["parent_candidate_id"]:
        raise ValueError("P7 Relay accepted parent changed")
    start = entry["timeline_end_frame"]
    if type(start) is not int or start < 1:
        raise ValueError("P7 Relay needs an exact accepted frame boundary")
    return start


def _project(contexts, global_plan, length, accepted_end_frame):
    request = _request(contexts)
    contexts.verify()
    if type(length) is not int or length < 5:
        raise ValueError("P7 Relay length must be an integer frame count")
    start = _accepted_start(contexts)
    end = (min(global_plan["frame_count"], start + length - request["context_frames"])
           if accepted_end_frame is None else accepted_end_frame)
    if type(end) is not int or end <= start:
        raise ValueError("P7 Relay accepted end must be an exact later frame")
    projected = project_prompt_relay_plan_to_long_video_window(
        global_plan, request["segment_index"], length, request["context_frames"],
        start / 24, end / 24)[0]
    contexts.verify()
    return projected


@dataclass(frozen=True)
class ProjectedP7Relay:
    contexts: p7.P7Contexts | p7.P7InitialContexts
    global_plan: dict
    projected: dict
    length: int
    accepted_end_frame: int | None
    contract_json: str

    def descriptor(self):
        current = _project(self.contexts, self.global_plan, self.length, self.accepted_end_frame)
        if current != self.projected:
            raise ValueError("P7 Relay projection changed")
        return {"schema": SCHEMA, "contexts_sha256": self.contexts.verify()["sha256"],
                "global_plan": self.global_plan, "projected": current,
                "length": self.length, "accepted_end_frame": self.accepted_end_frame}

    def verify(self):
        current = self.descriptor()
        current["sha256"] = sha(current)
        if canonical(current) != self.contract_json:
            raise ValueError("P7 Relay projection identity changed")
        return current


def project_relay(contexts, global_plan, length, *, accepted_end_frame=None):
    original = deepcopy(global_plan)
    projected = _project(contexts, original, length, accepted_end_frame)
    provisional = ProjectedP7Relay(contexts, original, projected, length, accepted_end_frame, "")
    descriptor = provisional.descriptor()
    descriptor["sha256"] = sha(descriptor)
    result = ProjectedP7Relay(contexts, original, projected, length, accepted_end_frame,
                              canonical(descriptor))
    result.verify()
    return result


def apply_relay(model, phase, clip, projected, *, query_chunk_rows=256, mode="apply_exp"):
    if type(phase) is not p7.P7Phase or type(projected) is not ProjectedP7Relay:
        raise ValueError("P7 Relay needs one prepared phase and projected window")
    if mode not in ("disabled", "apply_exp") or type(query_chunk_rows) is not int or not 32 <= query_chunk_rows <= 2048:
        raise ValueError("P7 Relay mode or query row count is invalid")
    phase.verify()
    projected.verify()
    if (projected.contexts is not phase.contexts or projected.length != phase.result[-1]["frame_count"]
            or projected.projected["compiled_prompt"] != phase.result[3]):
        raise ValueError("P7 Relay plan must match this phase's encoded prompt and window")
    positive, latent, _, prompt, _, _, details = phase.result
    if mode == "disabled":
        return model, positive, positive, canonical({"schema": SCHEMA, "mode": "disabled",
            "phase_sha256": phase.verify()["sha256"], "sampling_calls": 0})
    if model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is not None:
        raise ValueError("P7 Relay MODEL already has a Relay owner")
    binding = relay.build_prompt_relay_binding(clip, projected.projected, prompt, positive, details["tokens"])
    if binding["query_route"] == "joint_av_exp" and details["audio_mode"] == "lock_source":
        raise ValueError("P7 Relay joint AV route cannot modify locked source audio")
    segment_index = _request(phase.contexts)["segment_index"]
    task = details["resolved_task"].lower()
    if (segment_index == 0 and task != "t2va") or (segment_index > 0 and not task.endswith("-motion")):
        raise ValueError("P7 Relay requires its native segment-0 or continuation task")
    video, audio = latent["samples"].unbind()
    layout = build_packed_layout(binding["text_len"], *video.shape[2:], audio.shape[-1],
                                 keyframes=details["keyframes"], refs=details["refs"],
                                 frame_count=details["frame_count"])
    layout = long_video.repair_long_video_layout(layout, list(details["keyframes"]),
                                                 list(details["refs"]), details["frame_count"])
    binding = relay._bind_layout_contract(binding, layout, resolved_task=details["resolved_task"],
                                           keyframes=details["keyframes"], refs=details["refs"])
    base = long_video.patch_long_video_model(model)
    selected, _ = relay.patch_prompt_relay_model(base, binding, query_chunk_rows)
    marked = [[value, {**meta, relay.PROMPT_RELAY_BINDING_KEY: binding}] for value, meta in positive]
    paired = relay._attach_binding_model_cond(marked, binding["binding_hash"])
    phase.verify()
    report = {"schema": SCHEMA, "mode": mode, "phase_sha256": phase.verify()["sha256"],
              "projection_sha256": projected.verify()["sha256"],
              "binding_hash": binding["binding_hash"], "sampling_calls": 0,
              "boundary": "Actual routed calls must be checked on the completed Stage Result receipt."}
    return selected, paired, paired, canonical(report)
