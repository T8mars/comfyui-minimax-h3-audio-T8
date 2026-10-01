"""Independent accepted-window Relay projection and native motion-scoped EAV.

Each owner binds one prepared phase. No opposite-phase condition, no duplicate
guide resizing, no paired-conditions proxy, no hidden two-stage execution.
"""
from copy import deepcopy
from dataclasses import dataclass
import json

import torch

from .. import prompt_relay_advanced as relay
from .. import progressive_continuation_relay as legacy_relay
from .. import progressive_relay as relay_stages
from .. import progressive_eav as eav_stages
from ..conditioning import build_packed_layout
from ..long_video import LONG_VIDEO_PATCH_VERSION, patch_long_video_model, repair_long_video_layout
from ..progressive_eav_masks import NativeProgressiveMaskContract
from ..prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
from . import continuation as stages
from . import progressive_effects as effects
from .eav import EAVConfig


@dataclass(frozen=True)
class ProjectedRelay:
    contexts: stages.ContinuationContexts
    global_plan: dict
    projected: dict
    length: int
    accepted_end_frame: int | None
    contract_json: str

    def descriptor(self):
        contexts = self.contexts.verify()
        current = legacy_relay.project_for_source(self.contexts.source, self.global_plan,
                                                   self.length, self.accepted_end_frame)
        if current != self.projected:
            raise ValueError("Continuation Relay projection changed")
        return {"schema": "t8.modular-sampling.continuation-relay.v1", "contexts_sha256": contexts["sha256"],
            "source_sha256": self.contexts.source.sha256, "global_plan": self.global_plan,
            "projected": current, "length": self.length, "accepted_end_frame": self.accepted_end_frame}

    def verify(self):
        current = self.descriptor()
        if stages.stages.canonical(current) != self.contract_json:
            raise ValueError("Continuation Relay binding changed")
        return current


def project_relay(contexts, global_plan, length, *, accepted_end_frame=None):
    if type(contexts) is not stages.ContinuationContexts or type(length) is not int:
        raise ValueError("Relay projection requires authenticated contexts and integer frame length")
    with stages.chain_guard(contexts.source.root):
        contexts.verify()
        original = deepcopy(global_plan)
        projected = legacy_relay.project_for_source(contexts.source, original, length, accepted_end_frame)
        provisional = ProjectedRelay(contexts, original, projected, length, accepted_end_frame, "")
        contract = stages.stages.canonical(provisional.descriptor())
        result = ProjectedRelay(contexts, original, projected, length, accepted_end_frame, contract)
        result.verify()
        return result


def relay_attachment(prepared, report):
    return {"schema": 1, "global_plan_hash": report["global_plan_hash"],
        "projected_plan_hash": report["plan_hash"], "binding_hash": report["binding_hash"],
        "segment_index": prepared.contexts.source.binding["request"]["segment_index"],
        "accepted_source_sha256": prepared.contexts.source.sha256}


def apply_relay(model, prepared, plan, clip, projected, *, query_chunk_rows=256, mode="apply_exp"):
    if type(projected) is not ProjectedRelay or mode not in ("disabled", "apply_exp"):
        raise ValueError("Select an authenticated continuation Relay plan and execution mode")
    stages._checked_phase(prepared, prepared.phase, plan)
    if effects.owner(model) is not None or model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is not None:
        raise ValueError("Use one independent continuation Relay before stage EAV")
    if type(query_chunk_rows) is not int or not 32 <= query_chunk_rows <= 2048:
        raise ValueError("Continuation Relay query_chunk_rows must be32..2048")
    with stages.chain_guard(prepared.contexts.source.root):
        projected.verify()
        if (projected.contexts.verify() != prepared.contexts.verify()
                or projected.length != prepared.result[-1]["frame_count"]
                or projected.projected["compiled_prompt"] != prepared.result[3]):
            raise ValueError("Continuation Relay must match this phase's encoded prompt and accepted window")
        positive, latent, _, prompt, _, _, details = prepared.result
        base = patch_long_video_model(model)
        if mode == "disabled":
            return base, positive, positive
        binding = relay.build_prompt_relay_binding(clip, projected.projected, prompt, positive, details["tokens"])
        if binding["query_route"] == "joint_av_exp" and details["audio_mode"] == "lock_source":
            raise ValueError("Continuation joint_av_exp cannot route locked source audio; use video_only_paper")
        if not details["resolved_task"].lower().endswith("-motion"):
            raise ValueError("Continuation Relay requires native motion conditioning")
        video, audio = latent["samples"].unbind()
        layout = build_packed_layout(binding["text_len"], *video.shape[2:], audio.shape[-1],
            keyframes=details["keyframes"], refs=details["refs"], frame_count=details["frame_count"])
        layout = repair_long_video_layout(layout, list(details["keyframes"]), list(details["refs"]), details["frame_count"])
        binding = relay._bind_layout_contract(binding, layout, resolved_task=details["resolved_task"],
                                               keyframes=details["keyframes"], refs=details["refs"])
        inspection = base.clone()
        inspection.object_patches.pop("extra_conds")  # inspection only; runtime retains the real delegate
        selected, pos, neg, _, report = relay_stages.install_relay_stage(base, binding, positive, positive,
            query_chunk_rows, allowed_extra_conds_versions=(LONG_VIDEO_PATCH_VERSION,), inspection_model=inspection)
        report.update(global_plan_hash=projected.projected["global_plan_hash"],
            projection=json.loads(json.dumps(projected.projected["long_video_projection"])),
            accepted_source_sha256=prepared.contexts.source.sha256)
        selected.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, relay_attachment(prepared, report))
        selected.set_attachments(effects.KEY, effects.StageEffects(plan, prepared.phase, relay_report=report,
                                                                   continuation=prepared))
        prepared.verify()
        return selected, pos, neg


def apply_eav(model, config, prepared, plan, *, restart=None):
    if type(config) is not EAVConfig:
        raise ValueError("Connect external Stage EAV Config")
    stages._checked_phase(prepared, prepared.phase, plan)
    if config.mode == "disabled":
        return model
    with stages.chain_guard(prepared.contexts.source.root):
        prepared.verify()
        current = effects.owner(model)
        if current is not None:
            current.verify()
            if (current.plan != plan or current.phase != prepared.phase or current.eav_runtime is not None
                    or current.continuation is None or current.continuation.verify() != prepared.verify()):
                raise ValueError("Continuation EAV and Relay must bind the same independent phase")
        elif model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is not None:
            raise ValueError("Apply continuation Relay to this phase before EAV")
        if prepared.phase == "high":
            if type(restart) is not stages.stages.ProgressiveRestart or restart.plan != plan:
                raise ValueError("HIGH continuation EAV requires its own typed restart")
            restart.verify()
            if restart.metadata.get(stages.KEY, {}).get("high_phase") != prepared.verify():
                raise ValueError("HIGH continuation EAV restart has another accepted phase")
            latent = {"samples": restart.tensors["anchor"], "noise_mask": restart.tensors["mask"]}
        else:
            if restart is not None:
                raise ValueError("LOW continuation EAV does not consume a HIGH restart")
            latent = prepared.result[1]
        base = patch_long_video_model(model)
        video, audio = stages.stages._geometry(latent, plan, prepared.phase)
        normalized = stages.stages.masks.normalize_av_masks(latent.get("noise_mask"), video, audio)
        mask = NativeProgressiveMaskContract(base, normalized, video.shape, audio.shape,
                                              accepted_source=prepared.contexts.source)
        report = deepcopy(current.relay_report) if current is not None else None
        sigmas = torch.tensor(plan.sigmas, dtype=getattr(torch, plan.sigma_dtype.split(".")[-1]))
        selected, runtime = eav_stages.prepare_progressive_eav(base, sigmas, plan,
            mode=config.mode, tau=config.tau, start=config.start_video_progress, end=config.end_video_progress,
            workspace=config.max_workspace_mib, hard_limit=config.g_hard_limit, relay_report=report, mask_contract=mask)
        selected.set_attachments(effects.KEY, effects.StageEffects(plan, prepared.phase, relay_report=report,
            eav_runtime=runtime, mask=mask, continuation=prepared))
        prepared.verify()
        return selected
