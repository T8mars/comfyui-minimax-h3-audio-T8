"""Visible continuation choices from existing authenticated stage contracts, not a sampler."""
from copy import deepcopy
import re


def continuation_card(*, completed_quality_stage=None, completed_freevideo_stage=None,
                      freevideo_mid=None, completed_modular_stage=None,
                      res_checkpoint_path="", res_checkpoint_sha256="", checkpoint_root=None):
    selected = [completed_quality_stage, completed_freevideo_stage, freevideo_mid,
                completed_modular_stage, res_checkpoint_path or None]
    if sum(value is not None for value in selected) != 1:
        raise ValueError("Choose exactly one actual typed stage or explicit RES history checkpoint; MP4 is not a stage")
    result = dict(schema="t8.h07.continuation-card.v1", read_extra_NFE=0,
        sampling_started=False, automatic_accept=False, cache_reuse_authorized=False,
        MP4_recovery_supported=False, required_revalidation="existing destination validates current producer/model/conditions/clock")
    if completed_quality_stage is not None:
        from .freevideo_quality.runtime import validate_stage
        stage = completed_quality_stage
        receipt = validate_stage(stage)
        result.update(kind="quality_completed", complete_AV=True, completed_NFE=receipt["completed_nfe"],
            receipt_sha256=stage.receipt_sha256, request_sha256=receipt["request_sha256"],
            producer=receipt["freevideo_revision"], role=receipt["role"], clock=deepcopy(receipt["clock"]),
            restore_node="MiniMaxH3FreeVideoQualityStageLoadEXPT8")
        if receipt["role"] == "LOW":
            result.update(action="continue_completed_LOW8_with_independent_HIGH3", additional_NFE=3,
                next_node="MiniMaxH3FreeVideoCommunityHIGH3EXPT8",
                required_inputs=["same producer runtime", "HIGH conditioning", "completed_low",
                                 "learned-lifted AV with unchanged completed LOW audio", "explicit seed"])
        else:
            result.update(action="decode_completed_AV", additional_NFE=0, next_node="MiniMaxH3AVDecodeT8")
    elif completed_freevideo_stage is not None:
        from .freevideo_exp.runtime import validate_stage
        stage = completed_freevideo_stage
        receipt = validate_stage(stage)
        result.update(kind="legacy_FreeVideo_completed", complete_AV=True, completed_NFE=receipt["completed_nfe"],
            receipt_sha256=stage.receipt_sha256, request_sha256=receipt["request_sha256"],
            producer=receipt["freevideo_revision"], role=receipt["role"],
            restore_node="MiniMaxH3FreeVideoStageLoadEXPT8")
        if receipt["role"] == "LOW":
            result.update(action="continue_completed_LOW8_with_original_tail", additional_NFE=2,
                explicit_tail_options=list(range(1, 8)), default_unchanged=2,
                next_node="MiniMaxH3FreeVideoHIGHEXPT8",
                required_inputs=["original producer runtime", "HIGH conditioning", "completed_low",
                                 "learned-lifted AV with unchanged completed LOW audio", "explicit tail/seed"])
        else:
            result.update(action="decode_completed_AV", additional_NFE=0, next_node="MiniMaxH3AVDecodeT8")
    elif freevideo_mid is not None:
        from .freevideo_exp.split_runtime import validate_mid
        stage = freevideo_mid
        receipt = validate_mid(stage)
        result.update(kind="original_trajectory_MID4", complete_AV=False, completed_NFE=4,
            receipt_sha256=stage.receipt_sha256, request_sha256=receipt["request_sha256"],
            producer=receipt["freevideo_revision"], clock=deepcopy(receipt["split"]),
            restore_node="MiniMaxH3FreeVideoMIDLoadEXPT8", next_node="MiniMaxH3FreeVideoSplitHIGHEXPT8",
            action="continue_original_trajectory_remaining4", additional_NFE=4,
            required_inputs=["original producer runtime", "HIGH conditioning", "real noisy MID audio/state",
                             "externally lifted partial x0", "explicit seed"],
            warning="Partial x0 preview and unfinished audio are not a complete movie or completed LOW8")
    elif completed_modular_stage is not None:
        from .modular_sampling.results import StageResult
        if not isinstance(completed_modular_stage, StageResult):
            raise ValueError("Expected actual typed modular StageResult")
        stage = completed_modular_stage
        receipt = stage.verify()
        context = receipt["request"]["stage_context"]
        result.update(kind="completed_modular_result", receipt_sha256=receipt["receipt_sha256"],
            request_sha256=receipt["request_sha256"], recipe=context["recipe"], role=context["stage"],
            planned_interval=[context["start"], context["end"]],
            actual_observed_NFE=receipt["execution"]["denoiser_evaluations"],
            verified_recipe_completion=receipt["verified_recipe_completion"],
            portable_identity=receipt["portable_identity"], action="read_completed_result_not_resume_history",
            additional_NFE=0, complete_AV=receipt["verified_recipe_completion"] is True and context["trajectory_sigmas"][-1:] == [0.] and context["end"] == len(context["trajectory_sigmas"])-1,
            restore_node="MiniMaxH3RESStageLoadEXPT8" if context["recipe"] == "h05_RES_history_complete_v1" else "MiniMaxH3StageLoadEXPT8",
            warning="Unknown patches remain usable; this card never grants persistent reuse or CUDA/quality qualification")
    else:
        from .res_history_exp import read_checkpoint
        if checkpoint_root is None or not isinstance(res_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", res_checkpoint_sha256):
            raise ValueError("Supply the existing RES storage root and actual checkpoint SHA256")
        loaded = read_checkpoint(checkpoint_root, res_checkpoint_path)
        if loaded["file_sha256"].lower() != res_checkpoint_sha256.lower():
            raise ValueError("RES checkpoint SHA differs from the selected candidate")
        payload = loaded["payload"]
        result.update(kind="RES_post_step_history_not_x0", completed_global_intervals=payload["completed_steps"],
            complete_AV=payload["completed_steps"] == payload["total_steps"],
            action="resume_same_RES_trajectory", additional_NFE=payload["total_steps"]-payload["completed_steps"],
            checkpoint_path=loaded["relative_path"], checkpoint_sha256=loaded["file_sha256"],
            full_sigmas=loaded["tensors"]["full_sigmas"].tolist(),
            restore_node="MiniMaxH3RESHistoryEXPT8", restore_mode="resume",
            next_node="MiniMaxH3RESStageSamplerEXPT8",
            required_inputs=["same actual MODEL/ordered LoRA/conditions", "same full SIGMAS/AV coordinates",
                             "original noise/latent", "matching run contract/compiled assets"],
            warning="File integrity is not current MODEL compatibility; existing restore verifies it at execution")
    result["text"] = (f"{result['kind']}\n{result['action']}\n读取额外采样：0 NFE；所选下一动作额外采样："
                      f"{result['additional_NFE']} NFE\n恢复节点：{result['restore_node']}\n"
                      f"下一节点：{result.get('next_node', '按原工作流继续')}\n"
                      "只读提示，手工采用；不自动排队、不从MP4恢复latent、不放宽原有校验。")
    return result
