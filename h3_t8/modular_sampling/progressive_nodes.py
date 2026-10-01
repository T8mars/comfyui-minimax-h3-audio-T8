"""Explicit native Progressive graph surfaces; no hidden whole-run invocation."""
from pathlib import Path

from comfy_api.latest import io

from . import progressive as stages
from . import progressive_storage as storage
from . import progressive_high_result as high_results

PLAN = "T8_PROGRESSIVE_STAGE_PLAN"
BOUNDARY = "T8_PROGRESSIVE_LOW_BOUNDARY"
RESTART = "T8_PROGRESSIVE_HIGH_RESTART"
HIGH_RESULT = "T8_PROGRESSIVE_HIGH_RESULT"
CATEGORY = "T8/MiniMax H3/Modular Sampling/Progressive Experimental"


def schema(node, label, description, inputs, outputs, *, output_node=False):
    return io.Schema(node_id=node.__name__, display_name="H3 Progressive · " + label + " (T8 EXP)",
                     category=CATEGORY, is_experimental=True, description=description,
                     inputs=inputs, outputs=outputs, is_output_node=output_node)


def _store_root():
    import folder_paths
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "progressive_stage_artifacts"


def _noise_seed(noise):
    seed = getattr(noise, "seed", None)
    if type(seed) is not int or not 0 <= seed < 2**64 or not callable(getattr(noise, "generate_noise", None)):
        raise ValueError("Connect a Core NOISE provider with an unsigned 64-bit seed")
    return seed


def _generate_noise(noise, latent):
    _noise_seed(noise)
    before = stages.snapshot(latent)
    value = noise.generate_noise(latent)
    if before != stages.snapshot(latent):
        raise ValueError("NOISE provider changed the supplied Progressive source")
    return value


def _stage_progress(plan, phase):
    import comfy.utils
    start, count = (0, plan.low_evaluations) if phase == "low" else (plan.low_evaluations, plan.high_evaluations)
    progress = comfy.utils.ProgressBar(count)
    def notify(step, prediction, state, total):
        progress.update_absolute(step - start + 1, count)
    return notify


class MiniMaxH3ProgressiveStagePlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Plan + LOW Source", "Plan only: no MODEL, conditioning, sampling or learned lift. "
            "Original native Euler AV time grid. Empty T2VA/I2VA or explicit initialized AV. "
            "Not an accepted-parent continuation or standalone Avatar delivery contract.",
            [io.Latent.Input("high_source"), io.Sigmas.Input("full_sigmas"),
             io.Int.Input("low_evaluations", default=4, min=1, max=999),
             io.Float.Input("low_scale", default=.5, min=.25, max=.99, step=.01),
             io.Combo.Input("task", options=["t2va", "i2va"], default="t2va"),
             io.Combo.Input("input_mode", options=["empty", "initialized_av_exp"], default="empty")],
            [io.Custom(PLAN).Output("plan"), io.Latent.Output("low_source"),
             io.Sigmas.Output("low_sigmas"), io.Sigmas.Output("high_sigmas"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_source, full_sigmas, low_evaluations=4, low_scale=.5, task="t2va", input_mode="empty"):
        plan = stages.build_plan(high_source, full_sigmas, low_evaluations, low_scale, task, input_mode)
        low = stages.prepare_low_source(high_source, plan, input_mode)
        return io.NodeOutput(plan, low, full_sigmas[:low_evaluations+1].detach().clone(),
                             full_sigmas[low_evaluations:].detach().clone(), stages.canonical(plan.report()))


class MiniMaxH3ProgressiveStageConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "ONE Stage Conditioning", "Prepare one selected stage's native conditioning. "
            "Use separate nodes/inputs for independently editable LOW and HIGH prompts. "
            "Only LOW reference latent is resized; HIGH retains its original guide. No model call. "
            "Relay requires its own paired stage-layout adapter, not an unpaired copied binding.",
            [io.Conditioning.Input("positive"), io.Conditioning.Input("negative"), io.Custom(PLAN).Input("plan"),
             io.Combo.Input("phase", options=["low", "high"], default="low"),
             io.Combo.Input("guide_resize", options=["legacy_bilinear", "preserve_mean"], default="legacy_bilinear")],
            [io.Conditioning.Output("positive"), io.Conditioning.Output("negative")])

    @classmethod
    def execute(cls, positive, negative, plan, phase="low", guide_resize="legacy_bilinear"):
        stages.validate_plan(plan)
        if phase not in ("low", "high"):
            raise ValueError("Unknown Progressive conditioning phase")
        index = 0 if phase == "low" else 1
        pos = stages.legacy.prepare_stage_conditioning(positive, plan, positive=True, guide_resize=guide_resize)[index]
        neg = stages.legacy.prepare_stage_conditioning(negative, plan, positive=False, guide_resize=guide_resize)[index]
        return io.NodeOutput(pos, neg)


class MiniMaxH3ProgressiveLowStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "LOW Sampler Only", "Execute only LOW native Euler. No HIGH model/prompt/lifter input. "
            "Returns typed model-space clean_video + evolving audio_next + original audio noise, "
            "not completed AV or ordinary denoised output. Connect the dedicated Lift Input and Save nodes.",
            [io.Model.Input("model"), io.Sampler.Input("sampler"), io.Noise.Input("noise"),
             io.Custom(PLAN).Input("plan"), io.Latent.Input("low_source"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
             io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            [io.Custom(BOUNDARY).Output("low_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sampler, noise, plan, low_source, positive, negative, cfg=1., reserve_vram_mib=1024):
        seed = _noise_seed(noise)
        generated = _generate_noise(noise, low_source)
        boundary = stages.sample_low(model, sampler, plan, low_source, positive, negative, generated,
                                     seed=seed, cfg=cfg, reserve_vram_mib=reserve_vram_mib,
                                     callback=_stage_progress(plan, "low"))
        return io.NodeOutput(boundary, boundary.receipt_json)


class MiniMaxH3ProgressiveLiftInputEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Boundary to External Lift Input", "No sampling or learned network executes. "
            "Converts only clean video to original VAE coordinates; zero audio is a lifter placeholder. "
            "The original evolving audio stays in low_boundary for HIGH. Connect the existing learned "
            "3D latent upscaler using these target dimensions.",
            [io.Custom(BOUNDARY).Input("low_boundary"), io.Model.Input("model"), io.Sampler.Input("sampler")],
            [io.Latent.Output("lift_input"), io.Int.Output("target_width"), io.Int.Output("target_height"),
             io.Custom(PLAN).Output("plan"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_boundary, model, sampler):
        receipt = low_boundary.verify()
        plan = stages.plan_from_dict(receipt["request"]["plan"])
        value = stages.lift_input(low_boundary, model, sampler)
        return io.NodeOutput(value, plan.target_width, plan.target_height, plan,
            stages.canonical({"sampling_calls": 0, "learned_lift_executed": False,
                              "low_receipt_sha256": receipt["receipt_sha256"]}))


class MiniMaxH3ProgressiveHighHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "HIGH Handoff Only", "Connect actual external learned output, frozen LOW boundary "
            "and independent HIGH clean source. Draws only HIGH video noise, restores original native AV "
            "coordinates and clean-source masks; no LOW/HIGH sampling or learned lift. For the original "
            "noise policy use video_noise seed=(LOW seed+1) modulo 2^64. Optional HIGH sigma tail must start "
            "at the frozen boundary. Full pretrained/effect/continuation qualification remains separate.",
            [io.Custom(BOUNDARY).Input("low_boundary"), io.Model.Input("model"), io.Sampler.Input("sampler"),
             io.Latent.Input("lifted_av"), io.Latent.Input("high_source"), io.Noise.Input("video_noise"),
             io.Sigmas.Input("high_sigmas", optional=True)],
            [io.Custom(RESTART).Output("high_restart"), io.Custom(PLAN).Output("plan"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_boundary, model, sampler, lifted_av, high_source, video_noise, high_sigmas=None):
        plan = stages.plan_from_dict(low_boundary.verify()["request"]["plan"])
        video, _ = stages._geometry(lifted_av, plan, "high")
        generated = _generate_noise(video_noise, {"samples": video})
        state = stages.prepare_high(low_boundary, model, sampler, lifted_av, high_source, generated,
                                    high_sigmas=high_sigmas)
        return io.NodeOutput(state, state.plan, stages.canonical({"restart": state.verify(),
            "video_noise_seed": _noise_seed(video_noise), "sampling_calls": 0, "learned_lift_executed": False}))


class MiniMaxH3ProgressiveHighStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "HIGH Sampler Only", "Execute only the native Euler tail using explicit "
            "zero-noise reconstruction and the original clean source/masks. Independent MODEL/LoRA/conditions. "
            "seed is the sampler seed (original policy: same as LOW), not the HIGH video-noise seed. "
            "Never executes LOW or learned upscale; no automatic cache or retry.",
            [io.Custom(RESTART).Input("high_restart"), io.Model.Input("model"), io.Sampler.Input("sampler"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Int.Input("seed", default=1234, min=0, max=0xffffffffffffffff),
             io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
             io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            [io.Latent.Output("av_latent"), io.String.Output("report_json"),
             io.Custom(HIGH_RESULT).Output("high_result")])

    @classmethod
    def execute(cls, high_restart, model, sampler, positive, negative, seed=1234, cfg=1., reserve_vram_mib=1024):
        result, report = high_results.sample_high_result(high_restart, model, sampler, positive, negative,
                              seed=seed, cfg=cfg, reserve_vram_mib=reserve_vram_mib,
                              callback=_stage_progress(high_restart.plan, "high"))
        return io.NodeOutput(result.output, report, result)


class MiniMaxH3ProgressiveLowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Save Frozen LOW Boundary", "Writes a new typed artifact with exact SHA under "
            "output/MiniMaxH3/progressive_stage_artifacts. No overwrite. Not ordinary Stage Save: evolving "
            "audio and clean video remain distinct. Unknown executable identity may be archived but not "
            "certified for persistent reuse. Save both path and SHA.",
            [io.Custom(BOUNDARY).Input("low_boundary"), io.String.Input("prefix", default="LOW")],
            [io.Custom(BOUNDARY).Output("low_boundary"), io.String.Output("artifact_path"),
             io.String.Output("artifact_sha256"), io.String.Output("report_json")], output_node=True)

    @classmethod
    def execute(cls, low_boundary, prefix="LOW"):
        path, digest, report = storage.save_boundary(low_boundary, _store_root(), prefix)
        return io.NodeOutput(low_boundary, path, digest, report,
                             ui={"text": ("artifact_path: " + path, "artifact_sha256: " + digest, report)})


class MiniMaxH3ProgressiveLowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load Exact Frozen LOW Boundary", "Select completed LOW path and exact SHA. "
            "No MODEL loading or LOW sampling, and no assertion that today's LOW settings match this frozen "
            "artifact. Feed its boundary to Lift Input/HIGH Handoff; remove LOW output nodes from a HIGH-only "
            "recovery graph. Corruption/missing/busy artifacts fail, never silently regenerate.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Custom(BOUNDARY).Output("low_boundary"), io.Custom(PLAN).Output("plan"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        boundary, report = storage.load_boundary(_store_root(), artifact_path, artifact_sha256)
        return io.NodeOutput(boundary, stages.plan_from_dict(boundary.verify()["request"]["plan"]), report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        try:
            return storage.fingerprint_boundary(_store_root(), artifact_path)
        except (OSError, ValueError, RuntimeError):
            return float("nan")


class MiniMaxH3ProgressiveHighSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Save Completed HIGH AV", "Save typed completed HIGH with its execution receipt, "
            "exact output identity and metadata. Unique new artifact; never overwrites an old result. "
            "No additional sampling or learned lift. Save both path and SHA.",
            [io.Custom(HIGH_RESULT).Input("high_result"), io.String.Input("prefix", default="HIGH")],
            [io.Latent.Output("av_latent"), io.String.Output("artifact_path"),
             io.String.Output("artifact_sha256"), io.String.Output("report_json")], output_node=True)

    @classmethod
    def execute(cls, high_result, prefix="HIGH"):
        path, digest, report = high_results.save_result(high_result, _store_root(), prefix)
        return io.NodeOutput(high_result.output, path, digest, report,
                             ui={"text": ("artifact_path: " + path, "artifact_sha256: " + digest, report)})


class MiniMaxH3ProgressiveHighLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load Exact Completed HIGH AV", "Select frozen completed HIGH using path and SHA. "
            "Loads no models, runs no sampling/lift. Reconnect the returned AV to decoding or later stages. "
            "No implicit cache lookup, retry or claim of matching today's settings.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Latent.Output("av_latent"), io.Custom(HIGH_RESULT).Output("high_result"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        result, report = high_results.load_result(_store_root(), artifact_path, artifact_sha256)
        return io.NodeOutput(result.output, result, report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        try:
            return high_results.fingerprint_result(_store_root(), artifact_path)
        except (OSError, ValueError, RuntimeError):
            return float("nan")


NODES = (MiniMaxH3ProgressiveStagePlanEXPT8, MiniMaxH3ProgressiveStageConditioningEXPT8,
         MiniMaxH3ProgressiveLowStageEXPT8, MiniMaxH3ProgressiveLiftInputEXPT8,
         MiniMaxH3ProgressiveHighHandoffEXPT8, MiniMaxH3ProgressiveHighStageEXPT8,
         MiniMaxH3ProgressiveLowSaveEXPT8, MiniMaxH3ProgressiveLowLoadEXPT8,
         MiniMaxH3ProgressiveHighSaveEXPT8, MiniMaxH3ProgressiveHighLoadEXPT8)
