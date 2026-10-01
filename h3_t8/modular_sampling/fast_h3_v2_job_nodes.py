"""Current FastH3 V2 editable recipe fingerprint; not an acceptance shortcut."""

from comfy_api.latest import io

from ..nodes_learned_latent_upscale_advanced import MiniMaxH3LearnedLatentUpscaleT8Advanced
from ..nodes_prompt_relay_long_video_advanced import MiniMaxH3PromptRelayLongVideoConditioningT8Advanced
from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .fast_h3_v2_conditioning import condition_with_provenance
from .eav import CONFIG_TYPE
from .fast_h3_v2_job import make_current_recipe
from .fast_h3_v2_stage_attest import attest_current_stage
from .fast_h3_v2_upscale import upscale_with_provenance
from .fast_h3_v2_current_handoff import attest_current_handoff
from .fast_h3_v2_job_binding import bind_current_job
from .fast_h3_v2_media import prepare_current_media
from .fast_h3_v2_color import color_match_current_media
from .fast_h3_v2_candidate import save_current_candidate
from .fast_h3_v2_color_candidate import save_colored_candidate
from .fast_h3_v2_review import review_accept_current_candidate, verify_current_accepted_chain
from .fast_h3_v2_frozen_low import save_current_low_bundle, load_current_low_bundle
from ..nodes_long_video_delivery_exp import _preview_video


JOB_TYPE = "T8_FAST_H3_V2_CURRENT_RECIPE"
STAGE_ATTEST_TYPE = "T8_FAST_H3_V2_CURRENT_STAGE_ATTESTATION"
CONDITION_RECEIPT_TYPE = "T8_FAST_H3_V2_CONDITION_RECEIPT"
UPSCALE_RECEIPT_TYPE = "T8_FAST_H3_V2_UPSCALE_RECEIPT"
HANDOFF_ATTEST_TYPE = "T8_FAST_H3_V2_CURRENT_HANDOFF"
JOB_BINDING_TYPE = "T8_FAST_H3_V2_CURRENT_JOB_BINDING"
MEDIA_RECEIPT_TYPE = "T8_FAST_H3_V2_CURRENT_MEDIA_RECEIPT"
COLORED_MEDIA_RECEIPT_TYPE = "T8_FAST_H3_V2_COLORED_MEDIA_RECEIPT"


class MiniMaxH3FastH3V2FrozenLOWBundleSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Save Attested Frozen LOW Recipe (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True, is_output_node=True,
            description="Attach the verified current recipe, executed LOW stage and actual LOW condition "
                        "receipt to an exact completed Stage Save manifest. Does not change the generic "
                        "stage artifact or freeze newly edited LOW settings.",
            inputs=[io.Custom(JOB_TYPE).Input("current_recipe"),
                    io.Custom("T8_STAGE_RESULT").Input("low_stage_result"),
                    io.Custom(STAGE_ATTEST_TYPE).Input("low_attestation"),
                    io.Custom(CONDITION_RECEIPT_TYPE).Input("low_condition_receipt"),
                    io.String.Input("artifact_path", force_input=True),
                    io.String.Input("artifact_sha256", force_input=True)],
            outputs=[io.String.Output("bundle_path"), io.String.Output("bundle_sha256"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        from .nodes import _stage_store_root
        return io.NodeOutput(*save_current_low_bundle(storage_root=_stage_store_root(), **kwargs))


class MiniMaxH3FastH3V2FrozenLOWBundleLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Load Exact Frozen LOW Recipe (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Revalidate an exact completed LOW artifact and its bound recipe, LOW stage "
                        "attestation and condition call. No LOW MODEL or sampler runs; HIGH live "
                        "inputs must still be attested independently.",
            inputs=[io.String.Input("artifact_path", default=""),
                    io.String.Input("artifact_sha256", default=""),
                    io.Custom("T8_STAGE_RESULT").Input("low_stage_result")],
            outputs=[io.Custom(JOB_TYPE).Output("current_recipe"),
                     io.String.Output("current_recipe_sha256"),
                     io.String.Output("model_id"), io.String.Output("report_json"),
                     io.Custom(STAGE_ATTEST_TYPE).Output("low_attestation"),
                     io.Custom(CONDITION_RECEIPT_TYPE).Output("low_condition_receipt"),
                     io.String.Output("bundle_sha256")])

    @classmethod
    def execute(cls, **kwargs):
        from .nodes import _stage_store_root
        return io.NodeOutput(*load_current_low_bundle(storage_root=_stage_store_root(), **kwargs))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, low_stage_result):
        del artifact_sha256, low_stage_result
        from . import storage
        from .nodes import _stage_store_root
        try:
            root = storage._root(_stage_store_root())
            manifest = storage._path(root, artifact_path)
            sidecar = storage._path(root, (manifest.parent / "current-v2-low-bundle.json")
                                    .relative_to(root).as_posix())
            return storage.fingerprint_stage(root, artifact_path), storage.file_sha(sidecar)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


class MiniMaxH3FastH3V2CurrentAcceptedChainVerifyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Verify Accepted Two-Segment Source (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Read-only verification of both accepted 124+68-frame segments, retained "
                        "source-bound candidates, media/context hashes and one current job SHA "
                        "before the unchanged long-video composer. Not a quality review.",
            inputs=[io.String.Input("chain_id", default="")],
            outputs=[io.String.Output("chain_id"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, chain_id):
        return io.NodeOutput(*verify_current_accepted_chain(chain_id))


class MiniMaxH3FastH3V2CurrentCandidateReviewAcceptEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Verify, Review & Explicitly Accept (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True, is_output_node=True,
            description="A separate review step for a saved split candidate. Verify its MP4/context, "
                        "current-job/media sidecar and exact 124/68-frame boundary before calling "
                        "the unchanged reject-existing manifest transaction. Defaults to preview only; "
                        "never composes a movie.",
            inputs=[io.String.Input("candidate_json_path", default=""),
                    io.Boolean.Input("accept_candidate", default=False)],
            outputs=[io.Video.Output("video"), io.Boolean.Output("accepted"),
                     io.String.Output("manifest_path"), io.String.Output("chain_id"),
                     io.String.Output("parent_candidate_id"), io.Int.Output("parent_revision"),
                     io.String.Output("current_job_sha256"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, candidate_json_path, accept_candidate=False):
        movie, accepted, manifest, chain, parent, revision, job, report = (
            review_accept_current_candidate(candidate_json_path, accept_candidate))
        video, preview = _preview_video(movie)
        return io.NodeOutput(video, accepted, manifest, chain, parent, revision,
                             job, report, ui=preview)


class MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Save Bound Candidate, Not Accepted (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True, is_output_node=True,
            description="Only consume the typed completed-HIGH media receipt and current job binding. "
                        "Use the unchanged durable candidate writer, add a source sidecar and stop before "
                        "human Review & Accept or composition.",
            inputs=[io.Custom(MEDIA_RECEIPT_TYPE).Input("media_receipt"),
                    io.String.Input("candidate_id", default="", advanced=True)],
            outputs=[io.String.Output("candidate_json_path"),
                     io.String.Output("candidate_video_path"),
                     io.String.Output("source_provenance_path"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, media_receipt, candidate_id=""):
        return io.NodeOutput(*save_current_candidate(media_receipt, candidate_id))


class MiniMaxH3FastH3V2ExternalColorMatchEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · External Accepted-Picture Color Match (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Apply the unchanged dual-loop RGB Color Match against the authenticated accepted "
                        "parent after HIGH decode/trim. Audio and latent stay untouched; segment 0 is identity.",
            inputs=[io.Custom(MEDIA_RECEIPT_TYPE).Input("media_receipt"),
                    io.Boolean.Input("enabled", default=True),
                    io.Combo.Input("mode", options=["bounded_spatial_v2", "bounded_spatial_temporal_exp",
                                                    "bounded_motion_color_exp"],
                                   default="bounded_motion_color_exp")],
            outputs=[io.Image.Output("frames"), io.Audio.Output("audio"),
                     io.Custom(COLORED_MEDIA_RECEIPT_TYPE).Output("color_receipt"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, media_receipt, enabled=True, mode="bounded_motion_color_exp"):
        return io.NodeOutput(*color_match_current_media(media_receipt, enabled, mode))


class MiniMaxH3FastH3V2ColoredCandidateSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Save Externally Colored Candidate (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True, is_output_node=True,
            description="Save only a typed completed-HIGH + external Color Match result as an unaccepted "
                        "candidate. Keeps the original raw-media candidate writer and sidecar intact.",
            inputs=[io.Custom(COLORED_MEDIA_RECEIPT_TYPE).Input("color_receipt"),
                    io.String.Input("candidate_id", default="", advanced=True)],
            outputs=[io.String.Output("candidate_json_path"),
                     io.String.Output("candidate_video_path"),
                     io.String.Output("source_provenance_path"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, color_receipt, candidate_id=""):
        return io.NodeOutput(*save_colored_candidate(color_receipt, candidate_id))


class MiniMaxH3FastH3V2CurrentMediaPrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Decode + Trim Bound HIGH (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Call the unchanged native AV Decode and Output Trim once, using the authenticated "
                        "completed HIGH output. Produce exact 124/68-frame media plus a typed source receipt. "
                        "Does not save or accept a candidate.",
            inputs=[io.Custom(JOB_BINDING_TYPE).Input("job_binding"),
                    io.Custom("T8_STAGE_RESULT").Input("high_result"),
                    io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
                    io.Float.Input("start_seconds", default=0., force_input=True),
                    io.Float.Input("duration_seconds", default=124 / 24, force_input=True),
                    io.Float.Input("fps", default=24.)],
            outputs=[io.Image.Output("frames"), io.Audio.Output("audio"),
                     io.Custom(MEDIA_RECEIPT_TYPE).Output("media_receipt"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*prepare_current_media(**kwargs))


class MiniMaxH3FastH3V2CurrentJobBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Bind Current Job to Accepted Parent (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Compare the completed two-stage job SHA with the immediate accepted parent's "
                        "saved job SHA and context/model identities. Segment 0 requires no parent. "
                        "Does not write, accept or compose a candidate.",
            inputs=[io.Custom(JOB_TYPE).Input("current_recipe"),
                    io.Custom(HANDOFF_ATTEST_TYPE).Input("handoff_attestation"),
                    io.Int.Input("segment_index", default=0, min=0, max=1),
                    io.Custom("T8_CONTINUATION_STAGE_CONTEXTS").Input("contexts", optional=True)],
            outputs=[io.Custom(JOB_BINDING_TYPE).Output("job_binding"),
                     io.String.Output("job_sha256"), io.String.Output("model_id"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*bind_current_job(**kwargs))


class MiniMaxH3FastH3V2CurrentHandoffAttestEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Verify LOW→Upscale→HIGH Origin (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Read-only proof of completed LOW x0, one learned 3D call, deterministic "
                        "HIGH reconcile/prefix and actual HIGH sampler source. Does not accept a segment.",
            inputs=[io.Custom(JOB_TYPE).Input("current_recipe"),
                    io.Custom("T8_STAGE_RESULT").Input("low_result"),
                    io.Custom("T8_STAGE_RESULT").Input("high_result"),
                    io.Custom(STAGE_ATTEST_TYPE).Input("low_attestation"),
                    io.Custom(STAGE_ATTEST_TYPE).Input("high_attestation"),
                    io.Custom(CONDITION_RECEIPT_TYPE).Input("low_condition_receipt"),
                    io.Custom(CONDITION_RECEIPT_TYPE).Input("high_condition_receipt"),
                    io.Custom(UPSCALE_RECEIPT_TYPE).Input("upscale_receipt"),
                    io.Latent.Input("upscaled_latent"),
                    io.String.Input("upscale_report_json", force_input=True),
                    io.Latent.Input("high_template"),
                    io.Conditioning.Input("high_positive"),
                    io.Latent.Input("high_source_latent"),
                    io.Int.Input("segment_index", default=0, min=0, max=1),
                    io.Custom("T8_CONTINUATION_STAGE_CONTEXTS").Input("contexts", optional=True)],
            outputs=[io.Custom(HANDOFF_ATTEST_TYPE).Output("handoff_attestation"),
                     io.String.Output("handoff_sha256"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*attest_current_handoff(**kwargs))


class MiniMaxH3FastH3V2UpscaleProvenanceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3LearnedLatentUpscaleT8Advanced.define_schema()
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Learned 3D Upscale + Origin (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Execute the unchanged learned 3D upscaler once and attach a typed "
                        "input/output/report receipt. No second upscale or HIGH sampling runs here.",
            inputs=original.inputs,
            outputs=[*original.outputs, io.Custom(UPSCALE_RECEIPT_TYPE).Output("upscale_receipt")])

    @classmethod
    def execute(cls, av_latent, **kwargs):
        return io.NodeOutput(*upscale_with_provenance(av_latent, **kwargs))


class MiniMaxH3FastH3V2ConditionProvenanceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3PromptRelayLongVideoConditioningT8Advanced.define_schema()
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Long Video Relay Conditions + Origin (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Call the unchanged native Long Video Relay conditioner once, adding a typed origin "
                        "receipt for actual CLIP/VAEs, first frame, projected plan and output conditions. "
                        "This alone does not certify a HIGH handoff or accepted parent.",
            inputs=original.inputs,
            outputs=[*original.outputs, io.Custom(CONDITION_RECEIPT_TYPE).Output("condition_receipt")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*condition_with_provenance(**kwargs))


class MiniMaxH3FastH3V2CurrentRecipeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Current Editable Recipe Fingerprint (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Hash the CURRENT two raw MODEL/LoRA branches, CLIP/VAEs, first frame, "
                        "global Relay plans, external EAV configs and learned-upscaler bytes. "
                        "This is not the previous accepted job SHA and does NOT attest stage execution, "
                        "save a candidate or authorize acceptance.",
            inputs=[io.Model.Input("model_pass1"), io.Model.Input("model_pass2"),
                    io.Clip.Input("clip"), io.Vae.Input("video_vae"),
                    io.Vae.Input("audio_vae"), io.Image.Input("first_frame"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("low_global_plan"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("high_global_plan"),
                    io.Custom(CONFIG_TYPE).Input("low_eav_config"),
                    io.Custom(CONFIG_TYPE).Input("high_eav_config"),
                    io.String.Input("upscale_report_json", force_input=True),
                    io.String.Input("chain_id"),
                    io.Int.Input("low_width", default=256, min=32, step=32),
                    io.Int.Input("low_height", default=384, min=32, step=32),
                    io.Int.Input("width", default=512, min=32, step=32),
                    io.Int.Input("height", default=768, min=32, step=32),
                    io.Int.Input("total_accepted_frames", default=192, min=192, max=192),
                    io.Int.Input("first_seed", default=2609152201, min=0),
                    io.Int.Input("continuation_render_frames", default=90, min=90, max=124,
                                 optional=True)],
            outputs=[io.Custom(JOB_TYPE).Output("current_recipe"),
                     io.String.Output("current_recipe_sha256"),
                     io.String.Output("model_id"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*make_current_recipe(**kwargs))


class MiniMaxH3FastH3V2CurrentStageAttestEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Verify One Current Stage (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Compare one completed LOW/HIGH StageResult against the current raw MODEL/LoRA, "
                        "actual sampler MODEL/guider/noise/source/sigmas, projected Relay and EAV. "
                        "Does not accept or save a segment; both stages and parent delivery require separate gates.",
            inputs=[io.Custom(JOB_TYPE).Input("current_recipe"),
                    io.Custom("T8_STAGE_RESULT").Input("stage_result"),
                    io.Model.Input("raw_model"), io.Model.Input("stage_model"),
                    io.Guider.Input("guider"), io.Noise.Input("noise"),
                    io.Sigmas.Input("sigmas"), io.Latent.Input("source_latent"),
                    io.Custom("T8_STAGE_CONTEXT").Input("stage_context"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("global_plan"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("projected_plan"),
                    io.Custom(CONFIG_TYPE).Input("eav_config"),
                    io.Int.Input("segment_index", default=0, min=0, max=1),
                    io.Combo.Input("phase", options=["low_0_4", "high_4_8"], default="low_0_4")],
            outputs=[io.Custom(STAGE_ATTEST_TYPE).Output("stage_attestation"),
                     io.String.Output("attestation_sha256"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*attest_current_stage(**kwargs))


class MiniMaxH3FastH3V2OriginStageAttestEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3FastH3V2CurrentStageAttestEXPT8.define_schema()
        return io.Schema(node_id=cls.__name__,
            display_name="FastH3 V2 · Verify Stage + Condition Origin (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Continuation Experimental",
            is_experimental=True,
            description="Require the same executed condition call's typed origin receipt and latent, "
                        "then bind its CLIP/VAEs/first frame/positive to the completed sampler. "
                        "Does not certify HIGH handoff, accepted parent, candidate save or assembly.",
            inputs=[*original.inputs,
                    io.Custom(CONDITION_RECEIPT_TYPE).Input("condition_receipt"),
                    io.Latent.Input("conditioned_latent")],
            outputs=original.outputs)

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*attest_current_stage(**kwargs))


NODES = (MiniMaxH3FastH3V2CurrentRecipeEXPT8,
         MiniMaxH3FastH3V2CurrentStageAttestEXPT8,
         MiniMaxH3FastH3V2ConditionProvenanceEXPT8,
         MiniMaxH3FastH3V2OriginStageAttestEXPT8,
         MiniMaxH3FastH3V2UpscaleProvenanceEXPT8,
         MiniMaxH3FastH3V2CurrentHandoffAttestEXPT8,
         MiniMaxH3FastH3V2CurrentJobBindEXPT8,
         MiniMaxH3FastH3V2CurrentMediaPrepareEXPT8,
         MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8,
         MiniMaxH3FastH3V2ExternalColorMatchEXPT8,
         MiniMaxH3FastH3V2ColoredCandidateSaveEXPT8,
         MiniMaxH3FastH3V2CurrentCandidateReviewAcceptEXPT8,
         MiniMaxH3FastH3V2CurrentAcceptedChainVerifyEXPT8,
         MiniMaxH3FastH3V2FrozenLOWBundleSaveEXPT8,
         MiniMaxH3FastH3V2FrozenLOWBundleLoadEXPT8)
