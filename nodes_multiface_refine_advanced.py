from __future__ import annotations

from comfy_api.latest import io

from .multiface_refine_advanced import (
    assign_multiface_identities,
    build_multiface_character_profile,
    build_multiface_repair_job,
    build_sam31_multiface_track_plan,
    composite_multiface_candidate,
    merge_multiface_cast,
)
from .nodes_face_refine_parity_advanced import FaceRefineParityPlanIO


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Face Refine Multi-Person"
CharacterProfileIO = io.Custom("H3_T8_MULTIFACE_CHARACTER_PROFILE")
FaceCastIO = io.Custom("H3_T8_MULTIFACE_CAST")
TrackPlanIO = io.Custom("H3_T8_SAM31_MULTIFACE_TRACK_PLAN")
IdentityAssignmentIO = io.Custom("H3_T8_MULTIFACE_IDENTITY_ASSIGNMENT")
CompositeStateIO = io.Custom("H3_T8_MULTIFACE_COMPOSITE")


class MiniMaxH3FaceCharacterProfileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceCharacterProfileT8Advanced",
            display_name="MiniMax H3 Face Character Profile / 多人角色参考 (Advanced)",
            description=(
                "Builds one in-memory character profile from one or more single-person reference "
                "images. OpenCV Zoo YuNet+SFace run on CPU. The embedding is a matching aid, "
                "never persistent identity proof."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input("character_id", default="Character_A"),
                io.Image.Input("reference_images"),
                io.Combo.Input(
                    "reference_face_policy",
                    options=["dominant_face_auto", "require_single_face", "largest_face_exp"],
                    default="dominant_face_auto",
                    advanced=True,
                ),
            ],
            outputs=[
                CharacterProfileIO.Output("character_profile"),
                io.Image.Output("reference_preview"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_multiface_character_profile(**kwargs))


class MiniMaxH3FaceCastMergeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceCastMergeT8Advanced",
            display_name="MiniMax H3 Face Cast Merge / 2-3人角色表 (Advanced)",
            description=(
                "Chains two or three unique character profiles into one in-memory cast. "
                "It refuses duplicate IDs and never saves biometric embeddings to disk."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                CharacterProfileIO.Input("profile"),
                FaceCastIO.Input("previous_cast", optional=True),
            ],
            outputs=[
                FaceCastIO.Output("face_cast"),
                io.Image.Output("reference_contact_sheet"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, profile, previous_cast=None):
        return io.NodeOutput(*merge_multiface_cast(profile, previous_cast))


class MiniMaxH3SAM31MultiPersonTrackT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SAM31MultiPersonTrackT8Advanced",
            display_name="MiniMax H3 SAM3.1 Multi-Person Track / 多人分色追踪 (Advanced)",
            description=(
                "Runs current ComfyUI native SAM3.1 multiplex tracking per detected shot, caps "
                "the result at 2-3 people and emits color-coded shot-local track IDs. Default "
                "cleanup selectively unloads only the SAM model and its clones before H3 repair. "
                "Colors are not character identity."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Model.Input("model", tooltip="Load sam3.1_multiplex_fp16.safetensors."),
                io.Conditioning.Input(
                    "conditioning", tooltip="SAM3.1 CLIP text conditioning, normally: person"
                ),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Int.Input("maximum_people", default=3, min=2, max=3),
                io.Float.Input(
                    "detection_threshold", default=0.5, min=0.0, max=1.0, step=0.01
                ),
                io.Int.Input("detect_interval", default=3, min=1, max=24),
                io.Float.Input(
                    "scene_cut_threshold", default=0.28, min=0.01, max=1.0, step=0.01
                ),
                io.Combo.Input(
                    "analysis_max_side",
                    options=[512, 640, 768, 0],
                    default=640,
                    tooltip=(
                        "0 keeps source size. SAM3.1 still uses its fixed 1008-square backbone; "
                        "this mainly bounds input/preview tensors rather than backbone VRAM."
                    ),
                ),
                io.Int.Input("preview_stride", default=8, min=1, max=120, advanced=True),
                io.Combo.Input(
                    "release_policy",
                    options=["offload_sam31_after_track", "keep_loaded"],
                    default="offload_sam31_after_track",
                ),
            ],
            outputs=[
                TrackPlanIO.Output("track_plan"),
                io.Image.Output("colored_preview"),
                io.String.Output("report_json"),
                io.Int.Output("shot_count"),
                io.Int.Output("shot_local_track_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_sam31_multiface_track_plan(**kwargs))


class MiniMaxH3FaceTrackAssignT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceTrackAssignT8Advanced",
            display_name="MiniMax H3 Face Track Assign / 轨迹绑定角色 (Advanced)",
            description=(
                "Binds every shot-local SAM track to a reviewed character profile. CPU SFace "
                "suggestions are one-to-one within each shot and fail closed below similarity "
                "or margin thresholds. JSON overrides such as {\"0:0\":\"Alice\"} are authoritative."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                TrackPlanIO.Input("track_plan"),
                FaceCastIO.Input("face_cast"),
                io.Combo.Input(
                    "identity_mode",
                    options=["sface_cpu_suggest", "manual_only"],
                    default="sface_cpu_suggest",
                ),
                io.String.Input(
                    "manual_assignments_json",
                    multiline=True,
                    default="{}",
                    tooltip='Shot-local mapping, for example {"0:0":"Alice","0:1":"Bob"}.',
                ),
                io.Float.Input(
                    "minimum_similarity", default=0.40, min=-1.0, max=1.0, step=0.01
                ),
                io.Float.Input("minimum_margin", default=0.05, min=0.0, max=1.0, step=0.01),
                io.Int.Input(
                    "identity_samples_per_track", default=3, min=1, max=8, advanced=True
                ),
                io.Boolean.Input("strict_identity", default=True),
                io.Int.Input("preview_stride", default=8, min=1, max=120, advanced=True),
            ],
            outputs=[
                IdentityAssignmentIO.Output("identity_assignment"),
                io.Image.Output("assignment_preview"),
                io.String.Output("report_json"),
                io.Int.Output("track_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*assign_multiface_identities(**kwargs))


class MiniMaxH3MultiFaceRepairJobT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiFaceRepairJobT8Advanced",
            display_name="MiniMax H3 Multi-Face Repair Job / 单角色修复任务 (Advanced)",
            description=(
                "Selects one character in one shot and creates a 17n+5 Face Refine Parity job. "
                "YuNet localizes the face inside that person's SAM mask; SFace rejects identity-"
                "incompatible detections. Generate each person sequentially, then composite."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                IdentityAssignmentIO.Input("identity_assignment"),
                io.String.Input("character_id", default="Character_A"),
                io.Int.Input("shot_id", default=0, min=0, max=999),
                io.Int.Input("window_start_in_shot", default=0, min=0, max=100000),
                io.Int.Input(
                    "window_frame_count",
                    default=73,
                    min=5,
                    max=124,
                    tooltip=(
                        "Must be 5, 22, 39, 56, 73, 90, 107 or 124. "
                        "The 73-frame default is about 3.04 seconds at 24 fps. If a shot is "
                        "short by at most 16 frames, its last frame is used only as H3 context "
                        "and the padded tail is discarded during final composition."
                    ),
                ),
                io.Float.Input("crop_factor", default=2.5, min=1.2, max=8.0, step=0.1),
                io.Combo.Input(
                    "canvas_mode",
                    options=["manual_384", "manual_512", "manual_640", "auto_capped_768"],
                    default="manual_512",
                ),
                io.Int.Input("center_smooth_window", default=21, min=1, max=121, step=2),
                io.Int.Input("size_smooth_window", default=51, min=1, max=181, step=2),
                io.Combo.Input(
                    "identity_guard",
                    options=["sface_cpu", "sam_track_only_exp"],
                    default="sface_cpu",
                ),
                io.Float.Input(
                    "minimum_similarity", default=0.36, min=-1.0, max=1.0, step=0.01
                ),
                io.Int.Input(
                    "analysis_chunk_frames", default=4, min=1, max=32, advanced=True
                ),
                io.Combo.Input(
                    "crop_scale_mode",
                    options=["legacy_crop_factor", "target_face_px"],
                    default="legacy_crop_factor",
                    optional=True,
                    tooltip=(
                        "legacy_crop_factor preserves existing workflows. target_face_px "
                        "automatically resolves crop_factor so the face reaches at least about "
                        "the requested height in a manual H3 crop canvas. Source boundaries can "
                        "make it larger."
                    ),
                ),
                io.Float.Input(
                    "target_face_px",
                    default=300.0,
                    min=96.0,
                    max=512.0,
                    step=8.0,
                    optional=True,
                    tooltip=(
                        "Minimum target face height inside the H3 crop canvas. The reviewed 512px "
                        "single-person route covered about 205-312px; 300px targets its upper end."
                    ),
                ),
            ],
            outputs=[
                FaceRefineParityPlanIO.Output("face_plan"),
                io.Image.Output("base_window"),
                io.Image.Output("crops"),
                io.Image.Output("identity_references"),
                io.Image.Output("source_reference_crop"),
                io.Image.Output("preview"),
                io.String.Output("report_json"),
                io.Int.Output("absolute_start_frame"),
                io.Int.Output("window_frame_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_multiface_repair_job(**kwargs))


class MiniMaxH3MultiFaceCompositeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MultiFaceCompositeT8Advanced",
            display_name="MiniMax H3 Multi-Face Composite / 多人候选合成 (Advanced)",
            description=(
                "Applies one shot-local Face Refine candidate to the original full video and "
                "chains up to three people. Candidates are accepted by default so the example "
                "produces a repaired final video; disable accept_candidate for preview-only "
                "review. Overlapping masks still reject by default, pixels outside the audited "
                "changed mask remain bit-exact, and audio is untouched."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("base_frames"),
                io.Image.Input("candidate_window"),
                io.Mask.Input("changed_mask"),
                FaceRefineParityPlanIO.Input("face_plan"),
                io.Boolean.Input(
                    "accept_candidate",
                    default=True,
                    tooltip=(
                        "Enabled by default so the candidate reaches the final composite. Disable "
                        "only when you want a preview-only run that returns the original frames."
                    ),
                ),
                io.Combo.Input(
                    "overlap_policy",
                    options=["reject", "new_over_old_exp", "keep_old_exp"],
                    default="reject",
                ),
                CompositeStateIO.Input("previous_composite", optional=True),
            ],
            outputs=[
                io.Image.Output("composited_frames"),
                CompositeStateIO.Output("composite_state"),
                io.String.Output("report_json"),
                io.Int.Output("applied_job_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*composite_multiface_candidate(**kwargs))


MULTIFACE_REFINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceCharacterProfileT8Advanced,
    MiniMaxH3FaceCastMergeT8Advanced,
    MiniMaxH3SAM31MultiPersonTrackT8Advanced,
    MiniMaxH3FaceTrackAssignT8Advanced,
    MiniMaxH3MultiFaceRepairJobT8Advanced,
    MiniMaxH3MultiFaceCompositeT8Advanced,
]
