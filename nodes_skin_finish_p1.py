from __future__ import annotations

from comfy_api.latest import io, ui

from .nodes_multiface_refine_advanced import TrackPlanIO
from .skin_finish import PRESET_CONFIG
from .skin_finish_p1 import (
    finalize_skin_finish_video,
    run_multiface_skin_finish,
    stream_skin_finish_video,
)


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"
SkinFinishSequenceStateIO = io.Custom("H3_T8_SKIN_FINISH_SEQUENCE_STATE")


class MiniMaxH3SkinFinishMultiPersonT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishMultiPersonT8Advanced",
            display_name="MiniMax H3 Skin Finish Multi-Person / 多人肤质收尾 (Advanced)",
            description=(
                "Reuses an existing native SAM3.1 multi-person track plan without loading SAM "
                "again. Shot-local person masks are intersected with CPU YuNet face regions; "
                "small, profile, blurred or occluded faces are downweighted. A hashed causal "
                "state lets overlapped long-video chunks continue without temporal RGB averaging."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                TrackPlanIO.Input("track_plan"),
                io.Int.Input(
                    "absolute_start_frame",
                    default=0,
                    min=0,
                    max=10000000,
                    tooltip=(
                        "Absolute zero-based frame index of this chunk. The first chunk must "
                        "start at 0; later chunks may overlap a reviewed bounded tail."
                    ),
                ),
                io.Combo.Input(
                    "preset",
                    options=[name for name in PRESET_CONFIG if name in {"subtle", "oil_control"}],
                    default="subtle",
                    tooltip=(
                        "Multi-person mode intentionally exposes only shared color-neutral "
                        "finishing. Per-person hue/saturation changes are not automatic."
                    ),
                ),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input("texture_keep", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("shine_control", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "detection_threshold", default=0.45, min=0.10, max=0.95, step=0.01
                ),
                io.Float.Input(
                    "minimum_face_height_px",
                    default=24.0,
                    min=4.0,
                    max=512.0,
                    step=1.0,
                ),
                io.Float.Input(
                    "minimum_detail",
                    default=0.010,
                    min=0.001,
                    max=0.100,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Gradient-detail floor used only to downweight already blurred faces. "
                        "Skin Finish does not sharpen or reconstruct them."
                    ),
                ),
                io.Float.Input(
                    "bbox_ema_alpha",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Causal current-observation weight; resets naturally at shot-local tracks.",
                ),
                io.Int.Input("max_missing_frames", default=2, min=0, max=8, advanced=True),
                io.Boolean.Input("protect_features", default=True),
                io.Boolean.Input(
                    "include_neck",
                    default=False,
                    advanced=True,
                    tooltip="Off by default. A conservative neck strip is still clipped to the person track.",
                ),
                io.Int.Input(
                    "maximum_overlap_frames", default=8, min=0, max=32, advanced=True
                ),
                io.Int.Input("mask_feather_px", default=3, min=0, max=64, advanced=True),
                io.Int.Input(
                    "proxy_long_side", default=640, min=128, max=1280, step=32, advanced=True
                ),
                io.Int.Input("chunk_frames", default=4, min=1, max=32, advanced=True),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip="Source remains selected until the candidate has been reviewed.",
                ),
                SkinFinishSequenceStateIO.Input("previous_state", optional=True),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Passed through as the same AUDIO object. Do not concatenate duplicate "
                        "audio from overlapped chunks; use Video Finalize for file-level audio."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("used_skin_mask"),
                io.Mask.Output("rejected_mask"),
                io.Image.Output("difference"),
                SkinFinishSequenceStateIO.Output("sequence_state"),
                io.String.Output("report_json"),
                io.Int.Output("emitted_absolute_start"),
                io.Int.Output("emitted_frame_count"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*run_multiface_skin_finish(**kwargs))


class MiniMaxH3SkinFinishVideoFinalizeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
            display_name="MiniMax H3 Skin Finish Video Finalize / 原音频封装 (Advanced)",
            description=(
                "Finalizes reviewed full-length IMAGE frames against an untrimmed file-backed "
                "ComfyUI VIDEO. Video is encoded frame by frame while original compressed audio "
                "packet payloads are copied and SHA-256 verified. Unsupported codecs, trims, "
                "HDR/10-bit, geometry or packet drift fail closed."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Video.Input("source_video"),
                io.Image.Input("processed_frames"),
                io.String.Input(
                    "filename_prefix", default="MiniMaxH3/SkinFinish/skin_finish"
                ),
                io.Float.Input("crf", default=18.0, min=0.0, max=51.0, step=1.0),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "False returns the exact source VIDEO and writes nothing. Enable only "
                        "after reviewing the Skin Finish candidate."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        video, path, report, saved = finalize_skin_finish_video(**kwargs)
        if saved is None:
            return io.NodeOutput(video, path, report)
        filename, subfolder = saved
        preview = ui.PreviewVideo(
            [ui.SavedResult(filename, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(video, path, report, ui=preview)


class MiniMaxH3SkinFinishVideoStreamT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishVideoStreamT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Video Stream / 两遍低内存肤质收尾 (Advanced)"
            ),
            description=(
                "Consumes an untrimmed file-backed VIDEO without GetVideoComponents. Pass 1 "
                "retains only pinned CPU YuNet face metadata; pass 2 processes bounded frame "
                "chunks, incrementally encodes H.264 and packet-copies verified source audio. "
                "No complete IMAGE batch is materialized. Source is selected by default."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Video.Input("source_video"),
                io.Combo.Input(
                    "preset",
                    options=[
                        name for name in PRESET_CONFIG if name in {"subtle", "oil_control"}
                    ],
                    default="subtle",
                ),
                io.Float.Input("amount", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "texture_keep", default=0.90, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "shine_control", default=0.35, min=0.0, max=1.0, step=0.01
                ),
                io.Float.Input(
                    "detection_threshold", default=0.45, min=0.10, max=0.95, step=0.01
                ),
                io.Float.Input(
                    "minimum_face_height_px",
                    default=24.0,
                    min=4.0,
                    max=512.0,
                    step=1.0,
                ),
                io.Float.Input(
                    "minimum_detail",
                    default=0.010,
                    min=0.001,
                    max=0.100,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Downweights already blurred faces; this node never sharpens or "
                        "reconstructs missing detail."
                    ),
                ),
                io.Float.Input(
                    "bbox_ema_alpha",
                    default=0.55,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "scene_cut_threshold",
                    default=0.28,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input("maximum_faces", default=4, min=1, max=12, advanced=True),
                io.Int.Input("mask_feather_px", default=3, min=0, max=64, advanced=True),
                io.Int.Input(
                    "proxy_long_side", default=640, min=128, max=1280, step=32, advanced=True
                ),
                io.Int.Input("chunk_frames", default=4, min=1, max=32, advanced=True),
                io.String.Input(
                    "filename_prefix",
                    default="MiniMaxH3/SkinFinish/stream_skin_finish",
                ),
                io.Float.Input("crf", default=18.0, min=0.0, max=51.0, step=1.0),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip=(
                        "False returns the source VIDEO without analysis or file writes. "
                        "Enable only after choosing to render a review candidate."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        video, path, report, saved = stream_skin_finish_video(**kwargs)
        if saved is None:
            return io.NodeOutput(video, path, report)
        filename, subfolder = saved
        preview = ui.PreviewVideo(
            [ui.SavedResult(filename, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(video, path, report, ui=preview)


SKIN_FINISH_P1_NODE_CLASSES = [
    MiniMaxH3SkinFinishMultiPersonT8Advanced,
    MiniMaxH3SkinFinishVideoFinalizeT8Advanced,
    MiniMaxH3SkinFinishVideoStreamT8Advanced,
]
